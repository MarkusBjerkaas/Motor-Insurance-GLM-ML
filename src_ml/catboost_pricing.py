"""CatBoost-utfordrer for forventet ren egen-skadepremie."""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import GridSearchCV, GroupKFold


def prepare_catboost_features(frame, predictors, categorical_features):
    """Velg features; kategorier fylles med MISSING og konverteres til tekst."""
    features = frame.loc[:, predictors].copy()
    for column in categorical_features:
        features[column] = features[column].astype("object").where(
            features[column].notna(), "MISSING"
        )
        features[column] = features[column].astype(str)
    return features


def make_weighted_tweedie_scorer(sample_weight, power):
    """Returner scorer med vekter hentet fra valideringsradenes indeks."""

    def scorer(estimator, features, response, sample_weight=None):
        prediction = estimator.predict(features)
        weights = sample_weight.loc[features.index]
        return -mean_tweedie_deviance(
            response, prediction, sample_weight=weights, power=power
        )

    return scorer


def _make_estimator(power, seed):
    """Bygg CatBoost-estimatoren med de låste innstillingene."""
    return CatBoostRegressor(
        loss_function=f"Tweedie:variance_power={power}",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def fit_catboost_grid(
    features,
    response,
    sample_weight,
    groups,
    categorical_features,
    power,
    param_grid,
    *,
    n_splits=5,
    seed=100,
):
    """Kjør gruppe-CV, refit beste modell og lag OOF-prediksjoner."""
    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scorer = make_weighted_tweedie_scorer(sample_weight, power)
    search = GridSearchCV(
        _make_estimator(power, seed),
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        refit=True,
        error_score="raise",
        n_jobs=-1,
    )
    search.fit(
        features,
        response,
        groups=groups,
        sample_weight=sample_weight,
        cat_features=list(categorical_features),
    )

    oof_prediction = pd.Series(np.nan, index=features.index, name="oof_prediction")
    null_prediction = pd.Series(np.nan, index=features.index, name="null_prediction")
    fold_rows, splits = [], []
    for number, (train_pos, val_pos) in enumerate(cv.split(features, response, groups), 1):
        train_features, val_features = features.iloc[train_pos], features.iloc[val_pos]
        train_response, val_response = response.iloc[train_pos], response.iloc[val_pos]
        train_weight, val_weight = sample_weight.iloc[train_pos], sample_weight.iloc[val_pos]
        model = clone(search.best_estimator_)
        model.fit(
            train_features,
            train_response,
            sample_weight=train_weight,
            cat_features=list(categorical_features),
        )
        prediction = model.predict(val_features)
        oof_prediction.loc[val_features.index] = prediction
        null_mean = np.average(train_response, weights=train_weight)
        null_prediction.loc[val_features.index] = null_mean
        fold_rows.append(
            {
                "fold": number,
                "val_deviance": mean_tweedie_deviance(
                    val_response, prediction, sample_weight=val_weight, power=power
                ),
                "val_weight": val_weight.sum(),
                "val_null_deviance": mean_tweedie_deviance(
                    val_response,
                    np.full(len(val_response), null_mean),
                    sample_weight=val_weight,
                    power=power,
                ),
            }
        )
        splits.append((features.index[train_pos], features.index[val_pos]))

    return {
        "search": search,
        "best_estimator": search.best_estimator_,
        "oof_prediction": oof_prediction,
        "null_prediction": null_prediction,
        "fold_scores": pd.DataFrame(fold_rows),
        "splits": splits,
        "features": features,
        "response": response,
        "sample_weight": sample_weight,
        "groups": groups,
        "power": power,
        "param_grid": param_grid,
    }


def build_catboost_result_table(result):
    """Oppsummer valgt gridrad og OOF-resultatet i én rad."""
    response, weight, power = (
        result["response"],
        result["sample_weight"],
        result["power"],
    )
    model_deviance = mean_tweedie_deviance(
        response, result["oof_prediction"], sample_weight=weight, power=power
    )
    null_deviance = mean_tweedie_deviance(
        response, result["null_prediction"], sample_weight=weight, power=power
    )
    search = result["search"]
    return pd.DataFrame(
        [
            {
                **search.best_params_,
                "grid_mean_deviance": -search.best_score_,
                "grid_deviance_std": search.cv_results_["std_test_score"][
                    search.best_index_
                ],
                "pooled_oof_deviance": model_deviance,
                "null_oof_deviance": null_deviance,
                "oof_d2": 1 - model_deviance / null_deviance,
            }
        ]
    )


def build_final_model_table(result):
    """Hent hyperparameterne fra den refittede modellen og vis om de ligger på gridkanten."""
    model = result["best_estimator"]
    fitted = model.get_all_params()
    values = {
        "depth": fitted["depth"],
        "learning_rate": fitted["learning_rate"],
        "iterations": model.tree_count_,
        "l2_leaf_reg": fitted["l2_leaf_reg"],
    }
    rows = []
    for name, value in values.items():
        grid = sorted(result["param_grid"][name])
        rows.append(
            {
                "parameter": name,
                "verdi": value,
                "grid_min": grid[0],
                "grid_maks": grid[-1],
                "på_kant": value in (grid[0], grid[-1]),
            }
        )
    return pd.DataFrame(rows)


def _weighted_mean(values, weights):
    return np.average(values, weights=weights)


def _tweedie_deviance_residual(response, prediction, power):
    """Signer roten av hver observasjons bidrag til Tweedie-deviancen."""
    response = np.asarray(response, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    first_term = np.where(
        response == 0,
        0.0,
        response ** (2 - power) / ((1 - power) * (2 - power)),
    )
    deviance = 2 * (
        first_term
        - response * prediction ** (1 - power) / (1 - power)
        + prediction ** (2 - power) / (2 - power)
    )
    return np.sign(response - prediction) * np.sqrt(np.maximum(deviance, 0))


def _prediction_bins(prediction, n_bins=10):
    ranks = prediction.rank(method="first")
    return pd.qcut(ranks, q=min(n_bins, len(prediction)), duplicates="drop")


def plot_catboost_diagnostics(result, *, top_n=10):
    """Lag én figur: kalibrering, residualmønster og feature importance."""
    prediction = result["oof_prediction"]
    response, weight = result["response"], result["sample_weight"]
    bins = _prediction_bins(prediction)
    diagnostic_data = pd.DataFrame(
        {"prediction": prediction, "response": response, "weight": weight, "bin": bins}
    )
    calibration = diagnostic_data.groupby("bin", observed=True).apply(
        lambda part: pd.Series(
            {
                "predicted": _weighted_mean(part["prediction"], part["weight"]),
                "observed": _weighted_mean(part["response"], part["weight"]),
            }
        ),
        include_groups=False,
    )
    residual = _tweedie_deviance_residual(response, prediction, result["power"])
    residual_data = pd.DataFrame({"prediction": prediction, "residual": residual, "weight": weight, "bin": bins})
    residual_bins = residual_data.groupby("bin", observed=True).apply(
        lambda part: pd.Series(
            {
                "prediction": _weighted_mean(part["prediction"], part["weight"]),
                "residual": _weighted_mean(part["residual"], part["weight"]),
            }
        ),
        include_groups=False,
    )
    importance = pd.Series(
        result["best_estimator"].feature_importances_,
        index=result["features"].columns,
    ).nlargest(top_n).sort_values()

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(calibration["predicted"], calibration["observed"], marker="o")
    limits = [0, max(calibration.max())]
    axes[0].plot(limits, limits, linestyle="--", color="black", linewidth=1)
    axes[0].set(title="OOF-kalibrering", xlabel="Predikert ren premie", ylabel="Observert ren premie")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].plot(residual_bins["prediction"], residual_bins["residual"], marker="o")
    axes[1].set(title="Binnede Tweedie-residualer", xlabel="Predikert ren premie", ylabel="Residual")
    axes[2].barh(importance.index, importance.values)
    axes[2].set(title="Feature importance (deskriptiv)", xlabel="Importance")
    figure.tight_layout()
    return figure
