"""CatBoost-diagnostikk av residualstruktur i en låst GLM.

Metoden er låst i ``plans/glm_catboost_residual_diagnostics_plan.md`` (RD-01 til
RD-14) og fungerer likt for frekvens (Poisson), severity (Gamma) og ren
premie (Tweedie). CatBoost er et diagnostisk verktøy: den spør om GLM-ens
middelverdi fortsatt etterlater struktur en fleksibel modell kan finne, og
endrer aldri GLM-spesifikasjonen.

Foldflyten er nestet slik at ingen ytre valideringsrad noen gang påvirker
CatBoosts treningsmål:

1. Ytre fold: den låste GLM-en fittes på ytre trening og predikerer ytre validering.
2. Indre fold: fem nye gruppefolder *inne i ytre trening* gir indre OOF-prediksjoner.
3. CatBoost lærer korreksjonsfaktoren ``z = y / mu_inner_oof`` med vekt
   ``a = w * mu_inner_oof ** (2 - p)`` og scores på ytre validering som
   ``mu_glm * h_hat``.

Modulen inneholder fire blokker: (1) inputvalidering, (2) residualratio,
vekter og nested CV, (3) tabeller og (4) plott.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from patsy import PatsyError
from sklearn.metrics import mean_tweedie_deviance

from src.core_glm.glm_core import (
    check_fold_fit,
    cross_validate_glm,
    fit_glm,
    prepare_design_frame,
    prepare_fold_frames,
)
from src.core_glm.validation import build_group_folds

DEPTHS = (1, 3)
BENCHMARKS = ("glm", "constant", "catboost_depth_1", "catboost_depth_3")
DEFAULT_CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.03,
    "l2_leaf_reg": 5.0,
    "random_strength": 1.0,
}
ALLOWED_CATBOOST_PARAMS = (*DEFAULT_CATBOOST_PARAMS, "thread_count")
PREDICTION_COLUMNS = [
    "fold",
    "observed",
    "base_weight",
    "diagnostic_weight",
    *BENCHMARKS,
    "observed_ratio",
    "constant_factor",
    "catboost_factor_depth_1",
    "catboost_factor_depth_3",
]
FOLD_SCORE_COLUMNS = [
    "fold",
    "benchmark",
    "n_train",
    "n_val",
    "train_weight",
    "val_weight",
    "deviance",
    "delta_vs_glm",
    "delta_vs_constant",
    "delta_vs_depth_1",
    "relative_gain_vs_glm",
    "actual",
    "predicted",
    "ae",
    "converged",
    "valid",
]
# Tweedie-kraften hver GLM-familie må ha for at diagnostikkvekten skal være riktig
POWER_RULES = {
    "poisson": lambda power: power == 1,
    "gamma": lambda power: power == 2,
    "tweedie": lambda power: 1 < power < 2,
}


class _FoldError(Exception):
    """Fold- eller fitfeil under kjøringen: gir ``valid=False`` (ikke ``ValueError``)."""


def _check(condition, message):
    """Kast ``ValueError`` når en forutsetning brytes (før noe fittes)."""
    if not condition:
        raise ValueError(message)


# ---------------------------------------------------------------------------
# 1. Input- og foldvalidering
# ---------------------------------------------------------------------------


def _validate_inputs(
    spec,
    data,
    folds,
    feature_columns,
    categorical_columns,
    group_column,
    blocked_feature_columns,
    depths,
    inner_splits,
    permutation_repeats,
    catboost_params,
):
    """Kontroller alle forutsetninger og returner foldene skåret mot ``data.index``."""
    response, weight = spec["y"], spec["weight"]
    features = set(feature_columns)
    _check(data.index.is_unique, "data har duplikate radindekser.")
    required = {response, weight, group_column, *spec["required_columns"], *features}
    _check(
        required <= set(data.columns),
        f"Kolonner mangler: {sorted(required - set(data.columns))}",
    )
    _check(len(features) == len(feature_columns), "feature_columns må være unike.")
    _check(
        set(categorical_columns) <= features,
        "categorical_columns må være med i feature_columns.",
    )
    _check(
        features <= set(spec["required_columns"]),
        "feature_columns må være råkolonner fra spec['required_columns'].",
    )
    forbidden = features & {response, weight, group_column, *blocked_feature_columns}
    _check(
        not forbidden,
        f"Responsavledede/utelatte kolonner brukt som feature: {sorted(forbidden)}",
    )
    power_is_valid = POWER_RULES.get(spec["family"])
    _check(
        power_is_valid is not None and power_is_valid(spec["power"]),
        f"Familien {spec['family']!r} med power {spec['power']} støttes ikke.",
    )
    y, base_weight = data[response].to_numpy(float), data[weight].to_numpy(float)
    minimum_ok = (y > 0) if spec["family"] == "gamma" else (y >= 0)
    _check(
        np.isfinite(y).all() and minimum_ok.all(), "Responsen er ugyldig for familien."
    )
    _check(
        np.isfinite(base_weight).all() and (base_weight > 0).all(),
        "Basisvekten må være positiv.",
    )
    numeric = [
        column for column in feature_columns if column not in categorical_columns
    ]
    _check(
        not np.isinf(data[numeric].to_numpy(float)).any(),
        "Numeriske features har ±inf.",
    )
    _check(tuple(depths) == DEPTHS, f"depths må være {DEPTHS}.")
    _check(inner_splits >= 2, "inner_splits må være minst 2.")
    _check(permutation_repeats >= 1, "permutation_repeats må være minst 1.")
    unknown = set(catboost_params) - set(ALLOWED_CATBOOST_PARAMS)
    _check(
        not unknown,
        f"catboost_params kan bare inneholde {ALLOWED_CATBOOST_PARAMS}, fikk {sorted(unknown)}",
    )
    return _restrict_folds(folds, data, group_column, inner_splits)


def _restrict_folds(folds, data, group_column, inner_splits):
    """Skjær foldene mot ``data.index`` (som ``cross_validate_glm``) og valider dem.

    Foldene kan være bygget på en større populasjon (severity bruker foldene fra
    hele modellrammen), så indekser utenfor ``data`` ignoreres. Hver rad i
    ``data`` må derimot ligge i nøyaktig én valideringsfold og i treningen til
    alle andre, og ingen gruppe kan ligge på begge sider av en fold.
    """
    _check(len(folds) >= 2, "Minst to ytre folder kreves.")
    groups = data[group_column]
    restricted = []
    for fold in folds:
        name = fold["fold"]
        train = data.index[data.index.isin(fold["train_index"])]
        val = data.index[data.index.isin(fold["val_index"])]
        _check(
            len(train) > 0 and len(val) > 0, f"{name}: tom trening eller validering."
        )
        _check(not train.isin(val).any(), f"{name}: trening og validering overlapper.")
        _check(
            len(train) + len(val) == len(data),
            f"{name}: trening + validering dekker ikke alle rader.",
        )
        _check(
            not groups.loc[train].isin(groups.loc[val]).any(),
            f"{name}: samme {group_column} finnes i trening og validering.",
        )
        _check(
            groups.loc[train].nunique() >= inner_splits,
            f"{name}: for få grupper til indre CV.",
        )
        restricted.append({"fold": name, "train_index": train, "val_index": val})
    validation_rows = np.concatenate([fold["val_index"] for fold in restricted])
    _check(
        pd.Index(validation_rows).is_unique and len(validation_rows) == len(data),
        "Hver rad må ligge i nøyaktig én valideringsfold.",
    )
    return restricted


# ---------------------------------------------------------------------------
# 2. Residualratio, vekter og nested CV
# ---------------------------------------------------------------------------


def residual_ratio(response, prediction):
    """Returner ``y / mu`` etter domenevalidering (korrekt spesifisert modell gir 1)."""
    mu, y = np.asarray(prediction, dtype=float), np.asarray(response, dtype=float)
    _check(
        np.isfinite(mu).all() and (mu > 0).all(),
        "Prediksjonene må være endelige og positive.",
    )
    _check(
        np.isfinite(y).all() and (y >= 0).all(),
        "Responsen må være endelig og ikke-negativ.",
    )
    return response / prediction


def residual_ratio_weight(base_weight, prediction, power):
    """Returner ``w * mu ** (2 - p)``: log-linkens IRLS-vekt, gyldig for alle familier."""
    mu, w = np.asarray(prediction, dtype=float), np.asarray(base_weight, dtype=float)
    _check(
        np.isfinite(mu).all() and (mu > 0).all(),
        "Prediksjonene må være endelige og positive.",
    )
    _check(
        np.isfinite(w).all() and (w > 0).all(), "Vektene må være endelige og positive."
    )
    return base_weight * prediction ** (2 - power)


def _benchmark_table(observed, weight, predictions, power):
    """Deviance, deltaer, relativ gevinst og A/E per benchmark (fold eller pooled)."""
    deviance = pd.Series(
        {
            name: mean_tweedie_deviance(
                observed, predictions[name], sample_weight=weight, power=power
            )
            for name in BENCHMARKS
        },
        name="deviance",
    ).rename_axis("benchmark")
    table = deviance.to_frame()
    table["delta_vs_glm"] = deviance["glm"] - deviance
    table["delta_vs_constant"] = deviance["constant"] - deviance
    table["delta_vs_depth_1"] = deviance["catboost_depth_1"] - deviance
    # Null i nevneren gir NaN, aldri uendelig
    table["relative_gain_vs_glm"] = (
        table["delta_vs_glm"] / deviance["glm"] if deviance["glm"] != 0 else np.nan
    )
    table["actual"] = (weight * observed).sum()
    table["predicted"] = pd.Series(
        {n: (weight * predictions[n]).sum() for n in BENCHMARKS}
    )
    table["ae"] = table["actual"] / table["predicted"]
    return table


def _fit_outer_glm(spec, train, val, fit_kwargs, fold_name):
    """Fit den låste GLM-en på ytre trening og returner prediksjonen på ytre validering."""
    (train_design, val_design), _ = prepare_fold_frames(
        spec, train, [train, val], prepare_design_frame
    )
    result = fit_glm(spec, train_design, check_convergence=False, fit_kwargs=fit_kwargs)
    train_prediction = result.predict(train_design)
    val_prediction = pd.Series(
        np.asarray(result.predict(val_design), dtype=float), index=val.index
    )
    problems, _ = check_fold_fit(
        spec, result, train_design, val_design, train_prediction, val_prediction
    )
    if problems:
        raise _FoldError(f"{fold_name}: " + "; ".join(problems))
    return val_prediction


def _inner_oof_predictions(spec, train, inner_seed, config, fold_name):
    """Indre OOF-prediksjoner for alle rader i ytre trening (RD-03)."""
    inner_folds = build_group_folds(
        train, config["group_column"], config["inner_splits"], True, inner_seed
    )
    inner = cross_validate_glm(
        spec, train, inner_folds, fit_kwargs=config["fit_kwargs"]
    )
    if not inner["valid"] or inner["oof"].isna().any():
        reason = inner["error"] or "ufullstendig OOF-dekning"
        raise _FoldError(f"{fold_name} (indre CV): {reason}")
    return inner["oof"]


def _catboost_features(frame, config):
    """CatBoost-features uten lært tilstand: kategorier som tekst, tall uendret."""
    features = frame[list(config["feature_columns"])].copy()
    for column in features:
        if column in config["categorical_columns"]:
            features[column] = (
                features[column].astype("object").fillna("MISSING").astype(str)
            )
        else:
            features[column] = features[column].astype(
                float
            )  # NaN håndteres av CatBoost
    return features


def _fit_catboost(features, target, weight, depth, config):
    """Poisson-loss på ratioen med låste innstillinger (RD-05, RD-06, RD-09)."""
    model = CatBoostRegressor(
        **config["catboost_params"],
        depth=depth,
        loss_function="Poisson",
        random_seed=config["seed"],
        verbose=False,
        allow_writing_files=False,
        task_type="CPU",
        max_ctr_complexity=1,  # ingen kombinerte kategoriske CTR-er: depth 1 forblir additiv
    )
    pool = Pool(
        features,
        label=target,
        weight=weight,
        cat_features=list(config["categorical_columns"]),
    )
    return model.fit(pool)


def _correction_factor(model, features, config):
    """Positiv faktor ``h_hat`` på respons-skala. ``Exponent``, ikke rå log-skala."""
    pool = Pool(features, cat_features=list(config["categorical_columns"]))
    return model.predict(pool, prediction_type="Exponent")


def _permutation_rows(model, features, glm_prediction, val, base_deviance, context):
    """Held-out permutasjon: økning i deviance når én featurekolonne stokkes (RD-11)."""
    spec, config, fold_name, fold_number, depth = context
    rows = []
    for position, column in enumerate(features.columns):
        for repeat in range(config["permutation_repeats"]):
            seed = np.random.SeedSequence(
                [config["seed"], fold_number, depth, position, repeat]
            )
            permuted = features.copy()
            permuted[column] = np.random.default_rng(seed).permutation(
                features[column].to_numpy()
            )
            prediction = glm_prediction * _correction_factor(model, permuted, config)
            deviance = mean_tweedie_deviance(
                val[spec["y"]],
                prediction,
                sample_weight=val[spec["weight"]],
                power=spec["power"],
            )
            rows.append(
                {
                    "depth": depth,
                    "fold": fold_name,
                    "feature": column,
                    "repeat": repeat,
                    "deviance_increase": deviance - base_deviance,
                }
            )
    return rows


def _run_fold(spec, data, fold, number, config):
    """Kjør én ytre fold og returner dens bidrag til resultatet."""
    name, response, weight, power = (
        fold["fold"],
        spec["y"],
        spec["weight"],
        spec["power"],
    )
    train, val = data.loc[fold["train_index"]], data.loc[fold["val_index"]]
    try:
        mu_val = _fit_outer_glm(spec, train, val, config["fit_kwargs"], name)
        mu_inner = _inner_oof_predictions(
            spec, train, config["seed"] + number, config, name
        )
        z = residual_ratio(train[response], mu_inner)
        a = residual_ratio_weight(train[weight], mu_inner, power)
        diagnostic_weight = residual_ratio_weight(val[weight], mu_val, power)
        observed_ratio = residual_ratio(val[response], mu_val)
    except (ValueError, np.linalg.LinAlgError, PatsyError) as error:
        raise _FoldError(f"{name}: {error}") from error

    # Alt under er lært på ytre trening; ytre validering brukes bare til prediksjon og scoring
    x_train, x_val = _catboost_features(train, config), _catboost_features(val, config)
    constant = np.average(z, weights=a)
    models = {
        depth: _fit_catboost(x_train, z, a, depth, config) for depth in config["depths"]
    }
    factors = {
        depth: _correction_factor(m, x_val, config) for depth, m in models.items()
    }

    predictions = pd.DataFrame(
        {
            "fold": name,
            "observed": val[response],
            "base_weight": val[weight],
            "diagnostic_weight": diagnostic_weight,
            "glm": mu_val,
            "constant": mu_val * constant,
            "observed_ratio": observed_ratio,
            "constant_factor": constant,
        },
        index=val.index,
    )
    for depth, factor in factors.items():
        predictions[f"catboost_factor_depth_{depth}"] = factor
        predictions[f"catboost_depth_{depth}"] = mu_val * factor
    predictions = predictions[PREDICTION_COLUMNS]
    corrected = predictions[list(BENCHMARKS)].to_numpy(float)
    if not (np.isfinite(corrected).all() and (corrected > 0).all()):
        raise _FoldError(
            f"{name}: ikke-endelige eller ikke-positive korrigerte prediksjoner"
        )

    scores = _benchmark_table(
        val[response], val[weight], predictions, power
    ).reset_index()
    if not np.isfinite(scores["deviance"]).all():
        raise _FoldError(f"{name}: ikke-endelig deviance")
    # En fold som ikke består GLM-kontrollene avbryter hele kjøringen, så rader som
    # kommer hit er alltid konvergerte og gyldige
    scores = scores.assign(
        fold=name,
        n_train=len(train),
        n_val=len(val),
        train_weight=train[weight].sum(),
        val_weight=val[weight].sum(),
        converged=True,
        valid=True,
    )[FOLD_SCORE_COLUMNS]

    importance, shap_values, shap_base = [], {}, {}
    for depth, model in models.items():
        deviance = scores.set_index("benchmark").loc[
            f"catboost_depth_{depth}", "deviance"
        ]
        context = (spec, config, name, number, depth)
        importance += _permutation_rows(model, x_val, mu_val, val, deviance, context)
        if config["calculate_shap"]:
            pool = Pool(x_val, cat_features=list(config["categorical_columns"]))
            raw = model.get_feature_importance(
                pool, type="ShapValues"
            )  # log-skala; siste kolonne = forventet verdi
            shap_values[depth] = pd.DataFrame(
                raw[:, :-1], index=val.index, columns=x_val.columns
            )
            shap_base[depth] = pd.Series(raw[:, -1], index=val.index)
    return {
        "predictions": predictions,
        "fold_scores": scores,
        "importance": importance,
        "shap_values": shap_values,
        "shap_base": shap_base,
    }


def _invalid_result(data, config, error):
    """Komplett resultatobjekt for en feilet kjøring: ingen delvis gyldige tall."""
    return {
        "predictions": pd.DataFrame(
            np.nan, index=data.index, columns=PREDICTION_COLUMNS
        ),
        "fold_scores": pd.DataFrame(columns=FOLD_SCORE_COLUMNS),
        "permutation_importance": pd.DataFrame(),
        "shap_values": {},
        "shap_base_values": {},
        "valid": False,
        "error": error,
        "config": config,
    }


def _assemble_result(data, parts, config):
    """Sett foldbidragene sammen og kontroller eksakt dekning av ``data.index``."""
    predictions = pd.concat([part["predictions"] for part in parts])
    _check(
        predictions.index.is_unique and len(predictions) == len(data),
        "Ytre valideringsindekser dekker ikke data nøyaktig én gang.",
    )
    depth_frames = lambda key: {  # noqa: E731 - samler ett resultat per depth på tvers av folder
        depth: pd.concat([part[key][depth] for part in parts]).loc[data.index]
        for depth in (config["depths"] if config["calculate_shap"] else ())
    }
    return {
        "predictions": predictions.loc[data.index],
        "fold_scores": pd.concat(
            [part["fold_scores"] for part in parts], ignore_index=True
        ),
        "permutation_importance": pd.DataFrame(
            [r for part in parts for r in part["importance"]]
        ),
        "shap_values": depth_frames("shap_values"),
        "shap_base_values": depth_frames("shap_base"),
        "valid": True,
        "error": None,
        "config": config,
    }


def cross_validate_residual_catboost(
    spec,
    data,
    folds,
    feature_columns,
    *,
    categorical_columns=(),
    group_column="insured_id",
    blocked_feature_columns=(),
    depths=DEPTHS,
    inner_splits=5,
    fit_kwargs=None,
    catboost_params=None,
    permutation_repeats=3,
    calculate_shap=True,
    seed=100,
):
    """Kjør lekkasjesikker, nestet CatBoost-diagnostikk av en låst GLM.

    Respons, basisvekt og Tweedie-power hentes utelukkende fra ``spec``.

    Parameters
    ----------
    spec : dict
        Den låste spesifikasjonen fra ``glm_spec``.
    data : pandas.DataFrame
        Utviklingsrammen GLM-en fittes og scores på (unik indeks).
    folds : list of dict
        Ytre folder fra ``build_group_folds``. De skjæres mot ``data.index``.
    feature_columns : list of str
        Råkolonner fra ``spec["required_columns"]`` som CatBoost får som features.
    categorical_columns : tuple of str, optional
        Delmengde av ``feature_columns`` som behandles som kategorier.
    group_column : str, optional
        Gruppekolonnen for de indre foldene (samme som de ytre foldene bruker).
    blocked_feature_columns : tuple of str, optional
        Responsavledede kolonner som aldri kan være features (feiler hvis de er det).
    depths : tuple of int, optional
        Må være ``(1, 3)``: additiv referanse og mulig interaksjonsstruktur.
    inner_splits : int, optional
        Antall indre gruppefolder per ytre trening.
    fit_kwargs : dict or None, optional
        Sendes til alle ytre og indre GLM-fit (samme som i modellseleksjonen).
    catboost_params : dict or None, optional
        Overstyrer bare ``iterations``, ``learning_rate``, ``l2_leaf_reg``,
        ``random_strength`` og ``thread_count``. Alt annet er låst.
    permutation_repeats : int, optional
        Antall permutasjoner per feature, fold og depth.
    calculate_shap : bool, optional
        Om OOF-SHAP beregnes (kreves av ``plot_residual_dependence``).
    seed : int, optional
        Hovedseed for CatBoost, indre folder og permutasjoner.

    Returns
    -------
    dict
        ``predictions``, ``fold_scores``, ``permutation_importance``,
        ``shap_values``, ``shap_base_values``, ``valid``, ``error`` og ``config``.
        Kontraktbrudd gir ``ValueError`` før noe fittes. Feil under kjøringen
        gir ``valid=False`` med en foldmerket ``error``.
    """
    catboost_params = catboost_params or {}
    folds = _validate_inputs(
        spec,
        data,
        folds,
        list(feature_columns),
        categorical_columns,
        group_column,
        blocked_feature_columns,
        depths,
        inner_splits,
        permutation_repeats,
        catboost_params,
    )
    config = {
        "name": spec["name"],
        "family": spec["family"],
        "power": spec["power"],
        "feature_columns": tuple(feature_columns),
        "categorical_columns": tuple(categorical_columns),
        "group_column": group_column,
        "depths": tuple(depths),
        "inner_splits": inner_splits,
        "fit_kwargs": fit_kwargs,
        "catboost_params": {**DEFAULT_CATBOOST_PARAMS, **catboost_params},
        "permutation_repeats": permutation_repeats,
        "calculate_shap": calculate_shap,
        "seed": seed,
    }
    parts = []
    try:
        for number, fold in enumerate(folds, start=1):
            parts.append(_run_fold(spec, data, fold, number, config))
    except _FoldError as error:
        return _invalid_result(data, config, str(error))
    return _assemble_result(data, parts, config)


# ---------------------------------------------------------------------------
# 3. Oppsummering og valideringsbasert importance
# ---------------------------------------------------------------------------


def _require_valid(result):
    """Presenter aldri tall fra en ugyldig eller delvis kjøring."""
    _check(result["valid"], f"Diagnostikken er ugyldig: {result['error']}")


def build_residual_diagnostic_summary(result):
    """Én rad per benchmark med pooled OOF-score (ikke snitt av foldscorer) og foldstabilitet."""
    _require_valid(result)
    predictions, fold_scores = result["predictions"], result["fold_scores"]
    table = _benchmark_table(
        predictions["observed"],
        predictions["base_weight"],
        predictions,
        result["config"]["power"],
    )

    def better_folds(column):
        return fold_scores.groupby("benchmark")[column].apply(
            lambda s: int((s > 0).sum())
        )

    # Sammenligningen mot depth 1 gjelder bare depth 3 (maske på indeksnavn, ikke posisjon)
    depth_3_only = lambda values: values.where(values.index == "catboost_depth_3")  # noqa: E731
    table["delta_vs_depth_1"] = depth_3_only(table["delta_vs_depth_1"])
    table["better_than_glm_folds"] = better_folds("delta_vs_glm")
    table["better_than_constant_folds"] = better_folds("delta_vs_constant")
    table["better_than_depth_1_folds"] = depth_3_only(better_folds("delta_vs_depth_1"))
    table["valid"] = result["valid"]
    return table.rename(
        columns={"deviance": "pooled_oof_deviance", "ae": "oof_ae"}
    ).reset_index()[
        [
            "benchmark",
            "pooled_oof_deviance",
            "delta_vs_glm",
            "relative_gain_vs_glm",
            "delta_vs_constant",
            "delta_vs_depth_1",
            "better_than_glm_folds",
            "better_than_constant_folds",
            "better_than_depth_1_folds",
            "oof_ae",
            "valid",
        ]
    ]


def _importance_by_fold(result, depth):
    """Held-out deviance-økning per feature (rader) og fold (kolonner), snitt over repetisjoner."""
    _require_valid(result)
    _check(
        depth in result["config"]["depths"], f"depth {depth} finnes ikke i kjøringen."
    )
    importance = result["permutation_importance"]
    importance = importance[importance["depth"] == depth]
    return (
        importance.groupby(["feature", "fold"])["deviance_increase"]
        .mean()
        .unstack("fold")
    )


def build_residual_importance_table(result, *, depth=3, top_n=10):
    """Aggreger held-out permutation importance på tvers av folder (negative verdier beholdes)."""
    by_fold = _importance_by_fold(result, depth)
    table = (
        pd.DataFrame(
            {
                "mean_deviance_increase": by_fold.mean(axis=1),
                "fold_std": by_fold.std(axis=1),
                "positive_folds": (by_fold > 0).sum(axis=1),
            }
        )
        .rename_axis("feature")
        .reset_index()
    )
    ordered = table.sort_values(
        ["mean_deviance_increase", "feature"], ascending=[False, True]
    )
    return ordered.head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Plott
# ---------------------------------------------------------------------------

BENCHMARK_LABELS = {
    "constant": "Konstant korreksjon",
    "catboost_depth_1": "CatBoost dybde 1",
    "catboost_depth_3": "CatBoost dybde 3",
}
BENCHMARK_COLORS = {
    "constant": "#8a8a85",
    "catboost_depth_1": "#2a78d6",
    "catboost_depth_3": "#eb6834",
}
COLOR_GROUP_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
]  # de tre første validerte slottene


def plot_residual_diagnostic_overview(result, *, depth=3, top_n=10):
    """To paneler: foldvis deviancegevinst mot GLM og held-out importance."""
    _require_valid(result)
    fold_scores = result["fold_scores"]
    gain = fold_scores.pivot(index="fold", columns="benchmark", values="delta_vs_glm")
    gain = gain.reindex(fold_scores["fold"].unique())[list(BENCHMARK_LABELS)]

    fig, (gain_axis, importance_axis) = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.8 / len(BENCHMARK_LABELS)
    positions = np.arange(len(gain))
    for offset, (benchmark, label) in enumerate(BENCHMARK_LABELS.items()):
        gain_axis.bar(
            positions + (offset - 1) * width,
            gain[benchmark],
            width,
            label=label,
            color=BENCHMARK_COLORS[benchmark],
        )
    gain_axis.axhline(0, color="black", linewidth=0.8)
    gain_axis.set_xticks(positions, gain.index)
    gain_axis.set_ylabel("Deviancegevinst mot GLM (høyere er bedre)")
    gain_axis.set_title("Gevinst per ytre fold")
    gain_axis.legend(frameon=False, fontsize=8)

    by_fold = _importance_by_fold(result, depth)
    top = build_residual_importance_table(result, depth=depth, top_n=top_n)["feature"][
        ::-1
    ]
    rows = np.arange(len(top))
    means = by_fold.loc[top].mean(axis=1)
    importance_axis.barh(
        rows, means, color=BENCHMARK_COLORS["catboost_depth_3"], alpha=0.35
    )
    for row, feature in zip(rows, top):
        values = by_fold.loc[feature]
        importance_axis.scatter(
            values, np.full(len(values), row), s=14, color="#52514e", zorder=3
        )
    importance_axis.axvline(0, color="black", linewidth=0.8)
    importance_axis.set_yticks(rows, top)
    importance_axis.set_xlabel("Økning i deviance ved permutasjon (ytre validering)")
    importance_axis.set_title(f"Held-out importance, dybde {depth}")

    fig.suptitle(f"Residualdiagnostikk: {result['config']['name']}")
    fig.tight_layout()
    return fig


def _group_labels(values, weights, categorical, max_groups):
    """Grupper en feature for visning: nivåer (rare samlet i OTHER), enkeltverdier eller vektede kvantiler.

    Returnerer ``(labels, discrete)``. ``labels`` er en ordnet kategori; ``discrete``
    er ``True`` når gruppene er nivåer/enkeltverdier i stedet for tallintervaller.
    """
    if categorical:
        text = values.astype("object").fillna("MISSING").astype(str)
        by_support = weights.groupby(text).sum().sort_values(ascending=False)
        kept = list(by_support.index[:max_groups])
        labels = text.where(text.isin(kept), "OTHER")
        order = kept + (["OTHER"] if len(by_support) > max_groups else [])
        return pd.Series(
            pd.Categorical(labels, order, ordered=True), index=values.index
        ), True
    values = values.astype(float)
    if values.nunique() <= max_groups:
        return pd.Series(
            pd.Categorical(values, np.sort(values.unique()), ordered=True),
            index=values.index,
        ), True
    # Vektede kvantilgrenser: like mye diagnostikkvekt i hver gruppe
    order = np.argsort(values.to_numpy())
    cumulative = np.cumsum(weights.to_numpy()[order]) / weights.sum()
    edges = np.unique(
        np.interp(
            np.linspace(0, 1, max_groups + 1), cumulative, values.to_numpy()[order]
        )
    )
    return pd.Series(
        pd.cut(values, edges, include_lowest=True), index=values.index
    ), False


def _weighted_group_summary(frame):
    """Vektet snitt, spredning og støtte av OOF-SHAP per visningsgruppe."""

    def summarize(part):
        weight = part["weight"]
        mean = np.average(part["shap"], weights=weight)
        return pd.Series(
            {
                "x": np.average(part["x"], weights=weight)
                if pd.api.types.is_numeric_dtype(part["x"])
                else np.nan,  # kategorier plasseres etter nivå, ikke etter snitt
                "mean_shap": mean,
                "std_shap": np.sqrt(
                    np.average((part["shap"] - mean) ** 2, weights=weight)
                ),
                "weight": weight.sum(),
            }
        )

    return frame.groupby("x_group", observed=True)[["x", "shap", "weight"]].apply(
        summarize
    )


def plot_residual_dependence(
    result,
    data,
    feature,
    *,
    depth=3,
    color_feature=None,
    n_bins=12,
    feature_labels=None,
):
    """Vis OOF-SHAP mot én feature, eventuelt farget etter en mulig interaksjon.

    Binning, snitt og støtte vektes med ``predictions["diagnostic_weight"]``. Fargede
    kurver er en eksplorativ visning av mulig interaksjon, ikke en interaksjonstest.
    """
    _require_valid(result)
    config = result["config"]
    _check(
        config["calculate_shap"], "OOF-SHAP ble ikke beregnet (calculate_shap=False)."
    )
    _check(depth in config["depths"], f"depth {depth} finnes ikke i kjøringen.")
    for column in (feature, color_feature):
        _check(
            column is None or column in config["feature_columns"],
            f"{column} var ikke med i CatBoost-kjøringen.",
        )
    labels = feature_labels or {}
    predictions = result["predictions"]
    weights = predictions["diagnostic_weight"]
    categorical = config["categorical_columns"]

    frame = pd.DataFrame(
        {
            "x": data.loc[predictions.index, feature],
            "shap": result["shap_values"][depth][feature],
            "weight": weights,
        }
    )
    frame["x_group"], discrete = _group_labels(
        frame["x"], weights, feature in categorical, n_bins
    )
    if color_feature is None:
        frame["color_group"] = "alle"
    else:
        colors = data.loc[predictions.index, color_feature]
        frame["color_group"], _ = _group_labels(
            colors, weights, color_feature in categorical, 3
        )

    fig, axis = plt.subplots(figsize=(8, 4.2))
    color_groups = frame.groupby("color_group", observed=True)
    for index, (color_name, part) in enumerate(color_groups):
        summary = _weighted_group_summary(part)
        if (
            discrete
        ):  # nivåer/enkeltverdier på hver sin plass, forskjøvet per fargegruppe
            place = summary.index.map(frame["x_group"].cat.categories.get_loc).to_numpy(
                float
            )
            x, style = place + (index - (len(color_groups) - 1) / 2) * 0.12, "o"
        else:
            x, style = summary["x"], "o-"
        text = (
            str(color_name)
            if not isinstance(color_name, pd.Interval)
            else f"{color_name.left:.3g}–{color_name.right:.3g}"
        )
        axis.errorbar(
            x,
            summary["mean_shap"],
            yerr=summary["std_shap"],
            fmt=style,
            color=COLOR_GROUP_COLORS[index % 3],
            capsize=2,
            linewidth=1.4,
            markersize=5,
            label=text,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    if discrete:
        support = frame.groupby("x_group", observed=True)["weight"].sum()
        support = support / support.sum()
        axis.set_xticks(
            np.arange(len(support)),
            [f"{level}\n{share:.0%}" for level, share in support.items()],
        )
    axis.set_xlabel(
        labels.get(feature, feature)
        + (" (andel av diagnostikkvekt)" if discrete else "")
    )
    axis.set_ylabel("OOF-SHAP på log-korreksjonsskala")
    title = f"OOF-SHAP for {labels.get(feature, feature)} (dybde {depth})"
    if color_feature is not None:
        axis.legend(
            title=labels.get(color_feature, color_feature), frameon=False, fontsize=8
        )
        title += f", delt etter {labels.get(color_feature, color_feature)}"
    axis.set_title(title)
    fig.tight_layout()
    return fig
