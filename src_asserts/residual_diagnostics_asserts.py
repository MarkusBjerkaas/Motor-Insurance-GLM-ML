"""Kontroller av CatBoost-residualdiagnostikken (syntetiske data, ingen prosjektdata).

To nivåer:

- ``assert_residual_diagnostics_result`` kontrollerer et faktisk resultat og kan
  kalles direkte i notebooken.
- ``run_residual_diagnostics_checks`` kjører alle syntetiske tester (formler,
  forutsetninger, lekkasje, modellatferd for Poisson/Gamma/Tweedie, reproduserbarhet,
  SHAP-skala og presentasjon). Kjør den med
  ``uv run python -m src_asserts.residual_diagnostics_asserts``.

Modellatferd testes med forhåndssatt signalstyrke og romslige marginer, ikke som
eksakte tall: tilfeldig støy gir aldri nøyaktig null gevinst.
"""

import functools
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

matplotlib.use("Agg")  # ingen skjerm nødvendig

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import mean_tweedie_deviance  # noqa: E402

from src_asserts.common_glm_asserts import (  # noqa: E402
    assert_full_oof_coverage,
    assert_positive_finite,
    require,
)
from src_core_glm import residual_diagnostics as rd  # noqa: E402
from src_core_glm.glm_core import glm_spec  # noqa: E402
from src_core_glm.validation import build_group_folds  # noqa: E402

# Små, raske CatBoost-innstillinger for testene
FAST_PARAMS = {"iterations": 50, "thread_count": 1}
FEATURES = ["x1", "x2", "x3", "region"]
LOG_LINK = sm.families.links.Log()
# familie -> (statsmodels-familie, respons, vekt, power)
FAMILIES = {
    "poisson": (
        sm.families.Poisson(link=LOG_LINK),
        "claim_frequency",
        "total_exposure",
        1.0,
    ),
    "gamma": (sm.families.Gamma(link=LOG_LINK), "average_claim", "claim_count", 2.0),
    "tweedie": (
        sm.families.Tweedie(var_power=1.5, link=LOG_LINK),
        "pure_premium",
        "total_exposure",
        1.5,
    ),
}
# Sannhet utover GLM-ens additive lineære ledd
MISSPECIFICATIONS = {
    "additive": lambda f: 0.0 * f["x1"],
    "nonlinear": lambda f: 0.9 * (f["x2"].abs() > 1),  # begrenset U-form i x2
    "interaction": lambda f: 0.9 * np.sign(f["x1"]) * np.sign(f["x2"]),
}


# ---------------------------------------------------------------------------
# Syntetiske data
# ---------------------------------------------------------------------------


def make_synthetic_frame(family, scenario, n=4000, seed=7):
    """Syntetisk portefølje med kjent sannhet; ``scenario`` avgjør hva GLM-en overser."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "insured_id": rng.integers(0, n // 2, n),
            "total_exposure": rng.uniform(0.3, 1.0, n),
            "claim_count": 1 + rng.poisson(1.5, n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),  # ren støy
            "region": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
        }
    )
    region_effect = frame["region"].map({"A": 0.0, "B": 0.3, "C": -0.3})
    log_mu = 0.5 * frame["x1"] - 0.3 * frame["x2"] + region_effect
    log_mu += MISSPECIFICATIONS[scenario](frame)
    exposure, count = frame["total_exposure"], frame["claim_count"]
    if family == "poisson":
        frame["claim_frequency"] = (
            rng.poisson(np.exp(-1.5 + log_mu) * exposure) / exposure
        )
    elif (
        family == "gamma"
    ):  # gjennomsnittlig skade over ``count`` skader, formparameter 2
        frame["average_claim"] = rng.gamma(2 * count, np.exp(7 + log_mu) / (2 * count))
    else:  # Tweedie: Poisson-antall av Gamma-skader, uttrykt som rate per eksponering
        claims = rng.poisson(np.exp(-1.5 + log_mu) * exposure)
        total = rng.gamma(np.maximum(2 * claims, 1e-9), 1000 / 2) * (claims > 0)
        frame["pure_premium"] = total / exposure
    # Ikke-sammenhengende, stokket indeks: bevares nøyaktig av diagnostikken
    frame.index = rng.permutation(np.arange(n)) + 1000
    return frame


def build_case(family, scenario):
    """Returner ``(spec, frame, folds)`` for en syntetisk familie og feilspesifikasjon."""
    glm_family, response, weight, power = FAMILIES[family]
    frame = make_synthetic_frame(family, scenario)
    spec = glm_spec(
        f"{family}_{scenario}", FEATURES, response, frame, glm_family, weight, power
    )
    return spec, frame, build_group_folds(frame, "insured_id", 5, True, 100)


def run_case(spec, frame, folds, **kwargs):
    """Kjør diagnostikken med raske testinnstillinger (kan overstyres)."""
    options = {
        "categorical_columns": ("region",),
        "catboost_params": FAST_PARAMS,
        "permutation_repeats": 2,
        **kwargs,
    }
    return rd.cross_validate_residual_catboost(spec, frame, folds, FEATURES, **options)


@functools.cache
def cached_case(family, scenario):
    """Kjør hvert (familie, scenario) én gang og del resultatet mellom testene."""
    spec, frame, folds = build_case(family, scenario)
    return spec, frame, folds, run_case(spec, frame, folds)


def expect_value_error(function, needle=""):
    """Kontroller at ``function`` gir ``ValueError`` som nevner ``needle``."""
    try:
        function()
    except ValueError as error:
        require(needle in str(error), f"Feilmeldingen mangler {needle!r}: {error}")
        return
    raise AssertionError(
        f"Forventet ValueError ({needle!r}), men ingen feil ble kastet."
    )


# ---------------------------------------------------------------------------
# Resultatkontroll (kan kalles i notebooken)
# ---------------------------------------------------------------------------


def assert_residual_diagnostics_result(result, data, folds):
    """Gyldig kjøring, full dekning og positive, endelige faktorer på hver ytre valideringsrad."""
    require(
        result["valid"] and result["error"] is None,
        f"Ugyldig diagnostikk: {result['error']}",
    )
    predictions = result["predictions"]
    assert_full_oof_coverage(predictions["glm"], data.index, "glm")
    require(predictions.notna().all().all(), "predictions har manglende verdier.")
    for column in [
        *rd.BENCHMARKS,
        "catboost_factor_depth_1",
        "catboost_factor_depth_3",
    ]:
        assert_positive_finite(predictions[column], column)
    # Hver rad hører til nøyaktig den ytre folden den ble validert i
    for fold in folds:
        rows = predictions.index[predictions["fold"] == fold["fold"]]
        require(
            set(rows) == set(data.index.intersection(fold["val_index"])),
            f"{fold['fold']}: prediksjonene dekker ikke foldens valideringsrader.",
        )
    scores = result["fold_scores"]
    require(
        scores["valid"].all() and scores["converged"].all(),
        "Ugyldig eller ikke-konvergert fold.",
    )
    require(
        len(scores) == len(folds) * len(rd.BENCHMARKS)
        and np.isfinite(scores["deviance"]).all(),
        "fold_scores mangler benchmarkrader eller har ikke-endelig deviance.",
    )
    for depth, shap_values in result["shap_values"].items():
        require(
            shap_values.index.equals(predictions.index),
            f"SHAP (depth {depth}) har feil indeks.",
        )
        require(
            np.isfinite(shap_values.to_numpy()).all(),
            f"SHAP (depth {depth}) er ikke endelig.",
        )


# ---------------------------------------------------------------------------
# 1. Rene beregningstester
# ---------------------------------------------------------------------------


def check_ratio_and_weight_formulas():
    """Håndregnet z og a for p = 1, 1.5 og 2, y = 0, bevart indeks og ugyldige input."""
    index = pd.Index([10, 3, 7])
    y = pd.Series([0.0, 2.0, 3.0], index=index)
    mu = pd.Series([2.0, 4.0, 1.0], index=index)
    w = pd.Series([1.0, 0.5, 2.0], index=index)
    z = rd.residual_ratio(y, mu)
    require(
        np.allclose(z, [0.0, 0.5, 3.0]) and z.index.equals(index), "z = y/mu er feil."
    )
    expected = {1.0: [2.0, 2.0, 2.0], 1.5: [2**0.5, 1.0, 2.0], 2.0: [1.0, 0.5, 2.0]}
    for power, values in expected.items():
        a = rd.residual_ratio_weight(w, mu, power)
        require(
            np.allclose(a, values) and a.index.equals(index),
            f"a = w*mu^(2-p) er feil for p={power}.",
        )
    for bad_mu in ([0.0, 1, 1], [-1.0, 1, 1], [np.nan, 1, 1], [np.inf, 1, 1]):
        expect_value_error(lambda: rd.residual_ratio(y, pd.Series(bad_mu, index=index)))
    expect_value_error(lambda: rd.residual_ratio(y * -1, mu))
    for bad_w in ([0.0, 1, 1], [-1.0, 1, 1], [np.nan, 1, 1]):
        expect_value_error(
            lambda: rd.residual_ratio_weight(pd.Series(bad_w, index=index), mu, 1.0)
        )


def check_benchmark_arithmetic():
    """Konstant c, pooled deviance, deltaer og A/E mot uavhengig håndregning (Poisson)."""
    y, w = np.array([0.0, 1.0, 4.0]), np.array([1.0, 2.0, 1.0])
    mu = np.array([1.0, 2.0, 2.0])
    z, a = y / mu, w * mu  # Poisson: a = w * mu
    c = (a * z).sum() / a.sum()
    require(
        np.isclose(c, (w * y).sum() / (w * mu).sum()), "c skal være A/E for Poisson."
    )
    predictions = pd.DataFrame(
        {
            "glm": mu,
            "constant": mu * c,
            "catboost_depth_1": mu,
            "catboost_depth_3": mu * 1.1,
        }
    )
    table = rd._benchmark_table(y, w, predictions, 1.0)

    def poisson_deviance(pred):  # 2 * (y log(y/mu) - (y - mu)), med 0 log 0 = 0
        term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / pred), 0.0)
        return np.average(2 * (term - (y - pred)), weights=w)

    for name in rd.BENCHMARKS:
        require(
            np.isclose(
                table.loc[name, "deviance"], poisson_deviance(predictions[name])
            ),
            f"deviance {name}",
        )
    glm_deviance = poisson_deviance(mu)
    require(
        np.isclose(
            table.loc["constant", "delta_vs_glm"],
            glm_deviance - poisson_deviance(mu * c),
        ),
        "delta",
    )
    require(
        np.isclose(
            table.loc["constant", "relative_gain_vs_glm"],
            table.loc["constant", "delta_vs_glm"] / glm_deviance,
        ),
        "relativ gevinst",
    )
    require(np.isclose(table.loc["glm", "ae"], (w * y).sum() / (w * mu).sum()), "A/E")
    require(
        np.isclose(table.loc["constant", "ae"], 1.0),
        "A/E for konstantkorrigert modell skal være 1.",
    )


# ---------------------------------------------------------------------------
# 2. Fold-, lekkasje- og feiltester
# ---------------------------------------------------------------------------


def check_input_contracts():
    """Alle kontraktbrudd gir ``ValueError`` før noe fittes."""
    spec, frame, folds = build_case("poisson", "additive")
    gamma_spec, gamma_frame, gamma_folds = build_case("gamma", "additive")
    call = lambda s=spec, f=frame, fo=folds, **kw: run_case(s, f, fo, **kw)  # noqa: E731

    expect_value_error(lambda: call(f=pd.concat([frame, frame.iloc[:1]])), "duplikate")
    expect_value_error(lambda: call(blocked_feature_columns=("x3",)), "x3")
    expect_value_error(lambda: call(depths=(1,)), "depths")
    expect_value_error(lambda: call(catboost_params={"depth": 6}), "depth")
    expect_value_error(lambda: call(s={**spec, "power": 1.5}), "power")
    expect_value_error(lambda: call(f=frame.assign(total_exposure=0.0)), "Basisvekten")
    expect_value_error(
        lambda: call(
            s=gamma_spec, f=gamma_frame.assign(average_claim=0.0), fo=gamma_folds
        ),
        "Responsen",
    )

    first = folds[0]
    row_in_val = first["val_index"][0]
    same_group = frame.index[
        frame["insured_id"].eq(frame.loc[row_in_val, "insured_id"])
    ]
    partner = [i for i in same_group if i in first["val_index"] and i != row_in_val]
    broken = {
        "overlapper": {
            **first,
            "train_index": first["train_index"].append(pd.Index([row_in_val])),
        },
        "dekker ikke": {**first, "val_index": first["val_index"][1:]},
    }
    if partner:  # samme person i både trening og validering
        moved = first["val_index"].drop(partner[0])
        broken["samme insured_id"] = {
            **first,
            "train_index": first["train_index"].append(pd.Index([partner[0]])),
            "val_index": moved,
        }
    for needle, fold in broken.items():
        expect_value_error(lambda: call(fo=[fold, *folds[1:]]), needle)
    expect_value_error(lambda: call(fo=[folds[0], folds[0], *folds[2:]]), "")


def check_no_outcome_leakage():
    """Ytre validering når aldri GLM-, CatBoost- eller indre fit, og responsen påvirker ikke foldens prediksjoner."""
    spec, frame, folds = build_case("poisson", "nonlinear")
    events = []
    outer_fit, inner_fit, catboost_fit = rd.fit_glm, rd.cross_validate_glm, rd._fit_catboost

    def spy_outer(spec_, data, *args, **kwargs):
        events.append(("ytre GLM", set(data.index)))
        return outer_fit(spec_, data, *args, **kwargs)

    def spy_inner(spec_, data, *args, **kwargs):
        events.append(("indre GLM", set(data.index)))
        return inner_fit(spec_, data, *args, **kwargs)

    def spy_catboost(features, *args, **kwargs):
        events.append(("CatBoost", set(features.index)))
        return catboost_fit(features, *args, **kwargs)

    with (
        mock.patch.object(rd, "fit_glm", spy_outer),
        mock.patch.object(rd, "cross_validate_glm", spy_inner),
        mock.patch.object(rd, "_fit_catboost", spy_catboost),
    ):
        result = run_case(spec, frame, folds, calculate_shap=False)
    validation_sets = [set(f["val_index"]) for f in folds]
    require(len(events) == len(folds) * 4, f"Uventet antall fit: {len(events)}.")
    fold_events = [events[i : i + 4] for i in range(0, len(events), 4)]
    for fold_number, (fold, group) in enumerate(zip(folds, fold_events)):
        for name, index in group:
            require(
                not index & validation_sets[fold_number],
                f"{fold['fold']}: {name} har sett ytre valideringsrader.",
            )
    require(result["valid"], result["error"])

    # Endre bare responsen i én valideringsfold: dens prediksjoner skal ikke endres
    target = folds[2]["val_index"]
    changed = frame.copy()
    changed.loc[target, spec["y"]] = changed.loc[target, spec["y"]] * 3 + 1
    rerun = run_case(spec, changed, folds, calculate_shap=False)
    columns = ["glm", "constant", "catboost_depth_1", "catboost_depth_3"]
    same = np.allclose(
        result["predictions"].loc[target, columns],
        rerun["predictions"].loc[target, columns],
    )
    require(
        same,
        "Prediksjonene i en fold endres når bare foldens egen respons endres (lekkasje).",
    )
    scored = result["fold_scores"].query("fold == @folds[2]['fold']")["deviance"]
    rerun_scored = rerun["fold_scores"].query("fold == @folds[2]['fold']")["deviance"]
    require(
        not np.allclose(scored, rerun_scored),
        "Scoren skal derimot avhenge av responsen.",
    )


def check_missing_values_and_unseen_levels():
    """Manglende verdier håndteres som i GLM-en; ukjente nivåer gir ugyldig resultat, ikke stille lekkasje."""
    spec, frame, folds = build_case("poisson", "additive")
    missing = frame.copy()
    missing.loc[missing.sample(300, random_state=1).index, "region"] = np.nan
    missing.loc[missing.sample(300, random_state=2).index, "x1"] = np.nan
    spec_missing = glm_spec(
        "missing",
        FEATURES,
        spec["y"],
        missing,
        FAMILIES["poisson"][0],
        spec["weight"],
        1.0,
    )
    result = run_case(spec_missing, missing, folds, calculate_shap=False)
    require(
        result["valid"],
        f"Manglende verdier i trening og validering skal håndteres: {result['error']}",
    )

    unseen = frame.copy()
    only_val_group = unseen.loc[folds[0]["val_index"][0], "insured_id"]
    unseen.loc[unseen["insured_id"] == only_val_group, "region"] = "Z"
    result = run_case(spec, unseen, folds, calculate_shap=False)
    require(
        not result["valid"] and "Z" in result["error"],
        "Nivå kun i validering skal gi ugyldig resultat.",
    )


def check_incomplete_inner_oof_invalidates_result():
    """Ufullstendig eller feilet indre OOF gir ugyldig resultat uten delvis gyldige tall."""
    spec, frame, folds = build_case("poisson", "additive")
    real = rd.cross_validate_glm

    def with_holes(*args, **kwargs):
        inner = real(*args, **kwargs)
        inner["oof"] = inner["oof"].copy()
        inner["oof"].iloc[0] = np.nan
        return inner

    def failed(*args, **kwargs):
        return {**real(*args, **kwargs), "valid": False, "error": "indre feil"}

    for replacement in (with_holes, failed):
        with mock.patch.object(rd, "cross_validate_glm", replacement):
            result = run_case(spec, frame, folds)
        require(
            not result["valid"] and "indre CV" in result["error"],
            "Indre feil skal gi valid=False.",
        )
        require(
            result["predictions"].isna().all().all() and result["fold_scores"].empty,
            "Ugyldig resultat skal være tomt.",
        )
        expect_value_error(
            lambda: rd.build_residual_diagnostic_summary(result), "ugyldig"
        )


# ---------------------------------------------------------------------------
# 3. Modellatferd på syntetiske data
# ---------------------------------------------------------------------------


def summary_of(result):
    """Summary indeksert på benchmark."""
    return rd.build_residual_diagnostic_summary(result).set_index("benchmark")


def top_features(result, depth, n=2):
    """De ``n`` featurene med høyest held-out importance."""
    return set(
        rd.build_residual_importance_table(result, depth=depth, top_n=n)["feature"]
    )


def check_additive_model_gives_no_gain():
    """Korrekt spesifisert additiv GLM: ingen tydelig CatBoost-gevinst."""
    _, _, _, result = cached_case("poisson", "additive")
    summary = summary_of(result)
    for benchmark in ("catboost_depth_1", "catboost_depth_3"):
        require(
            summary.loc[benchmark, "relative_gain_vs_glm"] < 0.01,
            f"{benchmark}: uventet stor gevinst på korrekt modell.",
        )
    require(
        abs(summary.loc["constant", "relative_gain_vs_glm"]) < 0.005,
        "Konstantkorreksjonen skal være ubetydelig.",
    )


def check_omitted_nonlinearity_is_found_by_depth_1():
    """Utelatt ikke-lineær effekt i x2: tydelig gevinst og x2 øverst på importance."""
    _, _, _, result = cached_case("poisson", "nonlinear")
    summary = summary_of(result)
    require(
        summary.loc["catboost_depth_1", "relative_gain_vs_glm"] > 0.02,
        "Depth 1 finner ikke ikke-lineariteten.",
    )
    require(
        summary.loc["catboost_depth_1", "better_than_glm_folds"] == 5,
        "Gevinsten skal være stabil i alle folder.",
    )
    require(top_features(result, 1, 1) == {"x2"}, "x2 skal ha høyest importance.")


def check_interaction_needs_depth_3():
    """Ren interaksjon x1 x x2: depth 3 slår depth 1, og begge variablene rangeres øverst."""
    _, _, _, result = cached_case("poisson", "interaction")
    summary = summary_of(result)
    require(
        summary.loc["catboost_depth_3", "delta_vs_depth_1"] > 0,
        "Depth 3 skal slå depth 1.",
    )
    require(
        summary.loc["catboost_depth_3", "relative_gain_vs_glm"] > 0.02,
        "Depth 3 finner ikke interaksjonen.",
    )
    require(
        summary.loc["catboost_depth_1", "relative_gain_vs_glm"]
        < 0.5 * summary.loc["catboost_depth_3", "relative_gain_vs_glm"],
        "Depth 1 skal ikke finne en ren interaksjon.",
    )
    require(
        top_features(result, 3, 2) == {"x1", "x2"}, "x1 og x2 skal rangeres øverst."
    )


def check_pure_level_error_is_taken_by_constant():
    """Ren nivåfeil (GLM 20 % for lav): konstant benchmark tar hoveddelen av gevinsten."""
    spec, frame, folds = build_case("poisson", "additive")
    real_fit = rd.fit_glm

    class ScaledResult:  # GLM-resultat med alle prediksjoner skalert ned
        def __init__(self, result):
            self._result = result

        def predict(self, data):
            return self._result.predict(data) * 0.8

        def __getattr__(self, name):
            return getattr(self._result, name)

    scaled_fit = lambda *args, **kwargs: ScaledResult(real_fit(*args, **kwargs))  # noqa: E731
    with (
        mock.patch.object(rd, "fit_glm", scaled_fit),
        mock.patch("src_core_glm.glm_core.fit_glm", scaled_fit),
    ):
        result = run_case(spec, frame, folds, calculate_shap=False)
    require(result["valid"], result["error"])
    summary = summary_of(result)
    require(
        summary.loc["constant", "relative_gain_vs_glm"] > 0.005,
        "Nivåfeilen skal gi målbar gevinst.",
    )
    require(
        abs(summary.loc["constant", "oof_ae"] - 1) < 0.01,
        "Konstantkorreksjonen skal rette A/E.",
    )
    level_gain = summary.loc["constant", "delta_vs_glm"]
    for benchmark in ("catboost_depth_1", "catboost_depth_3"):
        extra = summary.loc[benchmark, "delta_vs_constant"]
        require(extra < 0.5 * level_gain, f"{benchmark}: mesteparten av gevinsten skal være rent nivå.")


def check_gamma_and_tweedie_families():
    """Gamma (vekt = skadeantall) og Tweedie: gyldig, positive faktorer og signal ved utelatt struktur."""
    for family in ("gamma", "tweedie"):
        spec, frame, folds, additive = cached_case(family, "additive")
        assert_residual_diagnostics_result(additive, frame, folds)
        require(
            summary_of(additive).loc["catboost_depth_3", "relative_gain_vs_glm"] < 0.005,
            f"{family}: uventet gevinst uten utelatt struktur.",
        )
        _, _, _, nonlinear = cached_case(family, "nonlinear")
        depth_1 = summary_of(nonlinear).loc["catboost_depth_1"]
        require(
            depth_1["relative_gain_vs_glm"] > 0.005 and depth_1["better_than_glm_folds"] == 5,
            f"{family}: finner ikke utelatt ikke-lineær effekt.",
        )
        require(
            "x2" in top_features(nonlinear, 1, 2),
            f"{family}: x2 skal ligge høyt på importance.",
        )
        power = FAMILIES[family][3]
        predictions = nonlinear["predictions"]
        expected = predictions["base_weight"] * predictions["glm"] ** (2 - power)
        require(
            np.allclose(predictions["diagnostic_weight"], expected),
            f"{family}: diagnostikkvekt feil.",
        )


def check_reproducibility():
    """Samme seed gir identiske prediksjoner, importance og SHAP."""
    spec, frame, folds = build_case("poisson", "nonlinear")
    first = run_case(spec, frame, folds)
    second = run_case(spec, frame, folds)
    pd.testing.assert_frame_equal(first["predictions"], second["predictions"])
    pd.testing.assert_frame_equal(
        first["permutation_importance"], second["permutation_importance"]
    )
    pd.testing.assert_frame_equal(first["shap_values"][3], second["shap_values"][3])


def check_shap_scale():
    """SHAP + basisverdi = rå log-prediksjon, dvs. log av korreksjonsfaktoren (ikke rå skala blandet med respons)."""
    _, _, _, result = cached_case("poisson", "nonlinear")
    for depth in rd.DEPTHS:
        raw = (
            result["shap_values"][depth].sum(axis=1) + result["shap_base_values"][depth]
        )
        log_factor = np.log(result["predictions"][f"catboost_factor_depth_{depth}"])
        require(
            np.allclose(raw, log_factor, atol=1e-6),
            f"depth {depth}: SHAP er ikke på log-skala.",
        )


# ---------------------------------------------------------------------------
# 4. Presentasjonstester
# ---------------------------------------------------------------------------


def check_summary_and_importance_tables():
    """Forventede kolonner/rader, pooled score direkte over OOF-radene, deterministisk og signert importance."""
    _, _, _, result = cached_case("poisson", "nonlinear")
    summary = rd.build_residual_diagnostic_summary(result)
    require(
        list(summary["benchmark"]) == list(rd.BENCHMARKS),
        "Summary skal ha fire benchmarkrader.",
    )
    predictions = result["predictions"]
    pooled = mean_tweedie_deviance(
        predictions["observed"],
        predictions["glm"],
        sample_weight=predictions["base_weight"],
        power=1.0,
    )
    require(
        np.isclose(summary.loc[0, "pooled_oof_deviance"], pooled),
        "Pooled deviance skal beregnes direkte over OOF-radene.",
    )
    require(
        summary["delta_vs_depth_1"].isna().iloc[:3].all(),
        "delta_vs_depth_1 gjelder bare depth 3.",
    )

    table = rd.build_residual_importance_table(result, depth=3, top_n=10)
    require(
        table.equals(rd.build_residual_importance_table(result, depth=3, top_n=10)),
        "Deterministisk sortering.",
    )
    require(
        table["mean_deviance_increase"].is_monotonic_decreasing,
        "Importance skal være sortert synkende.",
    )
    require(len(table) == len(FEATURES), "Importance skal ha én rad per feature.")

    # Håndbygd resultat: negative verdier beholdes, og like verdier sorteres på navn
    rows = [("b", 0.2), ("a", 0.2), ("c", -0.1)]
    importance = pd.DataFrame(
        [
            {"depth": 3, "fold": fold, "feature": feature, "repeat": 0, "deviance_increase": value}
            for fold in ("gruppe_1", "gruppe_2")
            for feature, value in rows
        ]
    )
    fake = {"valid": True, "error": None, "config": {"depths": (1, 3)}, "permutation_importance": importance}
    ranked = rd.build_residual_importance_table(fake, depth=3, top_n=10)
    require(list(ranked["feature"]) == ["a", "b", "c"], "Sortering skal være synkende, deretter på navn.")
    require(np.isclose(ranked.loc[2, "mean_deviance_increase"], -0.1), "Negative verdier skal beholdes.")
    require(list(ranked["positive_folds"]) == [2, 2, 0], "positive_folds er feil.")


def check_plots():
    """Oversikt: én figur med to akser. Dependence: numerisk og kategorisk feature, med og uten fargefeature."""
    spec, frame, folds, result = cached_case("poisson", "interaction")
    before = set(plt.get_fignums())
    overview = rd.plot_residual_diagnostic_overview(result)
    require(len(overview.axes) == 2, "Oversiktsplottet skal ha to akser.")
    for feature, color in (
        ("x1", None),
        ("x1", "x2"),
        ("region", None),
        ("x2", "region"),
    ):
        figure = rd.plot_residual_dependence(
            result, frame, feature, color_feature=color
        )
        require(
            len(figure.axes) == 1, f"Dependence-plott for {feature} skal ha én akse."
        )
    expect_value_error(
        lambda: rd.plot_residual_dependence(result, frame, "ukjent"), "ukjent"
    )
    plt.close("all")
    require(set(plt.get_fignums()) == before, "Figurer ble ikke lukket.")


# ---------------------------------------------------------------------------
# Hovedfunksjon
# ---------------------------------------------------------------------------

CHECKS = [
    check_ratio_and_weight_formulas,
    check_benchmark_arithmetic,
    check_input_contracts,
    check_no_outcome_leakage,
    check_missing_values_and_unseen_levels,
    check_incomplete_inner_oof_invalidates_result,
    check_additive_model_gives_no_gain,
    check_omitted_nonlinearity_is_found_by_depth_1,
    check_interaction_needs_depth_3,
    check_pure_level_error_is_taken_by_constant,
    check_gamma_and_tweedie_families,
    check_reproducibility,
    check_shap_scale,
    check_summary_and_importance_tables,
    check_plots,
]


def run_residual_diagnostics_checks():
    """Kjør alle syntetiske kontroller og skriv én linje per bestått test."""
    for check in CHECKS:
        check()
        print(f"OK  {check.__name__}")
    print(f"Alle {len(CHECKS)} kontroller bestått.")


if __name__ == "__main__":
    run_residual_diagnostics_checks()
