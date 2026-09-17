"""Separat ABESS-diagnostikk for frekvensmodellen.

Modulen er bevisst ikke koblet til fase-1s kandidatregister eller modell-ID-er.
Den bruker den allerede låste design-/CV-kontrakten via dependency injection fra
``glm_pricing_models.py``: samme modellramme, fem gruppefolder, Patsy-koding,
imputasjon og OOF-metrikk. ABESS-resultatene er derfor en eksplorativ
utvidelse, ikke en ny hovedmodell.

ABESS 0.4.11 har ikke et offset-argument for ``PoissonRegression``. For
``alpha=0`` estimeres derfor frekvensen med ``y=N/e`` og ``sample_weight=e``.
Det er ekvivalent med Poisson for antall med ``log(e)`` som offset, opp til ledd
i likelihooden som ikke avhenger av parameterne. ``predict`` gir frekvenser;
forventet antall er alltid ``exposure * prediction``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from abess import __version__ as ABESS_VERSION
from abess.linear import PoissonRegression
from sklearn.metrics import mean_tweedie_deviance

from src.phase_2.frequency_tables import (
    build_actual_expected_table,
    paired_deviance_gain,
)

REQUIRED_ABESS_VERSION = "0.4.11"
RESPONSE_COLUMN = "claim_frequency"
EXPOSURE_COLUMN = "total_exposure"
CLAIMS_COLUMN = "property_claims"
INSURED_ID_COLUMN = "insured_id"

# Rekkefølgen er låst før kjøring. De to første gruppene er obligatoriske;
# interseptet håndteres av ABESS og er ikke en selekterbar gruppe.
BLOCKS = (
    "policy_type",
    "year",
    "driver_age",
    "log_vehicle_value",
    "performance_hp_per_tonne",
    "fuel_type",
    "circulation_area",
    "municipality_type",
    "payment_frequency",
    "business_type",
    "vehicle_brand_pooled",
    "seats_group",
)
MANDATORY_BLOCKS = ("policy_type", "year")
OPTIONAL_BLOCKS = BLOCKS[len(MANDATORY_BLOCKS) :]

BLOCK_LABELS = {
    "policy_type": "produkt",
    "year": "år",
    "driver_age": "føreralder",
    "log_vehicle_value": "log(bilverdi)",
    "performance_hp_per_tonne": "ytelse",
    "fuel_type": "drivstoff",
    "circulation_area": "urban/rural",
    "municipality_type": "kommune",
    "payment_frequency": "betalingsfrekvens",
    "business_type": "NB/P",
    "vehicle_brand_pooled": "poolingsdefinert bilmerke",
    "seats_group": "setekategori",
}

# Begge spesifikasjonene bruker prosjektets eksisterende Patsy-syntaks.
CANDIDATE_SETUPS = {
    "A": {
        "label": "A: lineær ytelse",
        "terms": {
            "driver_age": "cr(driver_age, df=3, constraints='center')",
            "log_vehicle_value": "log_vehicle_value",
            "performance_hp_per_tonne": "performance_hp_per_tonne",
        },
    },
    "B": {
        "label": "B: spline for ytelse",
        "terms": {
            "driver_age": "cr(driver_age, df=3, constraints='center')",
            "log_vehicle_value": "log_vehicle_value",
            "performance_hp_per_tonne": (
                "cr(performance_hp_per_tonne, df=3, constraints='center')"
            ),
        },
    },
}

DEFAULT_CALIBRATION_SEGMENTS = (
    "portefølje",
    "prediksjonsdesil",
    "policy_type",
    "year",
    "business_type",
    "payment_frequency",
    "circulation_area",
    "municipality_type",
    "fuel_type",
    "driver_age",
    "log_vehicle_value",
    "performance_hp_per_tonne",
    "vehicle_brand_pooled",
    "seats_group",
)


def verify_abess_poisson_equivalence(seed: int = 20260917) -> dict:
    """Kontroller installert ABESS mot count-Poisson med offset syntetisk.

    Testen bruker tre sammenhengende grupper der de to første er obligatoriske.
    Alle tre grupper inngår med ``support_size=3``. Dermed kontrolleres både
    rate+vekt/offset-ekvivalensen, gruppekodingen og ``always_select`` uten å
    bruke prosjektdata.
    """
    if ABESS_VERSION != REQUIRED_ABESS_VERSION:
        raise RuntimeError(
            f"Krever abess=={REQUIRED_ABESS_VERSION}, fant {ABESS_VERSION}"
        )

    rng = np.random.default_rng(seed)
    n_rows = 12_000
    x = rng.normal(size=(n_rows, 4))
    exposure = rng.uniform(0.1, 1.5, size=n_rows)
    coefficients = np.array([0.35, -0.22, 0.18, 0.04])
    frequency = np.exp(-1.25 + x @ coefficients)
    claims = rng.poisson(exposure * frequency)
    groups = np.array([0, 0, 1, 2], dtype=np.int32)

    abess_fit = _fit_abess(
        x,
        claims / exposure,
        exposure,
        groups,
        support_size=3,
        always_select=np.array([0, 1], dtype=np.int32),
    )
    offset_fit = sm.GLM(
        claims,
        sm.add_constant(x),
        family=sm.families.Poisson(),
        offset=np.log(exposure),
    ).fit()
    abess_parameters = np.r_[abess_fit.intercept_, abess_fit.coef_]
    max_parameter_difference = float(
        np.max(np.abs(abess_parameters - np.asarray(offset_fit.params)))
    )
    max_prediction_difference = float(
        np.max(
            np.abs(
                abess_fit.predict(x)
                # Statsmodels-prediksjonen er forventet *antall* når offset
                # inngår. Del på eksponeringen før den sammenlignes med
                # ABESS' predikerte frekvens.
                - offset_fit.predict(sm.add_constant(x), offset=np.log(exposure))
                / exposure
            )
        )
    )
    selected_groups = _selected_group_indices(abess_fit.coef_, groups)
    if selected_groups != {0, 1, 2}:
        raise AssertionError(f"Syntetisk gruppetest feilet: {selected_groups=}")
    if max_parameter_difference > 1e-5 or max_prediction_difference > 1e-5:
        raise AssertionError(
            "ABESS rate+vekt er ikke numerisk ekvivalent med count+offset: "
            f"parameteravvik={max_parameter_difference:.2e}, "
            f"prediksjonsavvik={max_prediction_difference:.2e}"
        )
    return {
        "abess_version": ABESS_VERSION,
        "max_parameter_difference": max_parameter_difference,
        "max_prediction_difference": max_prediction_difference,
        "selected_groups": sorted(selected_groups),
    }


def run_frequency_abess_diagnostic(
    frame: pd.DataFrame,
    folds: Sequence[dict],
    glm_spec: Callable,
    prepare_design_frame: Callable,
    frequency_target: dict,
    locked_glm_oof: pd.Series,
    *,
    locked_glm_name: str,
    calibration_segments: Sequence[str] = DEFAULT_CALIBRATION_SEGMENTS,
) -> dict:
    """Kjør den foldvise, gruppede ABESS-diagnostikken for frekvens.

    ``glm_spec`` og ``prepare_design_frame`` kommer fra notebooken. Det gjør at
    referansekoding, imputasjon og Patsy-definisjoner er identiske med den
    etablerte GLM-pipelinen, samtidig som ABESS ikke legges i ``frequency_cv``.
    Hver kombinasjon av oppsett, fold og gruppestørrelse fittes på nytt; et
    subset velges aldri på hele utviklingssettet før OOF-evaluering.

    ``locked_glm_oof`` er bare en sammenligningsprediksjon fra den låste
    hovedmodellen. Den endres ikke og får ingen ABESS-modell-ID.
    """
    _validate_inputs(frame, folds, frequency_target, locked_glm_oof)
    verification = verify_abess_poisson_equivalence()
    all_oof: dict[str, pd.Series] = {}
    score_rows: list[dict] = []
    fold_rows: list[dict] = []
    selection_rows: list[dict] = []
    specs: dict[str, dict] = {}

    for setup, setup_definition in CANDIDATE_SETUPS.items():
        spec = _build_spec(setup, frame, glm_spec, frequency_target)
        specs[setup] = spec
        for support_size in range(len(MANDATORY_BLOCKS), len(BLOCKS) + 1):
            candidate_key = _candidate_key(setup, support_size)
            result = _cross_validate_size(
                spec=spec,
                frame=frame,
                folds=folds,
                prepare_design_frame=prepare_design_frame,
                support_size=support_size,
                setup=setup,
                setup_label=setup_definition["label"],
            )
            all_oof[candidate_key] = result["oof"]
            selection_rows.extend(result["selection_rows"])
            fold_rows.extend(
                _fold_comparison_rows(
                    frame,
                    folds,
                    locked_glm_oof,
                    result["fold_scores"],
                    setup,
                    support_size,
                    candidate_key,
                )
            )
            score_rows.append(
                _score_row(
                    frame,
                    locked_glm_oof,
                    result,
                    setup,
                    setup_definition["label"],
                    support_size,
                    candidate_key,
                )
            )

    score_curve = pd.DataFrame(score_rows).sort_values(["oppsett", "gruppestørrelse"])
    fold_comparisons = pd.DataFrame(fold_rows).sort_values(
        ["oppsett", "gruppestørrelse", "fold"]
    )
    selections = pd.DataFrame(selection_rows).sort_values(
        ["oppsett", "gruppestørrelse", "fold"]
    )
    representatives = _choose_representatives(score_curve, fold_comparisons)
    representative_keys = set(representatives["candidate_key"])
    selection_frequency = _selection_frequency(
        selections[selections["candidate_key"].isin(representative_keys)],
        representatives,
    )
    calibration = _calibration_tables(
        frame,
        all_oof,
        representatives,
        calibration_segments,
        selections,
    )
    full_development_selection, full_designs = _fit_full_development_subsets(
        frame,
        specs,
        representatives,
        prepare_design_frame,
    )
    full_development_grid = _fit_full_development_grid(
        frame,
        specs,
        list(CANDIDATE_SETUPS),
        range(len(MANDATORY_BLOCKS), len(BLOCKS) + 1),
        prepare_design_frame,
    )

    return {
        "verification": verification,
        "score_curve": score_curve,
        "fold_comparisons": fold_comparisons,
        "representatives": representatives,
        "selections": selections,
        "selection_frequency": selection_frequency,
        "calibration": calibration,
        "full_development_selection": full_development_selection,
        "full_development_grid": full_development_grid,
        "oof_predictions": pd.DataFrame(all_oof, index=frame.index),
        "full_designs": {
            setup: {
                key: design[key]
                for key in ("feature_names", "groups", "always_select", "column_blocks")
            }
            for setup, design in full_designs.items()
        },
        "locked_glm_name": locked_glm_name,
    }


def _build_spec(
    setup: str, frame: pd.DataFrame, glm_spec: Callable, frequency_target: dict
) -> dict:
    """Bygg ABESS-formelen gjennom den eksisterende GLM-formelbyggeren."""
    setup_terms = CANDIDATE_SETUPS[setup]["terms"]
    x = [setup_terms.get(block, block) for block in BLOCKS]
    settings = {**frequency_target, "data": frame}
    return glm_spec(
        f"ABESS_diagnostic_{setup}",
        x,
        **settings,
        required_columns=list(BLOCKS),
    )


def _cross_validate_size(
    *,
    spec: dict,
    frame: pd.DataFrame,
    folds: Sequence[dict],
    prepare_design_frame: Callable,
    support_size: int,
    setup: str,
    setup_label: str,
) -> dict:
    oof = pd.Series(np.nan, index=frame.index, name=_candidate_key(setup, support_size))
    fold_scores: list[dict] = []
    selection_rows: list[dict] = []
    for fold in folds:
        train = frame.loc[fold["train_index"]]
        validation = frame.loc[fold["val_index"]]
        _assert_disjoint_insured_ids(train, validation, fold["fold"])
        _validate_exposure(train, f"{fold['fold']} trening")
        _validate_exposure(validation, f"{fold['fold']} validering")
        train_design = prepare_design_frame(train, train, spec["required_columns"])
        validation_design = prepare_design_frame(
            train, validation, spec["required_columns"]
        )
        train_x, design_info = _training_matrix(spec, train_design)
        validation_x = _validation_matrix(design_info, validation_design)
        group_design = _group_design(train_x, design_info, spec)
        _assert_matching_design(train_x, validation_x, train, validation)
        estimator = _fit_abess(
            group_design["matrix"],
            train_design[spec["y"]].to_numpy(dtype=float),
            train_design[spec["weight"]].to_numpy(dtype=float),
            group_design["groups"],
            support_size,
            group_design["always_select"],
        )
        train_prediction = estimator.predict(group_design["matrix"])
        validation_prediction = estimator.predict(
            validation_x.loc[:, group_design["feature_names"]].to_numpy(dtype=float)
        )
        _validate_prediction(train_prediction, f"{fold['fold']} trening")
        _validate_prediction(validation_prediction, f"{fold['fold']} validering")
        selected_groups = _selected_group_indices(
            estimator.coef_, group_design["groups"]
        )
        mandatory_indices = set(group_design["always_select"])
        if not mandatory_indices <= selected_groups:
            raise AssertionError(
                f"{fold['fold']}: obligatoriske grupper mangler i ABESS-støtten "
                f"({selected_groups=}, {mandatory_indices=})"
            )
        selected_blocks = [BLOCKS[index] for index in sorted(selected_groups)]
        selection_rows.append(
            {
                "candidate_key": _candidate_key(setup, support_size),
                "oppsett": setup,
                "oppsett_tekst": setup_label,
                "gruppestørrelse": support_size,
                "fold": fold["fold"],
                "valgte_blokker": ", ".join(
                    BLOCK_LABELS[block] for block in selected_blocks
                ),
                "valgte_blokker_rå": tuple(selected_blocks),
                "antall_valgte_grupper": len(selected_groups),
                "antall_parametere": 1 + int(np.count_nonzero(estimator.coef_)),
                "antall_designkolonner": 1 + len(group_design["feature_names"]),
            }
        )
        oof.loc[validation.index] = validation_prediction
        fold_scores.append(
            {
                "fold": fold["fold"],
                "val_deviance": _poisson_deviance(
                    validation_design, validation_prediction
                ),
                "n_val": len(validation),
                "val_weight": validation_design[spec["weight"]].sum(),
                "val_actual": validation_design[CLAIMS_COLUMN].sum(),
                "val_predicted": float(
                    np.sum(validation_design[spec["weight"]] * validation_prediction)
                ),
                "pearson_phi": _pearson_phi(
                    train_design[CLAIMS_COLUMN].to_numpy(dtype=float),
                    train_design[spec["weight"]].to_numpy(dtype=float)
                    * train_prediction,
                    1 + int(np.count_nonzero(estimator.coef_)),
                ),
            }
        )
    if oof.isna().any():
        raise AssertionError("ABESS ga ikke én OOF-prediksjon per utviklingsrad")
    return {
        "oof": oof,
        "fold_scores": pd.DataFrame(fold_scores),
        "selection_rows": selection_rows,
    }


def _training_matrix(spec: dict, design: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Lær Patsy-basis på trening og fjern kun dens interseptkolonne."""
    _, matrix = patsy.dmatrices(spec["formula"], design, return_type="dataframe")
    if len(matrix) != len(design) or not matrix.index.equals(design.index):
        raise AssertionError("Patsy endret radantallet i treningsdesignen")
    if "Intercept" not in matrix.columns:
        raise AssertionError("Forventet Patsy-intersept mangler")
    if matrix.columns.tolist().count("Intercept") != 1:
        raise AssertionError("Patsy-designen har ikke entydig intersept")
    design_info = matrix.design_info
    matrix = matrix.drop(columns="Intercept")
    if matrix.empty:
        raise AssertionError("ABESS-designen mangler prediktorkolonner")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise AssertionError("Ikke-endelige verdier i ABESS-treningsdesign")
    return matrix, design_info


def _validation_matrix(
    design_info: object, validation_design: pd.DataFrame
) -> pd.DataFrame:
    """Bruk treningens Patsy-tilstand på valideringsdelen."""
    (matrix,) = patsy.build_design_matrices(
        [design_info], validation_design, return_type="dataframe"
    )
    if "Intercept" not in matrix.columns:
        raise AssertionError("Patsy-intersept mangler i valideringsdesignen")
    matrix = matrix.drop(columns="Intercept")
    if len(matrix) != len(validation_design) or not matrix.index.equals(
        validation_design.index
    ):
        raise AssertionError("Patsy endret radantallet i valideringsdesignen")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise AssertionError("Ikke-endelige verdier i ABESS-valideringsdesign")
    return matrix


def _group_design(matrix: pd.DataFrame, design_info: object, spec: dict) -> dict:
    """Koble sammenhengende Patsy-kolonner til de låste variabelblokkene."""
    if len(design_info.term_names) != len(BLOCKS) + 1 or len(spec["x"]) != len(BLOCKS):
        raise AssertionError("Uventet antall ABESS-termer; kandidatrommet er ikke låst")
    column_blocks: dict[str, tuple[str, ...]] = {}
    ordered_columns: list[str] = []
    group_values: list[int] = []
    for group_index, block in enumerate(BLOCKS):
        term_name = _term_name_for_block(block, spec)
        if term_name not in design_info.term_name_slices:
            raise AssertionError(
                f"Finner ikke forventet Patsy-term for {block}: {term_name}"
            )
        term_slice = design_info.term_name_slices[term_name]
        # Slicen refererer til designen med intersept. Etter fjerning er den
        # én-posisjon forskjøvet; alle blokk-kolonner må være sammenhengende.
        start, stop = term_slice.start - 1, term_slice.stop - 1
        if start < 0 or stop <= start:
            raise AssertionError(f"Ugyldig Patsy-slice for {block}: {term_slice}")
        positions = np.arange(start, stop)
        if not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
            raise AssertionError(f"Kolonnene for {block} er ikke sammenhengende")
        columns = tuple(matrix.columns[positions])
        column_blocks[block] = columns
        ordered_columns.extend(columns)
        group_values.extend([group_index] * len(columns))
    if len(ordered_columns) != matrix.shape[1] or len(set(ordered_columns)) != len(
        ordered_columns
    ):
        raise AssertionError("Patsy-kolonnene kan ikke entydig grupperes for ABESS")
    matrix = matrix.loc[:, ordered_columns]
    groups = np.asarray(group_values, dtype=np.int32)
    if not np.array_equal(np.unique(groups), np.arange(len(BLOCKS))):
        raise AssertionError("ABESS-gruppeindeksene dekker ikke eksakt alle blokker")
    for block, columns in column_blocks.items():
        if not columns:
            raise AssertionError(f"Blokken {block} mangler Patsy-kolonner")
    always_select = np.arange(len(MANDATORY_BLOCKS), dtype=np.int32)
    return {
        "matrix": matrix.to_numpy(dtype=float),
        "feature_names": matrix.columns.tolist(),
        "groups": groups,
        "always_select": always_select,
        "column_blocks": column_blocks,
    }


def _term_name_for_block(block: str, spec: dict) -> str:
    """Finn prosjektets eksakte Patsy-term for én låst blokk."""
    position = BLOCKS.index(block)
    term = spec["x"][position]
    if term != block:
        return term
    if block in spec["base_levels"]:
        return f"C({block}, Treatment({spec['base_levels'][block]!r}))"
    return block


def _fit_abess(
    x: np.ndarray,
    y: np.ndarray,
    exposure: np.ndarray,
    groups: np.ndarray,
    support_size: int,
    always_select: np.ndarray,
) -> PoissonRegression:
    """Fit én eksplisitt ABESS-størrelse uten EBIC eller intern CV."""
    _validate_exposure_values(exposure, "ABESS-fitting")
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Frekvensresponsen må være endelig og ikke-negativ")
    if support_size < len(always_select) or support_size > len(np.unique(groups)):
        raise ValueError("Ugyldig ABESS-gruppestørrelse")
    estimator = PoissonRegression(
        path_type="seq",
        support_size=[support_size],
        group=groups,
        always_select=always_select,
        alpha=0,
        fit_intercept=True,
        cv=1,
        ic_type="loss",
        thread=1,
        max_iter=50,
        primary_model_fit_max_iter=50,
        screening_size=-1,
    )
    return estimator.fit(x, y, sample_weight=exposure)


def _assert_matching_design(
    train_x: pd.DataFrame,
    validation_x: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    if train_x.columns.tolist() != validation_x.columns.tolist():
        raise AssertionError("Trenings- og valideringsdesign har ulike Patsy-kolonner")
    if len(train_x) != len(train) or len(validation_x) != len(validation):
        raise AssertionError("Utilsiktet radtap i ABESS-design")


def _fold_comparison_rows(
    frame: pd.DataFrame,
    folds: Sequence[dict],
    locked_glm_oof: pd.Series,
    candidate_scores: pd.DataFrame,
    setup: str,
    support_size: int,
    candidate_key: str,
) -> list[dict]:
    rows = []
    for fold in folds:
        validation = frame.loc[fold["val_index"]]
        baseline = _poisson_deviance(validation, locked_glm_oof.loc[validation.index])
        candidate = candidate_scores.set_index("fold").loc[fold["fold"], "val_deviance"]
        rows.append(
            {
                "candidate_key": candidate_key,
                "oppsett": setup,
                "gruppestørrelse": support_size,
                "fold": fold["fold"],
                "glm_deviance": baseline,
                "abess_deviance": candidate,
                "gevinst_mot_låst_glm": baseline - candidate,
            }
        )
    return rows


def _score_row(
    frame: pd.DataFrame,
    locked_glm_oof: pd.Series,
    result: dict,
    setup: str,
    setup_label: str,
    support_size: int,
    candidate_key: str,
) -> dict:
    oof = result["oof"]
    abess_deviance = _poisson_deviance(frame, oof)
    glm_deviance = _poisson_deviance(frame, locked_glm_oof)
    cluster = paired_deviance_gain(
        frame[RESPONSE_COLUMN],
        frame[EXPOSURE_COLUMN],
        locked_glm_oof,
        oof,
        frame[INSURED_ID_COLUMN],
    )
    parameters = [row["antall_parametere"] for row in result["selection_rows"]]
    return {
        "candidate_key": candidate_key,
        "oppsett": setup,
        "oppsett_tekst": setup_label,
        "gruppestørrelse": support_size,
        "oof_deviance": abess_deviance,
        "gevinst_mot_låst_glm": glm_deviance - abess_deviance,
        "SE_cluster_mot_låst_glm": cluster["SE_cluster"],
        "z_cluster_mot_låst_glm": cluster["z"],
        "gjennomsnittlige_parametere": float(np.mean(parameters)),
        "min_parametere": int(np.min(parameters)),
        "maks_parametere": int(np.max(parameters)),
        "pearson_phi": float(result["fold_scores"]["pearson_phi"].mean()),
    }


def _choose_representatives(
    score_curve: pd.DataFrame, fold_comparisons: pd.DataFrame
) -> pd.DataFrame:
    """Velg minimum og enkleste kandidat innen en eksplisitt 1-SE-heuristikk."""
    rows = []
    for setup, curve in score_curve.groupby("oppsett", sort=True):
        best = curve.sort_values(
            ["oof_deviance", "gruppestørrelse", "gjennomsnittlige_parametere"]
        ).iloc[0]
        best_fold = fold_comparisons[
            fold_comparisons["candidate_key"].eq(best["candidate_key"])
        ].set_index("fold")["abess_deviance"]
        candidates = []
        for _, candidate in curve.iterrows():
            candidate_fold = (
                fold_comparisons[
                    fold_comparisons["candidate_key"].eq(candidate["candidate_key"])
                ]
                .set_index("fold")["abess_deviance"]
                .reindex(best_fold.index)
            )
            losses = candidate_fold - best_fold
            mean_loss = float(losses.mean())
            se_loss = float(losses.std(ddof=1) / np.sqrt(len(losses)))
            candidates.append(
                candidate.to_dict()
                | {
                    "mean_tap_mot_minimum": mean_loss,
                    "SE_tap_mot_minimum": se_loss,
                    "innenfor_1SE": bool(mean_loss <= se_loss + 1e-12),
                }
            )
        candidates = pd.DataFrame(candidates)
        simple = (
            candidates[candidates["innenfor_1SE"]]
            .sort_values(
                ["gruppestørrelse", "gjennomsnittlige_parametere", "oof_deviance"]
            )
            .iloc[0]
        )
        for rule, candidate in (("minimum", best), ("enklest_1SE", simple)):
            candidate_data = candidate.to_dict()
            if rule == "minimum":
                candidate_data |= {
                    "mean_tap_mot_minimum": 0.0,
                    "SE_tap_mot_minimum": 0.0,
                    "innenfor_1SE": True,
                }
            candidate_data["regel"] = rule
            candidate_data["regel_tekst"] = (
                "lavest pooled OOF-deviance"
                if rule == "minimum"
                else "færrest grupper innen 1 SE av minimum (heuristikk)"
            )
            rows.append(candidate_data)
    return pd.DataFrame(rows).sort_values(["oppsett", "regel"])


def _selection_frequency(
    selections: pd.DataFrame, representatives: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for representative in representatives.itertuples(index=False):
        chosen = selections[
            selections["candidate_key"].eq(representative.candidate_key)
        ]
        for block in BLOCKS:
            count = sum(block in blocks for blocks in chosen["valgte_blokker_rå"])
            rows.append(
                {
                    "oppsett": representative.oppsett,
                    "regel": representative.regel,
                    "gruppestørrelse": representative.gruppestørrelse,
                    "blokk": BLOCK_LABELS[block],
                    "blokk_rå": block,
                    "valgt_i_folder": count,
                    "seleksjonsfrekvens": count / len(chosen),
                }
            )
    return pd.DataFrame(rows).sort_values(["oppsett", "regel", "blokk_rå"])


def _calibration_tables(
    frame: pd.DataFrame,
    oof_predictions: dict[str, pd.Series],
    representatives: pd.DataFrame,
    segments: Sequence[str],
    selections: pd.DataFrame,
) -> pd.DataFrame:
    tables = []
    for representative in representatives.itertuples(index=False):
        candidate_selections = selections[
            selections["candidate_key"].eq(representative.candidate_key)
        ]
        # Samme φ-definisjon som i eksisterende frekvens-CV: Pearson-statistikken
        # på treningsdelen i hver fold, deretter foldgjennomsnitt.
        table = build_actual_expected_table(
            frame.assign(portefølje="alle"),
            oof_predictions[representative.candidate_key],
            list(segments),
            categorical=("year",),
            dispersion=representative.pearson_phi,
        )
        tables.append(
            table.assign(
                oppsett=representative.oppsett,
                regel=representative.regel,
                gruppestørrelse=representative.gruppestørrelse,
                SE_forutsetning="Foldgjennomsnitt av ABESS Pearson-φ",
            )
        )
        if len(candidate_selections) != 5:
            raise AssertionError("Kalibrering mangler en ABESS-fold")
    return pd.concat(tables, ignore_index=True)


def _fit_full_development_candidate(
    frame: pd.DataFrame, design: dict, support_size: int
) -> dict:
    """Fit ABESS på hele utviklingssettet for én (oppsett, gruppestørrelse)."""
    estimator = _fit_abess(
        design["matrix"],
        frame[RESPONSE_COLUMN].to_numpy(dtype=float),
        frame[EXPOSURE_COLUMN].to_numpy(dtype=float),
        design["groups"],
        int(support_size),
        design["always_select"],
    )
    selected_groups = _selected_group_indices(estimator.coef_, design["groups"])
    if not set(design["always_select"]) <= selected_groups:
        raise AssertionError("Obligatorisk gruppe mangler i full ABESS-fit")
    selected_blocks = [BLOCKS[index] for index in sorted(selected_groups)]
    nonzero_mask = np.abs(estimator.coef_) > 1e-12
    koeffisienter = pd.Series(
        np.r_[estimator.intercept_, estimator.coef_[nonzero_mask]],
        index=["Intercept", *np.array(design["feature_names"])[nonzero_mask]],
        name="koeffisient",
    )
    return {
        "gruppestørrelse": support_size,
        "valgte_blokker": ", ".join(BLOCK_LABELS[block] for block in selected_blocks),
        "valgte_blokker_rå": tuple(selected_blocks),
        "antall_valgte_grupper": len(selected_groups),
        "antall_parametere": 1 + int(np.count_nonzero(estimator.coef_)),
        "koeffisienter": koeffisienter,
    }


def _fit_full_development_subsets(
    frame: pd.DataFrame,
    specs: dict[str, dict],
    representatives: pd.DataFrame,
    prepare_design_frame: Callable,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Fit etter CV, kun for å beskrive diagnostiske subset – aldri OOF-score."""
    rows = []
    full_designs = {}
    for representative in representatives.itertuples(index=False):
        setup = representative.oppsett
        if setup not in full_designs:
            # Først etter at gruppestørrelsen er valgt på CV, læres basis på
            # hele utviklingssettet for ren beskrivelse av slutt-subsettet.
            spec = specs[setup]
            full_design = prepare_design_frame(frame, frame, spec["required_columns"])
            matrix, design_info = _training_matrix(spec, full_design)
            full_designs[setup] = _group_design(matrix, design_info, spec)
        design = full_designs[setup]
        candidate = _fit_full_development_candidate(
            frame, design, int(representative.gruppestørrelse)
        )
        rows.append(
            {
                "oppsett": representative.oppsett,
                "regel": representative.regel,
                **candidate,
                "merknad": "Fit på hele utviklingssettet; ingen ny valideringsytelse.",
            }
        )
    return pd.DataFrame(rows).sort_values(["oppsett", "regel"]), full_designs


def _fit_full_development_grid(
    frame: pd.DataFrame,
    specs: dict[str, dict],
    setups: Sequence[str],
    support_sizes: Sequence[int],
    prepare_design_frame: Callable,
) -> pd.DataFrame:
    """Fit ABESS på hele utviklingssettet for alle (oppsett, gruppestørrelse).

    CV velger kun to representanter per oppsett (minimum og enklest-1SE), som
    ofte faller sammen på samme faktiske subset. For å vise de faktisk
    distinkte featurekombinasjonene i kandidatrommet fittes hele rutenettet
    én gang på utviklingssettet – fortsatt uten å berøre frequency_cv eller
    OOF-scoren, som fortsatt kommer fra den foldvise CV-en i score_curve.
    """
    rows = []
    full_designs: dict[str, dict] = {}
    for setup in setups:
        spec = specs[setup]
        full_design = prepare_design_frame(frame, frame, spec["required_columns"])
        matrix, design_info = _training_matrix(spec, full_design)
        full_designs[setup] = _group_design(matrix, design_info, spec)
        design = full_designs[setup]
        for support_size in support_sizes:
            candidate = _fit_full_development_candidate(frame, design, support_size)
            rows.append({"oppsett": setup, **candidate})
    return pd.DataFrame(rows)


def _poisson_deviance(frame: pd.DataFrame, prediction: pd.Series | np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=float)
    _validate_prediction(prediction, "deviance")
    return float(
        mean_tweedie_deviance(
            frame[RESPONSE_COLUMN],
            prediction,
            sample_weight=frame[EXPOSURE_COLUMN],
            power=1,
        )
    )


def _pearson_phi(
    claims: np.ndarray, expected_counts: np.ndarray, n_parameters: int
) -> float:
    """Pearson-dispersjon på antallsskala, som i den eksisterende CV-sløyfen."""
    _validate_prediction(expected_counts, "Pearson-forventning")
    degrees_of_freedom = len(claims) - n_parameters
    if degrees_of_freedom <= 0:
        raise AssertionError("For få frihetsgrader til ABESS Pearson-φ")
    return float(
        np.sum((claims - expected_counts) ** 2 / expected_counts) / degrees_of_freedom
    )


def _selected_group_indices(coefficients: np.ndarray, groups: np.ndarray) -> set[int]:
    coefficients = np.asarray(coefficients, dtype=float)
    if not np.isfinite(coefficients).all():
        raise AssertionError("ABESS returnerte ikke-endelige koeffisienter")
    return set(groups[np.abs(coefficients) > 1e-12].tolist())


def _validate_inputs(
    frame: pd.DataFrame,
    folds: Sequence[dict],
    frequency_target: dict,
    locked_glm_oof: pd.Series,
) -> None:
    required = set(BLOCKS) | {
        RESPONSE_COLUMN,
        EXPOSURE_COLUMN,
        CLAIMS_COLUMN,
        INSURED_ID_COLUMN,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ABESS-rammen mangler kolonner: {sorted(missing)}")
    if (
        frequency_target.get("y") != RESPONSE_COLUMN
        or frequency_target.get("weight") != EXPOSURE_COLUMN
    ):
        raise ValueError(
            "ABESS krever frekvensrespons og eksponeringsvekt fra eksisterende pipeline"
        )
    _validate_exposure(frame, "utviklingsdata")
    if not frame.index.is_unique or not frame.index.equals(locked_glm_oof.index):
        raise ValueError("ABESS og låst GLM må ha samme, unike utviklingsindeks")
    _validate_prediction(locked_glm_oof.to_numpy(dtype=float), "låst GLM OOF")
    if len(folds) != 5:
        raise ValueError("ABESS-diagnostikken krever de fem låste gruppefoldene")
    all_validation = pd.Index(np.concatenate([fold["val_index"] for fold in folds]))
    if not all_validation.is_unique or not all_validation.sort_values().equals(
        frame.index.sort_values()
    ):
        raise ValueError(
            "Foldenes valideringsindekser dekker ikke utviklingsrammen én gang"
        )


def _validate_exposure(frame: pd.DataFrame, context: str) -> None:
    _validate_exposure_values(frame[EXPOSURE_COLUMN].to_numpy(dtype=float), context)


def _validate_exposure_values(exposure: np.ndarray, context: str) -> None:
    if not (np.isfinite(exposure).all() and (exposure > 0).all()):
        raise ValueError(
            f"Eksponeringen må være endelig og strengt positiv ({context})"
        )


def _validate_prediction(prediction: np.ndarray, context: str) -> None:
    if not (np.isfinite(prediction).all() and (prediction > 0).all()):
        raise AssertionError(
            f"ABESS ga ikke-endelige eller ikke-positive prediksjoner ({context})"
        )


def _assert_disjoint_insured_ids(
    train: pd.DataFrame, validation: pd.DataFrame, fold_name: str
) -> None:
    overlap = set(train[INSURED_ID_COLUMN]) & set(validation[INSURED_ID_COLUMN])
    if overlap:
        raise AssertionError(f"{fold_name}: overlapper på insured_id")


def _candidate_key(setup: str, support_size: int) -> str:
    return f"ABESS_{setup}_groups_{support_size}"
