"""Fase 1, seksjon 3.7: to avgrensede produkt-interaksjoner (F6, B-28).

Støtteporten, kandidatbyggingen, seleksjonen og kurveplottet for de to
COMP_N-kontrasthelningene (alder og logverdi) ligger her. Notebooken kjører
2024-vernet, kaller inn og viser tabeller/plott. Avhengighetene som er
notebook-lokale (`cross_validate_candidate`, `prepare_design_frame`,
`pooled_oof_deviance`, dataene og valgt additiv modell) sendes inn som
argumenter for å unngå sirkulær import.
"""

import numpy as np
import pandas as pd

from src.phase_2 import frequency_candidates
from src.phase_2.frequency_plots import plot_fold_curves
from src.phase_2.frequency_tables import (
    build_candidate_table,
    format_comparison_table,
    report_near_misses,
)

INTERACTION_PRODUCTS = ["COMP_E", "COMP_N"]  # basis først (B-24); CC er utenfor (B-29)
INTERACTION_FEATURES = {"I1": "driver_age", "I2": "log_vehicle_value"}
_FEATURE_LABELS = {
    "driver_age": "Førers alder",
    "log_vehicle_value": "Log kjøretøyverdi",
}


def _product_slopes(feature):
    """B-28 etter B-29: kolonnen 1{COMP_N}·(x − c), der c læres i treningsfolden."""
    terms = [f"comp_n_slope_{feature}"]

    def learn(train_design):
        # Eksponeringsvektet snitt etter imputasjon (B-09)
        return np.average(train_design[feature], weights=train_design["total_exposure"])

    def apply(frame, center):
        centered = frame[feature] - center
        return frame.assign(**{terms[0]: frame["policy_type"].eq("COMP_N") * centered})

    return {
        "name": f"product_x_{feature}",
        "slope_feature": feature,
        "terms": terms,
        "learn": learn,
        "apply": apply,
    }


def _tertile_support(feature, model_frame, cv_folds, prepare_design_frame):
    """Støtte per produkt × felles eksponeringstredel i hver foldtrening (B-28)."""
    rows, edges = [], []
    for fold in cv_folds:
        train = model_frame.loc[fold["train_index"]]
        train = prepare_design_frame(train, train, [feature])
        values = train[feature].to_numpy()
        order = np.argsort(values, kind="stable")
        cumulative = np.cumsum(train["total_exposure"].to_numpy()[order])
        # Minste x der kumulativ sortert eksponering når 1/3 og 2/3
        q1, q2 = values[order][
            np.searchsorted(cumulative / cumulative[-1], [1 / 3, 2 / 3])
        ]
        edges.append((q1, q2))
        if not q1 < q2:
            continue  # sammenfallende grenser er svikt; det lages ikke nye grenser
        tertile = pd.cut(
            train[feature], [-np.inf, q1, q2, np.inf], labels=["T1", "T2", "T3"]
        )
        cells = train.groupby(["policy_type", tertile], observed=False)[
            ["total_exposure", "property_claims"]
        ].sum()
        product_claims = train.groupby("policy_type")["property_claims"].sum()
        rows.append(
            cells.assign(
                fold=fold["fold"],
                product_claims=product_claims.reindex(
                    cells.index.get_level_values("policy_type")
                ).to_numpy(),
            )
        )
    table = (
        pd.concat(rows)
        .groupby(level=[0, 1], observed=False)
        .agg(
            min_fold_eksponering=("total_exposure", "min"),
            min_fold_skader=("property_claims", "min"),
            maks_fold_skader=("property_claims", "max"),
            min_fold_skader_produkt=("product_claims", "min"),
        )
        .rename_axis(["produkt", "tredel"])
    )
    passes = bool(
        all(q1 < q2 for q1, q2 in edges)
        and len(rows) == len(cv_folds)
        and table["min_fold_eksponering"].ge(50).all()
        and table["min_fold_skader_produkt"].ge(20).all()
    )
    return table, passes, edges


def test_interaction_support(
    model_frame, cv_folds, prepare_design_frame, additive_spec
):
    """Kjører B-28-støtteporten (felles eksponeringstredeler) for I1 og I2.

    Parameters
    ----------
    model_frame : pandas.DataFrame
        Modelldatasettet.
    cv_folds : list of dict
        Foldene fra ``build_model_frames``.
    prepare_design_frame : callable
        Notebookens design-forberedelse (imputasjon m.m.).
    additive_spec : dict
        Spesifikasjonen til den additive modellen fra 3.6; features som er
        fjernet der utgår fra interaksjonstestingen.

    Returns
    -------
    support_table : pandas.DataFrame
        Tabell 1: minste støtte per celle over de fem foldtreningene.
    support_passes : dict
        Label ("I1"/"I2") → om støtteporten er bestått.
    """
    support_tables, support_passes = {}, {}
    for label, feature in INTERACTION_FEATURES.items():
        if feature not in additive_spec["forms"]:
            print(f"{label} utgår: {feature} er fjernet i F5")
            continue
        support_tables[label], support_passes[label], edges = _tertile_support(
            feature, model_frame, cv_folds, prepare_design_frame
        )
        q1s, q2s = zip(*edges)
        print(
            f"{label} ({feature}): tredelgrenser q1 {min(q1s):.3f}–{max(q1s):.3f}, "
            f"q2 {min(q2s):.3f}–{max(q2s):.3f} over foldene; "
            f"støtteport {'bestått' if support_passes[label] else 'SVIKTER'}"
        )
    support_table = pd.concat(support_tables, names=["interaksjon"]).round(1)
    return support_table, support_passes


def select_interactions(
    cross_validate_candidate,
    frequency_cv,
    additive_id,
    additive_spec,
    support_passes,
    pooled_oof_deviance,
):
    """Kryssvaliderer I1/I2, velger etter B-08 og bygger sammenligningstabellen.

    Parameters
    ----------
    cross_validate_candidate : callable
        Notebookens cachede CV-funksjon.
    frequency_cv : dict
        Delt CV-resultatregister; oppdateres i place med de nye F6-kandidatene.
    additive_id : str
        Modell-ID for den additive modellen fra 3.6 (B-08-referansen).
    additive_spec : dict
        Spesifikasjonen til ``additive_id``.
    support_passes : dict
        Fra ``test_interaction_support``.
    pooled_oof_deviance : callable
        Notebookens pooled-deviance-funksjon (for tie-break lavest deviance).

    Returns
    -------
    dict
        ``f6_ids`` (label → modell-ID for I1/I2), ``selected_id``,
        ``selected_reason`` og ``comparison_table`` (Tabell 2).
    """

    def interaction_candidate(model_id, features):
        """Additiv modell + COMP_N-helning for hver av ``features``, med produktvise kurver."""
        candidate = frequency_candidates.build_frequency_candidate(
            model_id,
            "F6",
            additive_spec["feature_blocks"],
            forms=additive_spec["forms"],
            parent_id=additive_id,
            derived=[_product_slopes(feature) for feature in features],
        )
        # Rangkontroll mot den additive modellen: nøyaktig én ny parameter per interaksjon
        assert candidate["n_parameters"] == additive_spec["n_parameters"] + len(
            features
        )
        return candidate | {"curve_products": INTERACTION_PRODUCTS}

    feature_short = {
        block["column"]: block["label"]
        for block in frequency_candidates.FEATURE_BLOCKS.values()
        if "label" in block
    }
    f6_ids = {}
    for label, feature in INTERACTION_FEATURES.items():
        if not support_passes.get(label, False):
            print(f"{label} estimeres ikke: støtteporten er ikke bestått")
            continue
        model_id = f"F6_{label}_product-x-{feature_short[feature]}"
        frequency_cv[model_id] = cross_validate_candidate(
            interaction_candidate(model_id, [feature])
        )
        f6_ids[label] = model_id

    comparisons = [
        frequency_candidates.compare_models(frequency_cv, additive_id, model_id)
        for model_id in f6_ids.values()
    ]
    passed = [row["candidate"] for row in comparisons if row["passes_b08"]]
    selected_id = additive_id
    if len(passed) == 2:
        combination_id = "F6_I1-I2_product-x-age-value"
        frequency_cv[combination_id] = cross_validate_candidate(
            interaction_candidate(combination_id, list(INTERACTION_FEATURES.values()))
        )
        combination_rows = [
            frequency_candidates.compare_models(frequency_cv, model_id, combination_id)
            for model_id in passed
        ]
        comparisons += combination_rows
        if all(row["passes_b08"] for row in combination_rows):
            selected_id, selected_reason = (
                combination_id,
                "kombinasjonen består B-08 mot begge",
            )
        else:
            better, worse = sorted(
                passed, key=lambda m: pooled_oof_deviance(frequency_cv[m])
            )
            pair_within = frequency_candidates.compare_models(
                frequency_cv, better, worse, "forenkling"
            )["passes_1se"]
            selected_id = f6_ids["I1"] if pair_within else better
            selected_reason = "kombinasjonen består ikke B-08 mot begge; " + (
                "I1 foretrekkes innen parets 1 SE" if pair_within else "lavest deviance"
            )
    elif len(passed) == 1:
        selected_id, selected_reason = passed[0], "eneste interaksjon som består B-08"
    else:
        selected_reason = (
            "ingen interaksjon består B-08; den additive modellen beholdes"
        )
    comparisons = pd.DataFrame(comparisons)
    report_near_misses(comparisons)
    print(f"Valgt etter 3.7: {selected_id} ({selected_reason})")

    def slope_summary(model_id):
        """COMP_N-helninger: snitt over foldene (snitt av fold-SE i parentes)."""
        result = frequency_cv[model_id]
        terms = [t for item in result["spec"]["derived"] for t in item.get("terms", [])]
        return ", ".join(
            f"{term}: {result['params'].loc[term].mean():+.4f} ({result['param_se'].loc[term].mean():.4f})"
            for term in terms
        )

    scores = build_candidate_table(
        frequency_cv, list(comparisons["candidate"].unique())
    )[["parametere", "oof_deviance", "oof_d2"]].assign(
        kontrasthelninger=lambda t: [slope_summary(m) for m in t.index]
    )
    comparison_table = (
        format_comparison_table(comparisons)
        .drop(columns=["Retning", "near_miss_3of5"])
        .join(scores, on="Kandidat")
        .set_index(["Referanse", "Kandidat"])
    )

    return {
        "f6_ids": f6_ids,
        "selected_id": selected_id,
        "selected_reason": selected_reason,
        "comparison_table": comparison_table,
    }


def plot_interaction_curves(frequency_cv, f6_ids, additive_spec, model_frame):
    """Produktvise kurver mot felles additiv kurve, avgrenset til felles støttet område.

    Returns
    -------
    matplotlib.figure.Figure or None
        ``None`` hvis ingen interaksjon ble estimert (``f6_ids`` er tom).
    """
    if not f6_ids:
        return None
    additive_curves = frequency_candidates.fit_full_curves(additive_spec)["curves"]
    plot_curves = []
    for label, model_id in f6_ids.items():
        feature = INTERACTION_FEATURES[label]
        product_grids = [
            frequency_candidates.build_curve_grid(
                model_frame[model_frame["policy_type"].eq(p)], feature
            )["x"]
            for p in INTERACTION_PRODUCTS
        ]
        low, high = max(g[0] for g in product_grids), min(g[-1] for g in product_grids)
        print(f"{feature}: felles støttet område {low:.2f}–{high:.2f}")
        own = pd.concat(
            [
                frequency_cv[model_id]["curves"],
                frequency_candidates.fit_full_curves(frequency_cv[model_id]["spec"])[
                    "curves"
                ],
            ]
        )
        common = pd.concat(
            [
                additive_curves.query("feature == @feature").assign(product=p)
                for p in INTERACTION_PRODUCTS
            ]
        )
        plot_curves += [
            part.query("feature == @feature and @low <= x <= @high")
            for part in (own, common)
        ]
    if not plot_curves:
        return None
    return plot_fold_curves(
        pd.concat(plot_curves, ignore_index=True),
        feature_labels=_FEATURE_LABELS,
        title="Produktvise kurver (B-28) mot felles additiv kurve (±2 SE)",
    )
