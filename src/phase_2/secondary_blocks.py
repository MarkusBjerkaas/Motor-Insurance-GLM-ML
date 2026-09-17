"""Fase 1, seksjon 3.5: sekundære blokker (F4) — merke og setegruppe oppå F2.

All beregning for seksjon 3.5 ligger her; notebooken viser bare resultatene.
Avhengighetene (`cross_validate_candidate`, `frequency_cv`, valgt F2-kandidat,
data og folder) er notebook-lokale og sendes inn som argumenter for å unngå
sirkulær import.
"""

import pandas as pd

from src.phase_2 import frequency_candidates
from src.phase_2.frequency_tables import (
    build_candidate_table,
    format_comparison_table,
    report_near_misses,
)

SECONDARY_OPTIONS = {
    "none": [],
    "brand": ["brand"],
    "seats": ["seats"],
    "brand-seats": ["brand", "seats"],
}


def run_secondary_blocks(
    cross_validate_candidate, frequency_cv, f2_id, f2_spec, model_frame, cv_folds
):
    """Kryssvaliderer F4-kandidatene (merke/seter oppå F2) og bygger resultattabellene.

    Parameters
    ----------
    cross_validate_candidate : callable
        Notebookens cachede CV-funksjon.
    frequency_cv : dict
        Delt CV-resultatregister; oppdateres i place med de nye F4-kandidatene.
    f2_id : str
        Modell-ID for valgt F2-kandidat (seksjon 3.3).
    f2_spec : dict
        Spesifikasjonen til ``f2_id``, som F4 utvider.
    model_frame : pandas.DataFrame
        Modelldatasettet, for støttetabellen per nivå.
    cv_folds : list of dict
        Foldene fra ``build_model_frames``, for minste støtte per foldtrening.

    Returns
    -------
    dict
        ``f4_ids`` (label → modell-ID), ``selection`` (fra
        ``select_from_candidate_set``), ``comparison_table`` (Tabell 1) og
        ``support_table`` (Tabell 2).
    """
    f4_ids = {}
    for label, extra_blocks in SECONDARY_OPTIONS.items():
        if not extra_blocks:
            f4_ids[label] = f2_id  # ingen ekstra blokk er valgt F2
            continue
        model_id = "F4_" + label
        frequency_cv[model_id] = cross_validate_candidate(
            frequency_candidates.build_frequency_candidate(
                model_id,
                "F4",
                f2_spec["feature_blocks"] + extra_blocks,
                forms=f2_spec["forms"],
                parent_id=f2_id,
            )
        )
        f4_ids[label] = model_id

    selection = frequency_candidates.select_from_candidate_set(
        frequency_cv, list(f4_ids.values()), f2_id
    )
    # Hver utvidelse mot F2, og betinget bidrag innen paret av sekundære blokker
    comparisons = pd.DataFrame(
        [
            frequency_candidates.compare_models(frequency_cv, f2_id, f4_ids[k])
            for k in ("brand", "seats", "brand-seats")
        ]
        + [
            frequency_candidates.compare_models(
                frequency_cv, f4_ids["seats"], f4_ids["brand-seats"]
            ),
            frequency_candidates.compare_models(
                frequency_cv, f4_ids["brand"], f4_ids["brand-seats"]
            ),
        ]
    )
    report_near_misses(comparisons)
    print(
        f"Beste: {selection['best']}. Innen 1 SE: {', '.join(selection['within_1se'])}."
    )
    print(f"Valgt: {selection['selected']} ({selection['reason']})")

    # Tabell 1: sammenligninger med kandidatens score, stabilitet og valgflagg
    scores = build_candidate_table(frequency_cv, list(f4_ids.values()))[
        ["parametere", "oof_deviance", "oof_d2"]
    ].assign(
        stabilitet=selection["table"]["stability"],
        i_1SE_sett=lambda t: t.index.isin(selection["within_1se"]),
        valgt=lambda t: t.index == selection["selected"],
    )
    comparison_table = (
        format_comparison_table(comparisons)
        .drop(columns=["Retning", "near_miss_3of5", "stability"])
        .join(scores, on="Kandidat")
        .set_index(["Referanse", "Kandidat"])
    )

    # Tabell 2: støtte per nivå i train-poolen og minste støtte i en foldtrening
    support_tables = {}
    for block in ("brand", "seats"):
        column = frequency_candidates.FEATURE_BLOCKS[block]["column"]
        counts = ["total_exposure", "property_claims"]
        fold_support = pd.concat(
            [
                model_frame.loc[fold["train_index"]].groupby(column)[counts].sum()
                for fold in cv_folds
            ]
        ).groupby(level=0)
        support_tables[block] = (
            model_frame.groupby(column)[counts]
            .sum()
            .join(fold_support.min().add_prefix("min_fold_"))
            .rename_axis("nivå")
            .sort_values("total_exposure", ascending=False)
        )
    support_table = pd.concat(support_tables, names=["blokk"]).round(1)

    return {
        "f4_ids": f4_ids,
        "selection": selection,
        "comparison_table": comparison_table,
        "support_table": support_table,
    }
