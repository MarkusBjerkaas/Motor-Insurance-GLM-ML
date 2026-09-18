"""Diagnostikk for train/test-splitten av egen-skade-modellpopulasjonen.

Selve splitt-funksjonen og fold-kontrollen ligger i notebooken (analysis.py)
siden de er sentrale for modelleringsløpet. Her ligger bare oppsummerings- og
balansediagnostikk som brukes til å inspisere splitten i etterkant.
"""

import pandas as pd


def build_split_summary(train_pool, test):
    """Oppsummer eksponering og skadevolum i train/CV-pool vs. test."""
    parts = {"Train/CV-pool (2022-2023)": train_pool, "Test (2024)": test}
    rows = [
        {
            "sett": label,
            "poliseår": len(part),
            "unike_poliser": part["insured_id"].nunique(),
            "eksponering": part["total_exposure"].sum(),
            "skadeantall": part["property_claims"].sum(),
            "andel_poliseår_med_skade_prosent": part["property_claims"].gt(0).mean()
            * 100,
            "incurred": part["property_incurred"].sum(),
        }
        for label, part in parts.items()
    ]
    summary = pd.DataFrame(rows).round(2)

    train_ids = set(train_pool["insured_id"])
    test_ids = set(test["insured_id"])
    overlap = train_ids & test_ids
    overlap_summary = pd.Series(
        {
            "unike_poliser_train_pool": len(train_ids),
            "unike_poliser_test": len(test_ids),
            "poliser_i_begge": len(overlap),
            "andel_av_test_som_er_videreført_prosent": round(
                len(overlap) / len(test_ids) * 100, 2
            ),
        },
        name="Verdi",
    ).to_frame()
    return summary, overlap_summary
