"""Diagnostikk for train/test-splitten av egen-skade-modellpopulasjonen.

Selve splitt-funksjonen og fold-kontrollen ligger i notebooken (analysis.py)
siden de er sentrale for modelleringsløpet. Her ligger bare oppsummerings- og
balansediagnostikk som brukes til å inspisere splitten i etterkant.
"""

import numpy as np
import pandas as pd

CATEGORICAL_BALANCE_VARS = [
    "policy_type",
    "bonus_score",
    "fuel_type",
    "municipality_type",
    "circulation_area",
]
NUMERIC_BALANCE_VARS = [
    "driver_age",
    "age_driving_licence",
    "vehicle_age",
    "vehicle_value",
    "power_to_weight_ratio",
]
SMD_FLAG_THRESHOLD = 0.1
SHARE_FLAG_THRESHOLD_PP = 5.0


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


def _categorical_balance(train_pool, test):
    """Fordeling (%) per kategori i train vs. test, med differanse i prosentpoeng."""
    rows = []
    for column in CATEGORICAL_BALANCE_VARS:
        train_share = train_pool[column].value_counts(normalize=True).mul(100)
        test_share = test[column].value_counts(normalize=True).mul(100)
        categories = sorted(set(train_share.index) | set(test_share.index), key=str)
        for category in categories:
            train_pct = train_share.get(category, 0.0)
            test_pct = test_share.get(category, 0.0)
            diff = train_pct - test_pct
            rows.append(
                {
                    "variabel": column,
                    "kategori": category,
                    "train_prosent": round(train_pct, 2),
                    "test_prosent": round(test_pct, 2),
                    "diff_prosentpoeng": round(diff, 2),
                    "betydelig_avvik": abs(diff) > SHARE_FLAG_THRESHOLD_PP,
                }
            )
    return pd.DataFrame(rows)


def _numeric_balance(train_pool, test):
    """Gjennomsnitt/std i train vs. test, med standardized mean difference (SMD)."""
    rows = []
    for column in NUMERIC_BALANCE_VARS:
        train_mean = train_pool[column].mean()
        test_mean = test[column].mean()
        train_std = train_pool[column].std()
        test_std = test[column].std()
        pooled_std = np.sqrt((train_std**2 + test_std**2) / 2)
        smd = (train_mean - test_mean) / pooled_std if pooled_std else np.nan
        rows.append(
            {
                "variabel": column,
                "train_snitt": round(train_mean, 2),
                "test_snitt": round(test_mean, 2),
                "train_std": round(train_std, 2),
                "test_std": round(test_std, 2),
                "smd": round(smd, 3),
                "betydelig_avvik": abs(smd) > SMD_FLAG_THRESHOLD,
            }
        )
    return pd.DataFrame(rows)


def build_key_variable_balance(train_pool, test):
    """Balansediagnostikk for nøkkelvariabler mellom train/CV-pool og test.

    Kategoriske variabler sammenlignes på andel (%) per kategori (diff i
    prosentpoeng, flagg ved avvik > 5pp). Numeriske variabler sammenlignes med
    standardized mean difference (SMD), flagg ved |SMD| > 0.1 — en vanlig
    tommelfingerregel for når et kovariatavvik er stort nok til å bry seg om.
    """
    return _categorical_balance(train_pool, test), _numeric_balance(train_pool, test)
