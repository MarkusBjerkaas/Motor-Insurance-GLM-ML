"""Felles kontroller for GLM-oppsettet (frekvens, severity og Tweedie).

Alle kontroller kaster ``AssertionError`` med en tydelig melding. De er bevisst
samlet i ``src_asserts/`` slik at pakken og importlinjene kan slettes ved
levering uten at modellkoden endres.
"""

import numpy as np
import pandas as pd

from src_core_glm.model_data import TRAIN_YEARS, assert_development_years  # noqa: F401


def require(condition, message):
    """Kast ``AssertionError`` (også med ``python -O``) når ``condition`` er usann."""
    if not condition:
        raise AssertionError(message)


def assert_unique_index(frame, name="modellrammen"):
    require(frame.index.is_unique, f"{name} har duplikate radindekser.")


def assert_positive_finite_exposure(frame, column="total_exposure"):
    exposure = frame[column]
    require(
        np.isfinite(exposure).all() and (exposure > 0).all(),
        f"{column} må være endelig og positiv i alle rader.",
    )


def assert_nonnegative(series, name):
    require(
        np.isfinite(series).all() and (series >= 0).all(),
        f"{name} må være endelig og ikke-negativ.",
    )


def assert_valid_folds(folds, frame, group_column="insured_id"):
    """Foldene dekker ``frame`` uten overlapp mellom trening, validering og personer."""
    groups = frame[group_column]
    for fold in folds:
        train, val = fold["train_index"], fold["val_index"]
        name = fold["fold"]
        require(
            train.isin(frame.index).all() and val.isin(frame.index).all(),
            f"{name}: foldindekser som ikke finnes i modellrammen.",
        )
        require(not train.isin(val).any(), f"{name}: trening og validering overlapper.")
        require(
            len(train) + len(val) == len(frame),
            f"{name}: trening + validering dekker ikke hele modellrammen.",
        )
        require(
            not groups.loc[train].isin(groups.loc[val]).any(),
            f"{name}: samme {group_column} finnes i trening og validering.",
        )
    validation_rows = pd.Index(np.concatenate([f["val_index"] for f in folds]))
    require(
        validation_rows.is_unique and len(validation_rows) == len(frame),
        "Hver rad må ligge i nøyaktig én valideringsfold.",
    )


def assert_full_oof_coverage(oof, expected_index, name="OOF"):
    """OOF-serien har nøyaktig forventet indeks og ingen manglende verdier."""
    require(
        oof.index.equals(expected_index),
        f"{name}: indeksen avviker fra forventet populasjon.",
    )
    require(
        oof.notna().all(),
        f"{name}: {int(oof.isna().sum())} rader mangler OOF-prediksjon.",
    )


def assert_positive_finite(values, name="prediksjoner"):
    values = np.asarray(values, dtype=float)
    require(
        np.isfinite(values).all() and (values > 0).all(),
        f"{name} må være endelige og positive.",
    )


def assert_comparable_oof(oofs, expected_index):
    """Senere benchmark: alle OOF-serier har lik indeks, full dekning og positive verdier."""
    for name, oof in oofs.items():
        assert_full_oof_coverage(oof, expected_index, name)
        assert_positive_finite(oof, name)


def assert_model_definition(definition, locked):
    """Låste termer er med, ingen duplikater, og hver interaksjon har begge hovedeffektene."""
    terms = definition["terms"]
    require(len(terms) == len(set(terms)), f"Duplikate termer: {terms}.")
    missing = [column for column in locked if column not in terms]
    require(not missing, f"Låste variabler mangler i modellen: {missing}.")
    for term in terms:
        if isinstance(term, tuple):
            require(
                all(part in terms for part in term),
                f"Interaksjonen {term} mangler en hovedeffekt.",
            )
    require(
        all(column in terms for column in definition["splines"]),
        "En spline gjelder en variabel som ikke er i modellen.",
    )
