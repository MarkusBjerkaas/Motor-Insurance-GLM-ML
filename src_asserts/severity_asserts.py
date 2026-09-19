"""Kontroller for severity-modellen (Gamma, ``average_severity`` vektet med skadeantall)."""

import numpy as np

from src_asserts.common_glm_asserts import (
    assert_development_years,
    assert_full_oof_coverage,
    assert_positive_finite_exposure,
    assert_unique_index,
    require,
)
from src_descriptives.own_damage_descriptives import CURRENCY_ZERO_TOLERANCE


def assert_severity_frame(model_frame, severity_frame, development):
    """Delmengdeindeks, låst responsmaske (B-15), positive skadevekter og responsidentitet."""
    assert_development_years(model_frame)
    assert_unique_index(model_frame)
    require(
        model_frame.index.equals(development.index),
        "model_frame må ha samme indeks som hele utviklingspopulasjonen.",
    )
    assert_positive_finite_exposure(model_frame)
    require(
        severity_frame.index.is_unique
        and severity_frame.index.isin(model_frame.index).all(),
        "severity_frame må være en delmengde av model_frame (samme radindekser).",
    )
    expected_mask = model_frame["property_claims"].gt(0) & model_frame[
        "property_incurred"
    ].gt(CURRENCY_ZERO_TOLERANCE)
    require(
        model_frame.index[expected_mask].equals(severity_frame.index),
        "severity_frame avviker fra maskene property_claims > 0 og property_incurred > toleranse.",
    )
    require(
        (severity_frame["property_claims"] > 0).all(),
        "Skadevektene (property_claims) må være positive.",
    )
    require(
        np.allclose(
            severity_frame["average_severity"],
            severity_frame["property_incurred"] / severity_frame["property_claims"],
        ),
        "average_severity må være property_incurred / property_claims.",
    )


def assert_severity_spec(spec):
    """Gamma på ``average_severity`` vektet med ``property_claims`` (ingen eksponering)."""
    require(
        spec["y"] == "average_severity", "Severityresponsen må være average_severity."
    )
    require(
        spec["weight"] == "property_claims", "Severityvekten må være property_claims."
    )
    require(spec["family"] == "gamma", "Severityfamilien må være Gamma.")


def assert_severity_fold_contract(result, model_frame, severity_frame, folds):
    """Fit og score på severity-delen av foldene; prediksjon for hele valideringsfolden."""
    assert_full_oof_coverage(result["oof"], severity_frame.index, "severity oof")
    assert_full_oof_coverage(
        result["prediction_oof"], model_frame.index, "severity prediction_oof"
    )
    scores = result["scores"].set_index("fold")
    for fold in folds:
        row = scores.loc[fold["fold"]]
        require(
            row["n_train"] == severity_frame.index.isin(fold["train_index"]).sum()
            and row["n_val"] == severity_frame.index.isin(fold["val_index"]).sum(),
            f"{fold['fold']}: fit/score-utvalget er ikke fullfoldens snitt med severity_frame.",
        )
