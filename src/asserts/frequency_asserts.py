"""Kontroller for frekvensmodellen (Poisson, ``claim_frequency`` med eksponeringsvekt)."""

import numpy as np

from src.asserts.common_glm_asserts import (
    assert_development_years,
    assert_nonnegative,
    assert_positive_finite_exposure,
    assert_unique_index,
    require,
)


def assert_frequency_frame(model_frame, development):
    """Fullpopulasjonsindeks, gyldig eksponering og ``claim_frequency = skader / eksponering``."""
    assert_development_years(model_frame)
    assert_unique_index(model_frame)
    require(
        model_frame.index.equals(development.index),
        "Frekvensrammen må ha samme indeks som hele utviklingspopulasjonen.",
    )
    assert_positive_finite_exposure(model_frame)
    assert_nonnegative(model_frame["property_claims"], "property_claims")
    require(
        np.allclose(
            model_frame["claim_frequency"],
            model_frame["property_claims"] / model_frame["total_exposure"],
        ),
        "claim_frequency må være property_claims / total_exposure.",
    )


def assert_frequency_spec(spec):
    """Poisson på ``claim_frequency`` med ``total_exposure`` som vekt."""
    require(
        spec["y"] == "claim_frequency", "Frekvensresponsen må være claim_frequency."
    )
    require(
        spec["weight"] == "total_exposure", "Frekvensvekten må være total_exposure."
    )
    require(spec["family"] == "poisson", "Frekvensfamilien må være Poisson.")
