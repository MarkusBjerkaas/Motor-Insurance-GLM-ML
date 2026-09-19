"""Kontroller for Tweedie-modellen (rate-respons med eksponeringsvekt og uten offset)."""

import numpy as np
import statsmodels.api as sm

from src_asserts.common_glm_asserts import (
    assert_development_years,
    assert_nonnegative,
    assert_positive_finite_exposure,
    assert_unique_index,
    require,
)


def assert_valid_power(power):
    """Tweedie-power må ligge i det åpne intervallet (1, 2)."""
    require(
        np.isfinite(power) and 1 < power < 2,
        f"Tweedie-power må være endelig og 1 < p < 2, fikk {power}.",
    )


def assert_tweedie_frame(model_frame, development):
    """Pure-premium-identitet, gyldig eksponering/incurred og alle poliseår (også nullene) beholdt."""
    assert_development_years(development)
    assert_unique_index(model_frame)
    require(
        model_frame.index.equals(development.index),
        "Tweedie-rammen må ha samme indeks som hele utviklingspopulasjonen.",
    )
    assert_positive_finite_exposure(model_frame)
    assert_nonnegative(model_frame["property_incurred"], "property_incurred")
    require(
        np.allclose(
            model_frame["pure_premium"],
            model_frame["property_incurred"] / model_frame["total_exposure"],
        ),
        "pure_premium må være property_incurred / total_exposure.",
    )
    zero_rows = model_frame["property_incurred"].eq(0)
    require(
        zero_rows.any() and model_frame["pure_premium"].eq(0).sum() == zero_rows.sum(),
        "Skadefrie poliseår (pure_premium = 0) må beholdes som nullobservasjoner.",
    )


def assert_tweedie_spec(spec):
    """Rate-respons, Tweedie med log-link, eksponeringsvekt, ingen offset og gyldig power."""
    require(spec["y"] == "pure_premium", "Tweedie-responsen må være pure_premium.")
    require(
        spec["weight"] == "total_exposure", "Tweedie-vekten må være total_exposure."
    )
    require(spec["family"] == "tweedie", "Familien må være Tweedie.")
    require(
        isinstance(spec["glm_family"].link, sm.families.links.Log),
        "Tweedie må bruke log-link.",
    )
    require(
        "offset" not in spec["formula"].lower(),
        "Tweedie skal ikke ha offset i formelen.",
    )
    assert_valid_power(spec["power"])
    require(
        spec["glm_family"].var_power == spec["power"],
        "Familiens var_power og spesifikasjonens power må være like.",
    )


def assert_tweedie_fit(result, design, spec):
    """Fittet modell bruker eksponering som var_weights og har ingen offset/eksponering."""
    require(
        np.allclose(result.model.var_weights, design[spec["weight"]]),
        "var_weights må være total_exposure.",
    )
    require(
        result.model.offset is None and result.model.exposure is None,
        "Tweedie-modellen skal ikke ha offset eller eksponeringsledd.",
    )


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
