"""Datagrunnlag for Tweedie-modellen: hele modellpopulasjonen med ``pure_premium``."""

import pandas as pd

from src.asserts.tweedie_asserts import assert_tweedie_frame
from src.core_glm.model_data import to_model_frame


def build_tweedie_frame(development, predictor_columns):
    """Bygg modellrammen for alle poliseår, også skadefrie (``pure_premium = 0``).

    ``pure_premium = property_incurred / total_exposure`` er observert
    skadekostnad per eksponeringsår. Indeksen er lik utviklingsrammens, slik at
    Tweedie-OOF senere kan sammenlignes rad for rad med frekvens × severity.
    Funksjonen velger ingen prediktorer: ``predictor_columns`` er bare
    kolonnene som gjøres tilgjengelige for kandidatregisteret.
    """
    columns = ["insured_id", "total_exposure", "property_incurred", *predictor_columns]
    model_frame = to_model_frame(development, list(dict.fromkeys(columns)))
    model_frame["pure_premium"] = (
        model_frame["property_incurred"] / model_frame["total_exposure"]
    )
    assert_tweedie_frame(model_frame, development)  # sanity-sjekk, kan fjernes
    return model_frame


def summarize_tweedie_frame(model_frame):
    """Én rad som dokumenterer Tweedie-populasjonen, inkludert andelen nullobservasjoner."""
    return pd.DataFrame(
        [
            {
                "Poliseår": len(model_frame),
                "Sum eksponering": model_frame["total_exposure"].sum(),
                "Sum incurred (EUR)": model_frame["property_incurred"].sum(),
                "Skadefrie poliseår (andel)": model_frame["pure_premium"].eq(0).mean(),
                "Unike insured_id": model_frame["insured_id"].nunique(),
            }
        ]
    )
