"""Severity-fasens datagrunnlag: ``model_frame`` og ``severity_frame`` for Gamma-GLM.

``build_severity_inputs`` tar imot utviklingsrammen fra
``src.core_glm.model_data.build_development_frames()["development"]`` (kun 2022–2023,
2024 er aldri lest) og returnerer:

- ``model_frame``: hele utviklingspopulasjonen, alle poliseår (også
  skadefrie), med bare ``insured_id``, kandidatkolonnene og utfallskolonnene.
  Severity-CV-foldene defineres på hele populasjonens ``insured_id``, og
  modellen skal senere predikere alle poliseår i valideringsfolden — derfor
  trengs hele populasjonen, ikke bare severity-utvalget.
- ``severity_frame``: delmengden med registrert skade og materiell kostnad
  (B-15: ``property_claims > 0`` og ``property_incurred >
  CURRENCY_ZERO_TOLERANCE``), med kolonnen ``average_severity =
  property_incurred / property_claims``. Indeksen er en ekte delmengde av
  ``model_frame``s indeks (ikke resatt), slik at radene kan kobles entydig.

Variabler utenfor ``available_columns`` (f.eks. ``bonus_score``) kommer aldri inn
i rammene, så de kan ikke lekke inn i modellen ved et uhell.
"""

import pandas as pd

from src.asserts.severity_asserts import assert_severity_frame
from src.core_glm.model_data import (
    add_seat_category,
    assert_development_years,
    to_model_frame,
)
from src.descriptives.own_damage_descriptives import CURRENCY_ZERO_TOLERANCE

# Samme kandidatkolonner som frekvens og Tweedie (``seats`` gir ``seat_category``).
AVAILABLE_COLUMNS = [
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
    "seats",
]
ID_COLUMNS = ["insured_id"]
OUTCOME_COLUMNS = ["total_exposure", "property_claims", "property_incurred"]


def build_severity_inputs(development, available_columns=AVAILABLE_COLUMNS):
    """Bygg ``model_frame`` (hele utviklingspopulasjonen) og ``severity_frame`` (B-15-utvalget).

    Kjører ``assert_development_years`` som inngangskontroll først, slik at
    funksjonen stopper umiddelbart dersom noe annet enn 2022–2023 skulle nå
    hit. Se modulens docstring for definisjonene av de to rammene.
    """
    assert_development_years(development)

    model_frame = to_model_frame(
        development, ID_COLUMNS + list(available_columns) + OUTCOME_COLUMNS
    )
    # Severity-prediktor: seter som <5, =5, >5. På model_frame slik at
    # også skadefrie poliseår kan predikeres.
    add_seat_category(model_frame)

    # B-15: severity er betinget på registrert skade med materiell kostnad.
    severity_mask = model_frame["property_claims"].gt(0) & model_frame[
        "property_incurred"
    ].gt(CURRENCY_ZERO_TOLERANCE)
    severity_frame = model_frame.loc[severity_mask].copy()
    severity_frame["average_severity"] = (
        severity_frame["property_incurred"] / severity_frame["property_claims"]
    )
    assert_severity_frame(model_frame, severity_frame, development)  # sanity-sjekk, kan fjernes
    return {"model_frame": model_frame, "severity_frame": severity_frame}


def summarize_severity_selection(model_frame, severity_frame):
    """Én lesbar oppsummeringsrad som dokumenterer severity-utvalget (B-15).

    Viser modellpopulasjonens størrelse og eksponering, skadeantallet og hvor
    mye av det som havner i severity-utvalget, samt de utelatte
    nær-null-kostnadsårene som fortsatt teller som skader i frekvens og ren
    premie, men ikke i severity.
    """
    excluded_mask = model_frame["property_claims"].gt(0) & model_frame[
        "property_incurred"
    ].le(CURRENCY_ZERO_TOLERANCE)

    return pd.DataFrame(
        [
            {
                "Poliseår i modellpopulasjonen": len(model_frame),
                "Sum eksponering": model_frame["total_exposure"].sum(),
                "Sum registrerte skader": int(model_frame["property_claims"].sum()),
                "Positive skadeår (severity-utvalg)": len(severity_frame),
                "Unike insured_id i severity-utvalget": severity_frame[
                    "insured_id"
                ].nunique(),
                "Sum skader i severity-utvalget": int(
                    severity_frame["property_claims"].sum()
                ),
                "Skadevektet snittseverity (EUR)": severity_frame[
                    "property_incurred"
                ].sum()
                / severity_frame["property_claims"].sum(),
                "Utelatte nær-null-poliseår": int(excluded_mask.sum()),
                "Registrerte skader i utelatte år": int(
                    model_frame.loc[excluded_mask, "property_claims"].sum()
                ),
            }
        ]
    )
