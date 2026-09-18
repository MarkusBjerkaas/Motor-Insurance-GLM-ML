"""Severity-fasens datagrunnlag: ``model_frame`` og ``severity_frame`` for Gamma-GLM.

Gjenbruker nøyaktig samme prediktorlister, ID-/utfallskolonner og B-15-masken
som ``glm_pricing_models.py`` (frekvensfasen, seksjon 1.1, rundt linje
203–245). Ingen nye kolonner eller terskler er lagt til her — konstruksjonen
er kopiert 1:1 derfra slik at severity bygger på nøyaktig samme datagrunnlag.

``build_severity_inputs`` tar imot utviklingsrammen fra
``src.model_data.build_development_frames()["development"]`` (kun 2022–2023,
2024 er aldri lest) og returnerer:

- ``model_frame``: hele utviklingspopulasjonen, alle poliseår (også
  skadefrie). Severity-CV-foldene defineres på hele populasjonens
  ``insured_id``, og modellen skal senere predikere alle poliseår i
  valideringsfolden — derfor trengs hele populasjonen, ikke bare
  severity-utvalget (se ``plans/Severity_plan.md``).
- ``severity_frame``: delmengden med registrert skade og materiell kostnad
  (B-15: ``property_claims > 0`` og ``property_incurred >
  CURRENCY_ZERO_TOLERANCE``), med kolonnen ``average_severity =
  property_incurred / property_claims``. Indeksen er en ekte delmengde av
  ``model_frame``s indeks (ikke resatt), slik at radene kan kobles entydig.
"""

import pandas as pd

from src.model_data import assert_development_years, to_model_frame
from src.own_damage_descriptives import CURRENCY_ZERO_TOLERANCE

# Identisk med glm_pricing_models.py (seksjon 1.1) — ingen nye kolonner.
CATEGORICAL_PREDICTORS = [
    "policy_type",
    "bonus_score",
    "fuel_type",
    "municipality_type",
    "circulation_area",
    "payment_frequency",
    "business_type",
    "vehicle_brand_pooled",
]
NUMERIC_PREDICTORS = [
    "driver_age",
    "driving_experience_years",
    "log_vehicle_value",
    "performance_hp_per_tonne",
    "seats",
]
PREDICTORS = CATEGORICAL_PREDICTORS + NUMERIC_PREDICTORS
ID_COLUMNS = ["insured_id", "year", "policy_status"]
OUTCOME_COLUMNS = ["total_exposure", "property_claims", "property_incurred"]


def build_severity_inputs(development):
    """Bygg ``model_frame`` (hele utviklingspopulasjonen) og ``severity_frame`` (B-15-utvalget).

    Kjører ``assert_development_years`` som inngangskontroll først, slik at
    funksjonen stopper umiddelbart dersom noe annet enn 2022–2023 skulle nå
    hit. Se modulens docstring for definisjonene av de to rammene.
    """
    assert_development_years(development)

    model_frame = to_model_frame(development, ID_COLUMNS + PREDICTORS + OUTCOME_COLUMNS)

    # B-15: severity er betinget på registrert skade med materiell kostnad.
    severity_mask = model_frame["property_claims"].gt(0) & model_frame[
        "property_incurred"
    ].gt(CURRENCY_ZERO_TOLERANCE)
    severity_frame = model_frame.loc[severity_mask].copy()
    severity_frame["average_severity"] = (
        severity_frame["property_incurred"] / severity_frame["property_claims"]
    )
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
