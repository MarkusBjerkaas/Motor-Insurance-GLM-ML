"""Kontroll av at alle ikke-utelatte variabler er kandidater i hver modell."""

import pandas as pd

# Variabler som bevisst ikke er kandidater, med begrunnelse (01_descriptiv §17 og planene).
EXCLUDED_VARIABLES = {
    "bonus_score": "tidspunkt for fastsettelse uavklart (lekkasjerisiko)",
    "policy_status": "ukjent om status er kjent ved periodestart",
    "vehicle_age": "uklar betydning i kildedokumentasjonen",
    "age_driving_licence": "rå verdi; erstattet av utledet erfaring",
    "driving_experience_years": "r = 0,88 mot driver_age; tolkningen er ikke fullt verifisert",
    "vehicle_value": "erstattet av log_vehicle_value",
    "power_to_weight_ratio": "erstattet av performance_hp_per_tonne",
    "vehicle_brand": "erstattet av vehicle_brand_pooled",
}
ID_COLUMNS = ("insured_id",)
OUTCOME_SUFFIXES = ("_premium", "_claims", "_incurred", "_exposure")
# Utledet term i kandidatregisteret -> kolonnen den bygger på
SOURCE_COLUMN = {"seat_category": "seats"}


def _is_outcome_or_id(column):
    return column in ID_COLUMNS or column.endswith(OUTCOME_SUFFIXES)


def build_variable_coverage(
    development, available_columns, tested_terms, final_columns
):
    """Én rad per variabel i utviklingsrammen, og feil hvis noe ikke er redegjort for.

    - ``available_columns``: kolonnene modellen får tilgang til.
    - ``tested_terms``: alle kolonner i kandidatregisteret (LOCKED, CORE, geografi,
      tillegg); må dekke ``available_columns``.
    - ``final_columns``: kolonnene i den endelige modellen.
    """
    tested = {SOURCE_COLUMN.get(term, term) for term in tested_terms}
    available = {SOURCE_COLUMN.get(col, col) for col in available_columns}
    final = {SOURCE_COLUMN.get(col, col) for col in final_columns}

    untested = available - tested
    if untested:
        raise ValueError(
            f"Tilgjengelige variabler som aldri ble testet: {sorted(untested)}"
        )
    unaccounted = [
        column
        for column in development.columns
        if not _is_outcome_or_id(column)
        and column not in available
        and column not in EXCLUDED_VARIABLES
    ]
    if unaccounted:
        raise ValueError(f"Variabler uten begrunnet status: {unaccounted}")

    rows = []
    for column in development.columns:
        if _is_outcome_or_id(column):
            continue
        if column in available:
            status = "i endelig modell" if column in final else "testet, ikke valgt"
            rows.append({"variabel": column, "status": status, "begrunnelse": ""})
        else:
            rows.append(
                {
                    "variabel": column,
                    "status": "utelatt",
                    "begrunnelse": EXCLUDED_VARIABLES[column],
                }
            )
    return pd.DataFrame(rows)
