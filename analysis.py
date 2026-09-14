# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Motorforsikring: innlasting, datarensing og kvalitetskontroll
#
# Notebooken viser resultatene fra datakvalitetsmodulen. Variabelordbok,
# rensing og integritetsdiagnostikk ligger i `src/data_quality.py`.

# %%
from pathlib import Path

import pandas as pd
from IPython.display import display

from src.data_quality import (
    build_integrity_diagnostics,
    build_variable_dictionary,
    clean_motor_data,
    find_variables,
    run_integrity_checks,
)

pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 120)
DATA_PATH = Path("data/Dataset of motor insurance portfolio.csv")
DESCRIPTION_PATH = Path("data/Descriptive of variables.xlsx")
if not DATA_PATH.exists() or not DESCRIPTION_PATH.exists():
    raise FileNotFoundError("Kjør notebooken fra prosjektroten slik at begge filene i data/ er tilgjengelige.")

# %% [markdown]
# ## 1. Uendret innlasting og første inspeksjon
#
# CSV-en leses før rensing. Excel-filen brukes som kildekontroll, mens den
# forbedrede variabelordboken bygges i scriptet.

# %%
raw_data = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8", low_memory=False)
source_dictionary_raw = pd.read_excel(DESCRIPTION_PATH, sheet_name="Cartera", usecols=["Variables", "Description", "Group"])
source_variables = source_dictionary_raw["Variables"].dropna().astype(str).tolist()
variable_dictionary = build_variable_dictionary()
raw_overview = pd.Series({
    "Antall rader": len(raw_data),
    "Antall kolonner": raw_data.shape[1],
    "Unike insured_id": raw_data["insured_id"].nunique(),
    "Duplikate hele rader": int(raw_data.duplicated().sum()),
    "Duplikate (insured_id, year)": int(raw_data.duplicated(["insured_id", "year"]).sum()),
    "Minste år": int(raw_data["year"].min()),
    "Største år": int(raw_data["year"].max()),
    "Variabler dokumentert i Excel": len(source_variables),
    "Excel- og CSV-variabler er identiske": source_variables == raw_data.columns.tolist(),
    "Minnestørrelse (MiB)": round(raw_data.memory_usage(deep=True).sum() / 1024**2, 1),
}, name="Verdi").to_frame()
display(raw_overview)
display(raw_data.head())

# %%
raw_schema = pd.DataFrame({
    "raw_dtype": raw_data.dtypes.astype(str),
    "missing_count": raw_data.isna().sum(),
    "missing_percent": raw_data.isna().mean().mul(100).round(4),
    "unique_count": raw_data.nunique(dropna=False),
    "example": [raw_data[column].dropna().iloc[0] if raw_data[column].notna().any() else pd.NA for column in raw_data.columns],
})
display(raw_schema)
display(raw_data.isna().groupby(raw_data["year"]).sum().loc[:, lambda frame: frame.sum().gt(0)].T)

# %% [markdown]
# ## 2. Variabelordbok
#
# `variable_dictionary` er en dataframe bygget i scriptet og kan filtreres
# senere med `find_variables(variable_dictionary, ...)`.

# %%
display(variable_dictionary)
display(find_variables(variable_dictionary, group="Skadeantall"))
display(find_variables(variable_dictionary, search="ansvar"))

# %% [markdown]
# ## 3. Rensing
#
# Rensingen låser skjema og datatyper, standardiserer kategorier og gjør kun den
# dokumenterte endringen av ikke-positive `power_to_weight_ratio`-verdier.

# %%
data, cleaning_log = clean_motor_data(raw_data, variable_dictionary)
display(cleaning_log)
display(pd.DataFrame({"raw_dtype": raw_data.dtypes.astype(str), "clean_dtype": data.dtypes.astype(str), "missing_after_cleaning": data.isna().sum()}))

# %% [markdown]
# ## 4. Integritetsdiagnostikk
#
# Alle kontroller kjøres i scriptet. Notebooken presenterer oppsummeringene og
# beholder dem som dataframes/datastrukturer for senere oppslag.

# %%
integrity_checks = run_integrity_checks(data, variable_dictionary)
diagnostics = build_integrity_diagnostics(data, integrity_checks)
critical_failures = diagnostics["critical_failures"]
warning_examples = diagnostics["warning_examples"]
coverage_diagnostics = diagnostics["coverage_diagnostics"]
temporal_diagnostics = diagnostics["temporal_diagnostics"]
display(integrity_checks)
display(warning_examples)
display(coverage_diagnostics)
display(temporal_diagnostics)
if not critical_failures.empty:
    raise AssertionError("Minst én kritisk integritetskontroll feilet. Se critical_failures.")

# %% [markdown]
# ## 5. Renseutfall og avgrensning for neste fase
#
# Det rensede analysegrunnlaget ligger i `data`. Ingen rader er slettet og ingen
# verdier er imputert. Diagnosene er funn som må vurderes før modellering.
