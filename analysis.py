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
#     display_name: MotorForsikring (3.12.14.final.0)
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
from src.pre_split_diagnostics import build_pre_split_diagnostics

pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 120)
DATA_PATH = Path("data/Dataset of motor insurance portfolio.csv")
DESCRIPTION_PATH = Path("data/Descriptive of variables.xlsx")
if not DATA_PATH.exists() or not DESCRIPTION_PATH.exists():
    raise FileNotFoundError(
        "Kjør notebooken fra prosjektroten slik at begge filene i data/ er tilgjengelige."
    )

# %% [markdown]
# ## 1. Uendret innlasting og første inspeksjon
#
# CSV-en leses før rensing. Excel-filen brukes som kildekontroll, mens den
# forbedrede variabelordboken bygges i scriptet.

# %%
raw_data = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8", low_memory=False)
source_dictionary_raw = pd.read_excel(
    DESCRIPTION_PATH,
    sheet_name="Cartera",
    usecols=["Variables", "Description", "Group"],
)
source_variables = source_dictionary_raw["Variables"].dropna().astype(str).tolist()
variable_dictionary = build_variable_dictionary()
raw_overview = pd.Series(
    {
        "Antall rader": len(raw_data),
        "Antall kolonner": raw_data.shape[1],
        "Unike insured_id": raw_data["insured_id"].nunique(),
        "Duplikate hele rader": int(raw_data.duplicated().sum()),
        "Duplikate (insured_id, year)": int(
            raw_data.duplicated(["insured_id", "year"]).sum()
        ),
        "Minste år": int(raw_data["year"].min()),
        "Største år": int(raw_data["year"].max()),
        "Variabler dokumentert i Excel": len(source_variables),
        "Excel- og CSV-variabler er identiske": source_variables
        == raw_data.columns.tolist(),
        "Minnestørrelse (MiB)": round(
            raw_data.memory_usage(deep=True).sum() / 1024**2, 1
        ),
    },
    name="Verdi",
).to_frame()
display(raw_overview)
display(raw_data.head())

# %%
raw_schema = pd.DataFrame(
    {
        "raw_dtype": raw_data.dtypes.astype(str),
        "missing_count": raw_data.isna().sum(),
        "missing_percent": raw_data.isna().mean().mul(100).round(4),
        "unique_count": raw_data.nunique(dropna=False),
        "example": [
            raw_data[column].dropna().iloc[0]
            if raw_data[column].notna().any()
            else pd.NA
            for column in raw_data.columns
        ],
    }
)
display(raw_schema)
display(
    raw_data.isna()
    .groupby(raw_data["year"])
    .sum()
    .loc[:, lambda frame: frame.sum().gt(0)]
    .T
)

# %% [markdown]
# ## 2. Variabelordbok
#
# `variable_dictionary` er en dataframe bygget i scriptet og kan filtreres
# senere med `find_variables(variable_dictionary, ...)`.

# %%
display(find_variables(variable_dictionary, search="policy_type"))

# %%
raw_data[raw_data['power_to_weight_ratio'] <= 0].head()

# %% [markdown]
# ## 3. Beslutninger etter datakontroll
#
# Følgende beslutninger gjelder dette datasettet:
#
# - `power_to_weight_ratio <= 0` settes til manglende. Tre rader omfattes.
# - `age_driving_licence` tolkes som antall år poliseholderen har hatt førerkort.
#   Verdier under 18 er derfor ikke et datakvalitetsavvik.
# - En kansellert polise med full eksponering beholdes uendret.
# - Skader med null `incurred` beholdes som nullskader.
# - Polise `(insured_id=23744, year=2022)` har en ansvarsskade, men null
#   eksponering. Både `total_exposure` og `liability_exposure` settes til 1, slik
#   at de fortsatt er konsistente.
# - Positiv ansvarseksponering med null ansvarspremie beholdes. Kontrollen fant 23
#   slike rader for 17 poliser; én av polisene har positiv ansvarspremie i senere
#   år. Nullpremien behandles derfor som et premieavvik, ikke som null risiko.
#   Radene skal flagges ved senere premieanalyse, men eksponeringen brukes i
#   skademodeller.

# %% [markdown]
# ## 4. Rensing
#
# Rensingen låser skjema og datatyper, standardiserer kategorier og utfører de
# dokumenterte korreksjonene over. Ingen rader slettes eller imputeres.

# %%
data, cleaning_log = clean_motor_data(raw_data, variable_dictionary)
display(cleaning_log)
display(
    pd.DataFrame(
        {
            "raw_dtype": raw_data.dtypes.astype(str),
            "clean_dtype": data.dtypes.astype(str),
            "missing_after_cleaning": data.isna().sum(),
        }
    ).sort_values('missing_after_cleaning', ascending=False).head(6)
)

# %%
display(
    data.loc[
        data["power_to_weight_ratio"].isna(),
        ["insured_id", "year", "power_to_weight_ratio"],
    ]
)

# %% [markdown]
# ## 5. Integritetsdiagnostikk
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
    raise AssertionError(
        "Minst én kritisk integritetskontroll feilet. Se critical_failures."
    )

# %% [markdown]
# ## 6. Renseutfall og avgrensning for neste fase
#
# Det rensede analysegrunnlaget ligger i `data`. Ingen rader er slettet og ingen
# verdier er imputert. Diagnosene er funn som må vurderes før modellering.

# %% [markdown]
# # Deskriptiv analyse
#
# Før train/test-splitt gjør vi bare strukturell porteføljediagnostikk. Tabellene
# under brukes til å definere risikopopulasjoner og vurdere tidsstrukturen, ikke
# til å velge prediktorer eller transformasjoner fra skadeutfall.

# %% [markdown]
# ## 7. Porteføljestørrelse og tidsstruktur
#
# `først_observert_i_datasettet` betyr første observasjon i denne treårsperioden;
# det er ikke nødvendigvis en reelt nytegnet polise. Eksponering oppsummeres som
# poliseår, og delårseksponering er strengt mellom null og én.

# %%
pre_split = build_pre_split_diagnostics(data)
display(pre_split["time_summary"])

# %% [markdown]
# ## 8. Panelstruktur
#
# Disse tabellene viser hvor lenge poliser observeres og hvor mye overlapp det er
# mellom år. De begrunner en tidsbasert split fremfor tilfeldig splitting på radnivå.

# %%
display(pre_split["panel_duration"])
display(pre_split["panel_transitions"])

# %% [markdown]
# ## 9. Porteføljesammensetning
#
# Sammensetningen vises per år uten skadeutfall. `vehicle_brand` oppsummeres
# separat fordi variabelen har mange nivåer.

# %%
display(pre_split["composition_summary"])
display(pre_split["brand_summary"])
display(pre_split["top_brand_summary"])

# %% [markdown]
# ## 10. Dekningsmatrise
#
# Ansvar anses aktivt når `liability_exposure > 0`. For andre dekninger brukes
# positiv totaleksponering og positiv dekningspremie som en praktisk proxy for at
# dekningen er aktiv. Avvik med skade eller incurred ved null premie vises, slik
# at regelen kan vurderes før modellering.

# %%
display(pre_split["coverage_summary"])
display(pre_split["coverage_by_policy_type"])

# %% [markdown]
# ## 11. Begrenset responsoppsummering
#
# Dette er kun en volumkontroll per dekning, ikke analyse av skadeutfall mot
# risikofaktorer. Klassifiseringen angir om skadevolumet grovt sett kan støtte en
# selvstendig modell; den erstatter ikke senere faglig vurdering av credibility.

# %%
display(pre_split["response_summary"])

# %% [markdown]
# ## 12. Eksponeringskontroll
#
# Eksponeringen skal senere brukes som offset i frekvensmodeller. Tabellen
# kontrollerer derfor nivåene og om skadeaktivitet forekommer ved null eksponering.

# %%
display(pre_split["exposure_summary"])
display(pre_split["exposure_checks"])
