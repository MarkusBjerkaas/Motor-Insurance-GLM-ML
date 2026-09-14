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
# Denne notebooken etablerer et reproduserbart og konservativt utgangspunkt for
# senere prising og skademodellering. Formålet her er **datakvalitet**, ikke
# deskriptiv analyse eller modellering.
#
# Arbeidsrekkefølgen er bevisst:
#
# 1. Les og inspiser den publiserte filen uten å endre den.
# 2. Dokumenter variablene og forventet semantikk.
# 3. Utfør kun sikre og sporbare type-/formatendringer.
# 4. Kjør integritetskontroller og skill mellom feil og forhold som må undersøkes.
#
# Datakilder:
#
# - `data/Dataset of motor insurance portfolio.csv`
# - `data/Descriptive of variables.xlsx`
# - [Espinosa, Lledó og Atance (2026), *A detailed dataset of motor insurance
#   policies with coverage-specific financial information*](https://pmc.ncbi.nlm.nih.gov/articles/PMC13234478/)
#
# Artikkelen beskriver 354 140 poliseår fra en spansk motorportefølje i
# 2022–2024. Forfatterne har allerede beholdt siste registrering per poliseår,
# standardisert kategorier og kontrollert numeriske områder og summer. CSV-en må
# derfor forstås som et publisert, forhåndsrenset datasett – ikke rådata fra
# selskapets kjernesystem.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

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
# CSV-en er UTF-8-kodet og semikolonseparert. Før rensefunksjonen defineres,
# leses filen med pandas sin ordinære typeinferens. Dette gir et etterprøvbart
# bilde av hva som faktisk finnes i kilden, og hindrer at typekonvertering skjuler
# uventede verdier.

# %%
raw_data = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8", low_memory=False)
source_dictionary_raw = pd.read_excel(
    DESCRIPTION_PATH,
    sheet_name="Cartera",
    usecols=["Variables", "Description", "Group"],
)
source_variables = source_dictionary_raw["Variables"].dropna().astype(str).tolist()

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

missing_by_year = (
    raw_data.isna()
    .groupby(raw_data["year"])
    .sum()
    .loc[:, lambda frame: frame.sum().gt(0)]
    .T
)
display(missing_by_year)

# %% [markdown]
# De følgende tabellene er inspeksjon for datarensing, ikke en porteføljeanalyse.
# De viser om kategorier, numeriske områder og manglende verdier stemmer med
# dokumentasjonen før vi låser skjemaet.

# %%
raw_categorical_levels = pd.DataFrame(
    [
        {
            "variable": column,
            "missing_count": int(raw_data[column].isna().sum()),
            "level_count": int(raw_data[column].nunique(dropna=True)),
            "observed_levels": ", ".join(
                sorted(raw_data[column].dropna().astype(str).unique())
            ),
        }
        for column in raw_data.select_dtypes(include=["object", "string"]).columns
    ]
)

raw_numeric_ranges = (
    raw_data.select_dtypes(include="number")
    .agg(["min", "max"])
    .T.rename(columns={"min": "observed_min", "max": "observed_max"})
)

display(raw_categorical_levels)
display(raw_numeric_ranges)

# %% [markdown]
# ### Resultat av første inspeksjon
#
# - Filen har de forventede 354 140 radene og 47 kolonnene. Nøkkelen
#   `(insured_id, year)` er unik, og det finnes ingen helt duplikate rader.
# - Variabellisten og rekkefølgen i Excel-filen er identisk med CSV-skjemaet.
# - Manglende verdier finnes bare i `vehicle_age` (2),
#   `age_driving_licence` (2), `fuel_type` (1 287) og `vehicle_value` (513).
# - Alle dokumenterte, lavkardinale kategorier bruker forventede koder.
# - Skadeantall, premier og påløpte skadekostnader er ikke-negative. Eksponering ligger i
#   intervallet 0–1.
# - Tre rader har `power_to_weight_ratio = 0`, selv om artikkelen sier at
#   ikke-positive verdier skal være manglende. Dette rettes eksplisitt nedenfor.
# - `age_driving_licence` er uklart dokumentert. Excel-tabellen kaller feltet et
#   kalenderår, artikkelteksten kaller det antall år siden førerkortet ble tatt,
#   mens observerte verdier (0–80, med stor konsentrasjon ved 18) ligner mest på
#   **alder da førerkortet ble tatt**. Feltet omdøpes ikke og verdiene endres ikke;
#   tolkningen må avklares før det brukes i en modell.

# %% [markdown]
# ## 2. Forbedret variabelordbok
#
# Tabellen under samler navn, forklaring, koding/enhet, analysefunksjon og dtype
# etter rensing. Forklaringene bygger på Excel-filen og artikkelen, men retter
# språklige uklarheter og markerer tvetydigheter. `insured_id` er en teknisk nøkkel,
# ikke en kontinuerlig forklaringsvariabel.

# %%
VARIABLE_DICTIONARY_ROWS = [
    (
        "insured_id",
        "Polise og kontrakt",
        "Anonym, sekvensiell identifikator for en polise. Samme ID kan forekomme i flere kalenderår.",
        "ID; ingen måleenhet",
        "Nøkkel, skal ikke brukes direkte som prediktor",
        "Int32",
        "Nei",
    ),
    (
        "year",
        "Polise og kontrakt",
        "Kalenderåret som poliseobservasjonen gjelder.",
        "2022, 2023 eller 2024",
        "Tidsindeks",
        "Int16",
        "Nei",
    ),
    (
        "policy_type",
        "Polise og kontrakt",
        "Overordnet produkt-/dekningsstruktur for polisen.",
        "TP=ansvar; TPG=ansvar+glass; CC=ansvar+minst to tilleggsdekninger; COMP_E=kasko med egenandel; COMP_N=kasko uten egenandel",
        "Kategorisk risikofaktor",
        "category",
        "Nei",
    ),
    (
        "policy_status",
        "Polise og kontrakt",
        "Status for polisen i det aktuelle kalenderåret.",
        "A=aktiv; C=kansellert/opphørt",
        "Kategorisk kontraktsinformasjon",
        "category",
        "Nei",
    ),
    (
        "business_type",
        "Polise og kontrakt",
        "Angir om observasjonen er nytegnet forretning eller eksisterende portefølje.",
        "NB=nytegning; P=portefølje/fornyelse",
        "Kategorisk kontraktsinformasjon",
        "category",
        "Nei",
    ),
    (
        "payment_frequency",
        "Polise og kontrakt",
        "Avtalt betalingsfrekvens for premien.",
        "A=årlig; S=halvårlig; Q=kvartalsvis",
        "Kategorisk kontraktsinformasjon",
        "category",
        "Nei",
    ),
    (
        "bonus_score",
        "Polise og kontrakt",
        "Grov klassifisering av forsikringstakerens tidligere skadeerfaring.",
        "G=god; N=nøytral; B=dårlig skadehistorikk",
        "Kategorisk/ordinal risikofaktor",
        "category",
        "Nei",
    ),
    (
        "driver_age",
        "Fører og kjøretøy",
        "Alder på hovedføreren som er registrert på polisen.",
        "Hele år",
        "Numerisk risikofaktor",
        "Int16",
        "Nei",
    ),
    (
        "vehicle_age",
        "Fører og kjøretøy",
        "Alder på det forsikrede kjøretøyet.",
        "Hele år",
        "Numerisk risikofaktor",
        "Int16",
        "Ja",
    ),
    (
        "age_driving_licence",
        "Fører og kjøretøy",
        "Uavklart førerkortvariabel. Verdiene tyder på alder da førerkortet ble tatt, men kildene beskriver feltet motstridende; må verifiseres før modellbruk.",
        "Hele år; foreløpig uklar semantikk",
        "Potensiell risikofaktor – ikke modellklar",
        "Int16",
        "Ja",
    ),
    (
        "fuel_type",
        "Fører og kjøretøy",
        "Drivstofftype for kjøretøyet.",
        "D=diesel; G=bensin",
        "Kategorisk risikofaktor",
        "category",
        "Ja",
    ),
    (
        "vehicle_value",
        "Fører og kjøretøy",
        "Oppgitt forsikringsverdi for kjøretøyet. Artikkelen bruker euro i premiepresentasjonen, men variabelarket oppgir ikke valuta eksplisitt for dette feltet.",
        "Beløp; valuta ikke eksplisitt angitt i variabelarket",
        "Numerisk risikofaktor",
        "float64",
        "Ja",
    ),
    (
        "seats",
        "Fører og kjøretøy",
        "Antall registrerte sitteplasser i kjøretøyet.",
        "Heltallsantall",
        "Numerisk risikofaktor",
        "Int16",
        "Nei",
    ),
    (
        "power_to_weight_ratio",
        "Fører og kjøretøy",
        "Til tross for feltnavnet beskriver kilden dette som kjøretøyets omtrentlige vekt per hestekraft, brukt som ytelsesindikator.",
        "kg per hk ifølge kilden",
        "Numerisk risikofaktor",
        "float64",
        "Ja",
    ),
    (
        "vehicle_brand",
        "Fører og kjøretøy",
        "Standardisert bilmerke registrert på polisen.",
        "68 observerte merkenavn",
        "Kategorisk risikofaktor",
        "category",
        "Nei",
    ),
    (
        "municipality_type",
        "Fører og kjøretøy",
        "Geografisk type for kommunen der kjøretøyet er forsikret.",
        "I=innland; C=kyst; IS=øyer",
        "Kategorisk geografisk faktor",
        "category",
        "Nei",
    ),
    (
        "circulation_area",
        "Fører og kjøretøy",
        "Om kjøretøyet hovedsakelig brukes i urbant eller ruralt område.",
        "U=urban; R=rural",
        "Kategorisk geografisk faktor",
        "category",
        "Nei",
    ),
]

PREMIUM_DESCRIPTIONS = {
    "total_premium": "Total nettopremie, lik summen av alle dekningspremiene i raden.",
    "liability_premium": "Nettopremie for ansvarsforsikring.",
    "property_damage_premium": "Nettopremie for egen kjøretøyskade/kaskoskade.",
    "theft_premium": "Nettopremie for tyveridekning.",
    "fire_premium": "Nettopremie for branndekning.",
    "glass_premium": "Nettopremie for glassdekning.",
    "legal_protection_premium": "Nettopremie for rettshjelpsdekning.",
    "occupants_premium": "Nettopremie for personskadedekning for passasjerer.",
}

for variable, description in PREMIUM_DESCRIPTIONS.items():
    VARIABLE_DICTIONARY_ROWS.append(
        (
            variable,
            "Premier",
            description,
            "Beløp (artikkelen presenterer premier i EUR); null betyr normalt at dekningen ikke er tegnet eller at eksponeringen er null",
            "Premie",
            "float64",
            "Nei",
        )
    )

CLAIM_DESCRIPTIONS = {
    "total_claims": "Totalt antall meldte skader på tvers av dekninger i eksponeringsperioden.",
    "liability_claims": "Totalt antall ansvarsskader; sum av materielle skader og personskader under ansvar.",
    "liability_property_claims": "Antall ansvarsskader med materiell skade på tredjepart.",
    "liability_injury_claims": "Antall ansvarsskader med personskade på tredjepart.",
    "property_claims": "Antall skader på eget kjøretøy under kaskodekning.",
    "theft_claims": "Antall tyveriskader.",
    "fire_claims": "Antall brannskader.",
    "glass_claims": "Antall glasskader.",
    "legal_protection_claims": "Antall rettshjelpsskader/-saker.",
    "occupants_claims": "Antall skader under passasjerdekningen.",
}

for variable, description in CLAIM_DESCRIPTIONS.items():
    VARIABLE_DICTIONARY_ROWS.append(
        (
            variable,
            "Skadeantall",
            description,
            "Ikke-negativt heltallsantall",
            "Respons-/utfallsvariabel",
            "Int16",
            "Nei",
        )
    )

INCURRED_DESCRIPTIONS = {
    "total_incurred": "Total påløpt skadekostnad, lik summen av incurred-beløpene på dekningsnivå.",
    "liability_incurred": "Påløpt kostnad for ansvarsskader; sum av materiell skade og personskade under ansvar.",
    "liability_property_incurred": "Påløpt kostnad for materielle ansvarsskader.",
    "liability_injury_incurred": "Påløpt kostnad for personskader under ansvar.",
    "property_incurred": "Påløpt kostnad for skade på eget kjøretøy.",
    "theft_incurred": "Påløpt kostnad for tyveriskader.",
    "fire_incurred": "Påløpt kostnad for brannskader.",
    "glass_incurred": "Påløpt kostnad for glasskader.",
    "legal_protection_incurred": "Påløpt kostnad for rettshjelpsskader/-saker.",
    "occupants_incurred": "Påløpt kostnad under passasjerdekningen.",
}

for variable, description in INCURRED_DESCRIPTIONS.items():
    VARIABLE_DICTIONARY_ROWS.append(
        (
            variable,
            "Påløpt skadekostnad",
            description,
            "Beløp (antatt EUR); betalt beløp pluss utestående reserve",
            "Respons-/utfallsvariabel",
            "float64",
            "Nei",
        )
    )

VARIABLE_DICTIONARY_ROWS.extend(
    [
        (
            "total_exposure",
            "Eksponering",
            "Tid polisen var eksponert for risiko i kalenderåret.",
            "Poliseår i intervallet [0, 1]",
            "Eksponering/offset",
            "float64",
            "Nei",
        ),
        (
            "liability_exposure",
            "Eksponering",
            "Eksponeringstid spesifikt for ansvarsdekningen.",
            "Poliseår i intervallet [0, 1]",
            "Eksponering/offset",
            "float64",
            "Nei",
        ),
    ]
)

variable_dictionary = pd.DataFrame(
    VARIABLE_DICTIONARY_ROWS,
    columns=[
        "variable",
        "group",
        "description",
        "unit_or_codes",
        "analysis_role",
        "clean_dtype",
        "missing_allowed",
    ],
)

if variable_dictionary["variable"].duplicated().any():
    raise ValueError("Variabelordboken inneholder duplikate variabelnavn.")

display(variable_dictionary)

# %%
def find_variables(search=None, group=None):
    """Finn variabler etter tekst og/eller eksakt gruppenavn."""
    result = variable_dictionary.copy()

    if group is not None:
        result = result.loc[result["group"].eq(group)]

    if search is not None:
        searchable_text = result[
            ["variable", "description", "unit_or_codes", "analysis_role"]
        ].astype("string").agg(" ".join, axis=1)
        result = result.loc[
            searchable_text.str.contains(search, case=False, regex=False, na=False)
        ]

    return result.reset_index(drop=True)


# Eksempler for senere bruk:
display(find_variables(group="Skadeantall"))
display(find_variables(search="ansvar"))

# %% [markdown]
# ## 3. Renseplan og beslutninger
#
# Rensingen er med vilje begrenset:
#
# - **Skjema og nøkkel:** Krev nøyaktig de 47 dokumenterte kolonnene og unik
#   `(insured_id, year)`. Uventet skjema skal gi feil, ikke passere stille.
# - **Tekst og kategorier:** Fjern omkringliggende mellomrom, bruk standardiserte
#   kodeverdier og konverter til `category`. Bilmerke normaliseres til store
#   bokstaver i samsvar med artikkelens beskrivelse.
# - **Datatyper:** ID, år, aldre, seter og skadeantall får eksplisitte nullable
#   heltallstyper. Beløp, forholdstall og eksponering beholdes som `float64` for å
#   unngå presisjonstap i summer og senere modeller.
# - **Sikre ugyldige verdier:** `power_to_weight_ratio <= 0` settes til manglende,
#   slik artikkelen sier. Ingen andre verdier endres.
# - **Ingen imputering eller radsletting:** De få manglende verdiene beholdes.
#   Null eksponering/premie og mulige longitudinelle avvik beholdes og rapporteres,
#   fordi automatisk sletting kan fjerne reelle kanselleringer, førerbytter eller
#   etterregistrerte skader.
# - **Ingen modellvariabler:** Frekvens, alvorlighet, skadeindikatorer, inflasjon og
#   eksponeringsjustering hører hjemme i senere analyse, ikke i datarensingen.

# %%
EXPECTED_COLUMNS = tuple(variable_dictionary["variable"])

CATEGORICAL_LEVELS = {
    "policy_type": ["TP", "TPG", "CC", "COMP_E", "COMP_N"],
    "policy_status": ["A", "C"],
    "business_type": ["NB", "P"],
    "payment_frequency": ["A", "S", "Q"],
    "bonus_score": ["G", "N", "B"],
    "fuel_type": ["D", "G"],
    "municipality_type": ["I", "C", "IS"],
    "circulation_area": ["U", "R"],
}

INTEGER_DTYPES = {
    "insured_id": "Int32",
    "year": "Int16",
    "driver_age": "Int16",
    "vehicle_age": "Int16",
    "age_driving_licence": "Int16",
    "seats": "Int16",
    **{variable: "Int16" for variable in CLAIM_DESCRIPTIONS},
}

FLOAT_COLUMNS = [
    *PREMIUM_DESCRIPTIONS,
    "vehicle_value",
    "power_to_weight_ratio",
    *INCURRED_DESCRIPTIONS,
    "total_exposure",
    "liability_exposure",
]


def validate_source_schema(frame):
    """Stopp før rensing dersom skjema, nøkkel eller faste koder har endret seg."""
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    extra_columns = sorted(set(frame.columns) - set(EXPECTED_COLUMNS))

    if missing_columns or extra_columns:
        raise ValueError(
            f"Uventet skjema. Mangler: {missing_columns}; ekstra: {extra_columns}."
        )

    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError("Kolonnerekkefølgen avviker fra det dokumenterte skjemaet.")

    if frame.columns.duplicated().any():
        raise ValueError("Kilden inneholder duplikate kolonnenavn.")

    duplicate_keys = int(frame.duplicated(["insured_id", "year"]).sum())
    if duplicate_keys:
        raise ValueError(f"Kilden inneholder {duplicate_keys} duplikate poliseår.")

    for column, allowed_levels in CATEGORICAL_LEVELS.items():
        observed_levels = set(
            frame[column].dropna().astype("string").str.strip().unique()
        )
        unexpected_levels = sorted(observed_levels - set(allowed_levels))
        if unexpected_levels:
            raise ValueError(
                f"{column} inneholder ukjente koder: {unexpected_levels}."
            )

    observed_years = set(pd.to_numeric(frame["year"], errors="raise").unique())
    if not observed_years.issubset({2022, 2023, 2024}):
        raise ValueError(f"Uventede kalenderår: {sorted(observed_years)}.")


def clean_motor_data(frame):
    """Utfør konservative og dokumenterte transformasjoner av publisert CSV."""
    validate_source_schema(frame)
    cleaned = frame.loc[:, EXPECTED_COLUMNS].copy()

    text_columns = [*CATEGORICAL_LEVELS, "vehicle_brand"]
    whitespace_changes = 0
    for column in text_columns:
        original = cleaned[column].astype("string")
        stripped = original.str.strip()
        whitespace_changes += int((original.notna() & original.ne(stripped)).sum())
        cleaned[column] = stripped

    cleaned["vehicle_brand"] = (
        cleaned["vehicle_brand"].str.upper().astype("category")
    )
    for column, levels in CATEGORICAL_LEVELS.items():
        category_type = pd.CategoricalDtype(categories=levels, ordered=False)
        cleaned[column] = cleaned[column].astype(category_type)

    for column, dtype in INTEGER_DTYPES.items():
        numeric_values = pd.to_numeric(cleaned[column], errors="raise")
        non_integer = numeric_values.dropna().mod(1).ne(0)
        if non_integer.any():
            raise ValueError(f"{column} inneholder verdier som ikke er heltall.")
        cleaned[column] = numeric_values.astype(dtype)

    for column in FLOAT_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise").astype(
            "float64"
        )

    invalid_power_ratio = cleaned["power_to_weight_ratio"].le(0)
    cleaned.loc[invalid_power_ratio, "power_to_weight_ratio"] = np.nan

    cleaning_log = pd.DataFrame(
        [
            {
                "step": "Kontroll av skjema, kategorikoder og unik poliseår-nøkkel",
                "affected_rows": 0,
                "result": "Bestått",
            },
            {
                "step": "Fjerning av omkringliggende mellomrom i tekstfelt",
                "affected_rows": whitespace_changes,
                "result": "Standardisert",
            },
            {
                "step": "Ikke-positiv power_to_weight_ratio satt til manglende",
                "affected_rows": int(invalid_power_ratio.sum()),
                "result": "Standardisert",
            },
            {
                "step": "Rader slettet eller imputert",
                "affected_rows": 0,
                "result": "Ingen",
            },
        ]
    )
    return cleaned, cleaning_log


data, cleaning_log = clean_motor_data(raw_data)

display(cleaning_log)
display(
    pd.DataFrame(
        {
            "raw_dtype": raw_data.dtypes.astype(str),
            "clean_dtype": data.dtypes.astype(str),
            "missing_after_cleaning": data.isna().sum(),
        }
    )
)

# %% [markdown]
# ## 4. Integritetsdiagnostikk etter rensing
#
# En kvalitetskontroll skal være reproduserbar og handlingsrettet. Tabellen skiller
# mellom:
#
# - **Kritisk:** brudd betyr at datasettet ikke bør brukes videre før feilen er
#   rettet.
# - **Undersøk:** kan være legitim forsikringslogikk eller datalivssyklus, men må
#   vurderes før den aktuelle variabelen inngår i analyse eller modell.
#
# Flyttallssummer sammenlignes med toleranse fordi binære flyttall kan gi svært
# små avrundingsdifferanser selv når kildesummene er konsistente.

# %%
PREMIUM_COMPONENTS = [
    "liability_premium",
    "property_damage_premium",
    "theft_premium",
    "fire_premium",
    "glass_premium",
    "legal_protection_premium",
    "occupants_premium",
]
CLAIM_COMPONENTS = [
    "liability_claims",
    "property_claims",
    "theft_claims",
    "fire_claims",
    "glass_claims",
    "legal_protection_claims",
    "occupants_claims",
]
INCURRED_COMPONENTS = [
    "liability_incurred",
    "property_incurred",
    "theft_incurred",
    "fire_incurred",
    "glass_incurred",
    "legal_protection_incurred",
    "occupants_incurred",
]


def run_integrity_checks(frame):
    """Returner radbaserte kvalitetskontroller uten å endre datasettet."""
    required_columns = variable_dictionary.loc[
        variable_dictionary["missing_allowed"].eq("Nei"), "variable"
    ].tolist()
    numeric_columns = frame.select_dtypes(include="number").columns
    finite_or_missing = np.isfinite(frame[numeric_columns]) | frame[
        numeric_columns
    ].isna()

    premium_sum = frame[PREMIUM_COMPONENTS].sum(axis=1)
    claim_sum = frame[CLAIM_COMPONENTS].sum(axis=1)
    incurred_sum = frame[INCURRED_COMPONENTS].sum(axis=1)

    checks = [
        (
            "Unik (insured_id, year)",
            "Kritisk",
            "Ingen duplikate poliseår",
            frame.duplicated(["insured_id", "year"]),
        ),
        (
            "Manglende i obligatoriske felt",
            "Kritisk",
            "Ingen manglende verdier",
            frame[required_columns].isna().any(axis=1),
        ),
        (
            "Endelige numeriske verdier",
            "Kritisk",
            "Ingen +inf eller -inf",
            ~finite_or_missing.all(axis=1),
        ),
        (
            "Ikke-negative premier",
            "Kritisk",
            "Alle premier >= 0",
            frame[["total_premium", *PREMIUM_COMPONENTS]].lt(0).any(axis=1),
        ),
        (
            "Ikke-negative skadeantall",
            "Kritisk",
            "Alle skadeantall >= 0",
            frame[["total_claims", *CLAIM_COMPONENTS]].lt(0).any(axis=1),
        ),
        (
            "Ikke-negative incurred-beløp",
            "Kritisk",
            "Alle incurred-beløp >= 0",
            frame[["total_incurred", *INCURRED_COMPONENTS]].lt(0).any(axis=1),
        ),
        (
            "Eksponering innenfor gyldig område",
            "Kritisk",
            "Begge eksponeringer i [0, 1]",
            ~frame[["total_exposure", "liability_exposure"]]
            .ge(0)
            .all(axis=1)
            | ~frame[["total_exposure", "liability_exposure"]]
            .le(1)
            .all(axis=1),
        ),
        (
            "Totalpremie summerer",
            "Kritisk",
            "total_premium = sum dekningspremier",
            ~np.isclose(frame["total_premium"], premium_sum, atol=1e-8, rtol=0),
        ),
        (
            "Totalt skadeantall summerer",
            "Kritisk",
            "total_claims = sum skadeantall per dekning",
            frame["total_claims"].ne(claim_sum),
        ),
        (
            "Ansvarsskadeantall summerer",
            "Kritisk",
            "liability_claims = property + injury under ansvar",
            frame["liability_claims"].ne(
                frame["liability_property_claims"]
                + frame["liability_injury_claims"]
            ),
        ),
        (
            "Total incurred summerer",
            "Kritisk",
            "total_incurred = sum incurred per dekning",
            ~np.isclose(frame["total_incurred"], incurred_sum, atol=1e-8, rtol=0),
        ),
        (
            "Ansvar incurred summerer",
            "Kritisk",
            "liability_incurred = property + injury under ansvar",
            ~np.isclose(
                frame["liability_incurred"],
                frame["liability_property_incurred"]
                + frame["liability_injury_incurred"],
                atol=1e-8,
                rtol=0,
            ),
        ),
        (
            "Total- og ansvarseksponering er like",
            "Kritisk",
            "total_exposure = liability_exposure",
            ~np.isclose(
                frame["total_exposure"],
                frame["liability_exposure"],
                atol=1e-12,
                rtol=0,
            ),
        ),
        (
            "Gyldig kjøretøyverdi",
            "Kritisk",
            "vehicle_value > 0 eller manglende",
            frame["vehicle_value"].notna() & frame["vehicle_value"].le(0),
        ),
        (
            "Gyldig vekt/effekt-forhold",
            "Kritisk",
            "power_to_weight_ratio > 0 eller manglende",
            frame["power_to_weight_ratio"].notna()
            & frame["power_to_weight_ratio"].le(0),
        ),
        (
            "Gyldig antall seter",
            "Kritisk",
            "seats er positivt heltall",
            frame["seats"].le(0),
        ),
        (
            "Plausibel føreralder",
            "Kritisk",
            "driver_age i [18, 100]",
            ~frame["driver_age"].between(18, 100),
        ),
        (
            "Kansellert polise med full eksponering",
            "Undersøk",
            "Kansellert polise forventes normalt å ha eksponering < 1",
            frame["policy_status"].eq("C") & frame["total_exposure"].eq(1),
        ),
        (
            "Positiv ansvarseksponering med null ansvarspremie",
            "Undersøk",
            "Positiv ansvarseksponering forventes normalt å ha positiv ansvarspremie",
            frame["liability_exposure"].gt(0)
            & frame["liability_premium"].eq(0),
        ),
        (
            "Skade eller incurred ved null eksponering",
            "Undersøk",
            "Null eksponering forventes normalt uten skadeaktivitet",
            frame["total_exposure"].eq(0)
            & (frame["total_claims"].gt(0) | frame["total_incurred"].gt(0)),
        ),
        (
            "Positiv incurred uten registrert skade",
            "Undersøk",
            "total_claims = 0 forventes normalt å gi total_incurred = 0",
            frame["total_claims"].eq(0) & frame["total_incurred"].gt(0),
        ),
        (
            "Registrert skade med null incurred",
            "Undersøk",
            "Kan være skader lukket uten utbetaling; vurderes før alvorlighetsmodell",
            frame["total_claims"].gt(0) & frame["total_incurred"].eq(0),
        ),
        (
            "Førerkortverdi høyere enn føreralder",
            "Undersøk",
            "age_driving_licence <= driver_age gitt foreløpig tolkning",
            frame["age_driving_licence"].gt(frame["driver_age"]),
        ),
        (
            "Førerkortverdi under 18",
            "Undersøk",
            "Verdier under 18 må vurderes når feltets semantikk er avklart",
            frame["age_driving_licence"].notna()
            & frame["age_driving_licence"].lt(18),
        ),
    ]

    results = pd.DataFrame(
        [
            {
                "check": name,
                "severity": severity,
                "expectation": expectation,
                "violating_rows": int(pd.Series(mask, index=frame.index).sum()),
            }
            for name, severity, expectation, mask in checks
        ]
    )
    results["status"] = np.where(
        results["violating_rows"].eq(0), "Bestått", "Undersøk"
    )
    return results


integrity_checks = run_integrity_checks(data)
display(integrity_checks)

critical_failures = integrity_checks.loc[
    integrity_checks["severity"].eq("Kritisk")
    & integrity_checks["violating_rows"].gt(0)
]
if not critical_failures.empty:
    raise AssertionError(
        "Minst én kritisk integritetskontroll feilet. Se critical_failures."
    )

# %% [markdown]
# For å gjøre advarslene enkle å følge opp vises høyst fem eksempelrader per
# utvalgt kontroll. `insured_id` og `year` kan brukes til å hente hele raden fra
# `data`; notebooken skriver ikke store mengder polisedata til resultatfeltet.

# %%
WARNING_MASKS = {
    "Kansellert polise med full eksponering": data["policy_status"].eq("C")
    & data["total_exposure"].eq(1),
    "Positiv ansvarseksponering med null ansvarspremie": data[
        "liability_exposure"
    ].gt(0)
    & data["liability_premium"].eq(0),
    "Skade eller incurred ved null eksponering": data["total_exposure"].eq(0)
    & (data["total_claims"].gt(0) | data["total_incurred"].gt(0)),
    "Registrert skade med null incurred": data["total_claims"].gt(0)
    & data["total_incurred"].eq(0),
    "Førerkortverdi under 18": data["age_driving_licence"].notna()
    & data["age_driving_licence"].lt(18),
}

EXAMPLE_COLUMNS = [
    "insured_id",
    "year",
    "policy_type",
    "policy_status",
    "business_type",
    "age_driving_licence",
    "liability_premium",
    "total_premium",
    "total_claims",
    "total_incurred",
    "total_exposure",
]

warning_samples = [
    data.loc[mask, EXAMPLE_COLUMNS].head(5).assign(check=name)
    for name, mask in WARNING_MASKS.items()
    if mask.any()
]
warning_examples = (
    pd.concat(warning_samples, ignore_index=True).loc[:, ["check", *EXAMPLE_COLUMNS]]
    if warning_samples
    else pd.DataFrame(columns=["check", *EXAMPLE_COLUMNS])
)
display(warning_examples)

# %% [markdown]
# ### Dekningssammenheng
#
# En skade på en dekning med null premie er ikke automatisk feil: premien kan være
# inkludert i en pakke, en kansellering kan gi null bokført årspremie, eller skaden
# kan være etterregistrert. Krysstabellen under gjør slike rader synlige uten å
# endre dem.

# %%
COVERAGE_COLUMNS = {
    "Ansvar": ("liability_premium", "liability_claims", "liability_incurred"),
    "Egen skade": (
        "property_damage_premium",
        "property_claims",
        "property_incurred",
    ),
    "Tyveri": ("theft_premium", "theft_claims", "theft_incurred"),
    "Brann": ("fire_premium", "fire_claims", "fire_incurred"),
    "Glass": ("glass_premium", "glass_claims", "glass_incurred"),
    "Rettshjelp": (
        "legal_protection_premium",
        "legal_protection_claims",
        "legal_protection_incurred",
    ),
    "Passasjer": ("occupants_premium", "occupants_claims", "occupants_incurred"),
}

coverage_diagnostics = pd.DataFrame(
    [
        {
            "coverage": coverage,
            "zero_premium_rows": int(data[premium].eq(0).sum()),
            "claim_with_zero_premium": int(
                (data[claims].gt(0) & data[premium].eq(0)).sum()
            ),
            "incurred_with_zero_premium": int(
                (data[incurred].gt(0) & data[premium].eq(0)).sum()
            ),
        }
        for coverage, (premium, claims, incurred) in COVERAGE_COLUMNS.items()
    ]
)
display(coverage_diagnostics)

# %% [markdown]
# ### Longitudinell konsistens
#
# Samme `insured_id` identifiserer en polise over tid, men registrert hovedfører
# eller kjøretøy kan endres. Derfor rapporteres hopp i aldersvariablene som
# diagnostikk, ikke som feil som skal overskrives. En uendret alder fra ett år til
# det neste kan også skyldes avrunding eller at kildesystemet ikke oppdaterte
# attributtet på årsskiftet.

# %%
def build_temporal_diagnostics(frame):
    """Oppsummer endringer i aldersfelt mellom observasjoner for samme polise."""
    ordered = frame.sort_values(["insured_id", "year"])
    grouped = ordered.groupby("insured_id", observed=True)
    year_gap = ordered["year"] - grouped["year"].shift()
    rows = []

    for column in ["driver_age", "vehicle_age", "age_driving_licence"]:
        change = ordered[column] - grouped[column].shift()
        comparable = year_gap.notna() & change.notna()
        rows.append(
            {
                "variable": column,
                "comparable_transitions": int(comparable.sum()),
                "unchanged": int((comparable & change.eq(0)).sum()),
                "decrease": int((comparable & change.lt(0)).sum()),
                "increase_larger_than_year_gap": int(
                    (comparable & change.gt(year_gap)).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


temporal_diagnostics = build_temporal_diagnostics(data)
display(temporal_diagnostics)

# %% [markdown]
# ## 5. Renseutfall og avgrensning for neste fase
#
# Det rensede analysegrunnlaget ligger i `data`. Renseprosessen har ikke slettet
# rader eller imputert verdier. Den har låst skjema og datatyper, standardisert
# kategorier og gjort tre dokumenterte nullverdier i
# `power_to_weight_ratio` om til manglende.
#
# Før modellering bør følgende avklares eller håndteres eksplisitt:
#
# 1. Bekreft semantikken til `age_driving_licence` med datasetteier/artikkelforfatter;
#    106 ikke-manglende verdier er under 18.
# 2. Bestem modellspesifikk behandling av null eksponering. Minst én slik rad har
#    skade og incurred-beløp og kan ikke brukes direkte med `log(exposure)` som
#    offset.
# 3. Undersøk én kansellert polise med full eksponering, positiv
#    ansvarseksponering med null ansvarspremie og de få dekningsskadene som har
#    null tilhørende premie.
# 4. For alvorlighetsmodell: bestem om registrerte skader med null incurred skal
#    ekskluderes, modelleres separat eller beholdes. Null incurred er ikke i seg
#    selv bevis på datakorrupsjon.
#
# Disse punktene er diagnostiske funn, ikke grunnlag for automatisk datatap i
# denne rensefasen.
