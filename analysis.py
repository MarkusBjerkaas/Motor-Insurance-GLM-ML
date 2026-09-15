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
#     display_name: Python 3
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
from sklearn.model_selection import GroupKFold

from src.coverage_visuals import (
    plot_coverage_by_policy_type_heatmap,
    plot_response_volume,
)
from src.data_quality import (
    build_integrity_diagnostics,
    build_variable_dictionary,
    clean_motor_data,
    find_variables,
    run_integrity_checks,
)
from src.portfolio_visuals import (
    plot_exposure_structure,
    plot_policy_type_composition,
)
from src.pre_split_diagnostics import (
    build_own_damage_scope_validation,
    build_pre_split_diagnostics,
)
from src.train_test_split import build_key_variable_balance, build_split_summary

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
raw_data[raw_data["power_to_weight_ratio"] <= 0].head()

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
    )
    .sort_values("missing_after_cleaning", ascending=False)
    .head(6)
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
# ## 7. Porteføljestørrelse, eksponering og panelstruktur
#
# Eksponeringsstatus per år (null / delår / full år) viser porteføljens vekst
# og at andelen fullårseksponerte poliser øker mot 2024. Panelstrukturen —
# hvor lenge poliser observeres — begrunner en tidsbasert train/test-splitt
# fremfor en tilfeldig radsplitt: samme polise opptrer typisk i flere år.

# %%
pre_split = build_pre_split_diagnostics(data)
display(plot_exposure_structure(data))

# %%
display(pre_split["panel_duration"])

# %% [markdown]
# 48 poliser har null eksponering og null premie i 2024 (aktive i 2022/2023,
# men ikke fornyet). De beholdes i datasettet for sporbarhet, men utelates fra
# skadeanalysen og train/test-splitten siden de ikke representerer risiko i
# noen periode med positiv eksponering.

# %%
assert pre_split["exposure_checks"]["antall_poliseår"].eq(0).all(), (
    "Uventet avvik i eksponeringskontrollen — se build_exposure_summary."
)

# %% [markdown]
# ## 8. Porteføljesammensetning: policy_type
#
# `policy_type` er den sentrale klassifiseringsvariabelen for dekning (se
# dekningsmatrisen under) og er stabil over tid. Øvrige
# sammensetningsvariabler er ikke kritiske for avgrensningsbeslutningen;
# `vehicle_brand` har særlig mange nivåer og egner seg bedre som prediktor
# senere i analysen enn som deskriptiv oversikt her.

# %%
display(plot_policy_type_composition(data))

# %% [markdown]
# ## 9. Dekningsmatrise
#
# Ansvar anses aktivt når `liability_exposure > 0`. For andre dekninger brukes
# positiv totaleksponering og positiv dekningspremie som en praktisk proxy for
# at dekningen er aktiv (ingen per-dekning eksponeringsvariabel finnes).
# Matrisen viser at egen skade (kasko) kun er aktiv for `policy_type` CC,
# COMP_E og COMP_N — konsistent med at TP/TPG er rene ansvarsprodukter.

# %%
display(plot_coverage_by_policy_type_heatmap(pre_split["coverage_by_policy_type"]))

# %%
display(pre_split["coverage_summary"])

# %% [markdown]
# ## 10. Responsvolum per dekning
#
# Ren volumkontroll per dekning, ikke analyse av skadeutfall mot
# risikofaktorer. Egen skade har det klart høyeste skadevolumet blant
# kasko-relevante dekninger og ligger godt over terskelen for selvstendig
# modellering.

# %%
display(plot_response_volume(pre_split["response_summary"]))

# %%
display(pre_split["response_summary"])

# %% [markdown]
# ## 11. Avgrensning til egen-skadedekning
#
# Basert på dekningsmatrisen (seksjon 9) og responsvolumet (seksjon 10)
# avgrenses videre analyse til poliseår med positiv `property_damage_premium`
# og positiv `total_exposure`. Premien brukes bare til å identifisere at
# dekningen er aktiv, ikke som prediktor. Denne avgrensningen gir en tydelig
# risikopopulasjon med tilstrekkelig skadevolum for både frekvens- og
# severitymodellering. Naturlige utvidelser er å gjenta samme analyse separat
# for de øvrige dekningene.

# %%
own_damage_mask = data["property_damage_premium"].gt(0) & data["total_exposure"].gt(0)
df = data.loc[own_damage_mask].copy()
scope_summary, scope_exceptions = build_own_damage_scope_validation(df)
display(scope_summary)
display(scope_exceptions)
assert df["property_damage_premium"].gt(0).all()
assert df["total_exposure"].gt(0).all()

# %% [markdown]
# Valideringen viser hvordan avgrensningen fordeler seg på produkttype og lister
# alle skader som faller utenfor regelen. Slike unntak beholdes som dokumenterte
# datavvik, men brukes ikke til å definere dekning fordi det ville innebære at
# skadeutfallet bestemmer modellpopulasjonen.

# %%
df.loc[(df["total_exposure"]) != (df["liability_exposure"])]

# %%
df.head()

# %% [markdown]
# ## 12. Train/test-splitt
#
# Splitten er tidsbasert (out-of-time), ikke en tilfeldig radsplitt: 2022 og
# 2023 slås sammen til en train/CV-pool, mens 2024 holdes urørt som endelig
# test. Dette speiler faktisk bruk av en prisingsmodell — fit på historikk,
# prises på neste års fornyelser — og fanger opp temporal drift (f.eks.
# skadeinflasjon) som en tilfeldig splitt ikke ville avslørt.
#
# Hyperparametertuning skjer med `GroupKFold` på `insured_id` innad i
# train/CV-pool, ikke vanlig radbasert k-fold: samme polise kan opptre i både
# 2022 og 2023, og uten gruppering kunne de to poliseårene til samme polise
# havnet i ulike foldere og gitt et for optimistisk CV-estimat. Testsettet
# rører vi ikke igjen før endelig evaluering. All videre deskriptiv/
# eksplorativ analyse av prediktorer bør fra nå av kun bruke `train_pool`.

# %%
TRAIN_YEARS = [2022, 2023]
TEST_YEAR = 2024


def make_train_test_split(frame, n_splits=5):
    """Del modellpopulasjonen i en train/CV-pool og en urørt testperiode."""
    train_pool = frame.loc[frame["year"].isin(TRAIN_YEARS)].copy()
    test = frame.loc[frame["year"].eq(TEST_YEAR)].copy()
    cv = GroupKFold(n_splits=n_splits)
    return train_pool, test, cv


def check_group_disjoint_folds(train_pool, cv, groups):
    """Bekreft at ingen insured_id opptrer i både train- og valideringsfold."""
    for fold, (train_idx, val_idx) in enumerate(cv.split(train_pool, groups=groups)):
        train_fold_ids = set(groups.iloc[train_idx])
        val_fold_ids = set(groups.iloc[val_idx])
        if train_fold_ids & val_fold_ids:
            raise ValueError(
                f"Fold {fold} har overlappende insured_id mellom train og val."
            )
    return True


# %%
train_pool, test, cv = make_train_test_split(df, n_splits=5)
split_summary, overlap_summary = build_split_summary(train_pool, test)
display(split_summary)
display(overlap_summary)

# %% [markdown]
# Overlappen mellom train/CV-pool og test er forventet og ønsket her — det
# er videreførte poliser som prises på nytt i 2024, ikke lekkasje. Til slutt
# bekreftes det at `GroupKFold` faktisk holder `insured_id` adskilt mellom
# train- og valideringsfold.

# %%
check_group_disjoint_folds(train_pool, cv, train_pool["insured_id"])

# %% [markdown]
# ### Balansediagnostikk: nøkkelvariabler i train/CV-pool vs. test
#
# Splitten er tidsbasert, ikke tilfeldig, så det er ikke gitt at fordelingen av
# sentrale risikofaktorer er lik i de to periodene. Tabellene under viser andel
# per kategori (kategoriske variabler) og standardized mean difference, SMD
# (numeriske variabler) mellom train/CV-pool og test. Store avvik er ikke i seg
# selv et problem — testsettet skal representere fremtidige fornyelser, ikke en
# tilfeldig delmengde av samme populasjon — men avvik bør være kjent før
# modellresultater tolkes.

# %%
categorical_balance, numeric_balance = build_key_variable_balance(train_pool, test)
display(categorical_balance)
display(numeric_balance)

# %%
