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
#     display_name: MotorForsikring (3.12.x)
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

import numpy as np
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
from src.own_damage_descriptives import (
    CATEGORICAL_VARS,
    NUMERIC_VARS,
    build_age_diagnostics,
    build_brand_distribution,
    build_brand_one_way,
    build_categorical_association,
    build_categorical_one_way,
    build_claim_count_distribution,
    build_missingness_outcome_comparison,
    build_numeric_correlation,
    build_numeric_one_way,
    build_overdispersion_summary,
    build_pure_premium_distribution_fit,
    build_pure_premium_summary,
    build_severity_distribution_fit,
    build_severity_summary,
    build_year_trend_summary,
    plot_age_diagnostics,
    plot_brand_distribution,
    plot_brand_one_way,
    plot_categorical_association_heatmap,
    plot_categorical_predictor_bars,
    plot_claim_count_distribution,
    plot_missingness_summary,
    plot_numeric_by_category_boxplot,
    plot_numeric_correlation_heatmap,
    plot_numeric_predictor_histograms,
    plot_one_way_grid,
    plot_pure_premium_distribution,
    plot_severity_distribution,
    plot_year_trend,
)
from src.portfolio_visuals import (
    plot_exposure_structure,
    plot_policy_type_composition,
)
from src.pre_split_diagnostics import (
    build_own_damage_scope_validation,
    build_pre_split_diagnostics,
)
from src.train_test_split import build_split_summary

pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 120)
DATA_PATH = Path("data/Dataset of motor insurance portfolio.csv")
DESCRIPTION_PATH = Path("data/Descriptive of variables.xlsx")
if not DATA_PATH.exists() or not DESCRIPTION_PATH.exists():
    raise FileNotFoundError(
        "Kjør notebooken fra prosjektroten slik at begge filene i data/ er tilgjengelige."
    )

seed = 100

# %% [markdown]
# # Arbeidsantakelser og sentrale begrensninger
#
# Før den videre analysen gjør vi følgende eksplisitte antakelser:
#
# - **Skadebeløpene brukes som levert i datasettet.** Artikkelen dokumenterer ikke om `property_incurred` er inflasjonsjustert eller indeksert, hvilket basisår som eventuelt er brukt, eller hvilken
# verdsettelsesdato beløpene gjelder. Vi foretar derfor ingen egen inflasjonsjustering og antar ikke at beløpene er indeksregulert.
#
# - **`property_incurred` behandles som beste tilgjengelige kostnadsmål, men ikke nødvendigvis som ultimate skadebeløp.** Artikkelen beskriver `incurred` som betalte beløp pluss utestående reserver, men gir ingen
# full dokumentasjon av skadeutvikling eller IBNR. Se [artikkelen](https://pmc.ncbi.nlm.nih.gov/articles/PMC13234478/).
#
# - **IBNR antas å være mindre problematisk for egen-skade enn for ansvarsskade**, fordi egen-skader normalt har kortere rapporterings- og oppgjørstid. Dette er likevel en praktisk arbeidsantakelse, ikke noe som
# kan verifiseres direkte fra datasettet.
#
# - **2024 behandles som et out-of-time-testsett for registrerte incurred-skader.** Publisering av datasettet i 2026 gjør det mulig at 2024-skadene er mer utviklet enn ved utgangen av 2024, men artikkelen oppgir
# ikke datauttrekksdato eller at skadebeløpene er ferdig utviklet. Testresultatene må derfor tolkes som ytelse mot registrerte/påløpte skadebeløp i uttrekket, ikke nødvendigvis mot ultimate kostnader.
#
# - **Modellvalg og variabelseleksjon baseres på 2022–2023.** 2024 holdes av til endelig evaluering og brukes ikke til å velge modell eller variabler. Vi må samtidig erkjenne at aggregerte testutfall allerede er
# synlige i den deskriptive analysen, slik at testsettet ikke lenger er fullstendig blindt.
#
# Disse antakelsene er en nødvendig forenkling for prosjektet. De innebærer at vi ikke tolker årseffekter som ren skadeinflasjon, og at vi omtaler resultatene som betinget på skadeutviklingen som foreligger i
# datasettet.
#

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
# - Kildedokumentasjonen for `age_driving_licence` og `vehicle_age` er
#   selvmotsigende. Arbeidshypotesen, som testes eksplisitt i seksjon 17, er at
#   `age_driving_licence` er alder ved førerkorterverv og at `vehicle_age` i
#   praksis er førerkortansiennitet. Dette er ikke endelig bevist siden
#   transformasjonskoden ikke er publisert. Reell kjøretøyalder behandles derfor
#   som utilgjengelig og `vehicle_age` brukes ikke som modellprediktor.
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
# ## 10. Responsvolum og eksponering per dekning
#
# Ren volumkontroll per dekning, ikke analyse av skadeutfall mot
# risikofaktorer. Egen skade har det klart høyeste skadevolumet blant
# kasko-relevante dekninger, men klart lavest eksponering — fordi
# risikopopulasjonen for egen skade er avgrenset til poliser med positiv
# `property_damage_premium` (se `_coverage_mask`), mens de øvrige dekningene
# dekker nær hele porteføljen.

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

# %%
# Som vi ser inneholder kun datasettet de forventede polisetypene etter avgrensningen.
df['policy_type'].unique()

# %% [markdown]
# Valideringen viser hvordan avgrensningen fordeler seg på produkttype og lister
# alle skader som faller utenfor regelen. Slike unntak beholdes som dokumenterte
# datavvik, men brukes ikke til å definere dekning fordi det ville innebære at
# skadeutfallet bestemmer modellpopulasjonen.

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
# # Deskriptiv analyse av modellvariablene
#
# Alt fra dette punktet bruker **kun `train_pool`** (2022-2023), i tråd med
# beslutningen i seksjon 12. Formålet er å forstå frekvens, severity og ren
# premie for egen skade, og å identifisere hvilke prediktorer som ser ut til
# å ha forklaringskraft, før modellspesifikasjon. `test` (2024) holdes urørt.

# %% [markdown]
# ## 13. Frekvens: fordeling og overspredning
#
# Fordelingen av `property_claims` viser hvor konsentrert skadeutfallet er
# (de fleste poliseår har 0 skader). Spredningsforholdet (varians/snitt)
# avgjør om Poisson (forhold ≈ 1) er en rimelig antakelse for frekvensmodellen,
# eller om overspredning krever negativ binomial.

# %%
claim_distribution = build_claim_count_distribution(train_pool)
display(claim_distribution)
display(plot_claim_count_distribution(claim_distribution))

# %%
display(build_overdispersion_summary(train_pool))

# %% [markdown]
# ## Implikasjoner:
# Variansen avviker betydelig fra gjennomsnittet og stemmer ikke overens med poisson fordeling. Undersøker videre om negativ binomialfordeling er mer egnet.

# %% [markdown]
# ## 14. Severity
#
# Severity (`property_incurred / property_claims`, kun poliseår med skade)
# er beregnet per poliseår, ikke per skademelding — ved flere skader i samme
# poliseår er tallet et gjennomsnitt. Gamma og lognormal sammenlignes med MLE
# og AIC på den strengt positive, **uvektede** delen; dette er en deskriptiv
# density-fit, ikke et endelig severitymodellvalg. Null-severity vises
# eksplisitt og inngår ikke i den kontinuerlige fitten. For
# fordelingsdiagnostikken behandles beløp med absoluttverdi ≤ 0,01 som numerisk
# null (én cent, gitt kildens EUR-presentasjon); rådata og sammendrag endres
# ikke. En eventuell
# claim-count-vektet fit ville besvare et annet spørsmål enn fordelingen av
# aggregerte poliseår og overlates til senere modellering.

# %%
display(build_severity_summary(train_pool))
severity_fit = build_severity_distribution_fit(train_pool)
display(severity_fit)
display(
    pd.Series(
        {
            "Numerisk nulltoleranse i fordelingsdiagnostikk": severity_fit.attrs["zero_tolerance"],
            "Eksakte null-severity": severity_fit.attrs["exact_zero_count"],
            "Nærnuller klassifisert som null kun i diagnostikk": severity_fit.attrs["near_zero_count"],
            "Minste substantielle positive severity": (
                train_pool.loc[train_pool["property_claims"].gt(0), "property_incurred"]
                .div(train_pool.loc[train_pool["property_claims"].gt(0), "property_claims"])
                .loc[lambda values: values.gt(severity_fit.attrs["zero_tolerance"])]
                .min()
            ),
            "Normalitetstest for log(severity), statistikk": severity_fit.attrs["normality_stat"],
            "Normalitetstest for log(severity), p-verdi": severity_fit.attrs["normality_pvalue"],
        },
        name="Verdi",
    ).to_frame()
)
display(plot_severity_distribution(train_pool, severity_fit))

# %% [markdown]
# Q-Q-panelet er relevant bare dersom lognormalfordeling vurderes. En Gamma GLM
# med log-link antar **ikke** lognormalitet. Formelle normalitetstester ved stort
# n avviser ofte små avvik og skal ikke velge modell alene; Q-Q, haletilpasning
# og senere out-of-sample deviance/kalibrering veier tyngre.
#
# Lognormal fordelingen bommer fremdeles betydelig i halen. Dette motiverer en vurdering av å modellere storskader og 

# %% [markdown]
# ## 15. Ren premie (severity × frekvens)
#
# Ren premie er `property_incurred / total_exposure` per poliseår — kombinerer
# frekvens og severity, og er størrelsen en Tweedie-modell til slutt skal
# predikere. Den har punktmasse i null (ingen skade) og en kontinuerlig,
# skjev hale for poliseår med kostnad. Gamma og lognormal sammenlignes kun for
# den positive delen med MLE/AIC; ingen observasjoner slettes, og absolutt-
# panelet begrenses transparent til p99 bare for lesbarhet.

# %%
display(build_pure_premium_summary(train_pool))
pure_premium_fit = build_pure_premium_distribution_fit(train_pool)
display(pure_premium_fit)
display(
    pd.Series(
        {
            "Numerisk nulltoleranse i fordelingsdiagnostikk": pure_premium_fit.attrs["zero_tolerance"],
            "Eksakte nuller i ren premie": pure_premium_fit.attrs["exact_zero_count"],
            "Nærnuller klassifisert som null kun i diagnostikk": pure_premium_fit.attrs["near_zero_count"],
            "Minste substantielle positive ren premie": (
                train_pool["property_incurred"] / train_pool["total_exposure"]
            ).loc[lambda values: values.gt(pure_premium_fit.attrs["zero_tolerance"])]
            .min(),
            "Normalitetstest for log(positiv ren premie), statistikk": pure_premium_fit.attrs["normality_stat"],
            "Normalitetstest for log(positiv ren premie), p-verdi": pure_premium_fit.attrs["normality_pvalue"],
        },
        name="Verdi",
    ).to_frame()
)
display(plot_pure_premium_distribution(train_pool, pure_premium_fit))

# %% [markdown]
# Den samlede renpremiefordelingen har punktmasse i null og er aktuarielt mer
# forenlig med compound Poisson–Gamma/Tweedie enn én kontinuerlig Gamma- eller
# lognormalfordeling. AIC-valget over beskriver derfor bare den betinget
# positive delen og erstatter ikke senere out-of-sample modellkontroll.

# %% [markdown]
# ## 16. Tidstrend i train_pool (2022 vs. 2023)
#
# Kun to år tilgjengelig i train_pool, men en sjekk av om frekvens/severity
# endrer seg mellom dem er relevant for om `year` bør inn som prediktor/offset,
# eller om skadeinflasjon bør vurderes separat.

# %%
year_trend = build_year_trend_summary(train_pool)
display(year_trend)
display(plot_year_trend(year_trend))

# %% [markdown]
# ## 17. Transparente transformasjoner og aldersdiagnostikk
#
# De publiserte kildene er selvmotsigende: variabelarket omtaler
# `age_driving_licence` som kalenderår, mens observerte verdier og de
# longitudinelle mønstrene er uforenlige med det. Arbeidshypotesen her er at
# feltet er **alder ved førerkorterverv**. Da blir
# `driving_experience_years = driver_age - age_driving_licence`, og
# `vehicle_age` er nesten det samme målet. Dette taler for at `vehicle_age`
# feilaktig representerer førerkortansiennitet, ikke reell kjøretøyalder.
#
# Hypotesen er svært godt støttet av data, men ikke endelig bevist: publisert
# transformasjonskode mangler. Reell kjøretøyalder behandles derfor som
# utilgjengelig. `vehicle_age` brukes bare i diagnosen nedenfor og aldri sammen
# med den utledede erfaringen som modellprediktor.

# %%
display(
    source_dictionary_raw.loc[
        source_dictionary_raw["Variables"].isin(
            ["driver_age", "age_driving_licence", "vehicle_age"]
        )
    ]
)
display(
    variable_dictionary.loc[
        variable_dictionary["variable"].isin(
            ["driver_age", "age_driving_licence", "vehicle_age"]
        )
    ]
)

# %% [markdown]
# Transformasjonene under er forhåndsbestemte og bruker ingen skadeutfall.
# Merke-poolingen læres utelukkende fra eksponering i `train_pool`: nivåer med
# minst 500 eksponeringsår beholdes, mens øvrige og hittil usette merker får
# `OTHER`. Den samme regelen brukes på testsettet. Eksponering er bare vekt/
# offset i rateberegningene, aldri en prediktor. `performance_hp_per_tonne =
# 1000 / power_to_weight_ratio` snur kildens kg-per-hk-mål: høyere verdi betyr
# flere hestekrefter per tonn og dermed høyere ytelse. Bare positive inputverdier
# transformeres; manglende input forblir manglende.

# %%
BRAND_MIN_EXPOSURE = 500.0
brand_exposure_train = train_pool.groupby("vehicle_brand", observed=True)[
    "total_exposure"
].sum()
retained_brands = brand_exposure_train.loc[
    brand_exposure_train.ge(BRAND_MIN_EXPOSURE)
].index.astype(str)


def add_transparent_predictors(frame, retained_brand_levels):
    """Legg til kun forhåndsdefinerte, ikke-responsbaserte prediktorer."""
    transformed = frame.copy()
    transformed["driving_experience_years"] = (
        transformed["driver_age"] - transformed["age_driving_licence"]
    )
    transformed["log_vehicle_value"] = np.log(transformed["vehicle_value"])
    transformed["performance_hp_per_tonne"] = 1000 / transformed[
        "power_to_weight_ratio"
    ]
    observed_brand = transformed["vehicle_brand"].astype("string")
    transformed["vehicle_brand_pooled"] = observed_brand.where(
        observed_brand.isin(retained_brand_levels), "OTHER"
    ).astype("category")
    return transformed


train_pool = add_transparent_predictors(train_pool, retained_brands)
test = add_transparent_predictors(test, retained_brands)
assert train_pool["driving_experience_years"].ge(0).all()
assert test["vehicle_brand_pooled"].isin([*retained_brands, "OTHER"]).all()
assert train_pool.loc[
    train_pool["power_to_weight_ratio"].notna(), "power_to_weight_ratio"
].gt(0).all()
assert train_pool["performance_hp_per_tonne"].isna().eq(
    train_pool["power_to_weight_ratio"].isna()
).all()
unseen_test_brands = set(test["vehicle_brand"].astype(str)) - set(
    train_pool["vehicle_brand"].astype(str)
)
assert test.loc[
    test["vehicle_brand"].astype(str).isin(unseen_test_brands),
    "vehicle_brand_pooled",
].eq("OTHER").all()
feature_summary = pd.Series(
    {
        "Merker beholdt ved minst 500 eksponeringsår": len(retained_brands),
        "Andel train-eksponering i beholdte merker (%)": (
            brand_exposure_train.loc[retained_brands].sum()
            / brand_exposure_train.sum()
            * 100
        ),
        "Testmerker mappet til OTHER": int(test["vehicle_brand_pooled"].eq("OTHER").sum()),
        "Usette testmerker mappet til OTHER": len(unseen_test_brands),
    },
    name="Verdi",
).to_frame()
display(feature_summary.round(2))

# %%
age_diagnostics = build_age_diagnostics(train_pool)
display(age_diagnostics["summary"])
display(age_diagnostics["residual_summary"])
display(age_diagnostics["temporal_changes"])
display(age_diagnostics["plausibility"])
display(plot_age_diagnostics(age_diagnostics))
assert age_diagnostics["plausibility"].loc["negativ_utledet_erfaring", "Verdi"] == 0
assert age_diagnostics["plausibility"].loc["førerkortalder_over_føreralder", "Verdi"] == 0

# %% [markdown]
# **Konklusjon.** Residualen `driver_age - age_driving_licence - vehicle_age`
# er praktisk talt null, og feltene endrer seg i tråd med dette mellom år for
# samme polise. Det støtter arbeidshypotesen svært sterkt. Begrensningen er at
# dette er en empirisk tolkning av et publisert datasett, ikke en verifisert
# datadefinisjon; modellen får derfor ikke en påstått reell kjøretøyalder.

# %% [markdown]
# ## 18. Bonus_score: foreløpig konklusjon
#
# Den separate [bonusdiagnostikken](bonus_score_analysis.ipynb) tyder på at
# `bonus_score` inneholder informasjon utover den eksplisitt observerte
# ettårige skadehistorikken. Timingmønsteret er mest konsistent med at scoren
# bygger på tidligere års skader, ikke inneværende års skadeutfall. Dette er
# ikke et endelig bevis på leakage-fri timing fordi eksakt *as-of*-dato mangler.
#
# `bonus_score` beholdes foreløpig som kategorisk prediktorkandidat, men
# vurderes på nytt i modelleringsfasen gjennom leakage-sikker sammenligning av
# modeller med/uten bonus og med eksplisitt lagget historikk, bare i train/CV.
# Teståret 2024 røres ikke.

# %% [markdown]
# ## 19. Prediktorer: univariate fordelinger
#
# Rene fordelinger for modellprediktorene, uten skadeutfall. De numeriske
# transformasjonene vises i stedet for råvariablene, slik at vi ikke senere
# inkluderer rå og transformert variant av samme informasjon samtidig.

# %%
display(plot_numeric_predictor_histograms(train_pool))
display(plot_categorical_predictor_bars(train_pool, columns=CATEGORICAL_VARS[:-1]))

# %%
brand_distribution = build_brand_distribution(train_pool)
display(brand_distribution)
display(plot_brand_distribution(brand_distribution))

# %% [markdown]
# ## 20. Manglende verdier i modellprediktorene
#
# Missingness følges for de transformerte prediktorene som faktisk går videre.
# `vehicle_age` er ikke med fordi det er et uavklart diagnostisk felt, ikke en
# modellvariabel.

# %%
missingness = build_missingness_outcome_comparison(train_pool)
display(missingness)
display(plot_missingness_summary(missingness))

# %% [markdown]
# ## 21. One-way-analyse: frekvens, severity og ren premie
#
# Oversiktsgridene viser eksponeringsandel og de tre ratene per nivå. Numeriske
# variabler er kun delt i faste diagnosebøtter her; senere modellering bør bruke
# kontinuerlige ledd eller splines. Stiplet linje er porteføljens rate. For
# frekvens vises omtrentlige 95 %-Poissonintervaller; severity og ren premie
# vises som punktestimat fordi aggregerte poliseår ikke gir skadeindivid-basert
# usikkerhet uten flere modellantakelser.

# %%
categorical_one_way = {
    column: build_categorical_one_way(train_pool, column)
    for column in CATEGORICAL_VARS[:-1]
}
numeric_one_way = {
    column: build_numeric_one_way(train_pool, column)
    for column in NUMERIC_VARS
}
display(plot_one_way_grid(categorical_one_way, train_pool, "kategoriske"))
display(plot_one_way_grid(numeric_one_way, train_pool, "numeriske"))

# %%
brand_one_way = build_brand_one_way(train_pool)
display(plot_brand_one_way(brand_one_way, train_pool))

# %% [markdown]
# ## 22. Samvariasjon blant prediktorer
#
# Relevant før GLM: sterkt korrelerte/assosierte prediktorer gir ustabile
# koeffisienter. Pearson-korrelasjon for de valgte numeriske prediktorene og
# Cramér's V (bias-korrigert) for kategoriske. `vehicle_age` og
# `age_driving_licence` er bevisst utelatt fra modellprediktor-listene.

# %%
correlation_matrix = build_numeric_correlation(train_pool)
display(correlation_matrix)
display(plot_numeric_correlation_heatmap(correlation_matrix))

# %%
association_matrix = build_categorical_association(train_pool)
display(association_matrix)
display(plot_categorical_association_heatmap(association_matrix))

# %%
display(plot_numeric_by_category_boxplot(train_pool, "log_vehicle_value", "policy_type"))

# %%
df['property_claims'].unique()

# %% [markdown]
# # 23. Oppsummering og konsekvenser for modelleringen
#
# ## Hva analysen samlet sett viser
#
# Analysen gir et godt nok grunnlag til å gå videre med en første modell for
# egen skade, men ikke til å låse endelig modell eller endelig variabelsett.
# Modellpopulasjonen består av 56 084 poliseår i 2022–2023, tilsvarende
# 37 904 eksponeringsår, 37 727 unike forsikringstakere og 10 126 skader.
# Datamengden er stor nok for ordinære GLM-er og noen kontrollerte, ikke-lineære
# effekter. Den er likevel ikke stor i alle undergrupper: særlig CC, sjeldne
# bilmerker, manglende-kategorier og eventuelle interaksjoner må behandles
# forsiktig.
#
# Egen-skadedekningen utgjør omtrent 31 % av eksponeringen i hele porteføljen og
# finnes bare for CC, COMP_E og COMP_N. I train-poolen dominerer COMP_E med
# 81,3 % av eksponeringen, mens COMP_N har 16,6 % og CC 2,1 %. Produktforskjellen
# er viktig: observert skadefrekvens er omtrent 0,18 for CC og COMP_E, men 0,69
# for COMP_N. Dette kan skyldes både risiko, egenandel og hvordan skader
# registreres. `policy_type` må derfor være med fra starten; den er mer enn en
# vanlig kontrollvariabel og beskriver ulike forsikringsvilkår.
#
# Skadeutfallet er nulltungt og høyreskjevt:
#
# - 89,7 % av poliseårene har ingen registrert skade.
# - Gjennomsnittlig skadeantall per rad er 0,181, mens variansen er 0,468
#   (rått varians/gjennomsnitt = 2,59).
# - Blant de 5 778 poliseårene med skade er uvektet gjennomsnittlig severity
#   1 059 EUR, medianen 617 EUR, p99 7 973 EUR og maksimum 27 333 EUR.
# - Porteføljens eksponeringsvektede frekvens er 0,267 skader per poliseår og
#   samlet ren premie er omtrent 232 EUR per eksponeringsår.
# - Ren premie per enkelt rad har en ekstrem hale. Maksimum på 1,19 mill. EUR
#   er en annualisert rate fra en skade på en svært kort eksponering, ikke en
#   skade på 1,19 mill. EUR. Slike rater må derfor ikke tolkes som skadebeløp
#   eller brukes til mekanisk trimming.
# - Bare omtrent en tredel av radene har fullårseksponering. Korrekt offset og
#   vekting er derfor avgjørende; radantall er ikke det samme som risikovolum.
#
# Det rå spredningsforholdet er et tydelig varsel om at en enkel Poisson-modell
# kan bli for snever, men det beviser ikke at negativ binomial er nødvendig.
# Beregningen er ujustert for ulik eksponering og ulik risikosammensetning.
# Dette avgjøres først etter at en Poisson-GLM med korrekt offset og sentrale
# risikofaktorer er tilpasset, ved å undersøke residualdispersjon, kalibrering
# og ytelse utenfor tilpasningsdataene. Den store nullandelen er heller ikke i
# seg selv grunnlag for en nullinflasjonsmodell.
#
# Severity og positiv ren premie passer beskrivende bedre til lognormal enn til
# Gamma målt med AIC, men ingen av fordelingene beskriver halen perfekt, og
# lognormalitet avvises klart. Sammenligningen er dessuten gjort på
# gjennomsnittlig skade per poliseår, ikke på individuelle skader. Den sier
# derfor lite om hvilken GLM som predikerer forventet kostnad best. Gamma med
# log-link er fortsatt et naturlig startpunkt for severity, med skadeantall som
# vekt når responsen er `property_incurred / property_claims`. Lognormal kan
# brukes som en utfordrermodell. De ni skadeårene med null eller praktisk talt
# null incurred må avklares og kan ikke inngå uendret i en Gamma-modell.
#
# Fra 2022 til 2023 øker frekvensen fra 0,235 til 0,282 (omtrent 20 %), severity
# faller fra 935 til 843 EUR (omtrent 10 %), og ren premie øker fra 220 til
# 238 EUR (omtrent 8 %). Dette kan være en reell tidseffekt, men også skyldes
# endret porteføljemiks, skadeoppgjørsmodning eller tilfeldig variasjon.
# Kalenderår kan brukes som kategorisk kontroll eller til senere rekalibrering;
# det er **ikke** en offset. To treningsår er for lite til å estimere en stabil
# lineær trend eller skadeinflasjon.
#
# ## Implikasjoner for variabelvalg
#
# One-way-analysene viser flere plausible kandidater, men er ikke kausale og
# skal ikke brukes som en automatisk utvelgelsesregel. Effektene kan endres når
# variablene vurderes samtidig.
#
# - **Høy prioritet:** `policy_type` har klart størst produktmessig betydning.
#   `bonus_score` har en sterk, monoton gradient i frekvens og ren premie
#   (omtrent 224, 351 og 567 EUR for G, N og B), men tas bare med dersom
#   tidspunktet for fastsettelsen kan dokumenteres som leakage-fritt.
# - **Kjøretøy:** Ren premie stiger tydelig med `log_vehicle_value` og
#   `performance_hp_per_tonne`, særlig i øverste intervall. Begge bør prøves som
#   kontinuerlige, fleksible ledd, for eksempel splines. Korrelasjonen på 0,56
#   betyr at stabilitet og marginalt bidrag må kontrolleres når begge er med.
# - **Alder og erfaring:** `driver_age` og `driving_experience_years` har
#   korrelasjon 0,88 og bør ikke uten videre inngå som to lineære hovedledd.
#   Alder viser tegn til en ikke-lineær effekt, med høyere frekvens blant de
#   eldste. Sammenlign fleksible aldersledd med en alternativ
#   erfaringsspesifikasjon i CV. `driver_age` er den sikreste hovedkandidaten
#   fordi den utledede erfaringen bygger på en ikke fullt verifisert
#   datatolkning.
# - **Geografi og kontrakt:** Diesel, urbant område, innland, kvartalsbetaling
#   og eksisterende portefølje har høyere observerte rater enn
#   sammenligningsgruppene. De er relevante kandidater, men forskjellene kan
#   være sammensetningseffekter. Det må også avklares at alle feltene er kjent
#   på prisingsdatoen og er akseptable tariffkriterier.
# - **Bilmerke:** Poolingen er leakage-sikker og beholder 17 merker som dekker
#   88,5 % av train-eksponeringen. One-way-ratene varierer, men merke henger
#   sammen med blant annet verdi og ytelse. Behold pooled merke som kandidat,
#   og krev dokumentert forbedring og stabile effekter i CV fremfor å tolke de
#   rå merkeratene direkte. Terskelen på 500 eksponeringsår bør også
#   sensitivitetskontrolleres.
# - **Lavere prioritet:** `seats` viser liten og lite entydig separasjon og kan
#   være en tidlig kandidat for utelatelse dersom den ikke tilfører verdi i en
#   multivariat modell. Lav one-way-effekt alene er likevel ikke nok til å
#   forkaste variabelen.
# - **Skal ikke brukes nå:** `vehicle_age` har feil eller uklar betydning og
#   utelates. Rå `age_driving_licence` bør heller ikke brukes ved siden av
#   utledet erfaring. Premier, skadeutfall, `insured_id` og informasjon som
#   først oppstår etter prisingsdatoen skal aldri brukes som risikoprediktorer.
#   `policy_status` holdes utenfor inntil det er dokumentert om statusen er kjent
#   ved periodens start; den sterke rå raten for kansellerte poliser kan skyldes
#   kort eksponering eller informasjon som oppstår i løpet av året.
#
# Manglendeverdiene er få, men ikke nødvendigvis tilfeldige. Manglende
# `fuel_type` (439 rader) og `vehicle_value` (33 rader) har høyere observert
# frekvens enn komplette rader, men gruppene er små. Bruk en eksplisitt og
# reproducerbar manglende-strategi som læres i hver CV-fold, og sørg for at
# produksjonspipelinen håndterer manglende verdier og nye kategorier selv om de
# ikke forekom i akkurat denne train-poolen.
#
# ## Anbefalt første modelleringsløp
#
# 1. Tilpass en enkel Poisson-frekvensmodell med `log(total_exposure)` som
#    offset og et lite, forhåndsdefinert sett av hovedledd. Sammenlign deretter
#    med negativ binomial på samme valideringsoppsett.
# 2. Tilpass en Gamma-severitymodell med log-link på positive skadebeløp, der
#    gjennomsnittlig severity vektes med antall skader. Sammenlign med en
#    lognormal utfordrer og undersøk kalibrering i halen.
# 3. Multipliser frekvens- og severityprediksjonene til forventet ren premie,
#    og sammenlign denne todelte modellen med en Tweedie-GLM på ren premie med
#    eksponering som vekt.
# 4. Legg til variabler i faglig begrunnede blokker. Velg kompleksitet etter
#    forbedring i out-of-fold deviance og kalibrering, samt fortegn og stabilitet
#    på tvers av foldene — ikke bare p-verdier eller one-way-rater.
# 5. Evaluer både samlet og innen viktige segmenter, spesielt `policy_type`,
#    år, nytegning/fornyelse og store/små predikerte risikoer. Rapporter
#    eksponeringsvektet kalibrering, deviance og rangering/lift.
#
# `GroupKFold` beskytter mot at samme `insured_id` havner på begge sider av en
# tilfeldig CV-fold, men tester ikke ren fremoverskuende generalisering. Med de
# tilgjengelige årene bør den suppleres med en sensitivitetsanalyse der 2022
# brukes til trening og 2023 til validering. Det er bare én slik tidsfold, så
# resultatet må tolkes med forsiktighet. Dersom koeffisientusikkerhet skal
# rapporteres, bør standardfeilene også ta hensyn til at samme forsikringstaker
# kan forekomme i flere år, for eksempel ved clustering på `insured_id`.
#
# ## Må undersøkes grundigere før endelig modellering
#
# Følgende er reelle avklaringspunkter, ikke bare ønskelige tillegg:
#
# - **Skademodning og prisnivå:** Finn verdsettelsesdato og kontroller om
#   `incurred` er like modent for alle år. Uten dette kan årsforskjeller og den
#   endelige testen blande modellfeil med etterslep i reserver. Avklar også om
#   beløpene skal inflasjonsjusteres til samme prisnivå.
# - **Egen-skadedekning:** Positiv premie er bare en proxy for aktiv dekning.
#   De to egen-skadesakene med null egen-skadepremie må undersøkes, og
#   dekningsregelen bør valideres mot produktvilkår eller et faktisk
#   dekningsflagg. Ellers kan modellpopulasjonen være selektert feil. I tillegg
#   kjøres unntakskontrollen i seksjon 11 på det allerede filtrerte `df`, slik
#   at den tomme unntakstabellen er sann per konstruksjon og ikke validerer hva
#   som falt utenfor. Kontrollen må kjøres på `data` før filteret anvendes.
# - **Eksponering og kansellering:** Kontroller frekvens og datakvalitet særskilt
#   for svært korte eksponeringer og kansellerte poliser. Avklar om skader kan
#   påvirke opphørstidspunkt/status. Den ekstreme annualiserte renpremien viser
#   hvorfor dette er viktig.
# - **Nullkostnadsskader og mange skader:** Undersøk de ni skadeårene med null
#   eller nær-null incurred og poliseårene med svært mange skader (opptil 15).
#   Avklar om dette er reelle nullskader, reserveringsstatus, dubletter eller en
#   aggregert registreringspraksis.
# - **Holdout-disiplin:** 2024 er omtalt som urørt test, men seksjon 12 viser
#   allerede skadeantall, skadeandel og incurred for 2024. Holdout-resultatet er
#   derfor ikke helt blindt. Lås nå modellvalg og akseptansekriterier basert på
#   2022–2023, og ikke bruk flere 2024-utfall før én samlet sluttevaluering.
# - **Porteføljedrift:** Sammenlign prediktor- og eksponeringsfordelinger mellom
#   2022, 2023 og 2024 uten å bruke 2024-skadeutfall. Sjekk særlig produktmiks,
#   nytegning/fornyelse, bonus, bilverdi, ytelse og andel korte eksponeringer.
# - **Bonus-timing:** Skaff en presis *as-of*-definisjon. Hvis den ikke finnes,
#   bør hovedresultatet vises både med og uten `bonus_score`; eksplisitt lagget
#   skadehistorikk må håndtere nytegning og manglende historikk separat.
# - **Påvirkningsanalyse:** Undersøk om et lite antall store skader driver
#   segmenteffekter og årsforskjeller. Vis resultater med robuste
#   sensitivitetsanalyser, men ikke kapp eller slett store skader uten en
#   aktuarfaglig begrunnelse.
# - **Usikkerhet i one-way-rater:** Frekvensintervallene er bare omtrentlige,
#   mens severity og ren premie mangler intervaller. Bruk gjerne bootstrap med
#   resampling på `insured_id` for å skille stabile mønstre fra tilfeldig støy,
#   særlig for små segmenter og bilmerker.
#
# Kort sagt: Start med transparente GLM-baseliner og et begrenset sett av
# faglig begrunnede variabler. De sterkeste kandidatene er produkt,
# bonus (dersom timing er trygg), kjøretøyverdi, ytelse, et fleksibelt aldersledd
# og noen få geografiske/kontraktsmessige faktorer. Den største risikoen nå er
# ikke mangel på modellkompleksitet, men uklar skade- og variabeltiming,
# skademodning, dekningsdefinisjon og at 2024-holdouten allerede er delvis sett.
