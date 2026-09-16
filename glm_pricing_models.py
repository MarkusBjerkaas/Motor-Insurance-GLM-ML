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
# # GLM-benchmark for egen skade: frekvens, severity og ren premie
#
# ## 0. Formål og omfang
#
# Målet er en transparent GLM-benchmark for forventet kaskokostnad (egen skade)
# per eksponeringsår. Benchmarken fryses og brukes som referanse senere
# ML-modeller må slå. Arbeidet er delt i fire faser:
#
# | Seksjon | Fase | Spørsmål som skal besvares |
# |---|---|---|
# | 3 | Frekvens | Hvilke variabler og hvilken form gir best out-of-fold frekvens? Poisson eller NB? |
# | 4 | Severity og storskader | Gamma eller lognormal? Bør storskader behandles separat? |
# | 5 | Ren premie | Er todelt modell (frekvens × severity) bedre enn én Tweedie-GLM? |
# | 6 | Konsolidert benchmark | Frys mesterspesifikasjonen og dokumenter relativiteter og svakheter |
#
# Seksjon 1 og 2 legger grunnmuren: datagrunnlag og evalueringsrammeverk.
# Seksjon 7 er beslutningsregisteret. Seksjon 3–6 legges til fase for fase.
#
# **Styringsdokument.** Planen ligger i
# [`plans/glm_pricing_models_plan.md`](plans/glm_pricing_models_plan.md). Den er
# levende: ved starten av hver ny fase leses den på nytt, vurderes mot
# resultatene fra forrige fase og skrives om ved behov. Hver revurdering logges i
# planens endringslogg.
#
# **Omfang.** Kun GLM, kun egen-skadedekning og kun train-poolen 2022–2023.
# Teståret 2024 åpnes ikke i denne notebooken (B-05). All modellsammenligning
# gjøres out-of-fold innenfor 2022–2023.
#
# **Beslutningsregister.** Alle valg som påvirker resultatet har en ID (B-xx)
# og er samlet i seksjon 7. Teksten henviser til ID-en der beslutningen brukes.
# Statusene betyr:
#
# - *Brukerbesluttet*: avklart eksplisitt i planleggingen.
# - *Foreslått*: faglig standardvalg som kan overprøves i etterkant.
# - *Datadrevet*: avgjøres av out-of-fold-resultater etter en regel som er låst
#   før modellene estimeres.

# %%
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from IPython.display import display
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import GroupKFold

from src.glm_diagnostics import (
    build_data_overview,
    build_fold_summary,
    build_glm_summary,
    build_relativity_table,
    summarize_cv_scores,
)
from src.model_data import TRAIN_YEARS, build_model_frames, to_model_frame
from src.own_damage_descriptives import CURRENCY_ZERO_TOLERANCE

pd.set_option("display.max_columns", 60)
SEED = 100
N_FOLDS = 5

# %% [markdown]
# ## 1. Datagrunnlag
#
# Datagrunnlaget bygges av `build_model_frames` i `src/model_data.py` (B-04).
# Funksjonen gjør nøyaktig de samme stegene som er begrunnet i
# [`analysis.ipynb`](analysis.ipynb), og begge notebookene importerer dem
# derfra:
#
# 1. **Innlesing og rensing** med `clean_motor_data` (seksjon 1–4 i analysis).
#    Ingen rader slettes og ingen verdier imputeres.
# 2. **Avgrensning** til poliseår med `property_damage_premium > 0` og
#    `total_exposure > 0` (seksjon 11). Premien er bare et dekningsflagg.
# 3. **Tidssplitt**: 2022–2023 er train-pool, 2024 er test (seksjon 12).
# 4. **Merke-pooling**: merker med minst 500 eksponeringsår i train-poolen får
#    eget nivå, resten blir `OTHER` (seksjon 17, B-14). Regelen bruker bare
#    eksponering, ikke skader, og læres derfor på hele train-poolen, ikke
#    inne i hver fold.
# 5. **Transparente prediktorer**: `driving_experience_years`,
#    `log_vehicle_value`, `performance_hp_per_tonne` og
#    `vehicle_brand_pooled` (seksjon 17).
#
# Kansellerte poliser beholdes med full vekt og fast eksponeringsoffset (B-01).
# `policy_status` brukes bare til sensitivitetsanalyse, aldri som prediktor
# (B-02). Testrammen fjernes med en gang, så den ikke kan brukes ved et uhell.

# %%
frames = build_model_frames()
frames.pop("test")  # B-05: 2024 holdes utenfor hele GLM-fasen
train_pool = frames["train_pool"]

# Kontroll mot tallene i analysis.ipynb (seksjon 12 og 17)
assert train_pool.index.is_unique
assert set(train_pool["year"]) == set(TRAIN_YEARS)
assert len(train_pool) == 56_084
assert round(train_pool["total_exposure"].sum(), 2) == 37_904.44
assert train_pool["property_claims"].sum() == 10_126
assert len(frames["retained_brands"]) == 17
display(frames["cleaning_log"])

# %%
frames

# %% [markdown]
# ### 1.1 Modellramme og responser
#
# Den rensede rammen bruker pandas' nullable dtypes (`Int16`, `category`).
# Formelmotoren patsy klarer ikke `Int16`, så `to_model_frame` i
# `src/model_data.py` velger kolonnene modellene trenger og konverterer dem til
# `float64`, `int64` og `object`. Konverteringen lærer ingenting fra data.
# Manglende verdier forblir manglende og håndteres i seksjon 2.6.
#
# Responsene defineres på én felles form, **rate med vekt**:
#
# | Målvariabel | Respons $y_i$ | Vekt $w_i$ | Rader |
# |---|---|---|---|
# | Frekvens | $N_i / e_i$ (`claim_frequency`) | $e_i$ | alle |
# | Severity | $\bar X_i = S_i / N_i$ (`average_severity`) | $N_i$ | $N_i > 0$ og $S_i > 0{,}01$ |
# | Ren premie | $S_i / e_i$ (`pure_premium`) | $e_i$ | alle |
#
# Her er $N_i$ skadeantall, $S_i$ incurred og $e_i$ eksponering. De 9
# skadeårene med incurred ≤ 0,01 er med i frekvens og ren premie, men ikke i
# severity (B-15).
#
# **Nullobservasjonene har ulike roller.** Poliseår uten registrert skade må
# være med i frekvensmodellen: uten dem ville modellen estimert skadeantall
# betinget på at en skade allerede har skjedd, ikke forventet skadefrekvens i
# porteføljen. De er ikke med i severity, som er betinget på registrert skade.
# De 9 radene med registrert skade, men null eller nesten null incurred, teller
# fortsatt som skader i frekvensen. Gamma-responsen må være positiv, så de
# utelates der og beholdes med sin observerte kostnad i ren premie. Den lille
# inkonsistensen dette kan skape i frekvens × severity, kvantifiseres før den
# todelte modellen sammenlignes med Tweedie (B-15).
#
# For frekvens er rate med vekt $e_i$ matematisk det samme som en Poisson-GLM
# for antall med offset $\log e_i$ (B-26). Estimeringsligningene er identiske:
# $\sum_i e_i (N_i/e_i - \mu_i)\,x_i = \sum_i (N_i - e_i\mu_i)\,x_i = 0$, og
# deviancen er den samme. Dette kontrolleres numerisk i seksjon 2.9. Fordelen
# er at én CV-sløyfe håndterer alle tre målvariablene.

# %%
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

model_frame = to_model_frame(train_pool, ID_COLUMNS + PREDICTORS + OUTCOME_COLUMNS)
assert model_frame.isna().sum().equals(train_pool[model_frame.columns].isna().sum())

# Responser for frekvens og ren premie: rater per eksponeringsår
model_frame["claim_frequency"] = (
    model_frame["property_claims"] / model_frame["total_exposure"]
)
model_frame["pure_premium"] = (
    model_frame["property_incurred"] / model_frame["total_exposure"]
)

# Severity: bare skadeår med positiv kostnad (B-15), snittskade per skade
severity_mask = model_frame["property_claims"].gt(0) & model_frame[
    "property_incurred"
].gt(CURRENCY_ZERO_TOLERANCE)
severity_frame = model_frame.loc[severity_mask].copy()
severity_frame["average_severity"] = (
    severity_frame["property_incurred"] / severity_frame["property_claims"]
)

data_overview = build_data_overview(model_frame, severity_frame)
display(data_overview.to_frame().round(3))
display(
    model_frame[PREDICTORS].isna().sum().loc[lambda s: s.gt(0)].to_frame("manglende")
)

# %% [markdown]
# ## 2. Evalueringsrammeverk
#
# ### 2.1 Prinsipp: scoren må belønne riktig forventningsverdi
#
# En tariff skal estimere forventet kostnad $\mathbb{E}[Y \mid x]$, ikke medianen
# eller den mest sannsynlige verdien. En metrikk som skal velge mellom modeller
# må derfor være **strengt konsistent for middelverdien**. Det betyr at den i
# forventning bare minimeres av den sanne middelverdien. Gneiting (2011) viser
# at de konsistente scoringsfunksjonene for middelverdien er Bregman-divergenser.
# Deviance-familien til Tweedie-fordelingene er slike divergenser (Wüthrich &
# Merz 2023, kap. 4).
#
# Det gir to praktiske regler:
#
# 1. **Samme score for alle kandidater til samme målvariabel,** uavhengig av
#    hvilken likelihood modellen er estimert med. NB og Poisson sammenlignes på
#    Poisson-deviance for middelverdien, og lognormal og Gamma på
#    Gamma-deviance. En bedre log-likelihood betyr ikke nødvendigvis at
#    middelverdien er bedre.
# 2. **Deviance krever ikke at fordelingen er riktig.** Poisson-deviance er
#    konsistent for middelverdien også når data er overspredt. Valget av
#    deviance bestemmer hvordan feil vektes, ikke hvilken fordeling som antas å
#    være sann.
#
# Alle scorer beregnes som vektet gjennomsnittlig Tweedie-deviance på
# valideringsdelen. Den primære OOF-scoren pooler alle OOF-prediksjonene, slik
# at hver observasjon får riktig eksponerings- eller skadevekt; foldscorene
# brukes til parvis usikkerhet og stabilitet (B-07):
#
# $$
# \bar D_p(y, \hat y) = \frac{\sum_i w_i\, d_p(y_i, \hat y_i)}{\sum_i w_i},
# \qquad
# d_p(y, \mu) =
# \begin{cases}
# 2\left(y \log\frac{y}{\mu} - y + \mu\right) & p = 1 \text{ (Poisson)} \\[4pt]
# 2\left(\log\frac{\mu}{y} + \frac{y}{\mu} - 1\right) & p = 2 \text{ (Gamma)} \\[4pt]
# 2\left(\frac{y^{2-p}}{(1-p)(2-p)} - \frac{y\,\mu^{1-p}}{1-p} + \frac{\mu^{2-p}}{2-p}\right) & 1 < p < 2 \text{ (Tweedie)}
# \end{cases}
# $$
#
# Intuitivt straffer $p = 1$ feil omtrent proporsjonalt med nivået, mens
# $p = 2$ straffer relative feil. Jo høyere $p$, desto mindre betyr en bom på
# en dyr observasjon.
#
# ### 2.2 Primærmetrikker, begrunnet i dette datasettet
#
# | Målvariabel | Primærscore (OOF) | Hvorfor den passer her |
# |---|---|---|
# | Frekvens | Poisson-deviance på $N/e$ med vekt $e$, og $D^2$ | 89,7 % av poliseårene har ingen skade, og bare om lag en tredel har full eksponering. Vektet Poisson-deviance håndterer begge deler og er konsistent selv med overspredningen fra pukkelen i skadeantall (N = 4–5) (Noll, Salzmann & Wüthrich 2018). |
# | Severity | Gamma-deviance på $\bar X$ med vekt $N$ | Positiv og høyreskjev respons. Relativ feil passer en multiplikativ tariff (Ohlsson & Johansson 2010). **Begrensning:** deviancen er skalafri, så en bom på en skade på 15 000 EUR veier like mye som en relativ bom på en skade på 500 EUR. Den suppleres derfor med A/E i EUR. |
# | Ren premie | Tweedie-deviance på $S/e$ med vekt $e$ ved $\hat p$ | Punktmasse i null og kontinuerlig positiv hale. Rangeringen kan avhenge av $p$, så i fase 3 rapporteres scoren også for $p \in \{1;\ 1{,}2;\ 1{,}5;\ 1{,}8\}$. En konklusjon regnes som robust bare når rangeringen er den samme på tvers av $p$ (Delong, Lindholm & Wüthrich 2021). |
#
# $D^2 = 1 - \bar D_p(\text{modell}) / \bar D_p(\text{nullmodell})$ er andelen
# av deviancen som forklares. Nullmodellen er det vektede snittet i
# treningsfolden. Med skadedata er $D^2$ typisk bare noen få prosent, og det er
# normalt: det meste av variasjonen i hvem som får skade er tilfeldig.
#
# ### 2.3 Sekundærmetrikker
#
# - **Kalibrering:** global balanse $\sum w\hat y / \sum w y$ og A/E per
#   prediksjonsdesil og per nivå av `policy_type`, år, `business_type` og
#   `bonus_score` (Goldburd et al. 2020). Kalibrering innen desiler er
#   autokalibrering (Denuit, Charpentier & Trufin 2021), altså at en premie
#   betaler for sine egne skader i snitt.
# - **Diskriminering:** Gini fra ordnet Lorenz-kurve (Frees, Meyers & Cummings
#   2011) og double lift i fase 3. Gini brukes **bare sekundært**: den er ikke en
#   konsistent score og kan bare sammenligne modeller som er autokalibrerte
#   (Wüthrich 2023). En modell kan få høyere Gini og likevel prise feil.
# - **Dekomponering** i fase 3–4: mean deviance = MCB − DSC + UNC via isotonisk
#   regresjon (Fissler, Lorentzen & Mayer 2023). Den skiller feilkalibrering fra
#   manglende diskriminering.
# - **Fordeling:** Pearson-dispersjon $\hat\phi$, Cameron–Trivedi-test (1990) og
#   rootogram (Kleiber & Zeileis 2016) for skadeantall.
# - **Inferens og stabilitet:** cluster-robuste standardfeil på `insured_id`
#   (Cameron & Miller 2015, B-21) og spenn i koeffisientene over foldene.
#
# Deviance er alltid første seleksjonskriterium. Den kan bare overstyres når
# den valgte modellen har en materiell og systematisk svakhet i en
# forhåndsdefinert diagnostikk — A/E totalt eller i vesentlige segmenter,
# halekalibrering, tidsrobusthet eller relativitetsstabilitet — og en
# konkurrerende modell tydelig reduserer svakheten. En slik overstyring skal
# begrunnes i beslutningsregisteret. Gini, p-verdi, AIC eller BIC kan aldri
# alene begrunne den (B-07).
#
# ### 2.4 Forkastede metrikker
#
# - **MAE** estimerer medianen, og median skadeantall og median ren premie er 0.
#   MAE ville belønnet en modell som predikerer 0 for alle.
# - **RMSE** på rater domineres av korte eksponeringer. Den høyeste annualiserte
#   renpremien i train-poolen er om lag 1,19 mill. EUR.
# - **R²** har ingen klar tolkning når nesten 90 % av responsene er null.
# - **AIC/BIC** kan bare sammenligne nøstede modeller med samme likelihood og
#   samme data. De brukes bare som kontroll.

# %% [markdown]
# ### 2.5 CV-design (B-06)
#
# - **Primært:** `GroupKFold(n_splits=5, shuffle=True, random_state=100)` på
#   `insured_id`. Samme polise kan forekomme både i 2022 og 2023, og begge
#   poliseårene må havne i samme fold. Ellers lærer modellen polisen i trening
#   og gjenkjenner den i validering. Uten `shuffle` fordeler `GroupKFold` like
#   store grupper i id-rekkefølge. Id-rekkefølgen kan henge sammen med
#   tegningstidspunkt, og det kan gi systematisk ulike folder (Roberts et al.
#   2017).
# - **Sensitivitet:** én tidsfold der 2022 er trening og 2023 validering. Den
#   tester fremoverskuende generalisering, men er bare én fold og må tolkes
#   forsiktig. En årseffekt `C(year)` kan ikke estimeres når valideringsåret
#   ikke finnes i treningsdataene. I tidsfolden brukes derfor spesifikasjonene
#   uten årsledd, og nivåskiftet mellom årene vises som global balanse (B-25).
#   Gruppe-CV velger variabler og funksjonsform. Tidsfolden er en obligatorisk
#   robusthetskontroll, ikke et nytt optimaliseringssett: et felles nivåskift
#   tolkes som kalenderdrift, mens klar forverring av både tidsdeviance og
#   segmentrelativiteter kan stoppe en kandidat (B-06).
# - **Alt som læres fra data, læres inne i treningsfolden:**
#   imputasjonsmedianer, spline-knuter (patsy `cr()` er stateful og gjenbruker
#   knutene fra trening ved prediksjon), NB-α, Tweedie-$p$ og
#   storskadetillegget $\lambda$.
#
# Foldene defineres som indeksmengder. Da kan de samme foldene brukes både på
# hele modellrammen og på severity-delmengden, og alle modeller for samme
# målvariabel sammenlignes på nøyaktig de samme radene.

# %%
group_kfold = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_folds = [
    {
        "fold": f"gruppe_{fold_number + 1}",
        "train_index": model_frame.index[train_positions],
        "val_index": model_frame.index[val_positions],
    }
    for fold_number, (train_positions, val_positions) in enumerate(
        group_kfold.split(model_frame, groups=model_frame["insured_id"])
    )
]
time_fold = [
    {
        "fold": "tid_2022_til_2023",
        "train_index": model_frame.index[model_frame["year"].eq(2022)],
        "val_index": model_frame.index[model_frame["year"].eq(2023)],
    }
]

# Ingen insured_id på begge sider, og hver rad valideres nøyaktig én gang
for fold in cv_folds:
    train_ids = set(model_frame.loc[fold["train_index"], "insured_id"])
    val_ids = set(model_frame.loc[fold["val_index"], "insured_id"])
    assert not train_ids & val_ids, f"{fold['fold']} har overlappende insured_id"
all_val_index = pd.Index(np.concatenate([fold["val_index"] for fold in cv_folds]))
assert all_val_index.is_unique and len(all_val_index) == len(model_frame)

display(build_fold_summary(model_frame, cv_folds + time_fold))

# %% [markdown]
# Gruppefoldene bør ha omtrent samme frekvens, samme andel kansellerte og
# samme andel 2023-rader. Store forskjeller her ville gjort fold-scorene
# vanskelige å sammenligne. Tidsfolden har per konstruksjon bare 2023 i
# valideringen.
#
# ### 2.6 Manglende verdier (B-09)
#
# Bare tre prediktorer har manglende verdier i train-poolen (se tabellen i
# seksjon 1.1). Strategien er enkel og læres i treningsfolden:
#
# - **Kategoriske:** manglende verdi blir et eget nivå, `MISSING`. For
#   `fuel_type` (439 rader) kan manglende drivstoff selv bære informasjon,
#   så nivået kan få sin egen relativitet.
# - **Numeriske:** median fra treningsfolden. For `log_vehicle_value` (33 rader)
#   og `driving_experience_years` (1 rad, fra manglende `age_driving_licence`)
#   er antallet så lite at valget knapt påvirker estimatene. Strategien dekker
#   også `performance_hp_per_tonne`, som mangler i tre rader utenfor
#   train-poolen.
#
# Et kategorinivå som mangler i treningsfolden, men finnes i valideringen, gir
# en tydelig feil i stedet for en stille feilprising.


# %%
def prepare_design_frame(train, apply):
    """B-09: fyll manglende verdier i ``apply`` med regler lært på ``train``."""
    prepared = apply.copy()
    train_medians = train[NUMERIC_PREDICTORS].median()
    prepared[NUMERIC_PREDICTORS] = prepared[NUMERIC_PREDICTORS].fillna(train_medians)
    for column in CATEGORICAL_PREDICTORS:
        prepared[column] = prepared[column].fillna("MISSING")
        unseen_levels = set(prepared[column]) - set(train[column].fillna("MISSING"))
        if unseen_levels:
            raise ValueError(f"{column} har nivåer uten treningsdata: {unseen_levels}")
    assert prepared[PREDICTORS].notna().all().all()
    return prepared


# %% [markdown]
# ### 2.7 GLM-byggesteiner og CV-sløyfen
#
# All GLM-estimering i notebooken går gjennom fire funksjoner:
#
# | Funksjon | Hva den gjør | Kalles |
# |---|---|---|
# | `glm_spec` | Bygger patsy-formelen fra respons `y` og prediktorer `x` | én gang per modell |
# | `fit_glm` | Estimerer en ferdig spesifikasjon på et datasett | på hele train-poolen og i hver fold |
# | `run_glm` | Spesifiserer, estimerer og presenterer én modell på hele train-poolen | én gang per modell |
# | `cross_validate_glm` | Kjører en ferdig spesifikasjon gjennom foldene | én gang per modell |
#
# **Målvariablene** samles i `TARGETS`. Hver målvariabel har respons, datasett,
# familie med log-link, vektkolonne og Tweedie-$p$ for scoren (B-07, B-26).
# Da er en modell fullt beskrevet av målvariabel og prediktorliste.
#
# **Formelen bygges fra dtypes.** `glm_spec` leser typen til hver prediktor:
#
# - Tekst/kategori (og `year`, B-12) blir `C(x, Treatment(basis))`, der
#   basisnivået er nivået med størst eksponering (B-24).
# - Numeriske kolonner går inn lineært.
# - Alt som ikke er en kolonne, f.eks. `cr(driver_age, df=3)`, sendes uendret
#   til patsy. Det gjør funksjonen klar for splines i fase 1.
#
# **Effektivitet.** Formelen, basisnivåene og familien bygges én gang av
# `glm_spec` og gjenbrukes i alle folder. I hver fold gjøres bare det som *må*
# læres på nytt: manglende-reglene (B-09) og designmatrisen, der patsy lærer
# eventuelle spline-knuter fra treningsdelen.
#
# I hver fold i `cross_validate_glm` skjer dette:
#
# 1. Manglende-reglene læres på treningsdelen og brukes på begge deler.
# 2. `fit_glm` estimerer modellen på treningsdelen med `var_weights` = vekten.
# 3. Trenings- og valideringsdelen predikeres og scores med vektet
#    Tweedie-deviance.
# 4. Nullmodellen (vektet snitt i treningsdelen) scores på valideringsdelen og
#    gir $D^2$.
#
# Funksjonen returnerer OOF-prediksjoner, én rad med scorer per fold og
# koeffisientene per fold (brukes til stabilitetssjekk i fase 1).

# %%
LOG_LINK = sm.families.links.Log()
TWEEDIE_POWER_PLACEHOLDER = 1.5

TARGETS = {
    "frequency": {
        "y": "claim_frequency",
        "data": model_frame,
        "family": sm.families.Poisson(link=LOG_LINK),
        "weight": "total_exposure",
        "power": 1,
    },
    "severity": {
        "y": "average_severity",
        "data": severity_frame,
        "family": sm.families.Gamma(link=LOG_LINK),
        "weight": "property_claims",
        "power": 2,
    },
    "pure_premium": {
        "y": "pure_premium",
        "data": model_frame,
        "family": sm.families.Tweedie(
            var_power=TWEEDIE_POWER_PLACEHOLDER, link=LOG_LINK, eql=True
        ),
        "weight": "total_exposure",
        "power": TWEEDIE_POWER_PLACEHOLDER,
    },
}


def glm_spec(name, x, y, data, family, weight, power, categorical=("year",)):
    """Bygg formel og innstillinger for én GLM. Kalles én gang per modell."""
    terms, base_levels = [], {}
    for predictor in x:
        if predictor not in data:  # ferdig patsy-ledd, f.eks. "cr(driver_age, df=3)"
            terms.append(predictor)
        elif predictor in categorical or not pd.api.types.is_numeric_dtype(
            data[predictor]
        ):
            # B-24: basisnivå = nivået med størst eksponering
            base = data.groupby(predictor)["total_exposure"].sum().idxmax()
            base = base.item() if isinstance(base, np.generic) else base
            base_levels[predictor] = base
            terms.append(f"C({predictor}, Treatment({base!r}))")
        else:
            terms.append(predictor)
    return {
        "name": name,
        "x": list(x),
        "y": y,
        "formula": f"{y} ~ {' + '.join(terms) or '1'}",
        "family": family,
        "weight": weight,
        "power": power,
        "base_levels": base_levels,
    }


def fit_glm(spec, data, cluster_groups=None):
    """Estimer en ferdig spesifikasjon. ``data`` må ha fylte manglende verdier."""
    covariance = (
        {}
        if cluster_groups is None
        else {"cov_type": "cluster", "cov_kwds": {"groups": cluster_groups}}  # B-21
    )
    with warnings.catch_warnings():
        # Statsmodels advarer generelt om cov_type + var_weights. Kontrollert:
        # cluster-SE er identiske med antall + offset og med manuell sandwich.
        warnings.filterwarnings("ignore", "cov_type not fully supported")
        result = smf.glm(
            spec["formula"],
            data=data,
            family=spec["family"],
            var_weights=data[spec["weight"]],
        ).fit(**covariance)
    if not result.converged:
        raise RuntimeError(f"{spec['name']} konvergerte ikke")
    return result


def run_glm(name, target, x, show=True):
    """Spesifiser, estimer og presenter én GLM på hele train-poolen."""
    settings = TARGETS[target]
    spec = glm_spec(name, x, **settings)
    design = prepare_design_frame(settings["data"], settings["data"])
    result = fit_glm(spec, design, cluster_groups=design["insured_id"])
    model = {
        "target": target,
        "spec": spec,
        "result": result,
        "summary": build_glm_summary(spec, result, design),
        "relativities": build_relativity_table(spec, result),
    }
    if show:
        display(model["summary"].to_frame().T.infer_objects().round(5))
        display(model["relativities"].round(4))
    return model


def cross_validate_glm(spec, data, folds):
    """Estimer og scor én ferdig spesifikasjon out-of-fold."""
    response, weight = spec["y"], spec["weight"]
    oof_predictions = pd.Series(np.nan, index=data.index, name=spec["name"])
    fold_scores, fold_params = [], []

    def score(part, prediction):
        return mean_tweedie_deviance(
            part[response], prediction, sample_weight=part[weight], power=spec["power"]
        )

    for fold in folds:
        train = data.loc[data.index.intersection(fold["train_index"])]
        val = data.loc[data.index.intersection(fold["val_index"])]
        train_design = prepare_design_frame(train, train)
        val_design = prepare_design_frame(train, val)

        result = fit_glm(spec, train_design)
        val_prediction = result.predict(val_design)
        oof_predictions.loc[val.index] = val_prediction
        null_prediction = np.full(
            len(val), np.average(train[response], weights=train[weight])
        )
        fold_scores.append(
            {
                "model": spec["name"],
                "fold": fold["fold"],
                "n_train": len(train),
                "n_val": len(val),
                "train_weight": train[weight].sum(),
                "val_weight": val[weight].sum(),
                "train_deviance": score(train, result.predict(train_design)),
                "val_deviance": score(val, val_prediction),
                "val_null_deviance": score(val, null_prediction),
                "val_actual": (val[response] * val[weight]).sum(),
                "val_predicted": (val_prediction * val[weight]).sum(),
                "val_balance": (val_prediction * val[weight]).sum()
                / (val[response] * val[weight]).sum(),
            }
        )
        fold_params.append(result.params.rename(fold["fold"]))

    scores = pd.DataFrame(fold_scores)
    scores["val_d2"] = 1 - scores["val_deviance"] / scores["val_null_deviance"]
    return {
        "oof": oof_predictions,
        "scores": scores,
        "params": pd.concat(fold_params, axis=1),
    }


# %% [markdown]
# ### 2.8 Seleksjonsregel, låst før første modell (B-08)
#
# En variabelblokk tas inn når alle disse er oppfylt:
#
# 1. **Gevinsten er større enn støyen:** gjennomsnittlig parvis forbedring i
#    OOF-deviance over de fem foldene er større enn én standardfeil av
#    forbedringen. Regelen er inspirert av 1-SE-regelen (Hastie, Tibshirani &
#    Friedman 2009). Parvis sammenligning på de samme foldene fjerner
#    variasjonen som skyldes at foldene er ulike.
# 2. **Gevinsten er bred:** kandidaten forbedrer deviancen i minst fire av fem
#    folder og gir positiv forbedring i samlet, vektet OOF-deviance.
# 3. **Effekten er stabil på en meningsfull skala:** et enkelt lineært ledd har
#    samme fortegn i minst fire av fem folder. Kategoriske blokker vurderes på
#    relativitetene for nivåer med vesentlig eksponering, og splines vurderes
#    på den predikerte kurven over de sentrale 95 % av eksponeringen — ikke på
#    fortegnet til de enkelte splinekoeffisientene.
#
# p-verdier og one-way-rater brukes ikke til å velge variabler. Med 56 084 rader
# blir nesten alt signifikant, og one-way-rater er forvridd av samvariasjon.
# Kandidatblokkene testes i den forhåndsbestemte rekkefølgen i faseplanen, uten
# interaksjoner, og fullmodellen får en bakoversjekk blokk for blokk. Når to
# kandidater ligger innenfor én standardfeil, velges den enkleste. Med fem
# folder er standardfeilen grov; derfor rapporteres foldresultatene alltid.


# %%
def paired_improvement(fold_scores, baseline, candidate):
    """B-08: parvis forbedring i OOF-deviance fra ``baseline`` til ``candidate``.

    ``baseline=None`` betyr nullmodellen, altså det vektede snittet i
    treningsfolden (``val_null_deviance``).
    """
    scores = fold_scores.set_index(["model", "fold"])
    candidate_deviance = scores.loc[candidate, "val_deviance"]
    if baseline is None:
        baseline_deviance = scores.loc[candidate, "val_null_deviance"]
    else:
        baseline_deviance = scores.loc[baseline, "val_deviance"]
    improvement = baseline_deviance - candidate_deviance
    candidate_weights = scores.loc[candidate, "val_weight"]
    pooled_improvement = np.average(improvement, weights=candidate_weights)
    standard_error = improvement.std(ddof=1) / np.sqrt(len(improvement))
    return pd.Series(
        {
            "baseline": baseline or "nullmodell",
            "kandidat": candidate,
            "pooled_forbedring": pooled_improvement,
            "snitt_forbedring": improvement.mean(),
            "standardfeil": standard_error,
            "folder_med_forbedring": int(improvement.gt(0).sum()),
            "passerer_1se": bool(improvement.mean() > standard_error),
            "passerer_b08": bool(
                pooled_improvement > 0
                and improvement.mean() > standard_error
                and improvement.gt(0).sum() >= 4
            ),
        }
    )


# %% [markdown]
# ### 2.9 Referansemodeller
#
# Før variabelseleksjonen starter, estimeres enkle referansemodeller for hver
# målvariabel. De tester at rammeverket fungerer og setter et nivå
# variabelmodellene må slå.
#
# **Frekvens.** $N_i$ er skadeantall og $e_i$ eksponering:
#
# $$
# \log \mathbb{E}\!\left[\frac{N_i}{e_i}\right] = \beta_0 + \beta_{\text{policy\_type}(i)} \;[+\; \beta_{\text{year}(i)}],
# \qquad N_i \sim \text{Poisson}(e_i\,\mu_i).
# $$
#
# **Severity.** $\bar X_i$ er snittskaden i poliseåret, med vekt $N_i$:
#
# $$
# \log \mathbb{E}[\bar X_i] = \gamma_0 + \gamma_{\text{policy\_type}(i)} \;[+\; \gamma_{\text{year}(i)}],
# \qquad \bar X_i \sim \text{Gamma},\; \operatorname{Var}(\bar X_i) = \frac{\phi\,\mu_i^2}{N_i}.
# $$
#
# **Ren premie.** $S_i/e_i$ er kostnad per eksponeringsår:
#
# $$
# \log \mathbb{E}\!\left[\frac{S_i}{e_i}\right] = \theta_0 + \theta_{\text{policy\_type}(i)} \;[+\; \theta_{\text{year}(i)}],
# \qquad \operatorname{Var}\!\left(\frac{S_i}{e_i}\right) = \frac{\phi\,\mu_i^{p}}{e_i}.
# $$
#
# Intuitivt gir hvert nivå av `policy_type` en multiplikativ faktor
# $\exp(\beta)$ på basisnivået. Basisnivået er `COMP_E`, nivået med størst
# eksponering, slik at relativitetene måles mot kjernen av porteføljen (B-24).
# For `year` er basisnivået 2023, som også er årsnivået prediksjonene skal
# ligge på (B-22).
# Tweedie-$p$ er satt til 1,5 som **midlertidig plassholder**. $\hat p$
# estimeres med EQL-profil i fase 3 (B-19).
#
# **Nullmodellen estimeres ikke.** Med bare intercept og log-link er $\mu$ lik
# for alle rader, og estimeringsligningen
# $\sum_i w_i (y_i - \mu)/V(\mu) = 0$ har løsningen
# $\hat\mu = \sum_i w_i y_i / \sum_i w_i$, det vektede snittet, for Poisson,
# Gamma og Tweedie. Porteføljenivåene står allerede i seksjon 1.1.
# `cross_validate_glm` bruker derfor snittet i treningsfolden direkte som
# nullmodell. Det gir `val_null_deviance`, $D^2$ og sammenligningen
# nullmodell → `policy_type`.
#
# De seks modellene estimeres på hele train-poolen med `run_glm`. Deretter
# kontrolleres at rate med vekt gir identiske koeffisienter og identisk
# deviance som antall med offset (B-26).

# %%
# Referansemodellene: samme tre prediktorsett for alle tre målvariabler
REFERENCE_PREDICTORS = {
    "policy_type": ["policy_type"],
    "policy_type_year": ["policy_type", "year"],
}
reference_models = {
    f"{target}_{label}": run_glm(f"{target}_{label}", target, x, show=False)
    for target in TARGETS
    for label, x in REFERENCE_PREDICTORS.items()
}
display(
    pd.DataFrame([model["summary"] for model in reference_models.values()])
    .drop(columns="formel")
    .infer_objects()
    .round(5)
)

# B-26: rate med vekt er samme modell som antall med offset
frequency_model = reference_models["frequency_policy_type_year"]
count_offset_fit = smf.glm(
    "property_claims ~" + frequency_model["spec"]["formula"].split("~")[1],
    data=model_frame,
    family=sm.families.Poisson(link=LOG_LINK),
    offset=np.log(model_frame["total_exposure"]),
).fit()
assert np.allclose(count_offset_fit.params, frequency_model["result"].params)
assert np.isclose(count_offset_fit.deviance, frequency_model["result"].deviance)

# %% [markdown]
# Relativitetene for modellene med `policy_type` og `year`, én tabell per
# målvariabel. Konfidensintervallene er cluster-robuste på `insured_id` (B-21).

# %%
display(
    pd.concat(
        {
            target: reference_models[f"{target}_policy_type_year"]["relativities"][
                ["relativitet", "ki_lav", "ki_høy"]
            ]
            for target in TARGETS
        },
        axis=1,
    ).round(3)
)

# %% [markdown]
# De to frekvensformuleringene gir samme koeffisienter og samme deviance. Poisson med log-link og intercept er
# eksakt balansert på treningsdataene (balanse = 1). Det gjelder bare den
# kanoniske linken, så Gamma og Tweedie med log-link er ikke garantert
# balanserte og kontrolleres separat senere.
#
# Nå kjøres referansemodellene gjennom gruppefoldene. I tidsfolden brukes bare
# modellene uten årsledd (B-25).

# %%
reference_cv = {
    name: cross_validate_glm(model["spec"], TARGETS[model["target"]]["data"], cv_folds)
    for name, model in reference_models.items()
}
reference_scores = pd.concat(
    [result["scores"] for result in reference_cv.values()], ignore_index=True
)
display(summarize_cv_scores(reference_scores).round(5))

# %%
reference_comparisons = pd.DataFrame(
    [
        paired_improvement(reference_scores, baseline, candidate)
        for target in TARGETS
        for baseline, candidate in [
            (None, f"{target}_policy_type"),
            (f"{target}_policy_type", f"{target}_policy_type_year"),
        ]
    ]
)
display(reference_comparisons.round(5))

# %%
time_fold_scores = pd.concat(
    [
        cross_validate_glm(model["spec"], TARGETS[model["target"]]["data"], time_fold)[
            "scores"
        ]
        for model in reference_models.values()
        if "year" not in model["spec"]["x"]
    ],
    ignore_index=True,
)
display(
    time_fold_scores[
        [
            "model",
            "fold",
            "train_deviance",
            "val_null_deviance",
            "val_deviance",
            "val_d2",
            "val_balance",
        ]
    ].round(5)
)

# %% [markdown]
# **Slik leses tabellene.**
#
# - `oof_deviance` er primærscoren. Lavere er bedre, og tallet kan bare
#   sammenlignes innen samme målvariabel.
# - `oof_null_deviance` er nullmodellens OOF-deviance, referansenivået for
#   målvariabelen. `oof_d2` er andelen av den som modellen forklarer.
# - `oof_balanse` er predikert/observert på valideringsdelen. Tall over 1 betyr
#   at modellen overpriser.
# - `train_deviance` skal normalt ikke være vesentlig lavere enn
#   `oof_deviance`. Er gapet stort, er modellen overtilpasset.
# - I tidsfolden viser `val_balance` hvor mye nivået flytter seg fra 2022 til
#   2023 når modellen ikke har et årsledd. Frekvensen steg fra 2022 til 2023,
#   så balansen forventes å ligge under 1.
#
# **Hva referansemodellene viser.**
#
# - **Rammeverket virker.** Balansen ligger på 1,00 i alle gruppefoldene, og
#   trenings- og OOF-deviance er nesten like. Det er ventet for modeller med
#   bare 3–4 parametere.
# - **`policy_type` alene forklarer mye:** OOF-$D^2$ er ca. 8 % for frekvens,
#   5 % for severity og 3 % for ren premie. COMP_N har nesten fire ganger så høy
#   frekvens som COMP_E, mens severity går motsatt vei (seksjon 14 i analysis).
#   Derfor forklarer produktet mindre av ren premie enn av hver komponent.
# - **Årseffekten er liten i gruppefoldene, men stor over tid.** Årsleddet
#   forbedrer frekvens i 4 av 5 folder. For severity passerer det 1-SE-grensen
#   med forbedring i bare 3 av 5 folder, og for ren premie passerer det ikke.
#   I tidsfolden underpredikerer en modell trent på 2022 frekvensen i 2023 med
#   ca. 17 % (balanse 0,83) og overpredikerer severity med ca. 10 % (balanse
#   1,10). Frekvens og severity drev altså i hver sin retning, og ren premie
#   ble 8 % underpredikert. Dette er relevant for årsnivået i prediksjonen
#   (B-22) og for den senere evalueringen på 2024.
# - **Merknad til B-08.** Severity-årsleddet viser at 1-SE-kriteriet alene kan
#   slippe gjennom en gevinst som bare finnes i 3 av 5 folder. Revurderingen
#   før fase 1 skjerpet derfor B-08 til minst 4 av 5 forbedrede folder og
#   typepasset stabilitet for lineære ledd, kategorier og splines.
#
# Referansemodellene er ikke kandidater til benchmarken. De er startpunktet
# blokkseleksjonen i fase 1 bygger videre på.

# %% [markdown]
# ### 2.10 Beslutningsport før fase 1
#
# Referansemodellene viser at rammeverket kan skille signal fra støy, men også
# hvorfor spillereglene må fryses før kandidatmodellene estimeres. Fra fase 1
# skal mange variabelblokker og funksjonsformer prøves på de samme dataene. Når
# resultatene først er sett, er det lett å endre respons, metrikk eller
# stabilitetskrav slik at en foretrukket modell vinner. Det ville gjort
# benchmarken mindre troverdig og 2024-evalueringen mindre informativ.
#
# Beslutningene under skiller derfor tre spørsmål som ellers lett blandes:
#
# 1. **Hva estimeres?** Observert incurred egen-skadekostnad per faktisk
#    eksponeringsår i hele dekningspopulasjonen, inklusive senere kansellerte.
#    Nullskadeår er nødvendige i frekvensen, mens severity per definisjon bare
#    bruker skadeår med positiv kostnad (B-01, B-03, B-15 og B-26).
# 2. **Hva kunne vært kjent ved prising?** En prediktor slipper bare inn hvis
#    den er kjent ved periodens start eller det definerte fornyelsestidspunktet.
#    Dette er en adgangsregel før OOF-seleksjon: god prediksjon kan ikke reparere
#    target leakage. `bonus_score` må derfor bestå as-of-kontrollen i
#    `bonus_score_analysis`; ellers kan den bare vises som sensitivitet (B-02,
#    B-13 og B-20).
# 3. **Hvordan avgjøres hva som generaliserer?** Gruppe-CV velger variabler og
#    form, tidsfolden utfordrer robustheten, og 2024 holdes lukket til både GLM
#    og ML er frosset. Pooled OOF-deviance er primær, mens foldene viser
#    usikkerhet og stabilitet (B-05–B-08 og B-25).
#
# Disse låsene gjør neste fase til en reell modelltest: dataene får avgjøre
# mellom forhåndsdefinerte kandidater, men får ikke endre konkurransereglene.
# De viktigste operative beslutningene er:
#
# | Område | Låst regel før fase 1 | Hvorfor den er viktig nå |
# |---|---|---|
# | Respons og populasjon | Alle dekkede poliseår beholdes. Nullskadeår inngår i frekvens, ikke severity. De 9 registrerte nullkostnadsskadene inngår i frekvens og ren premie, men ikke Gamma-severity. Incurred behandles som beste tilgjengelige kostnadsestimat; ukjent skadeutvikling oppgis som begrensning. | Hindrer at estimatet endres etter at vanskelige observasjoner eller relativiteter er sett. |
# | Informasjonstidspunkt | Bare opplysninger kjent ved periodestart/fornyelse er kvalifisert. Bonus tas inn i hovedmodellen bare ved dokumentert as-of-dato; status og eksisterende premie er aldri prediktorer. | CV beskytter ikke mot en variabel som allerede inneholder periodens skadeutfall. |
# | Validering | Fem gruppefolder er primære; 2022 → 2023 er obligatorisk robusthetskontroll; 2024 åpnes én gang etter at GLM og ML er frosset. `year` er kontroll i gruppe-CV, men utelates i tidsfolden. | Skiller generalisering mellom poliser, kalenderdrift og en reell fremtidstest. |
# | Blokkseleksjon | Fast rekkefølge, ingen interaksjoner, positiv pooled gevinst, parvis gevinst > 1 SE og forbedring i minst 4/5 folder; deretter bakoversjekk. | Reduserer rekkefølgefrihet og tilfeldige funn fra gjentatt bruk av de samme foldene. |
# | Stabilitet og enkelhet | Lineære ledd vurderes på fortegn, kategorier på eksponeringsstøttede relativiteter og splines på kurven i sentrale 95 %. Innen 1 SE velges enkleste modell. | Koeffisientfortegn betyr ikke det samme for en lineær effekt, en kategori og en spline. |
# | Beslutningshierarki | Pooled vektet OOF-deviance er primær. Den kan bare overstyres ved en dokumentert, materiell og systematisk svakhet i forhåndsdefinert A/E-, hale-, tids- eller stabilitetsdiagnostikk som en konkurrent tydelig reduserer. Gini, p-verdi og informasjonskriterier kan ikke alene overstyre. | Bevarer en konsistent hovedscore uten å tvinge frem en tariff som svikter på et vesentlig, dokumentert område. |
# | Kontinuerlige ledd | Lineær, `cr(df=3)` og `cr(df=4)` konkurrerer; enklere form vinner innen 1 SE. Alder og kjøreerfaring testes separat, aldri sammen; alder vinner ved resultat innen 1 SE. Merkegrensen er 500 eksponeringsår, med 250/1 000 som sensitivitet. | Låser tie-break før de mest attraktive kurvene er kjent og begrenser kollinearitet og haleustabilitet. |
#
# Gamma mot lognormal, Poisson mot NB, Tweedie-$p$ og storskadebehandling er
# med vilje ikke avgjort her. Dette er datadrevne beslutninger i senere faser,
# men kandidatene og scoringsreglene deres er definert før resultatene ses.

# %% [markdown]
# ### 2.11 Litteratur
#
# - Cameron, A. C. & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2).
# - Cameron, A. C. & Trivedi, P. K. (1990). Regression-based tests for overdispersion in the Poisson model. *Journal of Econometrics*, 46(3).
# - Delong, Ł., Lindholm, M. & Wüthrich, M. V. (2021). Making Tweedie's compound Poisson model more accessible. *European Actuarial Journal*, 11.
# - Denuit, M., Charpentier, A. & Trufin, J. (2021). Autocalibration and Tweedie-dominance for insurance pricing with machine learning. *Insurance: Mathematics and Economics*, 101.
# - Duan, N. (1983). Smearing estimate: a nonparametric retransformation method. *JASA*, 78(383).
# - Embrechts, P., Klüppelberg, C. & Mikosch, T. (1997). *Modelling Extremal Events for Insurance and Finance*. Springer.
# - Fissler, T., Lorentzen, C. & Mayer, M. (2023). Model comparison and calibration assessment: user guide for consistent scoring functions in machine learning and actuarial practice. arXiv:2202.12780.
# - Frees, E. W., Meyers, G. & Cummings, A. D. (2011). Summarizing insurance scores using a Gini index. *JASA*, 106(495).
# - Gneiting, T. (2011). Making and evaluating point forecasts. *JASA*, 106(494).
# - Goldburd, M., Khare, A., Tevet, D. & Guller, D. (2020). *Generalized Linear Models for Insurance Rating* (2. utg.). CAS Monograph 5.
# - Harrell, F. E. (2015). *Regression Modeling Strategies* (2. utg.). Springer.
# - Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning* (2. utg.). Springer.
# - Jørgensen, B. & de Souza, M. C. P. (1994). Fitting Tweedie's compound Poisson model to insurance claims data. *Scandinavian Actuarial Journal*, 1994(1).
# - Kleiber, C. & Zeileis, A. (2016). Visualizing count data regressions using rootograms. *The American Statistician*, 70(3).
# - Noll, A., Salzmann, R. & Wüthrich, M. V. (2018). Case study: French motor third-party liability claims. SSRN 3164764.
# - Ohlsson, E. & Johansson, B. (2010). *Non-Life Insurance Pricing with Generalized Linear Models*. Springer.
# - Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8).
# - Smyth, G. K. & Jørgensen, B. (2002). Fitting Tweedie's compound Poisson model to insurance claims data: dispersion modelling. *ASTIN Bulletin*, 32(1).
# - Wüthrich, M. V. (2023). Model selection with Gini indices under auto-calibration. *European Actuarial Journal*, 13.
# - Wüthrich, M. V. & Merz, M. (2023). *Statistical Foundations of Actuarial Learning and its Applications*. Springer.

# %% [markdown]
# ## 7. Beslutningsregister
#
# Registeret oppdateres ved slutten av hver fase. «Seksjon» viser hvor
# beslutningen brukes eller avgjøres.
#
# | ID | Beslutning | Begrunnelse | Status | Seksjon |
# |---|---|---|---|---|
# | B-01 | Alle egen-skade-poliseår inkluderes, også kansellerte, med fast $\log e$-offset. Sensitivitet uten kansellerte. | Kansellerte står for 13 % av skadene på 5 % av eksponeringen. Å fjerne dem senker nivået med ca. 9 % og demper relativitetene for bonus N, kvartalsbetaling og portefølje. En tariff må prise poliser som senere kanselleres. | Brukerbesluttet | 1, 3 |
# | B-02 | Bare opplysninger kjent ved periodestart eller definert fornyelsestidspunkt er kvalifisert. `policy_status` brukes aldri som prediktor. | Status er kjent først etter at perioden er ute og kan påvirkes av skaden selv; OOF-CV beskytter ikke mot tidsmessig lekkasje. | Brukerbesluttet | 1, 2.10 |
# | B-03 | Frekvensresponsen er skadeantall. Pukkelen ved N = 4–5 dokumenteres med rootogram. Tweedie er en robust kontroll. | Antall er standard tariffstruktur. Pukkelen kan være registreringspraksis, og Tweedie på kostnad påvirkes ikke av den. | Brukerbesluttet | 3, 5 |
# | B-04 | Datagrunnlaget ligger i `src/model_data.py` og dokumenteres i begge notebooks. | Én sannhet for populasjon, splitt og prediktorer. | Brukerbesluttet | 1 |
# | B-05 | 2024 brukes ikke i GLM-fasen og åpnes én gang for felles sluttevaluering etter at GLM og ML er frosset. | Aggregerte 2024-tall er allerede sett. Flere titt svekker testen ytterligere. | Brukerbesluttet | 0, 1, 2.10 |
# | B-06 | `GroupKFold(5, shuffle=True, random_state=100)` på `insured_id` velger modell. Tidsfold 2022 → 2023 er obligatorisk robusthetskontroll, ikke et nytt optimaliseringssett. | Gruppe-CV gir stabil sammenligning uten id-lekkasje; tidsfolden skiller fremoverskuende svikt fra generell kalenderdrift. | Brukerbesluttet | 2.5, 2.10 |
# | B-07 | Pooled vektet OOF Poisson-, Gamma- eller Tweedie-deviance er primær. Overstyring krever en dokumentert, materiell og systematisk svakhet i forhåndsdefinert A/E-, hale-, tids- eller stabilitetsdiagnostikk som en konkurrent tydelig reduserer. Gini, p-verdi, AIC eller BIC er aldri nok alene. | Deviance er konsistent for middelverdien, men en tariff skal ikke tvinges gjennom når hovedscoren skjuler en vesentlig og dokumentert praktisk svikt. | Brukerbesluttet | 2.1–2.3, 2.10 |
# | B-08 | En blokk krever positiv pooled OOF-gevinst, parvis gjennomsnittsgevinst > 1 SE og forbedring i minst 4/5 folder. Fast blokkfølge, typepasset stabilitet, bakoversjekk og enkleste modell innen 1 SE; ingen interaksjoner eller p-verdiutvalg. | Reduserer seleksjonsoptimisme og vurderer lineære ledd, kategorier og splines på meningsfulle skalaer. | Brukerbesluttet | 2.8, 2.10, 3 |
# | B-09 | Manglende kategorier blir `MISSING`. Numeriske variabler imputeres med median lært i treningsfolden. | Få manglende verdier. Enkelt, transparent og uten lekkasje. | Foreslått | 2.6 |
# | B-10 | Kontinuerlige ledd: lineært mot naturlige kubiske splines `cr(df=3)` og `cr(df=4)`, valgt på OOF. Innen 1 SE velges enklere form (`lineær` før `df=3` før `df=4`). | Fleksibel form tillates når den gir robust gevinst, uten å belønne unødig halevariasjon. | Brukerbesluttet regel / Datadrevet resultat | 2.10, 3 |
# | B-11 | `driver_age` og `driving_experience_years` testes separat, aldri sammen. Innen 1 SE velges `driver_age`. | Sterkt korrelerte; alder har bedre datadekning og enklere tolkning. | Brukerbesluttet regel / Datadrevet resultat | 2.10, 3 |
# | B-12 | `year` er kategorisk kontroll i gruppe-CV, ikke offset eller trend, og utelates i tidsfolden. Prediksjon på 2023-nivå er benchmarknivå, ikke et estimert fremtidig trendnivå. | To år gir ingen trend å estimere, og et ukjent årsnivå kan ikke predikeres direkte. | Brukerbesluttet | 2.5, 2.9–2.10, 3 |
# | B-13 | `bonus_score` er bare kvalifisert for hovedmodellen dersom `bonus_score_analysis` dokumenterer at verdien var kjent før skadeperioden. Deretter kreves vanlig OOF-gevinst og sensitivitet uten bonus. Uten dokumentert as-of-dato vises bonus bare som potensielt lekkende sensitivitet. | Prediksjonsstyrke kan ikke oppveie target leakage. | Brukerbesluttet adgangsregel / Datadrevet resultat | 2.10, 3 |
# | B-14 | Merke-pooling ved ≥ 500 eksponeringsår, lært på hele train-poolen fra eksponering alene. Sensitivitet med 250 og 1 000. | Regelen bruker ikke responsen, så den gir ingen responslekkasje. | Brukerbesluttet | 1, 2.10, 3 |
# | B-15 | Poliseår uten skade inngår i frekvens, ikke severity. De 9 skadeårene med incurred ≤ 0,01 beholdes i frekvens og ren premie, men utelates fra Gamma-severity; utslaget på todelt ren premie kvantifiseres. | Frekvens krever både nuller og skader; Gamma krever positiv respons. | Brukerbesluttet | 1.1, 2.10, 4–5 |
# | B-16 | Gamma med log-link og vekt $N$. Lognormal med Duan-smearing som utfordrer. | Standard multiplikativ severity. Lognormal kontrollerer halen. | Foreslått / Datadrevet | 4 |
# | B-17 | Storskadetersklene 5 000, 7 500 og 10 000 EUR er satt på forhånd. Kapping skjer på snittskaden. | Terskler valgt etter responsen ville gitt lekkasje i seleksjonen. | Foreslått | 4 |
# | B-18 | Separat storskadebehandling bare hvis overskridelsen er > 5 % av kostnaden **og** kapping + tillegg forbedrer OOF-score og A/E i øverste desil, eller tydelig stabiliserer relativitetene. | Halen er moderat. Kompleksitet må forsvares av data. | Foreslått / Datadrevet | 4 |
# | B-19 | Tweedie-$p$ estimeres med EQL-profil i treningsfolden. $p = 1{,}5$ er bare plassholder i referansemodellene. | $p$ styrer både estimat og score og må læres uten lekkasje. | Foreslått | 2.9, 5 |
# | B-20 | `property_damage_premium` brukes bare som benchmark i fase 4, aldri som prediktor. | Premien er dagens tariff og ville lekke eksisterende prisstruktur inn i modellen. | Brukerbesluttet | 2.10, 6 |
# | B-21 | Cluster-robuste standardfeil på `insured_id`. | Samme polise i to år gir korrelerte observasjoner. | Foreslått | 3–6 |
# | B-22 | Prediksjon ved $e = 1$ og årsnivå 2023. Dette er benchmarkens siste observerte kalendernivå, ikke et estimert fremtidig nivå. | Gir en sammenlignbar årspremie uten å late som to år identifiserer en trend. | Brukerbesluttet | 2.10, 6 |
# | B-23 | NB velges bare ved bedre OOF-score på middelverdien. Ellers Poisson med Pearson-skalert, cluster-robust inferens. | Overspredning påvirker usikkerhet, ikke nødvendigvis middelverdien. | Foreslått | 3 |
# | B-24 | Basisnivå for kategoriske variabler er nivået med størst eksponering, f.eks. `COMP_E` for `policy_type`. | Relativiteter mot kjernen av porteføljen blir stabile og lette å lese. Påvirker ikke prediksjonene. | Foreslått | 2.9 |
# | B-25 | I tidsfolden brukes spesifikasjonene uten `C(year)`. Nivåskiftet vises som global balanse. | Årsledd kan ikke predikere et år som ikke finnes i treningsdata. | Foreslått | 2.5, 2.9 |
# | B-26 | Alle responser modelleres som rate med vekt (`var_weights`). For frekvens er dette identisk med antall og $\log e$-offset. | Én CV-sløyfe for alle målvariabler. Ekvivalensen er kontrollert numerisk. | Foreslått | 1.1, 2.9 |
