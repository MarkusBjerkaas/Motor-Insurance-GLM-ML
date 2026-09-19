# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # Endelig modellsammenligning: out-of-sample 2024
#
# Tre prismodeller for ren egen-skadepremie (forventet skadekostnad per eksponeringsår)
# sammenlignes på ett ubrukt år: **Tweedie-GLM**, **toleddet GLM** (frekvens × severity)
# og **CatBoost**. Modellene er spesifisert og refittet på 2022–2023 i egne notebooks
# (`tweedie`, `glm_pricing_models`, `glm_pricing_severity`, `ml_pricing`) og lagret i
# `models/`. Ingen modell trenes eller justeres her.
#
# 2024 er ikke brukt til modellvalg, seleksjon eller tuning. Testlåsen er aktiv:
# ingen 2024-data leses før `RUN_2024_EVALUATION` og godkjenningskoden bevisst settes.

# %%
from IPython.display import display

from src_core_glm.model_data import build_development_frames
from src_model_comparison.locked_models import load_locked_model
from src_model_comparison.test_evaluation import (
    build_approved_test_frame,
    build_calibration_table,
    evaluate_pricing_models,
    plot_test_comparison,
)

TWEEDIE_POWER = 1.744  # samme avrundede p som i CatBoost og protokollen under
RUN_2024_EVALUATION = False
APPROVAL_CODE = None

# %% [markdown]
# ## 1. Modellene
#
# Alle tre estimerer $\mu_i=E[S_i/e_i]$, forventet skadekostnad per eksponeringsår for
# poliseår $i$ med totalkostnad $S_i$ og eksponering $e_i$. Prediktorene er de samme 12
# kandidatvariablene i alle modellene; hver modell velger selv hvilke som brukes.
#
# **Tweedie-GLM.** Ren premie modelleres direkte, med log-link og vekt $e_i$:
#
# $$
# \frac{S_i}{e_i}\sim\operatorname{Tweedie}\!\left(\mu_i,\tfrac{\phi}{e_i},p\right),
# \qquad
# \log\mu_i=\beta_0+\gamma^{\mathrm{type}}_i+\gamma^{\mathrm{year}}_i+\gamma^{\mathrm{area}}_i
# +\gamma^{\mathrm{pay}}_i+\gamma^{\mathrm{fuel}}_i+\gamma^{\mathrm{seat}}_i+\gamma^{\mathrm{bus}}_i
# +\beta_v\log V_i+f_3(\mathrm{age}_i).
# $$
#
# **Toleddet GLM.** Frekvens og severity modelleres hver for seg, og premien er produktet:
#
# $$
# \widehat R_i=\widehat\lambda_i\,\widehat s_i,
# \qquad
# \log\lambda_i=\beta_0+\gamma^{\mathrm{type}}_i+\gamma^{\mathrm{year}}_i+\gamma^{\mathrm{area}}_i
# +\gamma^{\mathrm{pay}}_i+\gamma^{\mathrm{bus}}_i+\beta_v\log V_i+f_4(\mathrm{age}_i),
# $$
#
# $$
# \log s_i=\alpha_0+\delta^{\mathrm{type}}_i+\delta^{\mathrm{year}}_i+\delta^{\mathrm{area}}_i
# +\delta^{\mathrm{bus}}_i+\alpha_v\log V_i+\alpha_a\,\mathrm{age}_i.
# $$
#
# Frekvensen er Poisson (vekt $e_i$), severity er Gamma på poliseår med skade (vekt
# antall skader). $f_k$ er en naturlig kubisk spline med $k$ frihetsgrader.
#
# **CatBoost.** Gradientboostede symmetriske trær med log-link og Tweedie-tap:
#
# $$
# \widehat\mu(x)=\exp\Big\{\sum_{m=1}^{300}\eta\,t_m(x)\Big\},\qquad \eta=0.03,\ \text{dybde }4.
# $$
#
# Tweedie-GLM og CatBoost bruker $p=1.744$. Den fullstendige spesifikasjonen (leddene,
# referansenivåene og antall parametere) står i hver modellnotebook.

# %%
# Lås opp de lagrede modellene. Brand-poolingen læres kun fra 2022–2023.
retained_brands = build_development_frames()["retained_brands"]
frequency_model = load_locked_model("frequency")
severity_model = load_locked_model("severity")
tweedie_model = load_locked_model("tweedie")
catboost_model = load_locked_model("catboost")

prediction_functions = {
    "Tweedie-GLM": tweedie_model,
    "Toleddet GLM": lambda frame: frequency_model(frame) * severity_model(frame),
    "CatBoost": catboost_model,
}

# %% [markdown]
# ## 2. Testprotokoll
#
# Med observert $R_i=S_i/e_i$ og predikert $\widehat R_i$ måles hver modell med
# eksponeringsvektet Tweedie-deviance ($p=1.744$), vektet MAE og porteføljebalanse
# $\sum_i e_i\widehat R_i\big/\sum_i S_i-1$. **Primærmål er deviance; lavere er bedre.**
# Kalibreringsplottet (ti like store prediksjonsgrupper) brukes som kontroll: en modell
# med lavest deviance regnes bare som best hvis den ikke viser vesentlig systematisk skjevhet.
#
# `year` er en låst kategorisk term og 2024 finnes ikke i treningsdata, så alle modeller skåres
# med 2023-nivået. Numeriske prediktorer klippes til treningsområdet ved skåring.

# %%
if RUN_2024_EVALUATION:
    test_frame = build_approved_test_frame(retained_brands, approval_code=APPROVAL_CODE)
    results, predictions = evaluate_pricing_models(
        test_frame, prediction_functions, power=TWEEDIE_POWER
    )
    calibration = build_calibration_table(test_frame, predictions)

# %% [markdown]
# ## 3. Resultater

# %%
if RUN_2024_EVALUATION:
    display(results.round(3))

# %%
if RUN_2024_EVALUATION:
    figure = plot_test_comparison(results, calibration)

# %% [markdown]
# ## 4. Begrensninger
#
# - **Ett testår.** Rangeringen dokumenterer generalisering til 2024, ikke en universell
#   eller kausal effekt. Forskjeller i deviance mellom modellene er ikke gitt usikkerhet;
#   dersom modellene ligger tett, bør forskjellene ikke tolkes som sikre.
# - **Årsdrift.** Alle modeller skåres med 2023-nivået. En prisendring eller endret
#   skadenivå fra 2023 til 2024 blir derfor liggende i porteføljebalansen og er ikke en
#   modellfeil.
# - **Optimistiske utviklingsscorer.** Variabelutvalg, spesifikasjon og hyperparametere er
#   valgt på samme fem CV-folder, og $p$ er estimert på hele utviklingssettet.
# - **CatBoost er skjev i nivå.** Modellen undervurderer porteføljen allerede på
#   utviklingsdata (−5,4 % balanse på 2023-radene i en tørrkjøring uten 2024). Den lave
#   læringsraten og $L_2$-regulariseringen gir ikke full nivåkalibrering; balansen på
#   teståret bør leses med det i minne.
# - **Lav forklart varians.** Skadekostnad per poliseår har få og skjeve skader; alle
#   modeller forklarer bare en liten andel av variasjonen (D² ≈ 0,02 for CatBoost).
#
# ## 5. Tolkning
#
# Fylles ut etter at 2024-evalueringen er kjørt og resultatene er vurdert mot
# kalibreringsplottet.
