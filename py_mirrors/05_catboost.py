# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py_mirrors//py:percent
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
# # CatBoost-utfordrer for ren egen-skadepremie
#
# Notebooken utvikler én CatBoost-modell med Tweedie-loss for forventet ren
# egen-skadepremie per eksponeringsår. Den utfordrer Tweedie-GLM-en på samme
# populasjon, med samme Tweedie-power, seed og gruppefolder. Bare
# utviklingsårene 2022–2023 er brukt; 2024 er ikke lest.

# %%
from IPython.display import display

from src_core_glm.model_data import build_development_frames
from src_ml.catboost_pricing import (
    build_catboost_result_table,
    build_final_model_table,
    fit_catboost_grid,
    plot_catboost_diagnostics,
    prepare_catboost_features,
)
from src_model_comparison.locked_models import lock_catboost
from src_model_comparison.variable_coverage import build_variable_coverage
from src_tweedie.tweedie_data import build_tweedie_frame

SEED = 100
N_SPLITS = 5
# Låst fra severity-implisert Tweedie-power i Tweedie-GLM-løpet.
TWEEDIE_POWER = 1.744

PREDICTORS = [
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

CATEGORICAL_FEATURES = [
    "policy_type",
    "year",
    "fuel_type",
    "circulation_area",
    "municipality_type",
    "payment_frequency",
    "business_type",
    "vehicle_brand_pooled",
]

# Bare parametrene som styrer kompleksitet og regularisering tunes. Læringsrate
# og antall trær veier mot hverandre, så begge varieres. Rutenettet er bredt
# nedover fordi et tidligere grid (depth 4–6, lr 0.03–0.07, 300–600 trær,
# l2 3–10) valgte laveste verdi i alle dimensjoner.
PARAM_GRID = {
    "depth": [2, 3, 4, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "iterations": [150, 300, 600],
    "l2_leaf_reg": [1.0, 3.0, 10.0, 30.0],
}

# %% [markdown]
# ## 1. Datagrunnlag
#
# Modellrammen bygges fra utviklingsårene 2022–2023. Kategoriske prediktorer
# gis direkte til CatBoost; manglende verdier får egen kategori.

# %%
development = build_development_frames()["development"]
model_frame = build_tweedie_frame(development, PREDICTORS)
features = prepare_catboost_features(
    model_frame, PREDICTORS, CATEGORICAL_FEATURES
)
response = model_frame["pure_premium"]
sample_weight = model_frame["total_exposure"]
groups = model_frame["insured_id"]

# %% [markdown]
# ## 2. Modell og seleksjonsregel
#
# For poliseår $i$ er observert ren premie
#
# $$
# R_i = \frac{S_i}{e_i},
# $$
#
# der $S_i$ er incurred egen-skadekostnad og $e_i$ er eksponering. CatBoost
# estimerer forventet ren premie $\widehat\mu_i = f_\theta(x_i)$ ved å minimere
# eksponeringsvektet Tweedie-deviance med låst $p$. Hyperparameterne velges av
# et rutenett over `depth`, `learning_rate`, `iterations` og `l2_leaf_reg`
# (192 kombinasjoner) ved å minimere
#
# $$
# D_p = \operatorname{mean\_tweedie\_deviance}
# \left(R, \widehat\mu;\, w=e,\, p\right)
# $$
#
# over fem gruppefolder på `insured_id`. Gruppefoldene hindrer at samme
# forsikringstaker står på begge sider av en fold. Etter valget refittes
# modellen på alle utviklingsdata.

# %%
result = fit_catboost_grid(
    features,
    response,
    sample_weight,
    groups,
    CATEGORICAL_FEATURES,
    TWEEDIE_POWER,
    PARAM_GRID,
    n_splits=N_SPLITS,
    seed=SEED,
)

# %% [markdown]
# ## 3. Resultat og diagnostikk
#
# Tabellen viser hyperparameterne rutenettet valgte. `grid_mean_deviance` er
# scoren som valgte dem. `pooled_oof_deviance` er beregnet fra de samme foldenes
# OOF-prediksjoner og er derfor optimistisk. Nullmodellen er treningsfoldens
# eksponeringsvektede gjennomsnitt.

# %%
result_table = build_catboost_result_table(result)
display(result_table.round(4))

# %%
figure = plot_catboost_diagnostics(result)

# %% [markdown]
# ## 4. Endelig modell
#
# Den endelige modellen er CatBoost med log-link, refittet på alle
# utviklingsdata:
#
# $$
# \widehat\mu(x) = \exp\{F_M(x)\}, \qquad
# F_M(x) = \sum_{m=1}^{M} \eta\, t_m(x),
# $$
#
# der $t_m$ er symmetriske (oblivious) beslutningstrær av dybde $d$ (med $2^d$
# blader per tre), $\eta$ er læringsraten, $M$ er antall trær og bladverdiene
# har $L_2$-regularisering $\lambda$. Trærne bygges sekvensielt for å minimere
# $D_p$ fra §2 med eksponeringsvekt $w=e$ og låst $p = 1.744$. Verdiene til
# $d$, $\eta$, $M$ og $\lambda$ leses fra den refittede modellen under, og
# `på_kant` viser om noen av dem ligger på kanten av rutenettet.

# %%
final_model_table = build_final_model_table(result)
display(final_model_table)

# %% [markdown]
# ## 5. Variabelsjekk og låst modell
#
# Alle prediktorer er tilgjengelige for CatBoost; ingen er utelatt av
# seleksjon. Kontrollen feiler hvis en kolonne i utviklingsdata verken er
# brukt, bevisst utelatt eller utfall/ID. Den låste modellen lagres i
# `models/catboost.joblib` og verifiseres mot notebookens egne prediksjoner.

# %%
coverage = build_variable_coverage(development, PREDICTORS, PREDICTORS, PREDICTORS)
display(coverage)

# %%
display(
    lock_catboost(
        "catboost", result, PREDICTORS, CATEGORICAL_FEATURES, development
    )
)

# %% [markdown]
# ## 6. Begrensninger
#
# - **Optimistisk utviklingsscore.** Samme fem folder velger hyperparameterne
#   og scorer dem; `pooled_oof_deviance` er derfor ikke et uavhengig estimat.
# - **Begrenset rutenett.** 192 kombinasjoner av fire hyperparametere er testet. Flere av de
#   valgte verdiene ligger på kanten av rutenettet (`på_kant` i §4): optimum kan
#   ligge utenfor, mot enda enklere trær og flere iterasjoner. CatBoost har ikke
#   tidlig stopp eller egen valideringsdel utover CV.
# - **Lav forklart deviance.** `oof_d2` i §3 er lav mot nullmodellen; skadekostnad er
#   svært støyende. $p$ er låst fra Tweedie-GLM-løpet og ikke tunet her.
# - **Feature importance er deskriptiv**, ikke kausal, og sier ikke noe om
#   retning eller størrelse på effekten.
# - **`year` er en låst kategorisk term**, og 2024 finnes ikke i treningsdata.
#   2024 skåres derfor med 2023-nivået: ingen trend-ekstrapolering, og
#   årsdrift blir liggende i porteføljebalansen.
# - **Numeriske prediktorer klippes til treningsområdet** ved skåring. Trærne er
#   flate utenfor det, og GLM-ene klippes tilsvarende (se
#   `src_model_comparison/locked_models.py`) slik at sammenligningen blir
#   rettferdig.
