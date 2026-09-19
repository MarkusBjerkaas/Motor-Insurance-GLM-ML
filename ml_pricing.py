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
# # CatBoost-utfordrer for ren egen-skadepremie
#
# Notebooken utvikler én CatBoost-modell for forventet ren egen-skadepremie.
# Den bruker bare utviklingsårene 2022–2023. Teståret 2024 er urørt og brukes
# først i en senere, låst sammenligning av alle ferdigspesifiserte modeller.

# %%
from IPython.display import display

from src_core_glm.model_data import build_development_frames
from src_ml.catboost_pricing import (
    build_catboost_result_table,
    fit_catboost_grid,
    plot_catboost_diagnostics,
    prepare_catboost_features,
)
from src_tweedie.tweedie_data import build_tweedie_frame

SEED = 100
N_SPLITS = 5
# Låst fra severity-implisert Tweedie-power i Tweedie-GLM-løpet (T-05).
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

PARAM_GRID = {
    "depth": [4, 6],
    "learning_rate": [0.03, 0.07],
    "iterations": [300, 600],
    "l2_leaf_reg": [3.0, 10.0],
}

# %% [markdown]
# ## 1. Datagrunnlag
#
# Modellrammen bygges gjennom den felles utviklingskjeden. Den filtrerer bort
# alle andre år enn 2022–2023 før rensing og feature engineering, slik at 2024
# aldri materialiseres i denne analysen.

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
# ## 2. Modell og utviklingsscore
#
# For poliseår $i$ er observert ren premie
#
# $$
# R_i = \frac{S_i}{e_i},
# $$
#
# der $S_i$ er incurred egen-skadekostnad og $e_i$ er eksponering. CatBoost
# estimerer $\widehat\mu_i = f_\theta(x_i)$ med Tweedie-loss. Hyperparameterne
# velges ved å minimere eksponeringsvektet Tweedie-deviance
#
# $$
# D_p = \operatorname{mean\_tweedie\_deviance}
# \left(R, \widehat\mu;\, w=e,\, p\right).
# $$
#
# Modellen anslår forventet kostnad per eksponeringsår; eksponeringen bestemmer
# hvor mye informasjon hvert poliseår bidrar med. Fem gruppefolder på
# `insured_id` hindrer at samme forsikringstaker inngår på begge sider av en
# fold. Resultatene nedenfor er utviklingsresultater, ikke uavhengige estimater.

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
# `grid_mean_deviance` er scoren som valgte hyperparameterne. `pooled_oof_deviance`
# er beregnet fra de samme foldenes OOF-prediksjoner etter at parameterne er
# valgt, og er derfor seleksjonspåvirket utviklingsdiagnostikk. Nullmodellen
# bruker bare treningsfoldens eksponeringsvektede gjennomsnitt i hver fold.

# %%
result_table = build_catboost_result_table(result)
display(result_table.round(4))

# %%
figure = plot_catboost_diagnostics(result)

# %% [markdown]
# ## 4. Konklusjon
#
# Den valgte CatBoost-spesifikasjonen er nå en direkte utfordrer for ren premie
# på samme utviklingspopulasjon, med samme Tweedie-power og gruppefolder som
# GLM-løpet. Endelig sammenligning med Tweedie-GLM og todelt GLM skjer først på
# det urørte teståret 2024, etter at alle spesifikasjoner og evalueringsregler
# er låst.
