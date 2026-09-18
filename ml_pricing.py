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
# # ML-utfordrere for egen-skadeprising
#
# Notebooken setter opp CatBoost og LightGBM for tre mål: frekvens, severity og
# pure premium. Den bruker bare utviklingsårene 2022–2023; modellene fittes og
# sammenlignes i en senere, eksplisitt CV-celle.

# %%
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupKFold

from src_core_glm.model_data import (
    assert_development_years,
    build_development_frames,
    to_model_frame,
)
from src_severity.severity_data import build_severity_inputs

SEED = 100
N_FOLDS = 5
TWEEDIE_POWER = 1.5

# %% [markdown]
# ## 1. Felles datagrunnlag og mål
#
# Frekvens bruker skadeantall med eksponering som vekt, severity bruker
# gjennomsnittsskade på positive skadeår med skadeantall som vekt, og pure
# premium modellerer samlet kostnad per eksponeringsår med eksponering som vekt.

# %%
frames = build_development_frames()
development = frames["development"]
assert_development_years(development)

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
model_columns = [
    "insured_id",
    "total_exposure",
    "property_claims",
    "property_incurred",
    *PREDICTORS,
]
model_frame = to_model_frame(development, model_columns)
model_frame["claim_frequency"] = (
    model_frame["property_claims"] / model_frame["total_exposure"]
)
model_frame["pure_premium"] = (
    model_frame["property_incurred"] / model_frame["total_exposure"]
)
severity_frame = build_severity_inputs(development)["severity_frame"]

# %%
TARGETS = {
    "frequency": {
        "frame": model_frame,
        "target": "property_claims",
        "weight": "total_exposure",
        "objective": "poisson",
    },
    "severity": {
        "frame": severity_frame,
        "target": "average_severity",
        "weight": "property_claims",
        "objective": "gamma",
    },
    "pure_premium": {
        "frame": model_frame,
        "target": "pure_premium",
        "weight": "total_exposure",
        "objective": "tweedie",
    },
}

# %% [markdown]
# ## 2. Modellfabrikker og preprocessing
#
# Samme foldvise preprocessing brukes for begge algoritmene. One-hot-koding
# gjør at kategorinivåer fra valideringsfolden ikke kan lekke inn i treningen.
# Senere kan CatBoost få en separat native-kategorivariant som utfordrer.

# %%
NUMERIC_COLUMNS = [
    column for column in PREDICTORS if column not in {
        "policy_type",
        "fuel_type",
        "circulation_area",
        "municipality_type",
        "payment_frequency",
        "business_type",
        "vehicle_brand_pooled",
    }
]
CATEGORICAL_COLUMNS = [column for column in PREDICTORS if column not in NUMERIC_COLUMNS]


def build_preprocessor():
    """Bygg preprocessing som kan fittes separat i hver CV-fold."""
    return ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median"),
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )


def build_catboost_model(objective):
    """Bygg én CatBoost-regressor med samme måltype som LightGBM."""
    loss = {
        "poisson": "Poisson",
        "gamma": "RMSE",
        "tweedie": f"Tweedie:variance_power={TWEEDIE_POWER}",
    }[objective]
    return CatBoostRegressor(
        loss_function=loss,
        iterations=500,
        depth=6,
        learning_rate=0.05,
        random_seed=SEED,
        verbose=False,
    )


def build_lightgbm_model(objective):
    """Bygg én LightGBM-regressor med eksplisitt objektiv."""
    parameters = {
        "objective": objective,
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": SEED,
        "verbosity": -1,
    }
    if objective == "tweedie":
        parameters["tweedie_variance_power"] = TWEEDIE_POWER
    return LGBMRegressor(**parameters)


def build_model(algorithm, objective):
    """Bygg en komplett, foldklar pipeline."""
    estimator = (
        build_catboost_model(objective)
        if algorithm == "catboost"
        else build_lightgbm_model(objective)
    )
    return Pipeline(
        [("preprocessor", build_preprocessor()), ("model", estimator)]
    )

# %% [markdown]
# ## 3. CV-oppsett
#
# Denne cellen definerer bare samme gruppefolder for hvert mål. Selve CV-loopen
# og scoringen skal implementeres etter at modellspesifikasjonene er gjennomgått.

# %%
cv_folds = {}
for target_name, target_config in TARGETS.items():
    frame = target_config["frame"]
    group_kfold = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_folds[target_name] = [
        {
            "fold": f"gruppe_{number + 1}",
            "train_index": frame.index[train_positions],
            "val_index": frame.index[val_positions],
        }
        for number, (train_positions, val_positions) in enumerate(
            group_kfold.split(frame, groups=frame["insured_id"])
        )
    ]

# %% [markdown]
# ## 4. Modellregister
#
# Hver måltype får én CatBoost- og én LightGBM-kandidat. Hyperparameterne er
# bevisst en liten startspesifikasjon; tuning og OOF-evaluering kommer senere.

# %%
model_registry = {
    target_name: {
        algorithm: build_model(algorithm, target_config["objective"])
        for algorithm in ("catboost", "lightgbm")
    }
    for target_name, target_config in TARGETS.items()
}

