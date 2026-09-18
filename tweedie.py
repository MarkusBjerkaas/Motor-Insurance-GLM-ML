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
# # Tweedie-modell for ren egen-skadepremie
#
# Denne notebooken setter opp en samlet Tweedie-GLM for forventet årlig
# egen-skadekostnad. Den bruker bare utviklingsårene 2022–2023. Prediktorer og
# CV-struktur holdes låst mens Tweedie-kraften `p` sammenlignes.

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold

from src_core_glm.glm_core import (
    cross_validate_glm,
    fit_glm,
    glm_spec,
    prepare_design_frame,
)
from src_core_glm.model_data import (
    TRAIN_YEARS,
    assert_development_years,
    build_development_frames,
    to_model_frame,
)

SEED = 100
N_FOLDS = 5
FIT_SETTINGS = {"maxiter": 200, "tol": 1e-8}
POWER_GRID = (1.1, 1.3, 1.5, 1.7, 1.9)

# %% [markdown]
# ## 1. Datagrunnlag
#
# For poliseår $i$ modelleres samlet kostnad per eksponeringsår som
#
# $$
# Y_i/e_i \sim \operatorname{Tweedie}(\mu_i,\phi e_i^{1-p}),
# \qquad \log(\mu_i)=\beta_0+\sum_j\beta_jx_{ij},
# $$
#
# med `total_exposure` som frekvens-/eksponeringsvekt. For $1<p<2$ har
# fordelingen både masse i null og en kontinuerlig positiv del.

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
    "property_incurred",
    *PREDICTORS,
]
model_frame = to_model_frame(development, model_columns)
model_frame["pure_premium"] = (
    model_frame["property_incurred"] / model_frame["total_exposure"]
)

# %% [markdown]
# ## 2. Gruppebasert CV
#
# Samme `insured_id` ligger ikke i både trening og validering. Dette er samme
# foldstruktur som i de øvrige GLM-fasene; 2024 inngår ikke i rammen.

# %%
group_kfold = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
cv_folds = [
    {
        "fold": f"gruppe_{number + 1}",
        "train_index": model_frame.index[train_positions],
        "val_index": model_frame.index[val_positions],
    }
    for number, (train_positions, val_positions) in enumerate(
        group_kfold.split(model_frame, groups=model_frame["insured_id"])
    )
]

# %% [markdown]
# ## 3. Tweedie-spesifikasjon
#
# `glm_spec` og `cross_validate_glm` er generiske: eneste modellparameter som
# endres her er `power`. Formelen og vekten er identiske for alle kandidater.

# %%
def build_tweedie_specification(power):
    """Bygg én samlet Tweedie-GLM for en forhåndsdefinert kraftparameter."""
    family = sm.families.Tweedie(
        var_power=power,
        link=sm.families.links.Log(),
    )
    return glm_spec(
        name=f"tweedie_p_{power:.1f}",
        x=PREDICTORS,
        y="pure_premium",
        data=model_frame,
        family=family,
        weight="total_exposure",
        power=power,
    )


tweedie_specs = {
    power: build_tweedie_specification(power) for power in POWER_GRID
}

# %% [markdown]
# ## 4. CV-oppsett for `p`
#
# Hver kandidat får samme fold, preprocessing og foldkontroller. Valg av `p`
# skal senere baseres på pooled OOF-Tweedie-deviance, ikke på treningsscore.

# %%
cv_results = {
    power: cross_validate_glm(
        specification,
        model_frame,
        cv_folds,
        fit_kwargs=FIT_SETTINGS,
    )
    for power, specification in tweedie_specs.items()
}


def summarize_power_cv(cv_results):
    """Samle foldscore til én sammenlignbar, eksponeringsvektet oversikt."""
    rows = []
    for power, result in cv_results.items():
        scores = result["scores"]
        rows.append(
            {
                "power": power,
                "pooled_oof_deviance": np.average(
                    scores["val_deviance"], weights=scores["val_weight"]
                ),
                "mean_fold_deviance": scores["val_deviance"].mean(),
                "valid": result["valid"],
            }
        )
    return pd.DataFrame(rows).sort_values("pooled_oof_deviance")


power_cv_summary = summarize_power_cv(cv_results)

# %% [markdown]
# ## 5. Finalist
#
# Denne cellen velger foreløpig laveste pooled OOF-deviance. Eventuell
# vurdering av flat scorekurve og praktisk avrunding dokumenteres før sluttfit.

# %%
selected_power = power_cv_summary.iloc[0]["power"]
selected_specification = tweedie_specs[selected_power]

# Sluttfit kjøres først når CV-resultatet er gjennomgått og spesifikasjonen er låst.
