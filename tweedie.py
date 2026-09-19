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
# Tweedie-GLM er utfordreren til den todelte modellen
# $\widehat{\text{pure premium}} = \widehat{\text{frekvens}} \times \widehat{\text{severity}}$.
# Notebooken bruker bare utviklingsårene 2022–2023; 2024 leses aldri.
#
# **Status:** infrastrukturen (datagrunnlag, felles folder, spesifikasjon og
# Pearson-estimering av $p$) er klar og testet. Bred ankermodell og
# kandidatregister er ikke låst, så ingen Tweedie-modell fittes ennå.
# Beslutningene ligger i `tweedie_plan.md` (register T-01–T-08).

# %%
import statsmodels.api as sm
from IPython.display import display

from src_asserts.common_glm_asserts import assert_valid_folds
from src_asserts.tweedie_asserts import assert_tweedie_spec
from src_core_glm.glm_core import glm_spec
from src_core_glm.model_data import build_development_frames
from src_core_glm.validation import build_group_folds
from src_tweedie.tweedie_data import build_tweedie_frame, summarize_tweedie_frame

SEED = 100
N_FOLDS = 5

# %% [markdown]
# ## 1. Datagrunnlag
#
# La $S_i$ være `property_incurred`, $e_i$ være `total_exposure` og
# $R_i = S_i/e_i$ observert pure premium per eksponeringsår. Alle poliseår er
# gyldige observasjoner, også skadefrie ($R_i = 0$), og flere skader på samme
# poliseår er tillatt fordi $S_i$ er aggregert årlig skadekostnad.

# %%
frames = build_development_frames()
development = frames["development"]

# Kolonner som er tilgjengelige for kandidatregisteret. Dette er ikke et modellvalg.
AVAILABLE_COLUMNS = [
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
model_frame = build_tweedie_frame(development, AVAILABLE_COLUMNS)
display(summarize_tweedie_frame(model_frame).round(3))

# %% [markdown]
# ## 2. Validering
#
# Fem gruppefolder på `insured_id` bygges på hele modellpopulasjonen. Samme
# indeks, gruppekolonne og seed gir identiske folder som i frekvens- og
# severity-notebookene, slik at OOF-prediksjonene kan sammenlignes rad for rad.

# %%
cv_folds = build_group_folds(
    model_frame,
    group_column="insured_id",
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED,
)
assert_valid_folds(cv_folds, model_frame)  # sanity-sjekk, kan fjernes

# %% [markdown]
# ## 3. Modellspesifikasjon
#
# $$
# R_i = \frac{S_i}{e_i} \sim \operatorname{Tweedie}\!\left(\mu_i, \frac{\phi}{e_i}, p\right),
# \qquad 1<p<2,
# $$
#
# $$
# \operatorname{E}[R_i \mid x_i] = \mu_i,
# \qquad
# \operatorname{Var}(R_i \mid x_i) = \frac{\phi}{e_i}\,\mu_i^{p},
# \qquad
# \log \mu_i = \beta_0 + \sum_j \beta_j x_{ij}.
# $$
#
# For $1<p<2$ er $R_i$ en sammensatt Poisson–Gamma-variabel: masse i null
# (skadefrie år) og en kontinuerlig positiv del. Eksponeringen $e_i$ er
# `var_weights`, uten offset: $R_i$ er allerede en rate, og et år med høy
# eksponering er en mer presis observasjon av samme $\mu_i$.
#
# Power $p$ estimeres med iterativ Pearson-estimering
# (`src_tweedie/tweedie_power.py`) på den låste ankermodellen og låses *før*
# variabelseleksjonen. Deviance med ulike $p$ er ikke sammenlignbar, så $p$
# velges ikke ved CV.


# %%
def build_tweedie_specification(
    name, power, predictors, spline_terms=(), interaction_terms=()
):
    """Bygg én Tweedie-spesifikasjon (log-link, eksponeringsvekt, ingen offset).

    ``predictors`` er råkolonner, ``spline_terms`` er ``(kolonne, df)`` og
    ``interaction_terms`` er ferdige patsy-ledd med råkolonnene i ``predictors``.
    """
    spline_map = dict(spline_terms)
    terms = [
        f"cr({column}, df={spline_map[column]}, constraints='center')"
        if column in spline_map
        else column
        for column in predictors
    ]
    terms += list(interaction_terms)
    family = sm.families.Tweedie(var_power=power, link=sm.families.links.Log())
    return glm_spec(
        name,
        terms,
        "pure_premium",
        model_frame,
        family,
        "total_exposure",
        power,
        required_columns=predictors,
    )


# Nøytral kontroll av byggeren (bare intercept): ikke en kandidat.
assert_tweedie_spec(build_tweedie_specification("intercept", 1.5, []))  # kan fjernes

# %% [markdown]
# ## 4. Sammenligning og valg
#
# Når bred ankermodell og kandidatregister er låst, følger Tweedie samme
# hovedstruktur som de andre GLM-ene: kjerne, alternative funksjonsformer,
# geografi og kjøretøyopplysninger, forhåndsdefinerte tillegg, eventuelle
# interaksjoner og eventuell ablasjon. Med låst $p$ brukes treleddsregelen:
# lavere pooled OOF-deviance enn foreldremodellen, bedre i minst fire av fem
# folder og gyldig fit i alle folder.
#
# Senere benchmark mot den todelte modellen krever identisk indeks, full
# dekning og positive prediksjoner for begge, samme `pure_premium`,
# eksponeringsvekt og én felles, forhåndslåst score-power:
# `two_part_oof = frequency_oof * severity_prediction_oof` og
# `tweedie_oof = tweedie_result["prediction_oof"]`
# (kontrollen `assert_comparable_oof` i `src_asserts/common_glm_asserts.py`).

# %%
print(
    "Kandidatregister og bred ankermodell er ikke låst. "
    "Ingen Tweedie-kandidater fittes eller velges i denne versjonen."
)
