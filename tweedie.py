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
# Beslutningene ligger i `tweedie_plan.md` (register T-01–T-09).

# %%
import pandas as pd
import statsmodels.api as sm
from IPython.display import display

from src_asserts.common_glm_asserts import assert_valid_folds
from src_asserts.tweedie_asserts import (
    assert_model_definition,
    assert_tweedie_spec,
    assert_valid_power,
)
from src_core_glm.glm_core import cross_validate_glm, glm_spec
from src_core_glm.model_data import add_seat_category, build_development_frames
from src_core_glm.model_selection import (
    can_add,
    propose_additions,
    propose_removals,
    run_round,
    run_stepwise,
    select_candidate_stage,
    summarize_selection_path,
)
from src_core_glm.validation import build_group_folds
from src_severity.severity_data import build_severity_inputs
from src_tweedie.tweedie_data import build_tweedie_frame, summarize_tweedie_frame
from src_tweedie.tweedie_power import severity_implied_power

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

# Kolonnene som er tilgjengelige for modellene. Selve modellvalget står i §3.
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
add_seat_category(model_frame)  # seter som <5, =5, >5 (som i severity)
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
# \log \mu_i = \beta_0 + \sum_j f_j(x_{ij}) + \sum_{(j,k)} \gamma_{jk}\, x_{ij} x_{ik}.
# $$
#
# Her er $f_j$ enten lineær ($\beta_j x_{ij}$) eller en naturlig kubisk spline, og
# $\gamma_{jk}$ er en interaksjon mellom to variabler. For $1<p<2$ er $R_i$ en
# sammensatt Poisson–Gamma-variabel: masse i null (skadefrie år) og en
# kontinuerlig positiv del. Eksponeringen $e_i$ er `var_weights`, uten offset:
# $R_i$ er allerede en rate, og et år med høy eksponering er en mer presis
# observasjon av samme $\mu_i$.
#
# Power $p$ utledes én gang fra severity-dispersjonen til en bred ankermodell (§4.1)
# og låses *før* seleksjonen (deviance med ulike $p$ er ikke sammenlignbar, så $p$
# velges ikke ved CV).

# %%
# Hvilke variabler som er hva. Seleksjonen under leser bare disse listene.
LOCKED = ["policy_type", "year"]  # alltid med, fjernes aldri
CORE = ["circulation_area", "log_vehicle_value", "driver_age"]  # startmodell K0
GEOGRAPHY = "municipality_type"  # testes mot og sammen med circulation_area
ADDITIONAL = [  # tillegg: legges til én om gangen, så lenge de forbedrer modellen
    "performance_hp_per_tonne",
    "fuel_type",
    "payment_frequency",
    "business_type",
    "vehicle_brand_pooled",
    "seat_category",
]
INTERACTIONS = [  # testes bare når begge hovedeffektene allerede er med
    ("driver_age", "performance_hp_per_tonne"),
    ("driver_age", "policy_type"),
    ("driver_age", "log_vehicle_value"),
]

# En modell er en liste med termer (kolonne eller interaksjon) og eventuelle splines.
definitions = {"K0": {"terms": [*LOCKED, *CORE], "splines": {}}}


# %%
def interaction_factor(column):
    """Faktor i et interaksjonsledd; kategoriske kolonner får samme referansenivå som hovedeffekten."""
    if pd.api.types.is_numeric_dtype(model_frame[column]) and column != "year":
        return column
    base = model_frame.groupby(column)["total_exposure"].sum().idxmax()
    return f"C({column}, Treatment({base!r}))"


def formula_term(term, splines):
    """Ett formelledd: interaksjon, spline eller vanlig kolonne."""
    if isinstance(term, tuple):
        return ":".join(interaction_factor(part) for part in term)
    if term in splines:
        return f"cr({term}, df={splines[term]}, constraints='center')"
    return term


# %%
def build_tweedie_specification(name, definition, power):
    """Bygg Tweedie-spesifikasjonen (log-link, eksponeringsvekt, ingen offset)."""
    terms = [formula_term(term, definition["splines"]) for term in definition["terms"]]
    family = sm.families.Tweedie(var_power=power, link=sm.families.links.Log())
    columns = [term for term in definition["terms"] if isinstance(term, str)]
    return glm_spec(
        name,
        terms,
        "pure_premium",
        model_frame,
        family,
        "total_exposure",
        power,
        required_columns=columns,
    )


assert_tweedie_spec(
    build_tweedie_specification("K0", definitions["K0"], 1.5)
)  # kan fjernes

# %% [markdown]
# ## 4. Sammenligning og valg
#
# Alle trinn bruker den samme treleddsregelen: en kandidat velges bare hvis den har
# **lavere pooled OOF-deviance** enn foreldremodellen, er **bedre i minst fire av
# fem folder** og har **gyldig fit i alle folder**.
#
# | Trinn | Type | Kandidater |
# |---|---|---|
# | 4.2–4.3 Funksjonsform | ett valg | alder: spline df 2, 3, 4; bilverdi: spline df 3 |
# | 4.4 Geografi | ett valg | `municipality_type` i stedet for, eller i tillegg til, `circulation_area` |
# | 4.5 Tillegg | forover-løkke | `ADDITIONAL`, én om gangen til ingen består |
# | 4.6 Interaksjoner | forover-løkke | `INTERACTIONS` |
# | 4.7 Ablasjon | bakover-løkke | alle termer unntatt `LOCKED` |
#
# Ablasjonen kan ikke fjerne en hovedeffekt så lenge en interaksjon bruker den.
# Utviklingsscoren er litt optimistisk fordi mange kandidater sammenlignes på de
# samme foldene; 2024 er den uavhengige kontrollen og er ikke berørt.
#
# ### 4.1 Låsing av $p$ fra severity-dispersjonen
#
# En Tweedie med $1<p<2$ er en sammensatt Poisson–Gamma-variabel. Er skadestørrelsen
# Gamma med dispersjon $\phi_s$ (= CV$^2$), er
#
# $$p = \frac{1 + 2\phi_s}{1 + \phi_s}.$$
#
# $\phi_s$ hentes fra en Gamma-GLM på snittskade (vekt = antall skader, samme
# severity-utvalg som i severity-notebooken) med en bred ankermodell: låste
# variabler, kjerne, `municipality_type` og alle tillegg, lineært.
#
# > **Limitation (bør vurderes utvidet).** $p$ er svakt identifisert i disse dataene,
# > og metoden hviler på to antakelser: Gamma-fordelte skader med konstant CV, og at
# > dispersjonen er lik for alle poliser. Estimatet er følsomt for storskadene: uten
# > de 1 % største snittskadene faller det fra ca. 1,74 til ca. 1,62. Pearson-estimering
# > direkte på Tweedie-modellen ble forsøkt (T-05), men ga ingen gyldig rot i $(1,2)$.
# > Mulige utvidelser: profilelikelihood over et $p$-gitter (krever eksakt
# > Tweedie-tetthet), eller en sensitivitetsanalyse av sluttmodellen ved andre $p$.

# %%
severity_frame = build_severity_inputs(development)["severity_frame"]
anchor_terms = [*LOCKED, *CORE, GEOGRAPHY, *ADDITIONAL]
power_estimate = severity_implied_power(severity_frame, anchor_terms)
TWEEDIE_POWER = power_estimate["power"]
assert_valid_power(TWEEDIE_POWER)  # sanity-sjekk, kan fjernes
print(
    f"Severity-dispersjon φ_s = {power_estimate['dispersion']:.3f} "
    f"({power_estimate['n_severity_rows']} skadepoliseår)  →  låst p = {TWEEDIE_POWER:.3f}"
)

# %%
cv_results, specifications = {}, {}


def evaluate_model(name):
    """Bygg spesifikasjonen og kjør CV for én modelldefinisjon."""
    assert_model_definition(definitions[name], LOCKED)  # sanity-sjekk, kan fjernes
    specifications[name] = build_tweedie_specification(
        name, definitions[name], TWEEDIE_POWER
    )
    cv_results[name] = cross_validate_glm(specifications[name], model_frame, cv_folds)


def select_stage(parent_name, candidate_names):
    """Treleddsregelen for Tweedie, med låst $p$."""
    return select_candidate_stage(
        cv_results,
        parent_name,
        candidate_names,
        model_frame["pure_premium"],
        model_frame["total_exposure"],
        power=TWEEDIE_POWER,
    )


evaluate_model("K0")

# %% [markdown]
# ### 4.2 Funksjonsform: alder
#
# Lineær alder (K0) utfordres av naturlig kubisk spline med 2, 3 og 4
# frihetsgrader. Bare én aldersform kan velges.

# %%
core = definitions["K0"]
age_alternatives = {
    f"+alder_df{df}": {**core, "splines": {"driver_age": df}} for df in (2, 3, 4)
}
age_model, age_table = run_round(
    "K0", age_alternatives, definitions, evaluate_model, select_stage
)
display(age_table.round(4))
print(f"Valgt etter aldersform: {age_model}")

# %% [markdown]
# ### 4.3 Funksjonsform: bilverdi
#
# `log_vehicle_value` utfordres av en spline med 3 frihetsgrader.

# %%
current = definitions[age_model]
value_alternatives = {
    "+bilverdi_df3": {
        **current,
        "splines": {**current["splines"], "log_vehicle_value": 3},
    }
}
value_model, value_table = run_round(
    age_model, value_alternatives, definitions, evaluate_model, select_stage
)
display(value_table.round(4))
print(f"Valgt etter bilverdi: {value_model}")

# %% [markdown]
# ### 4.4 Geografi
#
# `circulation_area` (U/R) er med i kjernen. `municipality_type` (I/C/IS) testes
# enten *i stedet for* `circulation_area` eller *i tillegg til* den. Bare ett
# alternativ kan velges.

# %%
current = definitions[value_model]
geography_alternatives = {
    "+municipality_type": {**current, "terms": [*current["terms"], GEOGRAPHY]},
    "_bytt_til_municipality_type": {
        **current,
        "terms": [
            GEOGRAPHY if t == "circulation_area" else t for t in current["terms"]
        ],
    },
}
geography_model, geography_table = run_round(
    value_model, geography_alternatives, definitions, evaluate_model, select_stage
)
display(geography_table.round(4))
print(f"Valgt etter geografi: {geography_model}")

# %% [markdown]
# ### 4.5 Tillegg
#
# Forover-seleksjon: beste tillegg som består regelen legges til, og de
# gjenstående testes på nytt mot den nye modellen. Løkken stopper når ingen
# tillegg består. Flere tillegg kan altså komme inn.

# %%
additions_model, additions_table = run_stepwise(
    geography_model,
    propose_additions(ADDITIONAL),
    definitions,
    evaluate_model,
    select_stage,
)
display(additions_table.round(4))
print(f"Valgt etter tillegg: {additions_model}")

# %% [markdown]
# ### 4.6 Interaksjoner
#
# Samme forover-løkke. En interaksjon testes bare når begge hovedeffektene er i
# modellen, ellers hadde testen lagt til to ting samtidig.

# %%
interaction_model, interaction_table = run_stepwise(
    additions_model,
    propose_additions(INTERACTIONS),
    definitions,
    evaluate_model,
    select_stage,
)
display(interaction_table.round(4))
final_definition = definitions[interaction_model]
not_tested = [
    term
    for term in INTERACTIONS
    if term not in final_definition["terms"] and not can_add(final_definition, term)
]
print(f"Valgt etter interaksjoner: {interaction_model}")
print(f"Ikke testet (mangler hovedeffekt): {not_tested or 'ingen'}")

# %% [markdown]
# ### 4.7 Ablasjon
#
# Bakover-løkke: én term fjernes om gangen fra modellen som nå er valgt (alt
# unntatt `LOCKED`), og fjerningen beholdes bare når den oppfyller den samme
# treleddsregelen. Gjentas til ingen fjerning består.

# %%
final_model, ablation_table = run_stepwise(
    interaction_model,
    propose_removals(set(LOCKED)),
    definitions,
    evaluate_model,
    select_stage,
)
display(ablation_table.round(4))
print(f"Valgt etter ablasjon: {final_model}")

# %% [markdown]
# ### 4.8 Sammendrag
#
# Pooled OOF-deviance for modellen som ble valgt i hvert trinn (samme $p$, samme
# folder). Dette er utviklingsscore, ikke en uavhengig evaluering.

# %%
selection_path = [
    ("K0", "K0"),
    ("alder", age_model),
    ("bilverdi", value_model),
    ("geografi", geography_model),
    ("tillegg", additions_model),
    ("interaksjoner", interaction_model),
    ("ablasjon", final_model),
]
display(
    summarize_selection_path(
        selection_path,
        cv_results,
        model_frame["pure_premium"],
        model_frame["total_exposure"],
        TWEEDIE_POWER,
    ).round(4)
)
assert_model_definition(definitions[final_model], LOCKED)  # sanity-sjekk, kan fjernes
print("Valgt Tweedie-modell:", definitions[final_model])
print("Spesifikasjonen er ikke låst før den er godkjent. 2024 er ikke berørt.")
