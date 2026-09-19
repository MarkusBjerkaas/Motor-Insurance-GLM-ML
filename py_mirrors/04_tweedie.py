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
# # Tweedie-modell for ren egen-skadepremie
#
# Formålet er å modellere forventet skadekostnad per eksponeringsår (pure premium)
# direkte med en Tweedie-GLM. Modellen utfordrer den todelte modellen
# $\widehat{\text{pure premium}} = \widehat{\text{frekvens}} \times \widehat{\text{severity}}$.
# Notebooken bruker bare utviklingsårene 2022–2023; 2024 er ikke brukt.

# %%
from functools import partial

import pandas as pd
import statsmodels.api as sm
from IPython.display import display

from src_asserts.common_glm_asserts import (
    assert_full_oof_coverage,
    assert_model_definition,
    assert_valid_folds,
)
from src_asserts.residual_diagnostics_asserts import assert_residual_diagnostics_result
from src_asserts.tweedie_asserts import assert_tweedie_spec, assert_valid_power
from src_core_glm.glm_core import cross_validate_glm, fit_glm, prepare_design_frame
from src_core_glm.model_data import add_seat_category, build_development_frames
from src_core_glm.model_selection import (
    build_specification,
    can_add,
    evaluate_models_parallel,
    propose_additions,
    propose_geography,
    propose_removals,
    propose_spline_forms,
    run_round,
    run_stepwise,
    select_candidate_stage,
)
from src_core_glm.residual_diagnostics import (
    build_residual_diagnostic_summary,
    cross_validate_residual_catboost,
    plot_residual_diagnostic_overview,
)
from src_core_glm.selection_report import (
    build_forward_selection_table,
    build_top_models_table,
)
from src_core_glm.validation import build_group_folds
from src_frequency.frequency_diagnostics import (
    plot_oof_residuals_against_fitted,
    plot_oof_residuals_by_continuous_predictor,
)
from src_model_comparison.locked_models import lock_glm
from src_model_comparison.variable_coverage import build_variable_coverage
from src_severity.severity_data import build_severity_inputs
from src_tweedie.tweedie_data import build_tweedie_frame, summarize_tweedie_frame
from src_tweedie.tweedie_diagnostics import tweedie_deviance_residuals
from src_tweedie.tweedie_power import severity_implied_power

SEED = 100
N_FOLDS = 5
SE_MULTIPLIER = 1.0  # fjerde seleksjonskrav (§4): snittgevinst > 1 standardfeil
N_JOBS = 8  # parallelle prosesser for kandidat-CV (resultatene er de samme som med 1)

# %% [markdown]
# ## 1. Datagrunnlag
#
# La $S_i$ være `property_incurred`, $e_i$ være `total_exposure` og
# $R_i = S_i/e_i$ observert pure premium per eksponeringsår. Alle poliseår er
# gyldige observasjoner, også skadefrie ($R_i = 0$); $S_i$ er aggregert årlig
# skadekostnad, så flere skader per poliseår er tillatt.

# %%
frames = build_development_frames()
development = frames["development"]

# Kolonnene som er tilgjengelige for modellene.
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
# Fem gruppefolder på `insured_id` (samme seed og folder som i frekvens- og
# severity-notebookene, så OOF-prediksjonene kan sammenlignes rad for rad).

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
# Power $p$ utledes fra severity-dispersjonen (§4.1) og låses før seleksjonen;
# deviance med ulike $p$ er ikke sammenlignbar, så $p$ velges ikke ved CV.

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
def build_tweedie_specification(name, definition, power):
    """Bygg Tweedie-spesifikasjonen (log-link, eksponeringsvekt, ingen offset)."""
    settings = {
        "y": "pure_premium",
        "family": sm.families.Tweedie(var_power=power, link=sm.families.links.Log()),
        "weight": "total_exposure",
        "power": power,
    }
    return build_specification(name, definition, model_frame, settings)


assert_tweedie_spec(
    build_tweedie_specification("K0", definitions["K0"], 1.5)
)  # kan fjernes

# %% [markdown]
# ## 4. Sammenligning og valg
#
# Alle trinn bruker den samme firleddsregelen: en kandidat velges bare hvis den har
# **lavere pooled OOF-deviance** enn foreldremodellen, er **bedre i minst fire av
# fem folder**, har **gyldig fit i alle folder** og har en **snittgevinst over én
# standardfeil**. Med $d_k$ = foreldrens minus kandidatens deviance i fold $k$ er
# siste krav
#
# $$\bar d > \frac{s_d}{\sqrt{5}}, \qquad \bar d = \tfrac15\sum_k d_k.$$
#
# Kravet er et grovt støyfilter mot små gevinster, ikke en formell test (fem folder
# gir et usikkert $s_d$).
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
#
# ### 4.1 Låsing av $p$ fra severity-dispersjonen
#
# Er skadestørrelsen Gamma med dispersjon $\phi_s$ (= CV$^2$), er den sammensatte
# Poisson–Gamma-variabelen Tweedie med
#
# $$p = \frac{1 + 2\phi_s}{1 + \phi_s}.$$
#
# $\phi_s$ hentes fra en Gamma-GLM på snittskade (vekt = antall skader, samme
# severity-utvalg som i severity-notebooken) med en bred ankermodell: låste
# variabler, kjerne, `municipality_type` og alle tillegg, lineært. Metoden
# forutsetter Gamma-fordelte skader med konstant CV og lik dispersjon for alle poliser.

# %%
severity_frame = build_severity_inputs(development)["severity_frame"]
anchor_terms = [*LOCKED, *CORE, GEOGRAPHY, *ADDITIONAL]
power_estimate = severity_implied_power(severity_frame, anchor_terms)
TWEEDIE_POWER = power_estimate["power"]
assert_valid_power(TWEEDIE_POWER)  # sanity-sjekk, kan fjernes
print(
    f"Severity-dispersjon φ_s = {power_estimate['dispersion']:.3f} "
    f"({power_estimate['n_severity_rows']} skadepoliseår)  →  låst p = {TWEEDIE_POWER:.5f}"
)

# %%
cv_results, specifications = {}, {}


def evaluate_model(name):
    """Bygg spesifikasjonen og kjør CV for én modelldefinisjon."""
    assert_model_definition(definitions[name], LOCKED)  # sanity-sjekk, kan fjernes
    specification = build_tweedie_specification(name, definitions[name], TWEEDIE_POWER)
    return specification, cross_validate_glm(specification, model_frame, cv_folds)


def evaluate_models(names):
    """Kjør CV for flere modelldefinisjoner parallelt (én prosess per kandidat)."""
    evaluate_models_parallel(evaluate_model, specifications, cv_results, N_JOBS, names)


# Firleddsregelen med låst p; kalles som select_stage(foreldre, kandidatnavn).
select_stage = partial(
    select_candidate_stage,
    cv_results,
    response=model_frame["pure_premium"],
    sample_weight=model_frame["total_exposure"],
    power=TWEEDIE_POWER,
    se_multiplier=SE_MULTIPLIER,
)

evaluate_models(["K0"])

# %% [markdown]
# ### 4.2 Funksjonsform: alder
#
# Lineær alder (K0) utfordres av naturlig kubisk spline med 2, 3 og 4
# frihetsgrader; bare én form kan velges.

# %%
core = definitions["K0"]
age_alternatives = propose_spline_forms(core, "driver_age", "alder", (2, 3, 4))
age_model, age_table = run_round(
    "K0", age_alternatives, definitions, evaluate_models, select_stage
)
display(age_table.round(4))
print(f"Valgt etter aldersform: {age_model}")

# %% [markdown]
# ### 4.3 Funksjonsform: bilverdi
#
# `log_vehicle_value` utfordres av en spline med 3 frihetsgrader.

# %%
current = definitions[age_model]
value_alternatives = propose_spline_forms(
    current, "log_vehicle_value", "bilverdi", (3,)
)
value_model, value_table = run_round(
    age_model, value_alternatives, definitions, evaluate_models, select_stage
)
display(value_table.round(4))
print(f"Valgt etter bilverdi: {value_model}")

# %% [markdown]
# ### 4.4 Geografi
#
# `circulation_area` (U/R) er med i kjernen. `municipality_type` (I/C/IS) testes
# enten *i stedet for* `circulation_area` eller *i tillegg til* den.

# %%
current = definitions[value_model]
geography_alternatives = propose_geography(current, GEOGRAPHY, "circulation_area")
geography_model, geography_table = run_round(
    value_model, geography_alternatives, definitions, evaluate_models, select_stage
)
display(geography_table.round(4))
print(f"Valgt etter geografi: {geography_model}")

# %% [markdown]
# ### 4.5 Tillegg
#
# Forover-seleksjon: beste tillegg som består regelen legges til, og de
# gjenstående testes på nytt mot den nye modellen, til ingen består.

# %%
additions_model, additions_table = run_stepwise(
    geography_model,
    propose_additions(ADDITIONAL),
    definitions,
    evaluate_models,
    select_stage,
)
display(additions_table.round(4))
print(f"Valgt etter tillegg: {additions_model}")

# %% [markdown]
# ### 4.6 Interaksjoner
#
# Samme forover-løkke. En interaksjon testes bare når begge hovedeffektene er i
# modellen.

# %%
interaction_model, interaction_table = run_stepwise(
    additions_model,
    propose_additions(INTERACTIONS),
    definitions,
    evaluate_models,
    select_stage,
)
display(interaction_table.round(4))
interaction_definition = definitions[interaction_model]
not_tested = [
    term
    for term in INTERACTIONS
    if term not in interaction_definition["terms"]
    and not can_add(interaction_definition, term)
]
print(f"Valgt etter interaksjoner: {interaction_model}")
print(f"Ikke testet (mangler hovedeffekt): {not_tested or 'ingen'}")

# %% [markdown]
# ### 4.7 Ablasjon
#
# Bakover-løkke: én term (alt unntatt `LOCKED`) fjernes om gangen, og fjerningen
# beholdes bare når den oppfyller den samme firleddsregelen.

# %%
final_model, ablation_table = run_stepwise(
    interaction_model,
    propose_removals(set(LOCKED)),
    definitions,
    evaluate_models,
    select_stage,
)
display(ablation_table.round(4))
print(f"Valgt etter ablasjon: {final_model}")

# %% [markdown]
# ### 4.8 Forover-seleksjon: hva tilfører verdi?
#
# Én rad per valgt steg. `oof_deviance` er pooled OOF etter steget,
# `snitt_gevinst` og `se_gevinst` er snitt og standardfeil av fold-gevinstene mot
# foreldremodellen, og `gevinst_i_se` er forholdet mellom dem. Rundt 1 er marginalt:
# når mange kandidater sammenlignes, ligger den beste ofte på 1–1,5 SE uten noe
# ekte signal.

# %%
display(
    build_forward_selection_table(
        [age_table, value_table, geography_table, additions_table, interaction_table]
    ).round(4)
)
final_definition = definitions[final_model]
assert_model_definition(final_definition, LOCKED)  # sanity-sjekk, kan fjernes
print("Valgt Tweedie-modell:", final_definition)

# %% [markdown]
# ### 4.9 Konkurrerende modeller
#
# De tre evaluerte modellene med lavest pooled OOF-deviance. `endring` viser hva
# som skiller dem fra den valgte modellen, `parametere` er antall koeffisienter
# (inkludert konstantledd) og `delta_mot_valgt` er OOF-deviance minus den valgtes
# (negativ = lavere). Er en enklere modell praktisk talt like god, er den å
# foretrekke; en mer kompleks modell teller bare hvis forskjellen er tydelig over
# støyen (jf. SE i §4.8).

# %%
top_models = build_top_models_table(
    cv_results,
    definitions,
    final_model,
    model_frame["pure_premium"],
    model_frame["total_exposure"],
    TWEEDIE_POWER,
)
with pd.option_context("display.max_colwidth", None):
    display(top_models.reset_index(drop=True).round(4))

# %% [markdown]
# ### 4.10 Endelig modell
#
# Tweedie-GLM med log-link, eksponeringen $e_i$ som vekt (uten offset) og låst
# $p \approx 1{,}744$:
#
# $$
# R_i = \frac{S_i}{e_i} \sim \operatorname{Tweedie}\!\left(\mu_i, \frac{\phi}{e_i}, p\right),
# \qquad p \approx 1{,}744,
# $$
#
# $$
# \begin{aligned}
# \log \mu_i ={}& \beta_0
# + \gamma^{\text{type}}_{\text{policy_type}_i}
# + \gamma^{\text{year}}_{\text{year}_i}
# + \gamma^{\text{area}}_{\text{circulation_area}_i}
# + \gamma^{\text{pay}}_{\text{payment_frequency}_i} \\
# &+ \gamma^{\text{fuel}}_{\text{fuel_type}_i}
# + \gamma^{\text{seat}}_{\text{seat_category}_i}
# + \gamma^{\text{bus}}_{\text{business_type}_i}
# + \beta_v\,\text{log_vehicle_value}_i
# + f(\text{driver_age}_i).
# \end{aligned}
# $$
#
# Her er $\gamma$ effekten av et nivå på en kategorisk variabel, og $\gamma = 0$ for
# referansenivået: `policy_type` = COMP_E, `year` = 2023, `circulation_area` = U,
# `payment_frequency` = A, `fuel_type` = D, `seat_category` = `=5`,
# `business_type` = NB. $f$ er en naturlig kubisk spline med 3 frihetsgrader.
# Modellen har 15 parametere: konstantledd (1) + kategoriske nivåer (10) +
# `log_vehicle_value` (1) + spline (3). Den estimeres på alle utviklingsdata
# (2022–2023) med cluster-robuste standardfeil på `insured_id`.

# %%
final_specification = specifications[final_model]
final_design = prepare_design_frame(
    model_frame, model_frame, final_specification["required_columns"]
)
final_fit = fit_glm(
    final_specification, final_design, cluster_groups=final_design["insured_id"]
)
# Kontroll: notebooken og LaTeX-teksten over beskriver samme modell.
assert final_definition["terms"] == [
    "policy_type",
    "year",
    "circulation_area",
    "log_vehicle_value",
    "driver_age",
    "payment_frequency",
    "fuel_type",
    "seat_category",
    "business_type",
]
assert final_definition["splines"] == {"driver_age": 3}
assert round(TWEEDIE_POWER, 3) == 1.744 and len(final_fit.params) == 15

# %% [markdown]
# ### 4.11 Koeffisienter
#
# Koeffisientene er log-relativiteter i dette datasettet, ikke kausale effekter.

# %%
display(final_fit.summary())

# %% [markdown]
# ## 5. CatBoost-diagnostikk av residualstruktur
#
# Finnes det struktur i middelverdien som den valgte GLM-en ikke fanger? CatBoost
# brukes kun som diagnose og velger aldri GLM-variabler.
#
# $$
# z_i=\frac{R_i}{\hat\mu_{i,\mathrm{inner\ OOF}}},
# \qquad
# a_i=e_i\,\hat\mu_{i,\mathrm{inner\ OOF}}^{\,2-p},
# \qquad
# \hat\mu_i^*=\hat\mu_{i,\mathrm{GLM}}\,\hat h(x_i).
# $$
#
# $z_i$ er observert pure premium relativt til GLM-prediksjonen (rundt 1 hvis
# GLM-en treffer), $a_i$ er radens vekt med det låste $p$, og $\hat h(x)$ er
# CatBoosts korreksjonsfaktor lært med Poisson-loss. **Indre CV** (fem nye
# `insured_id`-folder inne i hver ytre treningsdel) lager $z_i$, så CatBoost
# aldri trener på prediksjoner fra en GLM som har sett raden. **Ytre CV** måler
# gevinsten på forsikrede CatBoost ikke har sett. Dybde 1 er additiv; dybde 3 kan
# også fange interaksjoner.

# %%
residual_diagnostics = cross_validate_residual_catboost(
    final_specification,
    model_frame,
    cv_folds,
    final_specification["required_columns"],
    categorical_columns=tuple(final_specification["base_levels"]),
    blocked_feature_columns=("property_incurred", "pure_premium"),  # responsen
    seed=SEED,
)
assert_residual_diagnostics_result(
    residual_diagnostics, model_frame, cv_folds
)  # kan fjernes

# %% [markdown]
# ### 5.1 Finnes det residualstruktur?
#
# GLM-en sammenlignet med en konstant korreksjon (bare nivå) og to CatBoost-dybder
# på alle OOF-rader. `delta_*` er reduksjon i deviance (positivt er bedre),
# `better_than_*_folds` teller ytre folder med gevinst og `oof_ae` er faktisk delt
# på predikert skadekostnad.

# %%
display(build_residual_diagnostic_summary(residual_diagnostics).round(4))

# %%
residual_overview_figure = plot_residual_diagnostic_overview(residual_diagnostics)

# %% [markdown]
# ## 6. OOF-diagnostikk av Tweedie-GLM-en
#
# Eksponeringsvektede Tweedie deviance-residualer fra folden der poliseåret ikke
# inngikk i tilpasningen. Skadefrie år ligger i et eget negativt bånd (residualen er
# da bare en funksjon av $\hat\mu_i$), så spredningen er ikke normal. LOWESS-kurvene
# er visuelle hjelpemidler, ikke tester, og bør ligge nær null.

# %%
oof_premium = cv_results[final_model]["oof"]
assert_full_oof_coverage(oof_premium, model_frame.index, "Tweedie-OOF")  # kan fjernes
oof_residuals = tweedie_deviance_residuals(
    model_frame["pure_premium"],
    oof_premium,
    model_frame["total_exposure"],
    TWEEDIE_POWER,
)
oof_fitted_figure = plot_oof_residuals_against_fitted(
    oof_premium,
    oof_residuals,
    xlabel="OOF-predikert pure premium per eksponeringsår",
    title="OOF deviance-residualer mot predikert pure premium",
)

# %%
continuous_predictors = [
    predictor
    for predictor in ["driver_age", "log_vehicle_value", "performance_hp_per_tonne"]
    if predictor in final_specification["required_columns"]
]
oof_predictor_figure = plot_oof_residuals_by_continuous_predictor(
    model_frame, continuous_predictors, oof_residuals
)

# %% [markdown]
# ## 7. Variabelsjekk
#
# Alle tilgjengelige variabler skal ha vært kandidater, og alle øvrige kolonner i
# utviklingsdataene skal være utelatt med begrunnelse. Kontrollen feiler ellers.

# %%
tested_terms = [*LOCKED, *CORE, GEOGRAPHY, *ADDITIONAL]
variable_coverage = build_variable_coverage(
    development,
    AVAILABLE_COLUMNS,
    tested_terms,
    final_specification["required_columns"],
)
display(variable_coverage)

# %% [markdown]
# ## 8. Låst modell
#
# Den endelige modellen (§4.10) er allerede tilpasset på alle utviklingsdata.
# Den lagres som `models/tweedie.joblib`, lastes på nytt og kontrolleres mot
# notebookens egne prediksjoner.

# %%
display(lock_glm("tweedie", final_specification, final_fit, final_design, development))

# %% [markdown]
# ## 9. Begrensninger
#
# - **Optimistisk utviklingsscore.** Alle valg (funksjonsform, tillegg,
#   interaksjoner, ablasjon) er gjort på de samme fem foldene. Pooled OOF-deviance
#   (33,26) er derfor et optimistisk mål på ytelse på nye data. I §4.8 ligger
#   gevinstene på 1,3–1,6 standardfeil, unntatt `business_type` (2,9).
# - **Lite signal.** 55 246 poliseår, ca. 10 % med skade (5 698 skadepoliseår).
#   Cox–Snell pseudo-$R^2$ er 0,006, og koeffisientene har brede intervaller.
#   CatBoost-diagnostikken (§5.1) gir bare 0,0061 lavere pooled deviance for dybde 3,
#   men styrken er lav, så fravær av gevinst er ikke bevis for fravær av struktur.
# - **Låst $p$.** $p = 1{,}744$ er utledet fra severity-dispersjonen og hviler på
#   Gamma-fordelte skader med konstant CV. Estimatet er følsomt for storskader: uten
#   de 1 % største snittskadene faller det fra ca. 1,74 til ca. 1,62. Det er ikke
#   gjort noen sensitivitetsanalyse av sluttmodellen ved andre $p$.
# - **Modellantakelser.** Log-link gir multiplikative effekter, og dispersjonen er
#   konstant. Bare tre interaksjoner er testet; `driver_age*log_vehicle_value` ga
#   lavere deviance (−0,0061) men besto ikke seleksjonsregelen (§4.9). Koeffisientene
#   er assosiasjoner, ikke kausale effekter.
# - **Årseffekt.** `year` er en låst kategorisk term og 2024 finnes ikke i
#   treningsdata, så 2024 skåres med 2023-nivået. Det er ingen trend-ekstrapolering,
#   og årsdrift blir liggende i porteføljebalansen.
# - **Klipping.** Numeriske prediktorer klippes til treningsområdet ved skåring
#   (se `src_model_comparison/locked_models.py`), så modellen er flat utenfor det
#   observerte området.
