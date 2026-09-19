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
# # Endelig sammenligning på 2024-teståret
#
# Denne notebooken er presentasjonen av de låste prismodellene. Den viser én
# sammenligningstabell og én figur, begge basert på det urørte teståret.
#
# **Testlåsen er aktiv.** Ingen 2024-data leses, klargjøres eller evalueres før
# `RUN_2024_EVALUATION` og godkjenningskoden bevisst aktiveres etter eksplisitt
# godkjenning.

# %%
from IPython.display import display

from src_model_comparison.test_evaluation import (
    build_approved_test_frame,
    build_calibration_table,
    evaluate_pricing_models,
    plot_test_comparison,
)

TWEEDIE_POWER = 1.744
RUN_2024_EVALUATION = False
APPROVAL_CODE = None
retained_brands = None
prediction_functions = {}

# %% [markdown]
# ## 1. Låste modeller
#
# Fylles ut først når alle spesifikasjoner er låst. Hver funksjon skal motta
# den klargjorte test-rammen og returnere forventet ren premie per
# eksponeringsår. Den todelte modellen returnerer frekvens $\times$ severity.
#
# Modellobjekter finnes foreløpig bare i de enkelte modelleringsnotebookene.
# Derfor må de fittes på hele utviklingssettet og deres treningslærte
# imputasjon/transformasjoner tas med hit før testlåsen åpnes.

# %%
# prediction_functions = {
#     "Tweedie-GLM": lambda frame: tweedie_predictor(frame),
#     "Toleddet GLM": lambda frame: frequency_predictor(frame) * severity_predictor(frame),
#     "CatBoost": lambda frame: catboost_predictor(frame),
# }
# retained_brands = ...  # Kun poolingen lært fra utviklingsårene 2022–2023.

# %% [markdown]
# ## 2. Testprotokoll
#
# For poliseår $i$ sammenlignes observert og predikert ren premie,
# $R_i=S_i/e_i$ og $\widehat R_i$. Primærmålet er eksponeringsvektet
# Tweedie-deviance med den låste verdien $p=1.744$; lavere er bedre. Vektet MAE
# og porteføljebalanse supplerer med henholdsvis gjennomsnittlig feil og nivå.
#
# Teståret brukes én gang, etter at modellene og denne protokollen er låst.

# %%
if RUN_2024_EVALUATION:
    if retained_brands is None or not prediction_functions:
        raise ValueError("Koble inn alle låste modeller før testlåsen åpnes.")
    test_frame = build_approved_test_frame(
        retained_brands, approval_code=APPROVAL_CODE
    )
    results, predictions = evaluate_pricing_models(
        test_frame, prediction_functions, power=TWEEDIE_POWER
    )
    calibration = build_calibration_table(test_frame, predictions)

# %% [markdown]
# ## 3. Resultater
#
# Tabellen rangerer modellene etter primærmålet. Figuren er en kontroll av om
# riktig porteføljenivå også holdes på tvers av risikogruppene.

# %%
if RUN_2024_EVALUATION:
    display(results.round(3))
    figure = plot_test_comparison(results, calibration)

# %% [markdown]
# ## 4. Tolkning
#
# Den beste modellen har lavest Tweedie-deviance, men velges bare dersom
# kalibreringsplottet ikke viser en vesentlig systematisk skjevhet. Teståret
# er ett historisk år, så resultatet dokumenterer generalisering til denne
# perioden, ikke en universell eller kausal effekt.
