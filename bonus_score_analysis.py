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
# # Bonus_score: tidsplassering og informasjonsinnhold
#
# Denne notebooken vurderer om `bonus_score` kan være en kandidat i den senere
# prismodellen. Den er avgrenset til egen-skadeporteføljen og bruker bare
# train-pool (2022–2023). Teståret 2024 åpnes ikke, slik at featurebeslutningen
# ikke blir drevet av endelig testdata.

# %%
from pathlib import Path

import pandas as pd
from IPython.display import display

from src.data_quality import build_variable_dictionary, clean_motor_data
from src.own_damage_descriptives import (
    build_bonus_change_summary,
    build_bonus_history_strata,
    build_bonus_lagged_panel,
    build_bonus_transition_matrix,
    fit_bonus_timing_model,
    plot_bonus_change_summary,
    plot_bonus_history_strata,
    plot_bonus_transition_matrix,
)

DATA_PATH = Path("data/Dataset of motor insurance portfolio.csv")
DESCRIPTION_PATH = Path("data/Descriptive of variables.xlsx")
if not DATA_PATH.exists() or not DESCRIPTION_PATH.exists():
    raise FileNotFoundError(
        "Kjør notebooken fra prosjektroten slik at filene i data/ er tilgjengelige."
    )

# %% [markdown]
# ## 1. Reproduserbart analysegrunnlag
#
# Samme konservative rensing og kaskoscope som i hovednotebooken brukes her.
# Positiv egen-skadepremie identifiserer aktiv dekning, og positiv eksponering
# identifiserer poliseår med risiko. `train_pool` inneholder kun 2022–2023.

# %%
raw_data = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8", low_memory=False)
variable_dictionary = build_variable_dictionary()
data, cleaning_log = clean_motor_data(raw_data, variable_dictionary)
own_damage_mask = data["property_damage_premium"].gt(0) & data[
    "total_exposure"
].gt(0)
own_damage = data.loc[own_damage_mask].copy()
train_pool = own_damage.loc[own_damage["year"].isin([2022, 2023])].copy()
assert train_pool["year"].isin([2022, 2023]).all()
assert train_pool["property_damage_premium"].gt(0).all()
assert train_pool["total_exposure"].gt(0).all()
display(cleaning_log)
display(
    pd.Series(
        {
            "Poliseår i kaskoscope": len(own_damage),
            "Poliseår i train-pool (2022–2023)": len(train_pool),
            "Eksponering i train-pool": train_pool["total_exposure"].sum(),
            "2024-rader brukt i analysen": int(train_pool["year"].eq(2024).sum()),
        },
        name="Verdi",
    ).to_frame()
)

# %% [markdown]
# ## 2. Tidsplassering av bonus_score
#
# Kilden gir ingen eksakt *as-of*-dato for `bonus_score`. Vi kobler derfor bare
# sammenhengende par `t-1 → t` for samme `insured_id`. G < N < B brukes bare i
# denne diagnosen til å definere forbedring og forverring; i en modell skal
# `bonus_score` fortsatt behandles kategorisk.
#
# Dette er en observasjonsdiagnose, ikke kausal dokumentasjon. Den kan ikke i
# seg selv bevise leakage eller leakage-fri timing.

# %%
bonus_panel = build_bonus_lagged_panel(train_pool)
bonus_transition = build_bonus_transition_matrix(bonus_panel)
bonus_changes = build_bonus_change_summary(bonus_panel)
display(
    pd.Series(
        {"Antall sammenhengende t-1 → t-par": len(bonus_panel)}, name="Verdi"
    ).to_frame()
)
display(bonus_transition)
display(plot_bonus_transition_matrix(bonus_transition))
display(bonus_changes)
display(plot_bonus_change_summary(bonus_changes))

# %% [markdown]
# ## 3. Multivariat timingdiagnose
#
# Den logistiske diagnosen har bonusforverring i t som utfall og inkluderer
# både lagget og samtidig antall egen-skader og alle skader, med kontroll for
# bonusklasse i t-1. År tas med dersom det varierer. Her finnes bare overgangen
# 2022 → 2023 i den leakage-sikre train-poolen, så en årseffekt kan ikke
# estimeres uten å bruke 2024. B er dårligste klasse og kan ikke forverres;
# den holdes derfor utenfor risikosettet, men beholdes i overgangsmatrisen.
#
# Odds ratio over én betyr høyere observert odds for bonusforverring, gitt de
# andre leddene. Resultatet er en stabilitetssjekk, ikke kausal dokumentasjon.

# %%
bonus_timing_effects, bonus_timing_diagnostics, bonus_timing_formula = (
    fit_bonus_timing_model(bonus_panel)
)
display(bonus_timing_diagnostics)
display(bonus_timing_effects)
display(pd.Series({"Modellformel": bonus_timing_formula}, name="Verdi").to_frame())

# %%
lag_total_or = bonus_timing_effects.loc[
    bonus_timing_effects["ledd"].eq("total_claims_lag"), "odds_ratio"
].iloc[0]
current_total_or = bonus_timing_effects.loc[
    bonus_timing_effects["ledd"].eq("total_claims"), "odds_ratio"
].iloc[0]
timing_interpretation = pd.Series(
    {
        "Tolkning": (
            f"I {int(bonus_timing_diagnostics.loc['sammenhengende_par_i_modell', 'Verdi']):,} "
            f"par i risikosettet var odds ratio {lag_total_or:.2f} for én ekstra "
            f"lagget totalskade, mot {current_total_or:.2f} for samtidig totalskade. "
            "Sammen med intervallene i tabellen er dette konsistent med en lagget "
            "tolkning, men ikke et bevis på leakage-fri timing."
        )
    },
    name="Verdi",
).to_frame()
display(timing_interpretation)

# %% [markdown]
# ## 4. Informasjon utover eksplisitt ettårig skadehistorikk
#
# Nåværende kaskofrekvens vises per bonusklasse innen strata av lagget totalt
# skadeutfall og lagget egen-skadeutfall. Dette undersøker om scoren skiller
# risiko blant poliser med lik, eksplisitt observert ettårig historikk. Den kan
# fortsatt representere eldre skadehistorikk, underwriting-informasjon eller
# uavklart timing.

# %%
bonus_history_strata = build_bonus_history_strata(bonus_panel)
display(bonus_history_strata)
display(plot_bonus_history_strata(bonus_history_strata))

# %% [markdown]
# ## 5. Konklusjon og regel for modelleringsfasen
#
# Analysen tyder på at `bonus_score` inneholder informasjon utover den
# eksplisitt observerte ettårige skadehistorikken. Timingmønsteret er mest
# konsistent med at scoren bygger på tidligere års skader, ikke inneværende års
# skadeutfall. Dette er ikke et endelig bevis på leakage-fri timing fordi
# eksakt *as-of*-dato mangler.
#
# `bonus_score` beholdes derfor foreløpig som kategorisk prediktorkandidat. I
# modelleringsfasen skal den vurderes på nytt gjennom leakage-sikre
# sammenligninger kun i train/CV: basis; basis + lagget historikk; basis +
# bonus; og basis + begge, samt en obligatorisk sensitivitetsmodell uten bonus.
# Teståret 2024 røres ikke før endelig evaluering.
