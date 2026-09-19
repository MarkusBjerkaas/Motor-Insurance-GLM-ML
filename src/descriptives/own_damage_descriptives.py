"""Deskriptiv analyse av frekvens, severity og ren premie for egen skade (kasko).

Alle funksjoner tar en dataframe som parameter (ingen innlasting her) — kalles
fra notebooken med ``train_pool`` for å unngå at testsettet (2024) påvirker
variabelvalg før endelig modellevaluering. Målevariablene er:

- ``property_claims``  — skadeantall (frekvens-teller)
- ``property_incurred`` — påløpt skadekostnad (severity/ren premie)
- ``total_exposure``    — eksponering i poliseår (frekvens-/ren premie-nevner)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#2a78d6"
CONTEXT = "#9ec5f4"

SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "own_damage_blues",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
DIVERGING = LinearSegmentedColormap.from_list(
    "own_damage_diverging", ["#b23a2f", "#f2c9c2", SURFACE, "#9ec5f4", "#184f95"]
)

CLAIMS_COL = "property_claims"
INCURRED_COL = "property_incurred"
EXPOSURE_COL = "total_exposure"
# Beløpsfeltene presenteres i EUR i kilden. Én cent er derfor en transparent
# toleranse for numerisk null i *fordelingsdiagnostikk*, ikke en endring av data.
CURRENCY_ZERO_TOLERANCE = 0.01

CATEGORICAL_VARS = [
    "policy_type",
    "bonus_score",
    "fuel_type",
    "municipality_type",
    "circulation_area",
    "payment_frequency",
    "business_type",
    "vehicle_brand_pooled",
]
NUMERIC_VARS = [
    "driver_age",
    "driving_experience_years",
    "log_vehicle_value",
    "performance_hp_per_tonne",
    "seats",
]
# Manglendehet undersøkes for prediktorene som faktisk går videre til modellering.
# ``vehicle_age`` er kun et diagnostisk felt; se aldersdiagnostikken i notebooken.
MISSING_CHECK_VARS = [
    "driving_experience_years",
    "fuel_type",
    "log_vehicle_value",
    "performance_hp_per_tonne",
]


# ---------------------------------------------------------------------------
# Tabell: grunnlag for avgrensningsbeslutningen (B-29, CC ekskluderes)
# ---------------------------------------------------------------------------


def build_cc_scope_evidence(data, years):
    """Dokumenter grunnlaget for å ekskludere CC fra egen-skadepopulasjonen (B-29).

    Tar den urensede/ufiltrerte ``data`` (alle policy_type, ikke bare
    kaskoprodukter) og avgrenser internt til ``years`` (skal alltid være
    ``TRAIN_YEARS`` — aldri 2024). Ingen skadeutfall inngår i noen av de tre
    evidenspunktene:

    1. Andel CC-poliseår med positiv egen-skadepremie, per år og samlet — viser
       at egen-skade i praksis nesten aldri er aktiv for CC.
    2. Blant CC-poliseår med egen-skadepremie som også finnes i det andre
       treningsåret: andel som da er registrert som COMP_E/COMP_N. Høy andel
       indikerer produktbytte (kilden beholder bare siste registrering per
       poliseår), ikke en stabil CC-kaskodekning.
    3. Median egen-skadepremie per eksponeringsår per 1000 i bilverdi for
       CC vs. COMP_E vs. COMP_N — viser at prisingsgrunnlaget for CC-gruppen
       avviker fra de faktiske kaskoproduktene.
    """
    scope = data.loc[data["year"].isin(years)].copy()
    scope["policy_type"] = scope["policy_type"].astype(str)
    rows = []

    # 1) OD-premie-andel for CC, per år og samlet
    cc = scope.loc[scope["policy_type"].eq("CC")]
    for year, sub in cc.groupby("year"):
        rows.append(
            {
                "evidens": "Andel CC-poliseår med positiv egen-skadepremie",
                "gruppe": str(year),
                "verdi": sub["property_damage_premium"].gt(0).mean() * 100,
                "enhet": "% av CC-poliseår",
            }
        )
    rows.append(
        {
            "evidens": "Andel CC-poliseår med positiv egen-skadepremie",
            "gruppe": "-".join(str(y) for y in years),
            "verdi": cc["property_damage_premium"].gt(0).mean() * 100,
            "enhet": "% av CC-poliseår",
        }
    )

    # 2) Produktbytte: CC-med-OD som finnes i det andre året
    cc_od = cc.loc[cc["property_damage_premium"].gt(0), ["insured_id", "year"]]
    other = scope[["insured_id", "year", "policy_type"]].merge(
        cc_od, on="insured_id", suffixes=("", "_cc")
    )
    other = other.loc[other["year"].ne(other["year_cc"])]
    switch_share = (
        other["policy_type"].isin(("COMP_E", "COMP_N")).mean() if len(other) else np.nan
    )
    rows.append(
        {
            "evidens": "Andel CC-med-OD som er COMP_E/COMP_N i det andre året",
            "gruppe": f"n={other['insured_id'].nunique()}",
            "verdi": switch_share * 100,
            "enhet": "% av CC-med-OD funnet i annet år",
        }
    )

    # 3) Prisingsgrunnlag: OD-premie per eksponeringsår per 1000 i bilverdi
    od = scope.loc[
        scope["property_damage_premium"].gt(0) & scope["total_exposure"].gt(0)
    ].copy()
    od["premie_rate"] = (
        od["property_damage_premium"].astype(float)
        / od["total_exposure"].astype(float)
        / od["vehicle_value"].astype(float)
        * 1000
    )
    for product in ("CC", "COMP_E", "COMP_N"):
        rows.append(
            {
                "evidens": "Median OD-premie per eksp.år per 1000 i bilverdi",
                "gruppe": product,
                "verdi": od.loc[od["policy_type"].eq(product), "premie_rate"].median(),
                "enhet": "EUR-rate",
            }
        )

    return pd.DataFrame(rows).round(2)


# ---------------------------------------------------------------------------
# Tabeller: utfallsvariablene selv
# ---------------------------------------------------------------------------


def build_claim_count_distribution(frame):
    """Fordeling per heltallig skadeantall, med tilhørende eksponering.

    Alle heltall fra null til høyeste observerte skadeantall vises eksplisitt.
    Det gjør uventet opphopning ved bestemte skadeantall synlig.
    """
    order = pd.RangeIndex(frame[CLAIMS_COL].max() + 1)
    counts = frame[CLAIMS_COL].value_counts().reindex(order, fill_value=0).astype(int)
    exposure = (
        frame.groupby(CLAIMS_COL, observed=True)[EXPOSURE_COL]
        .sum()
        .reindex(order, fill_value=0)
    )
    return pd.DataFrame(
        {
            "skadeantall": order,
            "poliseår": counts.values,
            "andel_poliseår_prosent": (counts / counts.sum() * 100).values,
            "eksponering": exposure.values,
        }
    ).round(2)


def build_overdispersion_summary(frame):
    """Sjekk varians vs. gjennomsnitt for skadeantall (indikasjon for Poisson vs. NegBin).

    Uvektet på tvers av poliseår, ikke justert for at eksponering varierer per
    rad — gir en rask indikasjon, ikke et presist dispersjonsestimat.
    """
    counts = frame[CLAIMS_COL]
    mean = counts.mean()
    variance = counts.var()
    return (
        pd.Series(
            {
                "gjennomsnitt": mean,
                "varians": variance,
                "spredningsforhold_var_over_snitt": variance / mean,
                "andel_poliseår_uten_skade_prosent": counts.eq(0).mean() * 100,
            },
            name="Verdi",
        )
        .to_frame()
        .round(4)
    )


def build_severity_summary(frame):
    """Beskrivende statistikk for severity (incurred/skadeantall) gitt skade.

    Data er aggregert per poliseår, ikke per skademelding: ved flere skader i
    samme poliseår brukes gjennomsnittlig severity per skade i den raden.
    """
    with_claim = frame.loc[frame[CLAIMS_COL].gt(0)]
    severity = with_claim[INCURRED_COL] / with_claim[CLAIMS_COL]
    return (
        pd.Series(
            {
                "n_skaderader": severity.shape[0],
                "andel_med_severity_null_prosent": severity.eq(0).mean() * 100,
                "gjennomsnitt": severity.mean(),
                "median": severity.median(),
                "std": severity.std(),
                "p90": severity.quantile(0.9),
                "p99": severity.quantile(0.99),
                "maks": severity.max(),
                "skjevhet": stats.skew(severity),
            },
            name="Verdi",
        )
        .to_frame()
        .round(2)
    )


def _fit_positive_distributions(values, zero_tolerance=CURRENCY_ZERO_TOLERANCE):
    """Sammenlign Gamma og lognormal med MLE/AIC på strengt positive verdier.

    Lokasjon fikseres til null i begge fordelinger. Dette er en enkel
    deskriptiv density-fit, ikke en frekvens-/severitymodell eller en test av
    hvordan skadebeløp senere bør vektes.
    """
    all_values = pd.Series(values).dropna()
    diagnostic_zero = np.isclose(all_values, 0, atol=zero_tolerance, rtol=0)
    positive = all_values.loc[~diagnostic_zero & all_values.gt(zero_tolerance)]
    if positive.empty:
        raise ValueError(
            "Kan ikke tilpasse positiv fordeling uten positive observasjoner."
        )
    gamma_shape, _, gamma_scale = stats.gamma.fit(positive, floc=0)
    log_sigma, _, log_scale = stats.lognorm.fit(positive, floc=0)
    fits = [
        {
            "fordeling": "Gamma",
            "parameter_1": gamma_shape,
            "parameter_1_navn": "shape",
            "parameter_2": gamma_scale,
            "parameter_2_navn": "scale",
            "log_likelihood": stats.gamma.logpdf(
                positive, gamma_shape, loc=0, scale=gamma_scale
            ).sum(),
        },
        {
            "fordeling": "Lognormal",
            "parameter_1": log_sigma,
            "parameter_1_navn": "sigma",
            "parameter_2": log_scale,
            "parameter_2_navn": "scale = exp(mu)",
            "log_likelihood": stats.lognorm.logpdf(
                positive, log_sigma, loc=0, scale=log_scale
            ).sum(),
        },
    ]
    result = pd.DataFrame(fits)
    result["aic"] = 2 * 2 - 2 * result["log_likelihood"]
    result["valgt_best_aic"] = result["aic"].eq(result["aic"].min())
    normality_stat, normality_pvalue = stats.normaltest(np.log(positive))
    result.attrs["normality_stat"] = float(normality_stat)
    result.attrs["normality_pvalue"] = float(normality_pvalue)
    result.attrs["zero_tolerance"] = zero_tolerance
    result.attrs["exact_zero_count"] = int(all_values.eq(0).sum())
    result.attrs["near_zero_count"] = int((diagnostic_zero & all_values.ne(0)).sum())
    result.attrs["diagnostic_zero_count"] = int(diagnostic_zero.sum())
    result.attrs["diagnostic_zero_share"] = float(diagnostic_zero.mean() * 100)
    return result.round(3)


def build_severity_distribution_fit(frame):
    """Fit-sammendrag for positiv, uvektet severity per skaderad."""
    with_claim = frame.loc[frame[CLAIMS_COL].gt(0)]
    severity = with_claim[INCURRED_COL] / with_claim[CLAIMS_COL]
    result = _fit_positive_distributions(severity)
    result.attrs["positive_count"] = int(severity.gt(CURRENCY_ZERO_TOLERANCE).sum())
    return result


def build_pure_premium_summary(frame):
    """Beskrivende statistikk for ren premie (incurred/eksponering) per poliseår."""
    pure_premium = frame[INCURRED_COL] / frame[EXPOSURE_COL]
    positive = pure_premium[pure_premium.gt(0)]
    return (
        pd.Series(
            {
                "andel_poliseår_uten_skadekostnad_prosent": pure_premium.eq(0).mean()
                * 100,
                "gjennomsnitt_alle": pure_premium.mean(),
                "gjennomsnitt_gitt_positiv": positive.mean(),
                "median_gitt_positiv": positive.median(),
                "p90_gitt_positiv": positive.quantile(0.9),
                "p99_gitt_positiv": positive.quantile(0.99),
                "maks": pure_premium.max(),
            },
            name="Verdi",
        )
        .to_frame()
        .round(2)
    )


def build_pure_premium_distribution_fit(frame):
    """Fit-sammendrag for den strengt positive delen av ren premie."""
    pure_premium = frame[INCURRED_COL] / frame[EXPOSURE_COL]
    result = _fit_positive_distributions(pure_premium)
    result.attrs["positive_count"] = int(pure_premium.gt(CURRENCY_ZERO_TOLERANCE).sum())
    return result


def build_year_trend_summary(frame):
    """Frekvens, severity og ren premie per kalenderår (temporal trend i train_pool)."""
    grouped = frame.groupby("year", observed=True)
    exposure = grouped[EXPOSURE_COL].sum()
    claims = grouped[CLAIMS_COL].sum()
    incurred = grouped[INCURRED_COL].sum()
    return (
        pd.DataFrame(
            {
                "eksponering": exposure,
                "skadeantall": claims,
                "frekvens": claims / exposure,
                "incurred": incurred,
                "severity": incurred / claims,
                "ren_premie": incurred / exposure,
            }
        )
        .reset_index()
        .round(4)
    )


# ---------------------------------------------------------------------------
# Tabeller: manglende verdier
# ---------------------------------------------------------------------------


def build_missingness_outcome_comparison(frame, columns=MISSING_CHECK_VARS):
    """Missing-andel per prediktor, og om frekvens/severity skiller seg for manglende vs. ikke."""
    rows = []
    for column in columns:
        is_missing = frame[column].isna()
        for label, mask in [("Ikke manglende", ~is_missing), ("Manglende", is_missing)]:
            subset = frame.loc[mask]
            exposure = subset[EXPOSURE_COL].sum()
            claims = subset[CLAIMS_COL].sum()
            incurred = subset[INCURRED_COL].sum()
            rows.append(
                {
                    "variabel": column,
                    "gruppe": label,
                    "poliseår": len(subset),
                    "andel_prosent": mask.mean() * 100,
                    "frekvens": claims / exposure if exposure else np.nan,
                    "severity": incurred / claims if claims else np.nan,
                }
            )
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------------------
# Tabeller: bivariat one-way (prediktor vs. frekvens/severity/ren premie)
# ---------------------------------------------------------------------------


def _one_way_stats(frame, group_labels):
    """Kjerneberegning: eksponering, frekvens, severity og ren premie per nivå."""
    working = frame.assign(nivå=pd.Categorical(group_labels))
    grouped = working.groupby("nivå", observed=True)
    exposure = grouped[EXPOSURE_COL].sum()
    claims = grouped[CLAIMS_COL].sum()
    incurred = grouped[INCURRED_COL].sum()
    result = pd.DataFrame(
        {
            "poliseår": grouped.size(),
            "eksponering": exposure,
            "andel_eksponering_prosent": exposure / exposure.sum() * 100,
            "skadeantall": claims,
            "frekvens": claims / exposure,
            "severity": incurred / claims.replace(0, np.nan),
            "ren_premie": incurred / exposure,
        }
    )
    return result.reset_index().round(4)


def build_categorical_one_way(frame, column):
    """Frekvens/severity/ren premie per kategori (eksponeringsvektet). NaN utelates —
    se `build_missingness_outcome_comparison` for missing-andel og utfall."""
    return _one_way_stats(frame, frame[column])


def build_numeric_one_way(frame, column, n_bins=6):
    """Samme som `build_categorical_one_way`, men for en numerisk variabel delt i
    ~like store (kvantilbaserte) intervaller."""
    bucket = pd.qcut(frame[column], q=n_bins, duplicates="drop")
    return _one_way_stats(frame, bucket)


def build_brand_distribution(frame, top_n=15):
    """Toppmerker etter eksponering, uten å tegne en kunstig samlet restgruppe."""
    exposure_by_brand = frame.groupby("vehicle_brand", observed=True)[
        EXPOSURE_COL
    ].sum()
    exposure_by_brand = exposure_by_brand.sort_values(ascending=False)
    top = exposure_by_brand.head(top_n)
    result = (
        pd.DataFrame(
            {
                "vehicle_brand": top.index.astype(str),
                "eksponering": top.values,
                "andel_eksponering_prosent": top / exposure_by_brand.sum() * 100,
            }
        )
        .sort_values("eksponering", ascending=True)
        .reset_index(drop=True)
    )
    result.attrs["top_n"] = min(top_n, len(exposure_by_brand))
    result.attrs["top_share"] = float(top.sum() / exposure_by_brand.sum() * 100)
    result.attrs["other_share"] = 100 - result.attrs["top_share"]
    return result


def build_brand_one_way(frame, column="vehicle_brand_pooled", low_exposure=100):
    """One-way for eksponeringspoolte merker med et transparent lavvolum-flagg."""
    result = _one_way_stats(frame, frame[column])
    result["lavt_volum"] = result["eksponering"].lt(low_exposure) | result[
        "skadeantall"
    ].lt(20)
    return result.sort_values("eksponering", ascending=True).reset_index(drop=True)


def build_bonus_lagged_panel(frame):
    """Knytt kun sammen t-1 og t når poliseårene er sammenhengende.

    Panelet er en timingdiagnose. Det bruker ikke 2024 når notebooken sender
    inn ``train_pool`` og gjør dermed ikke featurevalget testdrevet.
    """
    ordered = frame.sort_values(["insured_id", "year"]).copy()
    grouped = ordered.groupby("insured_id", observed=True)
    ordered["year_lag"] = grouped["year"].shift()
    for column in ["bonus_score", CLAIMS_COL, "total_claims"]:
        ordered[f"{column}_lag"] = grouped[column].shift()
    panel = ordered.loc[ordered["year"].sub(ordered["year_lag"]).eq(1)].copy()
    score_order = {"G": 0, "N": 1, "B": 2}
    panel["bonus_score_lag"] = panel["bonus_score_lag"].astype("string")
    panel["bonus_score"] = panel["bonus_score"].astype("string")
    panel["bonus_worsened_t"] = panel["bonus_score"].map(score_order) > panel[
        "bonus_score_lag"
    ].map(score_order)
    panel["bonus_improved_t"] = panel["bonus_score"].map(score_order) < panel[
        "bonus_score_lag"
    ].map(score_order)
    for prefix, column in [("property", CLAIMS_COL), ("total", "total_claims")]:
        panel[f"{prefix}_claim_lag_indicator"] = panel[f"{column}_lag"].gt(0)
        panel[f"{prefix}_claim_t_indicator"] = panel[column].gt(0)
    return panel


def build_bonus_transition_matrix(bonus_panel):
    """Overganger t-1 til t i den diagnostiske G < N < B-ordningen."""
    order = ["G", "N", "B"]
    return pd.crosstab(
        pd.Categorical(bonus_panel["bonus_score_lag"], categories=order),
        pd.Categorical(bonus_panel["bonus_score"], categories=order),
        dropna=False,
    )


def build_bonus_change_summary(bonus_panel):
    """Endringsrater betinget på lagget og samtidig skade, separat per dekning."""
    rows = []
    for claim_type, prefix in [("Egen skade", "property"), ("Alle skader", "total")]:
        for timing, indicator in [
            ("Skade i t-1", f"{prefix}_claim_lag_indicator"),
            ("Skade i t", f"{prefix}_claim_t_indicator"),
        ]:
            for label, mask in [
                ("Ingen skade", ~bonus_panel[indicator]),
                ("Minst én skade", bonus_panel[indicator]),
            ]:
                subset = bonus_panel.loc[mask]
                rows.append(
                    {
                        "skadedefinisjon": claim_type,
                        "tidspunkt": timing,
                        "gruppe": label,
                        "sammenhengende_par": len(subset),
                        "forverret_prosent": subset["bonus_worsened_t"].mean() * 100,
                        "forbedret_prosent": subset["bonus_improved_t"].mean() * 100,
                    }
                )
    return pd.DataFrame(rows).round(2)


def fit_bonus_timing_model(bonus_panel):
    """Logistisk timingdiagnose for forverring av bonusklasse.

    Den estimeres kun når designet har minst to nivåer av utfallet og uten
    konstante ledd. Et konstant kalenderår i train-pool rapporteres eksplisitt
    i stedet for å late som om årseffekten er estimert.
    """
    required = [
        "bonus_worsened_t",
        "bonus_score_lag",
        "property_claims_lag",
        "property_claims",
        "total_claims_lag",
        "total_claims",
    ]
    # B er dårligste klasse og kan per definisjon ikke forverres; den gir
    # perfekt separasjon og holdes utenfor risikosettet for dette utfallet.
    model_data = (
        bonus_panel.dropna(subset=required)
        .loc[lambda frame: frame["bonus_score_lag"].ne("B")]
        .copy()
    )
    model_data["bonus_worsened_t"] = model_data["bonus_worsened_t"].astype(int)
    formula_terms = [
        "C(bonus_score_lag, Treatment(reference='G'))",
        "property_claims_lag",
        "property_claims",
        "total_claims_lag",
        "total_claims",
    ]
    year_estimable = model_data["year"].nunique() > 1
    if year_estimable:
        formula_terms.append("C(year)")
    formula = "bonus_worsened_t ~ " + " + ".join(formula_terms)
    if model_data["bonus_worsened_t"].nunique() < 2:
        raise ValueError(
            "Bonusforverring har bare ett observert utfall; logistisk test kan ikke estimeres."
        )
    fitted = smf.logit(formula, data=model_data).fit(disp=False, maxiter=100)
    interval = fitted.conf_int()
    coefficients = pd.DataFrame(
        {
            "ledd": fitted.params.index,
            "odds_ratio": np.exp(fitted.params.values),
            "nedre_95_prosent": np.exp(interval.iloc[:, 0].values),
            "øvre_95_prosent": np.exp(interval.iloc[:, 1].values),
            "p_verdi": fitted.pvalues.values,
        }
    )
    diagnostics = pd.Series(
        {
            "sammenhengende_par_i_modell": len(model_data),
            "lagget_B_utelatt_strukturelt": int(
                bonus_panel["bonus_score_lag"].eq("B").sum()
            ),
            "bonusforverringer": int(model_data["bonus_worsened_t"].sum()),
            "forverring_prosent": model_data["bonus_worsened_t"].mean() * 100,
            "kalenderår_estimert": "Ja"
            if year_estimable
            else "Nei; train-pool har bare overgang til 2023",
            "konvergerte": bool(fitted.mle_retvals.get("converged", False)),
        },
        name="Verdi",
    ).to_frame()
    return coefficients.round(3), diagnostics, formula


def build_bonus_history_strata(bonus_panel):
    """Nåværende kaskofrekvens per bonusklasse innen enkel lagget skadehistorikk."""
    rows = []
    order = ["G", "N", "B"]
    for claim_type, column in [
        ("Lagget total skadehistorikk", "total_claims_lag"),
        ("Lagget egen-skadehistorikk", "property_claims_lag"),
    ]:
        working = bonus_panel.assign(
            historikk_stratum=np.where(
                bonus_panel[column].gt(0), "1+ skade", "0 skader"
            )
        )
        for history, score in [
            (history, score) for history in ["0 skader", "1+ skade"] for score in order
        ]:
            subset = working.loc[
                working["historikk_stratum"].eq(history)
                & working["bonus_score"].eq(score)
            ]
            exposure = subset[EXPOSURE_COL].sum()
            rows.append(
                {
                    "historikkdefinisjon": claim_type,
                    "historikk_stratum": history,
                    "bonus_score": score,
                    "poliseår": len(subset),
                    "eksponering": exposure,
                    "skadeantall_t": subset[CLAIMS_COL].sum(),
                    "frekvens_t": subset[CLAIMS_COL].sum() / exposure
                    if exposure
                    else np.nan,
                }
            )
    return pd.DataFrame(rows).round(4)


def build_age_diagnostics(frame):
    """Undersøk den observerte relasjonen mellom alder, førerkortfelt og vehicle_age.

    Funnet dokumenterer en arbeidshypotese, ikke en bekreftet definisjon fra
    kildeleverandøren: ``age_driving_licence`` ser ut som alder ved erverv og
    ``vehicle_age`` som førerkortansiennitet.
    """
    age_columns = ["driver_age", "age_driving_licence", "vehicle_age"]
    summary_rows = []
    for column in age_columns:
        values = frame[column]
        summary_rows.append(
            {
                "variabel": column,
                "manglende": int(values.isna().sum()),
                "manglende_prosent": values.isna().mean() * 100,
                "min": values.min(),
                "p01": values.quantile(0.01),
                "median": values.median(),
                "p99": values.quantile(0.99),
                "maks": values.max(),
            }
        )
    residual = frame["driver_age"] - frame["age_driving_licence"] - frame["vehicle_age"]
    residual_summary = pd.Series(
        {
            "sammenlignbare_rader": int(residual.notna().sum()),
            "residual_lik_null_prosent": residual.loc[residual.notna()].eq(0).mean()
            * 100,
            "residual_median": residual.median(),
            "residual_p01": residual.quantile(0.01),
            "residual_p99": residual.quantile(0.99),
            "residual_min": residual.min(),
            "residual_maks": residual.max(),
        },
        name="Verdi",
    ).to_frame()

    ordered = frame.sort_values(["insured_id", "year"])
    prior = ordered.groupby("insured_id", observed=True)
    year_gap = ordered["year"] - prior["year"].shift()
    temporal_rows = []
    for column in [
        "driver_age",
        "age_driving_licence",
        "vehicle_age",
        "driving_experience_years",
    ]:
        change = ordered[column] - prior[column].shift()
        comparable = year_gap.notna() & change.notna()
        temporal_rows.append(
            {
                "variabel": column,
                "sammenlignbare_overganger": int(comparable.sum()),
                "uendret_prosent": (change[comparable].eq(0).mean() * 100),
                "lik_årsgapp_prosent": (
                    change[comparable].eq(year_gap[comparable]).mean() * 100
                ),
                "fall_prosent": (change[comparable].lt(0).mean() * 100),
            }
        )
    plausibility = pd.Series(
        {
            "manglende_utledet_erfaring": int(
                frame["driving_experience_years"].isna().sum()
            ),
            "negativ_utledet_erfaring": int(
                frame["driving_experience_years"].lt(0).sum()
            ),
            "førerkortalder_under_14": int(frame["age_driving_licence"].lt(14).sum()),
            "førerkortalder_over_føreralder": int(
                frame["age_driving_licence"].gt(frame["driver_age"]).sum()
            ),
        },
        name="Verdi",
    ).to_frame()
    comparison = frame.loc[
        residual.notna(),
        ["driving_experience_years", "vehicle_age"],
    ].copy()
    comparison["residual"] = residual.loc[comparison.index]
    return {
        "summary": pd.DataFrame(summary_rows).round(2),
        "residual_summary": residual_summary.round(2),
        "temporal_changes": pd.DataFrame(temporal_rows).round(2),
        "plausibility": plausibility,
        "comparison": comparison,
    }


# ---------------------------------------------------------------------------
# Tabeller: samvariasjon blant prediktorer
# ---------------------------------------------------------------------------


def build_numeric_correlation(frame, columns=NUMERIC_VARS):
    """Pearson-korrelasjon (lineær samvariasjon) mellom numeriske prediktorer."""
    return frame[columns].corr(method="pearson").round(3)


def _cramers_v(x, y):
    """Cramér's V, bias-korrigert (Bergsma, 2013)."""
    contingency = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(contingency, correction=False)[0]
    n = contingency.to_numpy().sum()
    phi2 = max(
        0, chi2 / n - (contingency.shape[1] - 1) * (contingency.shape[0] - 1) / (n - 1)
    )
    r_corr = contingency.shape[0] - (contingency.shape[0] - 1) ** 2 / (n - 1)
    k_corr = contingency.shape[1] - (contingency.shape[1] - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return np.sqrt(phi2 / denom) if denom > 0 else np.nan


def build_categorical_association(frame, columns=CATEGORICAL_VARS):
    """Cramér's V (bias-korrigert) parvis mellom kategoriske prediktorer, 0-1."""
    matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for a in columns:
        for b in columns:
            matrix.loc[a, b] = 1.0 if a == b else _cramers_v(frame[a], frame[b])
    return matrix.round(3)


def _correlation_ratio(frame, cat_col, num_col, weight_col=EXPOSURE_COL):
    """Eksponeringsvektet korrelasjonsforhold eta = sqrt(SS_between / SS_total).

    Rader der kategori, numerisk verdi eller vekt er manglende droppes parvis
    (kun for dette paret). Returnerer (eta, antall_droppede_rader).
    """
    subset = frame[[cat_col, num_col, weight_col]].dropna().copy()
    dropped = len(frame) - len(subset)
    weight = subset[weight_col].astype(float)
    value = subset[num_col].astype(float)
    total_mean = np.average(value, weights=weight)
    ss_total = np.sum(weight * (value - total_mean) ** 2)

    subset["_weighted_value"] = weight * value
    group_weight = subset.groupby(cat_col, observed=True)[weight_col].sum()
    group_mean = (
        subset.groupby(cat_col, observed=True)["_weighted_value"].sum() / group_weight
    )
    ss_between = np.sum(group_weight * (group_mean - total_mean) ** 2)

    eta = np.sqrt(ss_between / ss_total) if ss_total > 0 else np.nan
    return float(eta), dropped


def build_categorical_numeric_association(
    frame, cat_columns=CATEGORICAL_VARS, num_columns=NUMERIC_VARS
):
    """Eksponeringsvektet korrelasjonsforhold (eta) mellom kategoriske og numeriske
    prediktorer, 0-1. Kategoriske variabler som rader, numeriske som kolonner.

    Inneholder en innebygd sanity check: for det første kategoriske/numeriske
    paret verifiseres eta² mot R² fra en vektet OLS (WLS) av den numeriske
    variabelen på kategori-dummyer — de to skal være matematisk identiske.
    """
    matrix = pd.DataFrame(index=cat_columns, columns=num_columns, dtype=float)
    for cat in cat_columns:
        for num in num_columns:
            eta, dropped = _correlation_ratio(frame, cat, num)
            matrix.loc[cat, num] = eta
            if dropped:
                print(f"{cat} × {num}: droppet {dropped} rader (manglende verdier)")

    # Sanity check (kjører alltid): eta² for ett stabilt par skal matche WLS R².
    check_cat, check_num = cat_columns[0], num_columns[0]
    eta_check, _ = _correlation_ratio(frame, check_cat, check_num)
    check_data = frame[[check_cat, check_num, EXPOSURE_COL]].dropna()
    wls_fit = smf.wls(
        f"{check_num} ~ C({check_cat})",
        data=check_data,
        weights=check_data[EXPOSURE_COL],
    ).fit()
    assert np.isclose(eta_check**2, wls_fit.rsquared, rtol=1e-6), (
        f"eta² ({eta_check**2:.6f}) matcher ikke WLS R² ({wls_fit.rsquared:.6f}) "
        f"for {check_cat} × {check_num}."
    )
    return matrix.round(3)


def report_high_association_pairs(association_matrix, threshold=0.3):
    """Kategorisk/numerisk-par med eta over `threshold`, sortert synkende."""
    pairs = association_matrix.stack().rename("eta").reset_index()
    pairs.columns = ["kategorisk", "numerisk", "eta"]
    return (
        pairs.loc[pairs["eta"].gt(threshold)]
        .sort_values("eta", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Plot: felles stilhjelpere
# ---------------------------------------------------------------------------


def _style_axes(ax, title, ylabel, xlabel=""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left", pad=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)


def _heatmap(matrix, cmap, vmin, vmax, title, cbar_label):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=SURFACE)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=2,
        linecolor=SURFACE,
        cbar_kws={"label": cbar_label},
        annot_kws={"fontsize": 8, "color": INK_PRIMARY},
        ax=ax,
    )
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Plot: utfallsvariablene selv
# ---------------------------------------------------------------------------


def plot_claim_count_distribution(distribution):
    """Stolpediagram for alle skadeantall; log-skala synliggjør sjeldne grupper."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    ax.bar(
        distribution["skadeantall"],
        distribution["poliseår"],
        color=ACCENT,
        width=0.6,
    )
    ax.set_yscale("log")
    ax.set_xticks(distribution["skadeantall"])
    _style_axes(
        ax,
        "Fordeling av skadeantall per poliseår",
        "Antall poliseår (log-skala)",
        "Skadeantall",
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


def _fitted_density(x, fit_table):
    """Tetthet fra kandidaten med lavest AIC, på skalaen til x."""
    selected = fit_table.loc[fit_table["valgt_best_aic"]].iloc[0]
    if selected["fordeling"] == "Gamma":
        density = stats.gamma.pdf(
            x, selected["parameter_1"], loc=0, scale=selected["parameter_2"]
        )
    else:
        density = stats.lognorm.pdf(
            x, selected["parameter_1"], loc=0, scale=selected["parameter_2"]
        )
    return density, selected["fordeling"]


def _plot_positive_distribution_panels(positive, fit_table, title, absolute_label):
    """Absolutt, log og Q-Q-panel for en positiv, kontinuerlig del."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1), facecolor=SURFACE)
    absolute_limit = positive.quantile(0.99)
    visible = positive.loc[positive.le(absolute_limit)]
    outside = int(positive.gt(absolute_limit).sum())
    bins = np.linspace(0, absolute_limit, 35)
    axes[0].hist(visible, bins=bins, density=True, color=CONTEXT, edgecolor=SURFACE)
    x = np.linspace(positive.min(), absolute_limit, 300)
    density, selected_name = _fitted_density(x, fit_table)
    axes[0].plot(
        x, density, color="#b23a2f", linewidth=2, label=f"{selected_name}, best AIC"
    )
    axes[0].legend(frameon=False, fontsize=8)
    _style_axes(axes[0], f"{title}: absolutt skala", "Tetthet", absolute_label)
    axes[0].text(
        0.98,
        0.95,
        f"Viser t.o.m. p99; {outside:,} positive observasjoner over grensen\nble brukt i fit og vises i log-panelet.",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=INK_MUTED,
    )

    log_bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 45)
    axes[1].hist(positive, bins=log_bins, color=ACCENT, edgecolor=SURFACE)
    axes[1].set_xscale("log")
    axes[1].set_xlim(positive.min(), positive.max())
    _style_axes(
        axes[1],
        f"{title}: hele positive hale",
        "Antall poliseår",
        f"{absolute_label}, log-skala",
    )

    stats.probplot(np.log(positive), dist="norm", plot=axes[2])
    for line in axes[2].get_lines():
        line.set_color(ACCENT)
    _style_axes(
        axes[2],
        f"Q-Q: log({absolute_label})",
        "Observerte kvantiler",
        "Normale teoretiske kvantiler",
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_severity_distribution(frame, fit_table):
    """Tre deskriptive paneler for positiv severity; nuller rapporteres eksplisitt."""
    with_claim = frame.loc[frame[CLAIMS_COL].gt(0)]
    severity = with_claim[INCURRED_COL] / with_claim[CLAIMS_COL]
    positive = severity.loc[severity.gt(fit_table.attrs["zero_tolerance"])]
    fig = _plot_positive_distribution_panels(
        positive, fit_table, "Severity per skaderad", "Severity"
    )
    fig.text(
        0.01,
        0.005,
        f"{fit_table.attrs['diagnostic_zero_count']:,} av {len(severity):,} skaderader ({fit_table.attrs['diagnostic_zero_share']:.1f} %) er 0 eller ≤ {fit_table.attrs['zero_tolerance']:.2f} og inngår ikke i kontinuerlig fit.",
        color=INK_MUTED,
        fontsize=8,
    )
    return fig


def plot_pure_premium_distribution(frame, fit_table):
    """Fire paneler: nullmasse samt absolutt/log/Q-Q for positiv ren premie."""
    pure_premium = frame[INCURRED_COL] / frame[EXPOSURE_COL]
    positive = pure_premium.loc[pure_premium.gt(fit_table.attrs["zero_tolerance"])]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.1), facecolor=SURFACE)
    zero_share = fit_table.attrs["diagnostic_zero_share"]
    axes[0].bar(["0", "> 0"], [zero_share, 100 - zero_share], color=[INK_MUTED, ACCENT])
    _style_axes(axes[0], "Ren premie: nullmasse", "Andel poliseår (%)")
    axes[0].text(0, zero_share + 1, f"{zero_share:.1f}%", ha="center", fontsize=8)

    absolute_limit = positive.quantile(0.99)
    visible = positive.loc[positive.le(absolute_limit)]
    outside = int(positive.gt(absolute_limit).sum())
    axes[1].hist(
        visible,
        bins=np.linspace(0, absolute_limit, 35),
        density=True,
        color=CONTEXT,
        edgecolor=SURFACE,
    )
    x = np.linspace(positive.min(), absolute_limit, 300)
    density, selected_name = _fitted_density(x, fit_table)
    axes[1].plot(
        x, density, color="#b23a2f", linewidth=2, label=f"{selected_name}, best AIC"
    )
    axes[1].legend(frameon=False, fontsize=8)
    _style_axes(axes[1], "Gitt positiv ren premie: absolutt", "Tetthet", "Ren premie")
    axes[1].text(
        0.98,
        0.95,
        f"t.o.m. p99; {outside:,} over grensen",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=INK_MUTED,
    )

    log_bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 45)
    axes[2].hist(positive, bins=log_bins, color=ACCENT, edgecolor=SURFACE)
    axes[2].set_xscale("log")
    axes[2].set_xlim(positive.min(), positive.max())
    _style_axes(
        axes[2],
        "Gitt positiv ren premie: hele hale",
        "Antall poliseår",
        "Ren premie, log-skala",
    )

    stats.probplot(np.log(positive), dist="norm", plot=axes[3])
    for line in axes[3].get_lines():
        line.set_color(ACCENT)
    _style_axes(
        axes[3],
        "Q-Q: log(positiv ren premie)",
        "Observerte kvantiler",
        "Normale teoretiske kvantiler",
    )
    fig.text(
        0.01,
        0.005,
        f"{fit_table.attrs['diagnostic_zero_count']:,} null-/nærnullverdier (≤ {fit_table.attrs['zero_tolerance']:.2f}) er kun i nullmassepanelet; histogram og fit er betinget på positiv ren premie.",
        color=INK_MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    plt.close(fig)
    return fig


def plot_year_trend(year_trend):
    """Frekvens og severity per kalenderår, side om side."""
    fig, (ax_freq, ax_sev) = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    for ax, column, title, ylabel in [
        (ax_freq, "frekvens", "Frekvens per år", "Skader per poliseår"),
        (ax_sev, "severity", "Severity per år", "Snitt severity"),
    ]:
        ax.bar(
            year_trend["year"].astype(str), year_trend[column], color=ACCENT, width=0.5
        )
        _style_axes(ax, title, ylabel)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_age_diagnostics(age_diagnostics, max_points=12000):
    """Figur for å teste den observerte aldersidentiteten og tidsmønsteret."""
    comparison = age_diagnostics["comparison"]
    if len(comparison) > max_points:
        comparison = comparison.sample(max_points, random_state=100)
    temporal = age_diagnostics["temporal_changes"].set_index("variabel")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)
    ax_scatter, ax_residual, ax_temporal = axes
    ax_scatter.scatter(
        comparison["driving_experience_years"],
        comparison["vehicle_age"],
        s=5,
        alpha=0.12,
        color=ACCENT,
        rasterized=True,
    )
    limit = float(
        max(
            comparison["driving_experience_years"].max(),
            comparison["vehicle_age"].max(),
        )
    )
    ax_scatter.plot([0, limit], [0, limit], color="#b23a2f", linestyle="--")
    _style_axes(
        ax_scatter,
        "Vehicle_age mot utledet erfaring",
        "vehicle_age",
        "driving_experience_years",
    )

    residual = comparison["residual"]
    bins = np.arange(residual.min() - 0.5, residual.max() + 1.5, 1)
    ax_residual.hist(residual, bins=bins, color=ACCENT, edgecolor=SURFACE)
    ax_residual.axvline(0, color="#b23a2f", linestyle="--")
    _style_axes(
        ax_residual,
        "Residual: alder − førerkortalder − vehicle_age",
        "Poliseår",
        "Residual (år)",
    )

    columns = ["uendret_prosent", "lik_årsgapp_prosent", "fall_prosent"]
    temporal.loc[:, columns].plot.barh(
        ax=ax_temporal, color=[CONTEXT, ACCENT, "#b23a2f"]
    )
    _style_axes(
        ax_temporal,
        "Longitudinelle endringer per polise",
        "",
        "Andel overganger (%)",
    )
    ax_temporal.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_bonus_transition_matrix(transition_matrix):
    """Heatmap for bonusklasse t-1 til t."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=SURFACE)
    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt=".0f",
        cmap=SEQUENTIAL_BLUE,
        linewidths=1,
        linecolor=SURFACE,
        cbar_kws={"label": "Sammenhengende polisepar"},
        ax=ax,
    )
    ax.set_xlabel("Bonusklasse i t", color=INK_SECONDARY)
    ax.set_ylabel("Bonusklasse i t-1", color=INK_SECONDARY)
    ax.set_title("Bonusoverganger: G < N < B", loc="left", color=INK_PRIMARY, pad=12)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_bonus_change_summary(change_summary):
    """Kompakt sammenligning av forbedring/forverring mot skade i t-1 og t."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor=SURFACE, sharey=True)
    for row, claim_type in enumerate(["Egen skade", "Alle skader"]):
        for col, outcome in enumerate(["forverret_prosent", "forbedret_prosent"]):
            ax = axes[row, col]
            subset = change_summary.loc[
                change_summary["skadedefinisjon"].eq(claim_type)
            ].copy()
            labels = subset["tidspunkt"] + "\n" + subset["gruppe"]
            colors = np.where(subset["gruppe"].eq("Minst én skade"), ACCENT, CONTEXT)
            ax.bar(np.arange(len(subset)), subset[outcome], color=colors)
            ax.set_xticks(np.arange(len(subset)), labels, rotation=25, ha="right")
            outcome_label = (
                "Forverret" if outcome == "forverret_prosent" else "Forbedret"
            )
            _style_axes(ax, f"{claim_type}: {outcome_label.lower()}", "Andel par (%)")
    fig.suptitle(
        "Bonusendring betinget på skade i t-1 og t", x=0.01, ha="left", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.close(fig)
    return fig


def plot_bonus_history_strata(history_strata):
    """Frekvens per nåværende bonus innen lagget skadehistorikk."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=SURFACE, sharey=True)
    definitions = history_strata["historikkdefinisjon"].drop_duplicates().tolist()
    colors = {"G": CONTEXT, "N": ACCENT, "B": "#b23a2f"}
    for ax, definition in zip(axes, definitions):
        subset = history_strata.loc[
            history_strata["historikkdefinisjon"].eq(definition)
        ]
        positions = np.arange(2)
        width = 0.22
        for index, score in enumerate(["G", "N", "B"]):
            values = subset.loc[subset["bonus_score"].eq(score)].set_index(
                "historikk_stratum"
            )
            ordered = values.reindex(["0 skader", "1+ skade"])
            ax.bar(
                positions + (index - 1) * width,
                ordered["frekvens_t"],
                width=width,
                color=colors[score],
                label=score,
            )
            for x, y, exposure in zip(
                positions + (index - 1) * width,
                ordered["frekvens_t"],
                ordered["eksponering"],
            ):
                if exposure < 100:
                    ax.text(x, y, "*", ha="center", va="bottom", color="#b23a2f")
        ax.set_xticks(positions, ["0 skader", "1+ skade"])
        _style_axes(ax, definition, "Kaskofrekvens i t", "Lagget skadehistorikk")
        ax.legend(title="Bonus i t", frameon=False, fontsize=8)
    fig.text(
        0.01,
        0.005,
        "* under 100 eksponeringsår; tolk med varsomhet.",
        color=INK_MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Plot: univariate prediktorfordelinger
# ---------------------------------------------------------------------------


def plot_numeric_predictor_histograms(frame, columns=NUMERIC_VARS, n_cols=3):
    """Histogrammer for numeriske prediktorer."""
    n_rows = -(-len(columns) // n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows), facecolor=SURFACE
    )
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, column in zip(axes, columns):
        ax.hist(frame[column].dropna(), bins=30, color=ACCENT, edgecolor=SURFACE)
        _style_axes(ax, column, "Poliseår")
    for ax in axes[len(columns) :]:
        ax.set_visible(False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_categorical_predictor_bars(frame, columns=CATEGORICAL_VARS, n_cols=2):
    """Eksponeringsvektet andel per kategori, for kategoriske prediktorer."""
    n_rows = -(-len(columns) // n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 3.2 * n_rows), facecolor=SURFACE
    )
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, column in zip(axes, columns):
        share = frame.groupby(column, observed=True)[EXPOSURE_COL].sum()
        share = share / share.sum() * 100
        ax.bar(share.index.astype(str), share.values, color=ACCENT, width=0.6)
        _style_axes(ax, column, "Andel av eksponering (%)")
    for ax in axes[len(columns) :]:
        ax.set_visible(False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_brand_distribution(brand_distribution):
    """Horisontal toppmerke-figur uten å la restgruppen dominere bildet."""
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=SURFACE)
    ax.barh(
        brand_distribution["vehicle_brand"],
        brand_distribution["andel_eksponering_prosent"],
        color=ACCENT,
    )
    for y, share in enumerate(brand_distribution["andel_eksponering_prosent"]):
        ax.text(
            share + 0.05,
            y,
            f"{share:.1f}%",
            va="center",
            fontsize=8,
            color=INK_SECONDARY,
        )
    top_n = brand_distribution.attrs.get("top_n", len(brand_distribution))
    top_share = brand_distribution.attrs.get("top_share", np.nan)
    other_share = brand_distribution.attrs.get("other_share", np.nan)
    _style_axes(
        ax,
        f"Topp {top_n} bilmerker etter eksponeringsandel",
        "",
        "Andel av eksponering (%)",
    )
    ax.text(
        0,
        -0.16,
        f"Topp {top_n}: {top_share:.1f}% av eksponeringen. Øvrige merker: {other_share:.1f}% (ikke tegnet som én stolpe).",
        transform=ax.transAxes,
        fontsize=8,
        color=INK_MUTED,
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_missingness_summary(missingness_table):
    """Andel manglende verdier per prediktor."""
    missing = missingness_table.loc[missingness_table["gruppe"].eq("Manglende")]
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=SURFACE)
    ax.bar(missing["variabel"], missing["andel_prosent"], color=ACCENT, width=0.5)
    _style_axes(ax, "Andel manglende verdier per prediktor", "Manglende (%)")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Plot: bivariat one-way og samvariasjon
# ---------------------------------------------------------------------------


def plot_one_way(
    one_way_table,
    title,
    rate_column="frekvens",
    rate_label="Frekvens (skader per poliseår)",
):
    """Generisk one-way-plot: eksponering (stolpe) og en rate (linje) per nivå.

    `rate_column` kan være "frekvens", "severity" eller "ren_premie" —
    samme tabellformat fra `_one_way_stats` brukes til alle tre.
    """
    fig, ax_exposure = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    x = one_way_table["nivå"].astype(str)
    ax_exposure.bar(x, one_way_table["eksponering"], color=CONTEXT, width=0.6)
    ax_exposure.set_facecolor(SURFACE)
    ax_exposure.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    ax_exposure.set_ylabel("Eksponering (poliseår)", color=INK_SECONDARY, fontsize=9)
    ax_exposure.tick_params(colors=INK_MUTED, labelsize=8)
    ax_exposure.spines[["top", "right", "left"]].set_visible(False)
    ax_exposure.spines["bottom"].set_color(AXIS)

    ax_rate = ax_exposure.twinx()
    ax_rate.plot(x, one_way_table[rate_column], color=ACCENT, marker="o", linewidth=2)
    ax_rate.set_ylabel(rate_label, color=ACCENT, fontsize=9)
    ax_rate.tick_params(axis="y", colors=ACCENT, labelsize=8)
    ax_rate.spines[["top", "left"]].set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return fig


def _plot_one_way_rate(ax, table, column, label, portfolio_rate, horizontal=False):
    """Tegn punktestimat og en enkel Poisson-usikkerhet for frekvens."""
    if horizontal:
        position = np.arange(len(table))
        ax.scatter(table[column], position, color=ACCENT, s=28, zorder=3)
        if column == "frekvens":
            error = 1.96 * np.sqrt(table["skadeantall"]) / table["eksponering"]
            ax.errorbar(
                table[column],
                position,
                xerr=error,
                fmt="none",
                color=ACCENT,
                alpha=0.55,
            )
        ax.axvline(portfolio_rate, color=INK_MUTED, linestyle="--", linewidth=1)
        ax.set_yticks(position, table["nivå"].astype(str))
        _style_axes(ax, label, "", label)
    else:
        position = np.arange(len(table))
        ax.scatter(position, table[column], color=ACCENT, s=24, zorder=3)
        if column == "frekvens":
            error = 1.96 * np.sqrt(table["skadeantall"]) / table["eksponering"]
            ax.errorbar(
                position,
                table[column],
                yerr=error,
                fmt="none",
                color=ACCENT,
                alpha=0.55,
            )
        ax.axhline(portfolio_rate, color=INK_MUTED, linestyle="--", linewidth=1)
        ax.set_xticks(position, table["nivå"].astype(str), rotation=35, ha="right")
        _style_axes(ax, label, label)


def plot_one_way_grid(one_way_tables, frame, kind):
    """Kompakt dashboard med eksponering og tre eksponeringsvektede rater.

    Frekvenspunktene har omtrentlige 95 %-Poissonintervaller. Severity og ren
    premie markeres med punkter, fordi aggregerte poliseår ikke gir et
    skadeindivid-basert intervall uten flere antakelser.
    """
    variables = list(one_way_tables)
    fig, axes = plt.subplots(
        len(variables), 4, figsize=(17, 3.35 * len(variables)), facecolor=SURFACE
    )
    axes = np.atleast_2d(axes)
    portfolio_rates = {
        "frekvens": frame[CLAIMS_COL].sum() / frame[EXPOSURE_COL].sum(),
        "severity": frame[INCURRED_COL].sum() / frame[CLAIMS_COL].sum(),
        "ren_premie": frame[INCURRED_COL].sum() / frame[EXPOSURE_COL].sum(),
    }
    labels = {
        "frekvens": "Frekvens",
        "severity": "Severity",
        "ren_premie": "Ren premie",
    }
    for row, variable in enumerate(variables):
        table = one_way_tables[variable]
        positions = np.arange(len(table))
        ax_exposure = axes[row, 0]
        ax_exposure.bar(positions, table["andel_eksponering_prosent"], color=CONTEXT)
        ax_exposure.set_xticks(
            positions, table["nivå"].astype(str), rotation=35, ha="right"
        )
        _style_axes(ax_exposure, f"{variable}: eksponering", "Andel (%)")
        for col, rate in enumerate(["frekvens", "severity", "ren_premie"], start=1):
            _plot_one_way_rate(
                axes[row, col], table, rate, labels[rate], portfolio_rates[rate]
            )
    fig.suptitle(
        f"One-way-dashboard: {kind} prediktorer (stiplet linje = porteføljereferanse)",
        x=0.01,
        ha="left",
        color=INK_PRIMARY,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    plt.close(fig)
    return fig


def plot_brand_one_way(brand_one_way, frame):
    """Lesbar merke-one-way med felles rekkefølge og lavvolum-markering.

    Åpne punkter markerer nivåer med under 100 eksponeringsår eller under 20
    skader. Frekvens har omtrentlige 95 %-Poissonintervaller; øvrige ratepanel
    viser punktestimat og må tolkes særlig varsomt for åpne punkter.
    """
    table = brand_one_way.copy()
    fig, axes = plt.subplots(
        1, 4, figsize=(18, max(5.5, len(table) * 0.34)), facecolor=SURFACE
    )
    y = np.arange(len(table))
    axes[0].barh(y, table["andel_eksponering_prosent"], color=CONTEXT)
    axes[0].set_yticks(y, table["nivå"].astype(str))
    _style_axes(axes[0], "Eksponeringsandel", "", "Andel (%)")

    portfolio_rates = {
        "frekvens": frame[CLAIMS_COL].sum() / frame[EXPOSURE_COL].sum(),
        "severity": frame[INCURRED_COL].sum() / frame[CLAIMS_COL].sum(),
        "ren_premie": frame[INCURRED_COL].sum() / frame[EXPOSURE_COL].sum(),
    }
    for ax, rate, title in zip(
        axes[1:],
        ["frekvens", "severity", "ren_premie"],
        ["Frekvens", "Severity", "Ren premie"],
    ):
        solid = ~table["lavt_volum"]
        ax.scatter(table.loc[solid, rate], y[solid], color=ACCENT, s=28, zorder=3)
        ax.scatter(
            table.loc[~solid, rate],
            y[~solid],
            facecolors="none",
            edgecolors="#b23a2f",
            s=35,
            zorder=3,
        )
        if rate == "frekvens":
            error = 1.96 * np.sqrt(table["skadeantall"]) / table["eksponering"]
            ax.errorbar(
                table[rate], y, xerr=error, fmt="none", color=ACCENT, alpha=0.55
            )
        ax.axvline(portfolio_rates[rate], color=INK_MUTED, linestyle="--", linewidth=1)
        ax.set_yticks(y, table["nivå"].astype(str))
        _style_axes(ax, title, "", title)
    fig.suptitle(
        "Bilmerke: eksponeringspoolte nivåer (stiplet = portefølje)",
        x=0.01,
        ha="left",
        fontsize=13,
    )
    fig.text(
        0.01,
        0.005,
        "Åpne røde punkter: lavt volum; tolk rateestimatene varsomt.",
        color=INK_MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    plt.close(fig)
    return fig


def plot_numeric_correlation_heatmap(corr_matrix):
    """Pearson-korrelasjon mellom numeriske prediktorer."""
    return _heatmap(
        corr_matrix,
        DIVERGING,
        -1,
        1,
        "Korrelasjon mellom numeriske prediktorer",
        "Pearson-korrelasjon",
    )


def plot_categorical_association_heatmap(cramers_v_matrix):
    """Cramér's V mellom kategoriske prediktorer."""
    return _heatmap(
        cramers_v_matrix,
        SEQUENTIAL_BLUE,
        0,
        1,
        "Assosiasjon mellom kategoriske prediktorer (Cramér's V)",
        "Cramér's V",
    )


def plot_categorical_numeric_association_heatmap(correlation_ratio_matrix):
    """Eksponeringsvektet korrelasjonsforhold (eta) mellom kategoriske og numeriske prediktorer."""
    return _heatmap(
        correlation_ratio_matrix,
        SEQUENTIAL_BLUE,
        0,
        1,
        "Assosiasjon mellom kategoriske og numeriske prediktorer (η)",
        "η",
    )


def plot_numeric_by_category_boxplot(frame, numeric_column, category_column):
    """Boxplot av en numerisk prediktor fordelt på en kategorisk prediktor (confounding-sjekk)."""
    categories = sorted(frame[category_column].dropna().astype(str).unique())
    data = [
        frame.loc[frame[category_column].astype(str).eq(c), numeric_column].dropna()
        for c in categories
    ]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    boxes = ax.boxplot(
        data,
        tick_labels=categories,
        patch_artist=True,
        medianprops={"color": INK_PRIMARY},
    )
    for patch in boxes["boxes"]:
        patch.set_facecolor(CONTEXT)
        patch.set_edgecolor(AXIS)
    _style_axes(ax, f"{numeric_column} per {category_column}", numeric_column)
    fig.tight_layout()
    plt.close(fig)
    return fig
