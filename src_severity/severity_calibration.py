"""EUR-kalibrering og stabilitet ETTER seleksjon (S0 mot finalisten C1).

Se ``plans/Severity_plan.md``, seksjonen "Etter seleksjon: EUR-kalibrering og
stabilitet". Alle funksjoner er rene: de tar imot ferdige ``severity_frame``
og OOF-prediksjoner (eller CV-resultater) som argumenter, laster ingen data
selv og åpner ingen nye kandidater/terskler. Ingen funksjon her leser 2022-2023
utviklingsdata på egen hånd — de får alt fra ``src_severity.severity_data`` og
``src_severity.severity_cv`` via kall i notebooken.

``predictions`` er gjennomgående en dict modellnavn -> ``pandas.Series`` med
OOF-prediksjoner, indeksert som ``severity_frame`` (typisk
``{"S0": severity_results["S0"]["oof"], "C1": severity_results["C1"]["oof"]}``).

Gamma-enhetsdeviansen gjenbruker ``severity_scoring.GAMMA_FAMILY`` (samme
statsmodels-familieobjekt som den parvise gevinstestimatoren bruker), slik at
deviansformelen aldri implementeres to ganger i severity-fasen.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.own_damage_descriptives import (
    ACCENT,
    AXIS,
    CONTEXT,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
)
from src_severity.severity_scoring import GAMMA_FAMILY

N_DECILES = 10  # prediksjonsdesiler, låst i planen
# Faste beløpsklasser på average_severity (planens stoppregel for små beløp).
_SEVERITY_BAND_EDGES = [0, 1, 10, 100, 1000, 5000, np.inf]
_SEVERITY_BAND_LABELS = [
    "≤1",
    "(1, 10]",
    "(10, 100]",
    "(100, 1000]",
    "(1000, 5000]",
    ">5000",
]
_LINE_COLORS = [ACCENT, INK_PRIMARY, INK_SECONDARY]  # sykles per modell i plottet


def _seat_category(seats):
    """Setekategori `<5` / `=5` / `>5` (samme faste inndeling som i notebooken)."""
    return pd.Series(
        np.select([seats.lt(5), seats.eq(5)], ["<5", "=5"], default=">5"),
        index=seats.index,
    )


def _claim_count_group(claims):
    """Skadeantallsgruppe N=1 / N=2-3 / N>=4 (rekkefølgen i np.select er avgjørende)."""
    return pd.Series(
        np.select([claims.eq(1), claims.le(3)], ["N=1", "N=2-3"], default="N>=4"),
        index=claims.index,
    )


def _style_axes(ax, title, ylabel, xlabel=""):
    """Samme aksestil som resten av severity-diagnostikken (severity_descriptives)."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left", pad=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)


def build_ae_table(severity_frame, predictions, segments):
    """A/E = faktisk kostnad / sum(N x predikert severity), per segment og modell.

    ``segments`` er en liste kolonnenavn i ``severity_frame`` (f.eks.
    ``["policy_type", "year", "municipality_type", "circulation_area"]``) som
    brytes ut i tillegg til de faste segmentene planen krever: en «Totalt»-rad,
    setekategori (avledet fra ``seats``) og skadeantallsgruppe (avledet fra
    ``property_claims``). Summen av ``forventet_kostnad`` over nivåene i én
    segmentvariabel er alltid lik totalen, siden hver rad havner i nøyaktig
    ett nivå per variabel.
    """
    variable_groups = {"Totalt": pd.Series("Totalt", index=severity_frame.index)}
    for column in segments:
        variable_groups[column] = severity_frame[column]
    variable_groups["seat_category"] = _seat_category(severity_frame["seats"])
    variable_groups["skadeantallsgruppe"] = _claim_count_group(
        severity_frame["property_claims"]
    )

    rows = []
    for variable, group_values in variable_groups.items():
        for level, sub in severity_frame.groupby(group_values):
            actual_cost = sub["property_incurred"].sum()
            for model_name, prediction in predictions.items():
                expected_cost = (
                    sub["property_claims"] * prediction.loc[sub.index]
                ).sum()
                rows.append(
                    {
                        "variabel": variable,
                        "nivå": level,
                        "modell": model_name,
                        "antall_skadeår": len(sub),
                        "unike_personer": sub["insured_id"].nunique(),
                        "skadeantall": int(sub["property_claims"].sum()),
                        "faktisk_kostnad": actual_cost,
                        "forventet_kostnad": expected_cost,
                        "A_E": actual_cost / expected_cost,
                    }
                )
    return pd.DataFrame(rows)


def build_decile_table(severity_frame, predictions, reference_model="S0"):
    """Prediksjonsdesiler fra ``reference_model``s OOF-prediksjon, skadeantallsvektet.

    Radene sorteres etter referansemodellens prediksjon, ``property_claims``
    kumuleres, og grensene settes slik at hver av de ti desilene får omtrent
    like mange skader (ikke like mange poliseår). Samme gruppetilhørighet
    brukes for alle modeller i ``predictions`` — det er hele poenget: en
    forskjell mellom modellene i samme desil kan da ikke skyldes at desilene
    er ulikt definert.
    """
    reference_prediction = predictions[reference_model].dropna()
    order = reference_prediction.sort_values().index
    claims = severity_frame.loc[order, "property_claims"]
    cumulative_share = claims.cumsum() / claims.sum()
    decile = pd.Series(
        np.ceil(cumulative_share.to_numpy() * N_DECILES)
        .clip(max=N_DECILES)
        .astype(int),
        index=order,
    )

    rows = []
    for decile_number, sub in severity_frame.loc[order].groupby(decile):
        weight = sub["property_claims"]
        for model_name, prediction in predictions.items():
            predicted = np.average(prediction.loc[sub.index], weights=weight)
            actual = np.average(sub["average_severity"], weights=weight)
            rows.append(
                {
                    "desil": int(decile_number),
                    "antall_skadeår": len(sub),
                    "unike_personer": sub["insured_id"].nunique(),
                    "skadeantall": int(weight.sum()),
                    "modell": model_name,
                    "gjennomsnitt_predikert": predicted,
                    "faktisk_severity": actual,
                    "A_E": actual / predicted,
                }
            )
    return pd.DataFrame(rows).sort_values(["desil", "modell"]).reset_index(drop=True)


def build_contribution_table(severity_frame, predictions):
    """Bidrag til Gamma-deviance og EUR-feil fra faste beløpsklasser på ``average_severity``.

    Beløpsklassen bygger bare på observert severity, så den er identisk for
    alle modeller — det som varierer per modell er prediksjonen, og dermed
    deviance-/EUR-bidraget. ``deviance_bidrag`` er
    ``sum(property_claims * enhetsdevians)`` i klassen; ``deviance_andel`` og
    ``kostnadsandel`` er andeler av modellens/utvalgets totalsum, så de
    summerer til 1 per modell.
    """
    band = pd.cut(
        severity_frame["average_severity"],
        bins=_SEVERITY_BAND_EDGES,
        labels=_SEVERITY_BAND_LABELS,
        include_lowest=True,
    )
    total_claims = severity_frame["property_claims"].sum()
    total_cost = severity_frame["property_incurred"].sum()

    rows = []
    for model_name, prediction in predictions.items():
        unit_deviance = (
            GAMMA_FAMILY.resid_dev(severity_frame["average_severity"], prediction) ** 2
        )
        deviance_contrib = severity_frame["property_claims"] * unit_deviance
        expected_cost = severity_frame["property_claims"] * prediction
        eur_error = severity_frame["property_incurred"] - expected_cost
        total_deviance = deviance_contrib.sum()
        total_eur_error = eur_error.sum()

        grouped = pd.DataFrame(
            {
                "claims": severity_frame["property_claims"],
                "actual_cost": severity_frame["property_incurred"],
                "deviance": deviance_contrib,
                "eur_error": eur_error,
            }
        ).groupby(band, observed=True)

        for level, sub in grouped:
            rows.append(
                {
                    "beløpsklasse": level,
                    "modell": model_name,
                    "antall_skadeår": len(sub),
                    "skadeantall": int(sub["claims"].sum()),
                    "andel_skadeantall": sub["claims"].sum() / total_claims,
                    "kostnadsandel": sub["actual_cost"].sum() / total_cost,
                    "deviance_bidrag": sub["deviance"].sum(),
                    "deviance_andel": sub["deviance"].sum() / total_deviance,
                    "eur_feil": sub["eur_error"].sum(),
                    "eur_feil_andel": sub["eur_error"].sum() / total_eur_error,
                }
            )
    return pd.DataFrame(rows)


def build_numeric_correlation_table(severity_frame, columns):
    """Skadeantallsvektet korrelasjonsmatrise mellom numeriske severity-prediktorer.

    Vektet med ``property_claims`` — samme vekt severity-modellen selv bruker
    — via ``numpy.cov(..., aweights=...)``. Returnert som lang DataFrame (alle
    par, inkludert diagonalen) slik at den er enkel å pivotere eller lese rett
    av i notebooken.
    """
    columns = list(columns)
    values = severity_frame[columns].to_numpy(dtype=float)
    weights = severity_frame["property_claims"].to_numpy(dtype=float)
    covariance = np.cov(values, rowvar=False, aweights=weights)
    std = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(std, std)

    return pd.DataFrame(
        [
            {
                "variabel_1": columns[i],
                "variabel_2": columns[j],
                "korrelasjon": correlation[i, j],
            }
            for i in range(len(columns))
            for j in range(len(columns))
        ]
    )


def build_fold_coefficient_table(results, model_names):
    """Foldstabilitet: min/maks/gjennomsnitt/spredning per koeffisient over fem folder.

    ``results[name]["params"]`` er ``cross_validate_glm``s koeffisientmatrise
    (én kolonne per fold, fra ``glm_core.cross_validate_glm``).
    ``relativ_spredning`` er fold-spennet (maks - min) relativt til
    ``|gjennomsnitt|`` — hvor mye koeffisienten flytter seg mellom folder,
    målt i forhold til sitt eget nivå.
    """
    rows = []
    for name in model_names:
        for coefficient, fold_values in results[name]["params"].iterrows():
            fold_min, fold_max, mean = (
                fold_values.min(),
                fold_values.max(),
                fold_values.mean(),
            )
            rows.append(
                {
                    "modell": name,
                    "koeffisient": coefficient,
                    "fold_min": fold_min,
                    "fold_maks": fold_max,
                    "gjennomsnitt": mean,
                    "standardavvik": fold_values.std(),
                    "relativ_spredning": (fold_max - fold_min) / abs(mean),
                }
            )
    return pd.DataFrame(rows)


def plot_calibration(decile_table, ae_table):
    """Én figur: (a) desilkalibrering med skadeantallsstøtte, (b) A/E per segment.

    Bare visualisering — ingen beregning utover det ``decile_table`` og
    ``ae_table`` allerede inneholder (fra ``build_decile_table``/``build_ae_table``).
    """
    fig, (ax_decile, ax_ae) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE)
    models = list(
        dict.fromkeys(decile_table["modell"])
    )  # bevar rekkefølgen fra predictions

    # (a) Desilplott: skadeantall per desil som søyler på sekundærakse, og
    # faktisk vs. predikert severity som linjer for hver modell.
    support = decile_table.drop_duplicates("desil").sort_values("desil")
    support_ax = ax_decile.twinx()
    support_ax.bar(
        support["desil"], support["skadeantall"], color=CONTEXT, alpha=0.5, width=0.6
    )
    support_ax.set_ylabel("Skadeantall", color=INK_MUTED, fontsize=9)
    support_ax.tick_params(colors=INK_MUTED, labelsize=8)

    for model_name, color in zip(models, _LINE_COLORS):
        subset = decile_table[decile_table["modell"] == model_name].sort_values("desil")
        ax_decile.plot(
            subset["desil"],
            subset["gjennomsnitt_predikert"],
            marker="o",
            color=color,
            label=f"Predikert ({model_name})",
            zorder=3,
        )
    first_model = decile_table[decile_table["modell"] == models[0]].sort_values("desil")
    ax_decile.plot(
        first_model["desil"],
        first_model["faktisk_severity"],
        marker="s",
        linestyle="--",
        color=INK_PRIMARY,
        label="Faktisk",
        zorder=3,
    )
    _style_axes(
        ax_decile, "Desilkalibrering (referansemodellens desiler)", "EUR", "Desil"
    )
    ax_decile.legend(frameon=False, fontsize=8, loc="upper left")

    # (b) A/E per segment (uten «Totalt»), med 0,80-1,20-båndet markert.
    segment_rows = ae_table.loc[ae_table["variabel"] != "Totalt"].copy()
    segment_rows["segment"] = (
        segment_rows["variabel"] + ": " + segment_rows["nivå"].astype(str)
    )
    ax_ae.axhspan(0.8, 1.2, color=CONTEXT, alpha=0.3, zorder=0)
    ax_ae.axhline(1.0, color=INK_MUTED, linewidth=1, zorder=1)
    for model_name, color in zip(models, _LINE_COLORS):
        subset = segment_rows[segment_rows["modell"] == model_name]
        ax_ae.scatter(
            subset["segment"], subset["A_E"], color=color, label=model_name, zorder=3
        )
    _style_axes(ax_ae, "A/E per segment (0,80-1,20-bånd)", "A/E")
    ax_ae.tick_params(axis="x", rotation=90, labelsize=7)
    ax_ae.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    return fig
