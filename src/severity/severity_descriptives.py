"""Deskriptiv diagnostikk FØR severity-fitting: kostnadskonsentrasjon, små beløp og støtte.

Alle funksjoner tar imot ``severity_frame`` (poliseår med registrert egen-skade,
``property_claims > 0`` og ``property_incurred > 0.01``, med en tilhørende
``average_severity = property_incurred / property_claims``-kolonne bygget i
notebooken) og returnerer en ``pandas.DataFrame``/``dict`` med DataFrames eller
en ``matplotlib.figure.Figure``. Ingen funksjon her laster data, fitter noe,
eller åpner nye kandidater/terskler i seleksjonen (se
``plans/Severity_plan.md``, "Før fitting: skadebeløp og støtte").

Stil og fargepalett følger husstilen i ``src/descriptives/own_damage_descriptives.py`` og
``src/phase_2/frequency_plots.py``.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.descriptives.own_damage_descriptives import (
    ACCENT,
    AXIS,
    CONTEXT,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
)

# Terskler for "storskade" på begge grunnlag (samlet årskostnad og gjennomsnittsskade).
_LARGE_CLAIM_THRESHOLDS = (5000, 7500, 10000)
# Grensen som markerer svak visuell støtte i deskriptiv diagnostikk (IKKE en
# gyldighetsregel — se docstring for build_support_table).
_WEAK_SUPPORT_PERSONS = 50


def _style_axes(ax, title, ylabel, xlabel=""):
    """Samme aksestil som own_damage_descriptives._style_axes / frequency_plots._style_axes."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left", pad=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)


def _seat_category(seats):
    """Lås setekategorien til `<5` / `=5` / `>5` — ingen annen binning er tillatt."""
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


# ---------------------------------------------------------------------------
# Seksjon A — kostnadskonsentrasjon og storskader
# ---------------------------------------------------------------------------


def plot_cost_concentration(severity_frame, thresholds=_LARGE_CLAIM_THRESHOLDS):
    """Lorenz-kurve (poliseår/person) og histogram av log(gjennomsnittsskade) med tersklene.

    Returnerer en ``fig`` med to paneler: (i) Lorenz-kurve for kostnadsfordelingen
    per poliseår og per person, (ii) fordelingen av ``log(average_severity)``
    med de tre storskade-tersklene inntegnet som vertikale linjer.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)

    bases = {
        "poliseår": severity_frame["property_incurred"],
        "person": severity_frame.groupby("insured_id")["property_incurred"].sum(),
    }
    for (label, values), color in zip(bases.items(), (ACCENT, CONTEXT)):
        sorted_values = np.sort(values.to_numpy())
        cumulative_share = np.concatenate(
            ([0], np.cumsum(sorted_values) / sorted_values.sum())
        )
        population_share = np.concatenate(
            ([0], np.arange(1, len(sorted_values) + 1) / len(sorted_values))
        )
        axes[0].plot(
            population_share, cumulative_share, color=color, linewidth=2, label=label
        )
    axes[0].plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    _style_axes(
        axes[0],
        "Lorenz-kurve: kostnadskonsentrasjon",
        "Kumulativ andel av kostnad",
        "Kumulativ andel av enheter",
    )

    log_severity = np.log(severity_frame["average_severity"])
    axes[1].hist(log_severity, bins=40, color=CONTEXT, edgecolor=SURFACE)
    for threshold in thresholds:
        axes[1].axvline(
            np.log(threshold), color="#b23a2f", linewidth=1.4, linestyle="--"
        )
        axes[1].text(
            np.log(threshold),
            axes[1].get_ylim()[1] * 0.97,
            f"{threshold:,.0f}",
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            color="#b23a2f",
        )
    _style_axes(
        axes[1],
        "Fordeling av log(gjennomsnittsskade)",
        "Antall skadeår",
        "log(average_severity)",
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Seksjon B — små beløp og beløpsklumper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Seksjon C — støtte per segment
# ---------------------------------------------------------------------------

_DEFAULT_SUPPORT_COLUMNS = (
    "policy_type",
    "year",
    "municipality_type",
    "circulation_area",
    "fuel_type",
    "business_type",
    "payment_frequency",
    "vehicle_brand_pooled",
    "setekategori",
    "skadeantallsgruppe",
)


def _add_support_columns(severity_frame):
    """Legg til de avledede kategoriene setekategori og skadeantallsgruppe."""
    frame = severity_frame.copy()
    frame["setekategori"] = _seat_category(frame["seats"])
    frame["skadeantallsgruppe"] = _claim_count_group(frame["property_claims"])
    return frame


def build_support_table(severity_frame, group_columns=None):
    """Støtte per segment for standardvariablene i kandidatdiagnostikken.

    ``group_columns`` er som standard ``policy_type``, ``year``,
    ``municipality_type``, ``circulation_area``, ``fuel_type``,
    ``business_type``, ``payment_frequency``, ``vehicle_brand_pooled``,
    setekategori (`<5`/`=5`/`>5`) og skadeantallsgruppe
    (`N=1`/`N=2-3`/`N>=4`). For hvert nivå i hver variabel rapporteres
    ``antall_skadeår``, ``unike_personer``, ``skadeantall``, ``kostnadsandel``
    (andel av total ``property_incurred``) og ``skadevektet_severity`` (= sum
    ``property_incurred`` / sum ``property_claims`` i segmentet).

    Returnerer én lang DataFrame med kolonnene ``variabel``, ``nivå`` og
    målene over, slik at alle variablene vises i samme tabell.

    ``svakt_støttet`` er True når ``unike_personer < 50``. Denne grensen er
    KUN en visuell støttemarkering i denne deskriptive diagnostikken — den er
    IKKE gyldighetsregelen for treningsstøtte per fold, som defineres og
    håndheves et annet sted (se ``plans/Severity_plan.md``).
    """
    frame = _add_support_columns(severity_frame)
    columns = (
        list(group_columns)
        if group_columns is not None
        else list(_DEFAULT_SUPPORT_COLUMNS)
    )
    total_incurred = frame["property_incurred"].sum()

    rows = []
    for column in columns:
        summary = frame.groupby(column, observed=True).agg(
            antall_skadeår=("property_incurred", "size"),
            unike_personer=("insured_id", "nunique"),
            skadeantall=("property_claims", "sum"),
            _sum_incurred=("property_incurred", "sum"),
        )
        summary["kostnadsandel"] = summary["_sum_incurred"] / total_incurred
        summary["skadevektet_severity"] = (
            summary["_sum_incurred"] / summary["skadeantall"]
        )
        summary["svakt_støttet"] = summary["unike_personer"] < _WEAK_SUPPORT_PERSONS
        summary = summary.drop(columns="_sum_incurred").reset_index()
        summary = summary.rename(columns={column: "nivå"})
        summary["nivå"] = summary["nivå"].astype(str)
        summary.insert(0, "variabel", column)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


