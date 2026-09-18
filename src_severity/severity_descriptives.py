"""Deskriptiv diagnostikk FØR severity-fitting: kostnadskonsentrasjon, små beløp og støtte.

Alle funksjoner tar imot ``severity_frame`` (poliseår med registrert egen-skade,
``property_claims > 0`` og ``property_incurred > 0.01``, med en tilhørende
``average_severity = property_incurred / property_claims``-kolonne bygget i
notebooken) og returnerer en ``pandas.DataFrame``/``dict`` med DataFrames eller
en ``matplotlib.figure.Figure``. Ingen funksjon her laster data, fitter noe,
eller åpner nye kandidater/terskler i seleksjonen (se
``plans/Severity_plan.md``, "Før fitting: skadebeløp og støtte").

Stil og fargepalett følger husstilen i ``src/own_damage_descriptives.py`` og
``src/phase_2/frequency_plots.py``.
"""

import math

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


def _grid_shape(n_panels, max_cols=2):
    n_cols = min(max_cols, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    return n_rows, n_cols


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


def build_cost_concentration_table(severity_frame):
    """Kostnadskonsentrasjon per poliseår og per person (`insured_id`).

    For hvert grunnlag ("poliseår" = hver rad i ``severity_frame``, "person" =
    ``property_incurred`` summert per ``insured_id``) rapporteres:

    - andelen av samlet ``property_incurred`` som kommer fra de øverste
      0,1 %, 1 %, 5 % og 10 % (radene ``type="andel_topp"``, rangert etter
      antall enheter — øverste ``ceil(p*n)`` enheter),
    - de fem største enkeltbidragene på hvert grunnlag, med beløp og andel av
      totalen (radene ``type="topp_bidrag"``).

    Returnerer én lang DataFrame med kolonnene ``grunnlag``, ``type``,
    ``nivå``, ``antall_enheter``, ``beløp`` og ``andel``.
    """
    bases = {
        "poliseår": severity_frame["property_incurred"],
        "person": severity_frame.groupby("insured_id")["property_incurred"].sum(),
    }
    top_shares = (0.001, 0.01, 0.05, 0.10)

    rows = []
    for grunnlag, values in bases.items():
        total = values.sum()
        sorted_values = values.sort_values(ascending=False)
        cumulative = sorted_values.cumsum()
        n = len(sorted_values)

        for share in top_shares:
            k = max(1, math.ceil(share * n))
            amount = cumulative.iloc[k - 1]
            rows.append(
                {
                    "grunnlag": grunnlag,
                    "type": "andel_topp",
                    "nivå": f"topp {share * 100:g}%",
                    "antall_enheter": k,
                    "beløp": amount,
                    "andel": amount / total,
                }
            )
        for rank, amount in enumerate(sorted_values.head(5), start=1):
            rows.append(
                {
                    "grunnlag": grunnlag,
                    "type": "topp_bidrag",
                    "nivå": f"#{rank}",
                    "antall_enheter": 1,
                    "beløp": amount,
                    "andel": amount / total,
                }
            )
    return pd.DataFrame(rows)


def build_large_claim_table(severity_frame, thresholds=_LARGE_CLAIM_THRESHOLDS):
    """Overskridelser ved ``thresholds`` (EUR), på to ulike grunnlag.

    Grunnlag ``"samlet årskostnad"`` bruker ``property_incurred`` (summen for
    hele poliseåret); grunnlag ``"gjennomsnittsskade"`` bruker
    ``average_severity`` (= ``property_incurred / property_claims``). For hver
    (terskel, grunnlag)-kombinasjon rapporteres:

    - ``antall_skadeår``: antall rader i ``severity_frame`` over terskelen,
    - ``unike_personer``: antall unike ``insured_id`` blant disse radene,
    - ``kostnadsandel`` = summen av ``property_incurred`` i de overskridende
      radene, delt på total ``property_incurred`` i hele ``severity_frame``,
    - ``overskytende_kostnadsandel`` = **bare** beløpet OVER terskelen, delt
      på samme totaltall.

    ADVARSEL: ``kostnadsandel`` og ``overskytende_kostnadsandel`` er to
    forskjellige størrelser og må ALDRI slås sammen eller forveksles.
    ``kostnadsandel`` teller HELE kostnaden i de overskridende radene;
    ``overskytende_kostnadsandel`` teller kun differansen mot terskelen.
    ``overskytende_kostnadsandel`` er derfor alltid <= ``kostnadsandel``.

    Overskytende beregnes slik:

    - "samlet årskostnad": sum(max(0, property_incurred - terskel))
    - "gjennomsnittsskade": sum(property_claims * max(0, average_severity - terskel))

    OBS: et tall på grunnlaget "gjennomsnittsskade" er IKKE antall
    individuelle storskader. Har et poliseår flere registrerte skader, kan én
    stor enkeltskade skjules i (dvs. dempes av) gjennomsnittet over de andre
    skadene i samme poliseår — se ``plans/Severity_plan.md``.
    """
    total_incurred = severity_frame["property_incurred"].sum()
    incurred = severity_frame["property_incurred"]
    severity = severity_frame["average_severity"]
    claims = severity_frame["property_claims"]

    rows = []
    for threshold in thresholds:
        over_incurred = severity_frame.loc[incurred.gt(threshold)]
        rows.append(
            {
                "terskel": threshold,
                "grunnlag": "samlet årskostnad",
                "antall_skadeår": len(over_incurred),
                "unike_personer": over_incurred["insured_id"].nunique(),
                "kostnadsandel": over_incurred["property_incurred"].sum()
                / total_incurred,
                "overskytende_kostnadsandel": (incurred - threshold).clip(lower=0).sum()
                / total_incurred,
            }
        )
        over_severity = severity_frame.loc[severity.gt(threshold)]
        rows.append(
            {
                "terskel": threshold,
                "grunnlag": "gjennomsnittsskade",
                "antall_skadeår": len(over_severity),
                "unike_personer": over_severity["insured_id"].nunique(),
                "kostnadsandel": over_severity["property_incurred"].sum()
                / total_incurred,
                "overskytende_kostnadsandel": (
                    claims * (severity - threshold).clip(lower=0)
                ).sum()
                / total_incurred,
            }
        )
    return pd.DataFrame(rows)


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


def build_small_amount_table(
    severity_frame, model_frame=None, small_thresholds=(1, 10), n_clumps=15
):
    """Små beløp, utelatte nær-nullår og hyppige beløpsklumper.

    Returnerer en ``dict`` med tre DataFrames (ikke én lang tabell): delene
    har ulik radstruktur (én rad per terskel, én oppsummeringsrad, én rad per
    beløp), og et samlet langt format ville gitt mange tomme celler uten å
    bli mer lesbart.

    - ``"små_terskler"``: for hver terskel i ``small_thresholds``, antall
      skadeår og skadeantall med ``average_severity <= terskel``, med
      tilhørende kostnadsandel av total ``property_incurred``.
    - ``"utelatte_nullår"``: kun hvis ``model_frame`` (utviklingspopulasjonen
      FØR severity-filteret) er gitt. Antall poliseår med registrert skade
      (``property_claims > 0``) men ``property_incurred <= 0.01`` — dvs.
      poliseårene som er utelatt fra ``severity_frame`` — og hvor mange
      registrerte skader de representerer.
    - ``"beløpsklumper"``: de ``n_clumps`` hyppigste eksakte
      ``property_incurred``-beløpene i ``severity_frame`` (kandidater for
      forhåndsavtalte CICOS-oppgjørsbeløp), med antall skadeår og andel av
      ``severity_frame``.
    """
    total_incurred = severity_frame["property_incurred"].sum()

    small_rows = []
    for threshold in small_thresholds:
        under = severity_frame.loc[severity_frame["average_severity"].le(threshold)]
        small_rows.append(
            {
                "terskel": threshold,
                "antall_skadeår": len(under),
                "skadeantall": int(under["property_claims"].sum()),
                "kostnadsandel": under["property_incurred"].sum() / total_incurred,
            }
        )
    small_table = pd.DataFrame(small_rows)

    excluded_table = pd.DataFrame(
        columns=["antall_utelatte_poliseår", "sum_registrerte_skader"]
    )
    if model_frame is not None:
        excluded_mask = model_frame["property_claims"].gt(0) & model_frame[
            "property_incurred"
        ].le(0.01)
        excluded_table = pd.DataFrame(
            [
                {
                    "antall_utelatte_poliseår": int(excluded_mask.sum()),
                    "sum_registrerte_skader": int(
                        model_frame.loc[excluded_mask, "property_claims"].sum()
                    ),
                }
            ]
        )

    clump_counts = severity_frame["property_incurred"].value_counts().head(n_clumps)
    clump_table = (
        clump_counts.rename("antall_skadeår")
        .rename_axis("property_incurred")
        .reset_index()
    )
    clump_table["andel_skadeår"] = clump_table["antall_skadeår"] / len(severity_frame)

    return {
        "små_terskler": small_table,
        "utelatte_nullår": excluded_table,
        "beløpsklumper": clump_table,
    }


def plot_small_amounts(severity_frame, tail_limit=100, n_clumps=15):
    """Nedre hale av gjennomsnittsskaden (lineær skala) og de hyppigste eksakte beløpene.

    Returnerer en ``fig`` med to paneler: (i) empirisk fordeling av
    ``average_severity`` under ``tail_limit`` EUR på lineær skala,
    (ii) de ``n_clumps`` hyppigste eksakte ``property_incurred``-beløpene som
    et horisontalt stolpediagram.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)

    tail = severity_frame.loc[
        severity_frame["average_severity"].le(tail_limit), "average_severity"
    ]
    axes[0].hist(tail, bins=40, color=CONTEXT, edgecolor=SURFACE)
    _style_axes(
        axes[0],
        f"Nedre hale: gjennomsnittsskade ≤ {tail_limit:g} EUR",
        "Antall skadeår",
        "Gjennomsnittsskade (EUR)",
    )

    clumps = (
        severity_frame["property_incurred"].value_counts().head(n_clumps).sort_values()
    )
    axes[1].barh([f"{v:,.2f}" for v in clumps.index], clumps.to_numpy(), color=ACCENT)
    _style_axes(
        axes[1], "Hyppigste eksakte skadebeløp", "Beløp (EUR)", "Antall skadeår"
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Seksjon C — støtte per segment
# ---------------------------------------------------------------------------

_DEFAULT_SUPPORT_COLUMNS = (
    "policy_type",
    "year",
    "municipality_type",
    "circulation_area",
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
    """Støtte per segment for policy_type, year, geografi, setekategori og skadeantallsgruppe.

    ``group_columns`` er som standard de seks variablene ``policy_type``,
    ``year``, ``municipality_type``, ``circulation_area``, setekategori
    (`<5`/`=5`/`>5`) og skadeantallsgruppe (`N=1`/`N=2-3`/`N>=4`). For hvert
    nivå i hver variabel rapporteres ``antall_skadeår``, ``unike_personer``,
    ``skadeantall``, ``kostnadsandel`` (andel av total ``property_incurred``)
    og ``skadevektet_severity`` (= sum ``property_incurred`` / sum
    ``property_claims`` i segmentet).

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


def plot_support(severity_frame):
    """Skadevektet severity per nivå for de seks støttevariablene, med personantall synlig.

    Én figur med ett panel per variabel (policy_type, year, municipality_type,
    circulation_area, setekategori, skadeantallsgruppe). Antall unike personer
    bak hvert nivå annoteres over søylen, slik at svakt støttede nivåer er
    synlige direkte i plottet.
    """
    table = build_support_table(severity_frame)
    variables = list(dict.fromkeys(table["variabel"]))
    n_rows, n_cols = _grid_shape(len(variables), max_cols=3)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 3.8 * n_rows),
        facecolor=SURFACE,
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for ax, variable in zip(flat_axes, variables):
        part = table.loc[table["variabel"].eq(variable)]
        x = np.arange(len(part))
        colors = [CONTEXT if weak else ACCENT for weak in part["svakt_støttet"]]
        ax.bar(x, part["skadevektet_severity"], color=colors)
        for xi, n_persons in zip(x, part["unike_personer"]):
            ax.text(
                xi,
                ax.get_ylim()[1] * 0.02,
                f"n={n_persons}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK_MUTED,
                rotation=90,
            )
        ax.set_xticks(x, part["nivå"], rotation=35, ha="right")
        _style_axes(ax, variable, "Skadevektet severity (EUR)")

    for ax in flat_axes[len(variables) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Skadevektet severity per segment, med antall personer bak hvert nivå",
        x=0.01,
        ha="left",
        color=INK_PRIMARY,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.close(fig)
    return fig
