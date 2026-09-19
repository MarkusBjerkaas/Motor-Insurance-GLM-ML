"""Visualiseringer for porteføljestruktur: eksponering og produktmiks per år."""

import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# Eksponeringsstatus er ordinal (null < delår < full år) -> én valgt blåtone per nivå.
EXPOSURE_ORDER = ["Null", "Delår", "Full år"]
EXPOSURE_COLORS = {"Null": "#86b6ef", "Delår": "#2a78d6", "Full år": "#104281"}
# policy_type er nominal -> faste kategoriske fargeslot i dokumentert rekkefølge.
POLICY_TYPE_ORDER = ["TP", "TPG", "CC", "COMP_E", "COMP_N"]
POLICY_TYPE_COLORS = dict(
    zip(POLICY_TYPE_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"])
)


def _style_axes(ax, title, ylabel):
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    ax.set_xlabel("")
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)


def _stacked_bar(ax, shares, order, colors, label_totals):
    """Stablet stolpe med 2px overflategap mellom segmenter og valgfrie totalsum-etiketter."""
    bottom = pd.Series(0.0, index=shares.index)
    for category in order:
        values = shares[category]
        ax.bar(
            shares.index.astype(str),
            values,
            bottom=bottom,
            width=0.6,
            color=colors[category],
            edgecolor=SURFACE,
            linewidth=1.5,
            label=category,
        )
        bottom = bottom + values
    if label_totals:
        for x, total in zip(shares.index.astype(str), bottom):
            ax.text(
                x,
                total,
                f" {total:,.0f}",
                ha="center",
                va="bottom",
                color=INK_SECONDARY,
                fontsize=9,
            )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
        labelcolor=INK_SECONDARY,
        fontsize=9,
    )


def plot_exposure_structure(frame):
    """Poliseår per år, fordelt på eksponeringsstatus (null/delår/full år)."""
    category = pd.cut(
        frame["total_exposure"], bins=[-0.01, 0, 0.999999, 1.0], labels=EXPOSURE_ORDER
    )
    counts = (
        frame.assign(eksponeringskategori=category)
        .groupby(["year", "eksponeringskategori"], observed=True)
        .size()
        .unstack("eksponeringskategori")
        .reindex(columns=EXPOSURE_ORDER)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    _stacked_bar(ax, counts, EXPOSURE_ORDER, EXPOSURE_COLORS, label_totals=True)
    _style_axes(ax, "Poliseår per år, etter eksponeringsstatus", "Poliseår")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_policy_type_composition(frame):
    """Andel poliseår per policy_type, per år (100 %-stablet)."""
    counts = (
        frame.groupby(["year", "policy_type"], observed=True)
        .size()
        .unstack("policy_type")
    )
    shares = (
        counts.div(counts.sum(axis=1), axis=0)
        .mul(100)
        .reindex(columns=POLICY_TYPE_ORDER)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    _stacked_bar(ax, shares, POLICY_TYPE_ORDER, POLICY_TYPE_COLORS, label_totals=False)
    _style_axes(ax, "Andel poliseår per policy_type, per år", "Andel av poliseår (%)")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    plt.close(fig)
    return fig
