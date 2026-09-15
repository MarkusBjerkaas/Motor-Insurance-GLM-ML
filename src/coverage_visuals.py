"""Visualiseringer som begrunner avgrensningen til egen-skadedekning."""

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
AXIS = "#c3c2b7"
ACCENT = "#2a78d6"
CONTEXT = "#9ec5f4"

# Sekvensiell (magnitude) ramp: én blåtone, lys -> mørk.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "own_damage_blues",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
MODEL_SCOPE_THRESHOLDS = [
    (1_000, "Enklere modell / sammenslåing"),
    (10_000, "Selvstendig modellering"),
]


def plot_coverage_by_policy_type_heatmap(coverage_by_policy_type):
    """Andel poliseår med positiv dekningspremie, per policy_type og dekning."""
    premie_sum = coverage_by_policy_type.pivot_table(
        index="policy_type", columns="dekning", values="positiv_premie", aggfunc="sum"
    )
    total_sum = coverage_by_policy_type.pivot_table(
        index="policy_type", columns="dekning", values="poliseår", aggfunc="sum"
    )
    aggregated = premie_sum.div(total_sum).mul(100).astype("float64")
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURFACE)
    sns.heatmap(
        aggregated,
        annot=True,
        fmt=".1f",
        cmap=SEQUENTIAL_BLUE,
        vmin=0,
        vmax=100,
        linewidths=2,
        linecolor=SURFACE,
        cbar_kws={"label": "Andel med positiv premie (%)"},
        annot_kws={
            "fontsize": 9,
            "color": INK_PRIMARY,
            "path_effects": [pe.withStroke(linewidth=2.5, foreground=SURFACE)],
        },
        ax=ax,
    )
    ax.set_title(
        "Hvilke policy_type har faktisk hvilken dekning aktiv?",
        color=INK_PRIMARY,
        fontsize=12,
        loc="left",
        pad=12,
    )
    ax.set_xlabel("Dekning", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("policy_type", color=INK_SECONDARY, fontsize=10)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    fig.tight_layout()
    return fig


def plot_response_volume(response_summary):
    """Skadeantall per dekning (log-skala), med terskler for selvstendig modellering."""
    ordered = response_summary.sort_values("skadeantall")
    colors = [ACCENT if d == "Egen skade" else CONTEXT for d in ordered["dekning"]]
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(ordered["dekning"], ordered["skadeantall"], color=colors, height=0.6)
    ax.set_xscale("log")
    ax.set_xlim(20, 60_000)
    for value, y in zip(ordered["skadeantall"], range(len(ordered))):
        ax.text(
            value * 1.1,
            y,
            f"{value:,.0f}",
            va="center",
            color=INK_SECONDARY,
            fontsize=9,
        )
    for threshold, label in MODEL_SCOPE_THRESHOLDS:
        ax.axvline(threshold, color=INK_MUTED, linewidth=1)
        ax.text(
            threshold,
            len(ordered) - 0.15,
            f" {label}",
            color=INK_MUTED,
            fontsize=8,
            ha="left",
            va="bottom",
        )
    ax.set_title(
        "Skadevolum per dekning, med modelleringsterskler",
        color=INK_PRIMARY,
        fontsize=12,
        loc="left",
        pad=12,
    )
    ax.set_xlabel("Skadeantall (log-skala)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    fig.tight_layout()
    return fig
