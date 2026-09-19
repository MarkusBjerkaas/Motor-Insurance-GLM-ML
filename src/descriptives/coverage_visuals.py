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
    plt.close(fig)
    return fig


def _style_response_axis(ax, title, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def plot_response_volume(response_summary):
    """Skadeantall (log-skala) og eksponering per dekning, side om side."""
    ordered = response_summary.sort_values("skadeantall")
    colors = [ACCENT if d == "Egen skade" else CONTEXT for d in ordered["dekning"]]
    fig, (ax_claims, ax_exposure) = plt.subplots(
        1, 2, figsize=(12, 4), facecolor=SURFACE
    )

    ax_claims.barh(ordered["dekning"], ordered["skadeantall"], color=colors, height=0.6)
    ax_claims.set_xscale("log")
    ax_claims.set_xlim(20, 60_000)
    for value, y in zip(ordered["skadeantall"], range(len(ordered))):
        ax_claims.text(
            value * 1.1,
            y,
            f"{value:,.0f}",
            va="center",
            color=INK_SECONDARY,
            fontsize=9,
        )
    _style_response_axis(ax_claims, "Skadevolum per dekning", "Skadeantall (log-skala)")

    exposure = ordered["samlet_eksponering"]
    ax_exposure.barh(ordered["dekning"], exposure, color=colors, height=0.6)
    ax_exposure.set_xlim(0, exposure.max() * 1.15)
    for value, y in zip(exposure, range(len(ordered))):
        ax_exposure.text(
            value + exposure.max() * 0.02,
            y,
            f"{value:,.0f}",
            va="center",
            color=INK_SECONDARY,
            fontsize=9,
        )
    _style_response_axis(
        ax_exposure, "Eksponering per dekning", "Samlet eksponering (poliseår)"
    )

    fig.tight_layout()
    plt.close(fig)
    return fig
