"""Plott for fase 1 (frekvens): foldkurver, gevinstintervaller, A/E og rootogram.

Stil og fargebruk følger ``src/own_damage_descriptives.py`` (samme
hus-palett): nøytral bakgrunn, hårfine akser, aksent-farge for hovedserie og
en lysere kontekstfarge for eksponering/støtte.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

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

# Ekstra kategoriske farger (samme palett-familie som ACCENT/CONTEXT) for
# sammenligning av inntil tre modeller/kandidater i samme panel.
_ORANGE = "#eb6834"
_AQUA = "#1baf7a"
_MODEL_COLORS = [ACCENT, _ORANGE, _AQUA]

_GOOD = "#0ca30c"  # status: kandidat består B-08
_WARN = "#eda100"  # status: nær-treff (3/5 folder)
_CRITICAL = "#d03b3b"  # status: flagget avvik (A/E)

# Pene, runde relativitetstall for log-skalerte y-akser (unngår 2x10^0-etiketter).
_NICE_RELATIVES = [
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1,
    1.1,
    1.25,
    1.5,
    1.75,
    2,
    2.5,
    3,
    4,
    5,
    6,
    8,
    10,
]


def _format_relative(x, _pos=None):
    """Norsk tallformat (komma) for relativitetsverdier, uten unødvendige desimaler."""
    return f"{x:g}".replace(".", ",")


def _set_relative_yticks(ax, max_ticks=6):
    """Faste, lesbare ticks på en log-skalert relativitetsakse (ingen minor-ticks)."""
    ymin, ymax = ax.get_ylim()
    ticks = [t for t in _NICE_RELATIVES if ymin <= t <= ymax]
    if 1.0 not in ticks and ymin <= 1.0 <= ymax:
        ticks.append(1.0)
    ticks = sorted(set(ticks))
    if len(ticks) > max_ticks:
        # Tynn ut jevnt, men behold alltid 1,0 som referanse.
        keep_idx = np.unique(
            np.linspace(0, len(ticks) - 1, max_ticks).round().astype(int)
        )
        ticks = sorted(
            {ticks[i] for i in keep_idx} | ({1.0} if 1.0 in ticks else set())
        )
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(_format_relative))


def _style_axes(ax, title, ylabel, xlabel=""):
    """Samme aksestil som own_damage_descriptives._style_axes."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left", pad=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)


def _grid_shape(n_panels, max_cols=3):
    n_cols = min(max_cols, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    return n_rows, n_cols


def plot_fold_curves(
    curves,
    features=None,
    support_frame=None,
    exposure_col="total_exposure",
    feature_labels=None,
    title=None,
):
    """Log-relativitetskurver per feature (og per produkt ved interaksjoner).

    Tynne linjer viser hver treningsfolds kurve (stabilitet/spredning); en
    tydelig linje viser fit på hele train_pool (``fold == "full"``). Flere
    ``model_id`` i samme feature/produkt fargelegges separat (til og med 3).
    Gis ``support_frame``, tegnes et svakt eksponeringshistogram langs x
    nederst i panelet så man ser hvor dataene faktisk støtter kurven.
    """
    feature_labels = feature_labels or {}
    feature_list = (
        list(dict.fromkeys(curves["feature"])) if features is None else features
    )

    panels = []  # (feature, product)
    for feature in feature_list:
        products = list(
            dict.fromkeys(curves.loc[curves["feature"] == feature, "product"])
        )
        for product in products:
            panels.append((feature, product))

    n_rows, n_cols = _grid_shape(len(panels))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 4 * n_rows),
        facecolor=SURFACE,
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for ax, (feature, product) in zip(flat_axes, panels):
        panel = curves[(curves["feature"] == feature) & (curves["product"] == product)]
        models = list(dict.fromkeys(panel["model_id"]))
        colors = dict(zip(models, _MODEL_COLORS))

        for model_id, model_part in panel.groupby("model_id", sort=False):
            color = colors[model_id]
            for fold, fold_part in model_part.groupby("fold", sort=False):
                fold_part = fold_part.sort_values("x")
                is_full = fold == "full"
                ax.plot(
                    fold_part["x"],
                    fold_part["relative"],
                    color=color,
                    linewidth=2.2 if is_full else 0.9,
                    alpha=1.0 if is_full else 0.35,
                    label=model_id if is_full else None,
                    zorder=3 if is_full else 2,
                )
                # Svakt +/-2*SE-bånd (på log-skala) rundt full-fold-linjen, hvis oppgitt.
                if is_full and "se_log" in fold_part.columns:
                    lower = fold_part["relative"] * np.exp(-2 * fold_part["se_log"])
                    upper = fold_part["relative"] * np.exp(2 * fold_part["se_log"])
                    ax.fill_between(
                        fold_part["x"],
                        lower,
                        upper,
                        color=color,
                        alpha=0.15,
                        zorder=2.5,
                    )
        ax.axhline(1.0, color=AXIS, linewidth=1, zorder=1)
        ax.set_yscale("log")
        _set_relative_yticks(ax)

        if support_frame is not None and feature in support_frame:
            ax_support = ax.twinx()
            values = support_frame[feature].dropna()
            weights = support_frame.loc[values.index, exposure_col]
            counts, edges = np.histogram(values, bins=20, weights=weights)
            ax_support.bar(
                (edges[:-1] + edges[1:]) / 2,
                counts,
                width=(edges[1] - edges[0]) * 0.9,
                color=CONTEXT,
                alpha=0.5,
                zorder=0,
            )
            # Histogrammet skal bare fylle bunnen (~25%) av panelet.
            ax_support.set_ylim(0, counts.max() * 4)
            ax_support.set_yticks([])
            ax_support.spines[["top", "right", "left", "bottom"]].set_visible(False)

        label = feature_labels.get(feature, feature)
        panel_title = label if product == "ALLE" else f"{label} ({product})"
        _style_axes(ax, panel_title, "Relativitet (log-skala)", label)
        if len(models) > 1:
            ax.legend(fontsize=7, frameon=False, labelcolor=INK_SECONDARY)

    for ax in flat_axes[len(panels) :]:
        ax.set_visible(False)

    fig.suptitle(
        title or "Foldkurver: log-relativitet per prediktor",
        x=0.01,
        ha="left",
        color=INK_PRIMARY,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.close(fig)
    return fig


def plot_gain_intervals(comparisons, label_col="candidate", title=None):
    """Punktestimat for mean_gain +/- 1 SE, sortert, med B-08/nær-treff markert."""
    table = comparisons.sort_values("mean_gain").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(table))), facecolor=SURFACE)
    y = np.arange(len(table))

    colors, markers = [], []
    for _, row in table.iterrows():
        if row.get("passes_b08", False):
            colors.append(_GOOD)
            markers.append("o")
        elif row.get("near_miss_3of5", False):
            colors.append(_WARN)
            markers.append("D")
        else:
            colors.append(INK_MUTED)
            markers.append("o")

    ax.axvline(0, color=AXIS, linewidth=1)
    for yi, (_, row), color, marker in zip(y, table.iterrows(), colors, markers):
        ax.errorbar(
            row["mean_gain"],
            yi,
            xerr=row["se_gain"],
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6,
        )
    ax.set_yticks(y, table[label_col])
    _style_axes(
        ax,
        title or "Gevinst per kandidat (snitt over folder, +/- 1 SE)",
        "",
        "Snittgevinst i deviance (kandidat bedre til høyre)",
    )
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color=_GOOD, linestyle="", label="Består B-08"
        ),
        plt.Line2D(
            [0], [0], marker="D", color=_WARN, linestyle="", label="Nær-treff (3/5)"
        ),
        plt.Line2D(
            [0], [0], marker="o", color=INK_MUTED, linestyle="", label="Ingen av delene"
        ),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_actual_expected(ae_table, n_cols=3, title=None):
    """Småmultipler av A/E per segment, med eksponering som svake søyler."""
    segments = list(dict.fromkeys(ae_table["segment"]))
    n_rows, n_cols = _grid_shape(len(segments), max_cols=n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 3.6 * n_rows),
        facecolor=SURFACE,
        squeeze=False,
    )
    flat_axes = axes.flatten()
    has_models = "model" in ae_table.columns
    models = list(dict.fromkeys(ae_table["model"])) if has_models else [None]
    colors = dict(zip(models, _MODEL_COLORS))

    for ax, segment in zip(flat_axes, segments):
        part = ae_table[ae_table["segment"] == segment]
        levels = list(dict.fromkeys(part["level"]))
        x = np.arange(len(levels))

        exposure_by_level = (
            part.groupby("level", sort=False)["exposure"].sum().reindex(levels)
        )
        ax_exp = ax.twinx()
        ax_exp.bar(x, exposure_by_level, color=CONTEXT, alpha=0.4, width=0.6, zorder=0)
        ax_exp.set_ylim(0, exposure_by_level.max() * 3)
        ax_exp.set_yticks([])
        ax_exp.spines[["top", "right", "left", "bottom"]].set_visible(False)

        ax.axhline(1.0, color=AXIS, linewidth=1, zorder=1)
        has_se = "ae_se" in part.columns
        has_flag = "flag" in part.columns
        n_models = len(models)
        for i, model in enumerate(models):
            sub = part[part["model"] == model] if has_models else part
            sub = sub.set_index("level").reindex(levels)
            offset = (i - (n_models - 1) / 2) * 0.15
            color = colors[model]
            if has_se:
                ax.errorbar(
                    x + offset,
                    sub["ae"],
                    yerr=2 * sub["ae_se"],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    elinewidth=1,
                    capsize=2,
                    markersize=5.5,
                    zorder=3,
                )
            else:
                ax.scatter(x + offset, sub["ae"], color=color, zorder=3, s=32)
            if has_flag:
                flagged = sub["flag"].fillna(False).to_numpy()
                if flagged.any():
                    ax.scatter(
                        (x + offset)[flagged],
                        sub["ae"].to_numpy()[flagged],
                        facecolors="none",
                        edgecolors=_CRITICAL,
                        linewidths=1.8,
                        s=90,
                        zorder=4,
                    )
        ax.set_xticks(x, [str(v) for v in levels], rotation=35, ha="right")
        _style_axes(ax, str(segment), "A/E (observert/forventet)")

        handles = (
            [
                plt.Line2D([0], [0], marker="o", color=colors[m], linestyle="", label=m)
                for m in models
            ]
            if has_models
            else []
        )
        if has_flag and part["flag"].fillna(False).any():
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    markerfacecolor="none",
                    markeredgecolor=_CRITICAL,
                    linestyle="",
                    markersize=8,
                    label="Flagget: |A/E-1| > 2×SE",
                )
            )
        if handles:
            ax.legend(
                handles=handles, fontsize=7, frameon=False, labelcolor=INK_SECONDARY
            )

    for ax in flat_axes[len(segments) :]:
        ax.set_visible(False)

    fig.suptitle(
        title or "Faktisk mot forventet (A/E) per segment",
        x=0.01,
        ha="left",
        color=INK_PRIMARY,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.close(fig)
    return fig


def plot_rootogram(tables, title=None):
    """Hengende rootogram (Kleiber & Zeileis): kvadratrot-skala, søyler henger fra forventet."""
    labels = list(dict.fromkeys(tables["label"]))
    fig, axes = plt.subplots(
        1, len(labels), figsize=(5 * len(labels), 4.2), facecolor=SURFACE, squeeze=False
    )
    axes = axes[0]

    for ax, label in zip(axes, labels):
        part = tables[tables["label"] == label]
        x = np.arange(len(part))
        sqrt_obs = np.sqrt(part["observed"].to_numpy())
        sqrt_exp = np.sqrt(part["expected"].to_numpy())
        ax.bar(
            x,
            sqrt_obs,
            bottom=sqrt_exp - sqrt_obs,
            color=CONTEXT,
            width=0.65,
            label="Observert (hengende)",
            zorder=2,
        )
        ax.plot(
            x,
            sqrt_exp,
            color=ACCENT,
            marker="o",
            markersize=4,
            linewidth=1.6,
            label="Forventet",
            zorder=3,
        )
        ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
        ax.set_xticks(x, part["k"])
        _style_axes(ax, label, "Kvadratrot av antall", "Skadeantall k")
        ax.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY)

    fig.suptitle(
        title or "Rootogram: observert vs. forventet skadeantall",
        x=0.01,
        ha="left",
        color=INK_PRIMARY,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.close(fig)
    return fig
