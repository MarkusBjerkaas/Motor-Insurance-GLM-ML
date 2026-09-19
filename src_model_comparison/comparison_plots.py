"""Figurer for modellsammenligningen: Lorenz/Gini, kalibrering, balanse og parvise forskjeller."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from src_model_comparison.gini_bootstrap import NULL_MODEL, lorenz_curve

# Okabe-Ito-palett (fargeblind-vennlig); samme farge for samme modell i alle figurer.
MODEL_COLORS = {
    NULL_MODEL: "#7f7f7f",
    "Toleddet GLM": "#0072B2",
    "Tweedie-GLM": "#009E73",
    "CatBoost": "#D55E00",
    "CatBoost (rekalibrert)": "#D55E00",
}
DASHED = {"CatBoost (rekalibrert)"}
RC = {
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
}


def _style(name):
    return {
        "color": MODEL_COLORS.get(name, "black"),
        "linestyle": "--" if name in DASHED else "-",
    }


def _dot_with_interval(ax, table, value, low, high, models, xlabel, title, digits):
    """Punkt med 95 %-intervall per modell, en modell per rad."""
    positions = np.arange(len(models))[::-1]
    for y, name in zip(positions, models):
        row = table.loc[name]
        color = MODEL_COLORS.get(name, "black")
        ax.errorbar(
            row[value],
            y,
            xerr=[[row[value] - row[low]], [row[high] - row[value]]],
            fmt="o",
            color=color,
            markerfacecolor="white" if name in DASHED else color,
            capsize=4,
            markersize=8,
        )
        ax.annotate(
            f"{row[value]:.{digits}f}",
            (row[high], y),
            xytext=(6, -3),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_yticks(positions, models)
    ax.set(xlabel=xlabel, title=title)
    ax.margins(x=0.2)


def plot_model_comparison(test_frame, predictions, calibration, summary):
    """Fire paneler: Lorenz-kurver, Gini med KI, kalibrering per desil og balanse med KI."""
    cost = test_frame["property_incurred"].to_numpy(dtype=float)
    exposure = test_frame["total_exposure"].to_numpy(dtype=float)
    models = [name for name in summary.index if name != NULL_MODEL]
    with plt.rc_context(RC):
        figure, axes = plt.subplots(2, 2, figsize=(13, 10))
        lorenz, gini, calib, balance = axes.ravel()

        # (a) Lorenz-kurver: jo lenger under diagonalen, desto bedre rangering.
        lorenz.plot(
            [0, 1],
            [0, 1],
            color=MODEL_COLORS[NULL_MODEL],
            lw=1.2,
            label=f"{NULL_MODEL} (Gini 0)",
        )
        for name in models:
            x, y = lorenz_curve(cost, exposure, predictions[name].to_numpy(dtype=float))
            lorenz.plot(
                x,
                y,
                lw=2,
                label=f"{name} (Gini {summary.loc[name, 'gini']:.3f})",
                **_style(name),
            )
        x, y = lorenz_curve(cost, exposure, cost / exposure)
        lorenz.plot(x, y, color="black", lw=1, linestyle=":", label="Perfekt rangering")
        lorenz.set(
            title="Lorenz-kurver",
            xlabel="Kumulativ andel av eksponering",
            ylabel="Kumulativ andel av skadekostnad",
        )
        lorenz.legend(fontsize=8, loc="upper left")

        # (b) Gini med bootstrap-intervall.
        _dot_with_interval(
            gini,
            summary,
            "gini",
            "gini_lo",
            "gini_hi",
            models,
            "Gini (95 % KI)",
            "Gini med usikkerhet",
            3,
        )

        # (c) Kalibrering: observert/predikert per prediksjonsdesil.
        for name, part in calibration.groupby("Modell", observed=True):
            ratio = part["Observert ren premie"] / part["Predikert ren premie"]
            calib.plot(
                part["Prediksjonsgruppe"], ratio, marker="o", label=name, **_style(name)
            )
        calib.axhline(1, color="black", lw=1)
        calib.set(
            title="Kalibrering per prediksjonsdesil",
            xlabel="Prediksjonsdesil (1 = lavest risiko)",
            ylabel="Observert / predikert",
            xticks=range(1, 11),
        )
        calib.legend(fontsize=8)

        # (d) Porteføljebalanse med bootstrap-intervall.
        _dot_with_interval(
            balance,
            summary,
            "balance",
            "balance_lo",
            "balance_hi",
            models,
            "Predikert minus observert (%)",
            "Porteføljebalanse med usikkerhet",
            1,
        )
        balance.axvline(0, color="black", lw=1)
        figure.tight_layout()
    return figure


def plot_pairwise_forest(pairwise):
    """To paneler med parvise forskjeller og 95 %-intervall; positivt = første modell bedre."""
    labels = [name.replace(" mot ", "\nmot ") for name in pairwise.index]
    positions = np.arange(len(labels))[::-1]
    with plt.rc_context(RC):
        figure, axes = plt.subplots(
            1, 2, figsize=(13, 0.9 * len(labels) + 2.2), sharey=True
        )
        panels = [
            ("deviance", "Devians-gevinst (lavere devians hos A)", 3),
            ("gini", "Gini-gevinst (høyere Gini hos A)", 4),
        ]
        for ax, (metric, title, digits) in zip(axes, panels):
            gain = pairwise[f"{metric}_gain"].to_numpy()
            low, high = (
                pairwise[f"{metric}_lo"].to_numpy(),
                pairwise[f"{metric}_hi"].to_numpy(),
            )
            significant = (low > 0) | (high < 0)
            colors = np.where(significant, "#D55E00", "#4d4d4d")
            for y, g, lo, hi, c in zip(positions, gain, low, high, colors):
                ax.errorbar(
                    g, y, xerr=[[g - lo], [hi - g]], fmt="o", color=c, capsize=4
                )
            ax.axvline(0, color="black", lw=1)
            for y, p in zip(positions, pairwise[f"{metric}_p_better"]):
                if np.isnan(p):
                    continue
                ax.annotate(
                    f"P(A bedre) = {p:.0%}",
                    (1, y),
                    xycoords=("axes fraction", "data"),
                    xytext=(-4, 8),
                    textcoords="offset points",
                    ha="right",
                    fontsize=8,
                )
            ax.set(title=title, xlabel="Forskjell (positivt = A bedre)")
            ax.margins(x=0.25)
        axes[0].set_yticks(positions, labels)
        figure.suptitle(
            "Parvis bootstrap: oransje = intervallet utelukker 0", fontsize=11
        )
        figure.tight_layout()
    return figure


def plot_top_decile_decomposition(component_summary):
    """Tre kompakte paneler: frekvens, rapportert severity og ren premie i toppdesilen."""
    labels = {
        "Frekvens": ("Skader per eksponeringsår", "{x:.2f}"),
        "Rapportert severity": ("EUR per rapportert skade", "€{x:,.0f}"),
        "Ren premie": ("EUR per eksponeringsår", "€{x:,.0f}"),
    }
    colors = ["#0072B2", "#222222"]
    with plt.rc_context(RC):
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for axis, (component, row) in zip(axes, component_summary.iterrows()):
            ylabel, formatter = labels[component]
            values = row[["Predikert", "Observert"]].to_numpy(dtype=float)
            bars = axis.bar(["Predikert", "Observert"], values, color=colors, width=0.62)
            axis.set(title=component, ylabel=ylabel)
            axis.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _, fmt=formatter: fmt.format(x=x))
            )
            axis.set_ylim(0, values.max() * 1.28)
            for bar, value in zip(bars, values):
                axis.annotate(
                    formatter.format(x=value),
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
            axis.annotate(
                f"A/E = {row['A/E']:.3f}",
                (0.5, 0.92),
                xycoords="axes fraction",
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="#D55E00" if abs(row["A/E"] - 1) > 0.03 else "#222222",
            )
        figure.suptitle(
            "Høyeste prediksjonsdesil, toleddet GLM — frekvens forklarer premieunderskuddet",
            fontsize=13,
            fontweight="bold",
            y=1.03,
        )
        figure.tight_layout()
    return figure


def plot_top_decile_profile(profile):
    """Dumbbell-plot for modellenes tydeligste, observerbare høyrisikokjennetegn."""
    ordered = profile.sort_values("Høyeste desil")
    positions = np.arange(len(ordered))
    with plt.rc_context(RC):
        figure, axis = plt.subplots(figsize=(9, 3.8))
        for y, (_, top_share, other_share) in zip(
            positions, ordered.itertuples(index=False, name=None)
        ):
            axis.plot(
                [other_share, top_share],
                [y, y],
                color="#B0B0B0",
                linewidth=2,
                zorder=1,
            )
        axis.scatter(
            ordered["Øvrige 90 %"], positions, s=70, color="#7f7f7f", label="Øvrige 90 %", zorder=2
        )
        axis.scatter(
            ordered["Høyeste desil"], positions, s=70, color="#0072B2", label="Høyeste desil", zorder=3
        )
        for y, (_, top_share, _) in zip(
            positions, ordered.itertuples(index=False, name=None)
        ):
            axis.annotate(
                f"{top_share:.1%}",
                (top_share, y),
                xytext=(7, -3),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="#0072B2",
            )
        axis.set(
            xlabel="Andel av eksponering",
            yticks=positions,
            yticklabels=ordered["Kjennetegn"],
            xlim=(0, 1.08),
        )
        axis.set_title("Hva kjennetegner modellens høyeste risikodesil?", pad=26)
        axis.text(
            0.5,
            1.02,
            "Blå = høyeste desil  ·  grå = øvrige 90 %",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#4d4d4d",
        )
        axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
        figure.tight_layout()
    return figure
