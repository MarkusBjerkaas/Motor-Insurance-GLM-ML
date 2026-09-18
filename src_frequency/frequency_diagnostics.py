"""Tabeller og diagnostikk for GLM-modelleringen i ``glm_pricing_models.py``.

Modellspesifikasjoner, CV-folder og CV-sløyfen ligger i notebooken. Her ligger
bare funksjoner som beskriver data og folder og oppsummerer resultatene.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import mean_tweedie_deviance
from statsmodels.nonparametric.smoothers_lowess import lowess


def build_glm_summary(spec, result, data):
    """Én rad med nøkkeltall for en GLM estimert på ``data`` (in-sample).

    Deviance er vektet gjennomsnittlig Tweedie-deviance med ``spec["power"]``,
    altså samme skala som CV-scorene. $D^2$ er mot det vektede snittet i ``data``.
    """
    response, weight = data[spec["y"]], data[spec["weight"]]
    prediction = result.predict(data)
    null_prediction = np.full(len(data), np.average(response, weights=weight))

    def deviance(predicted):
        return mean_tweedie_deviance(
            response, predicted, sample_weight=weight, power=spec["power"]
        )

    return pd.Series(
        {
            "formel": spec["formula"],
            "familie": type(spec["glm_family"]).__name__,
            "rader": len(data),
            "parametere": len(result.params),
            "deviance": deviance(prediction),
            "d2": 1 - deviance(prediction) / deviance(null_prediction),
            "balanse": (prediction * weight).sum() / (response * weight).sum(),
        },
        name=spec["name"],
    )


def build_coefficient_table(result):
    """Koeffisienttabell med robuste standardfeil og 95 %-intervaller."""
    intervals = result.conf_int()
    return pd.DataFrame(
        {
            "koeffisient": result.params,
            "standardfeil": result.bse,
            "z": result.tvalues,
            "p_verdi": result.pvalues,
            "ki_95_lav": intervals.iloc[:, 0],
            "ki_95_høy": intervals.iloc[:, 1],
            "relativitet": np.exp(result.params),
        }
    )


def compare_candidate_coefficients(fits, reference_name):
    """Sammenlign felles koeffisienter mot en valgt referansekandidat.

    Bare ledd som finnes i referansemodellen beholdes. Endringen er derfor
    direkte lesbar som kandidatens koeffisient minus den valgte modellens.
    """
    if reference_name not in fits:
        raise KeyError(f"Referansemodellen finnes ikke: {reference_name}")

    reference = fits[reference_name].params.rename(reference_name)
    comparison = reference.to_frame()
    for name, fit in fits.items():
        if name == reference_name:
            continue
        coefficients = fit.params.rename(name)
        comparison = comparison.join(coefficients, how="left")
        comparison[f"endring_fra_{reference_name}_{name}"] = (
            comparison[name] - comparison[reference_name]
        )
    return comparison


def poisson_deviance_residuals(observed_claims, expected_claims):
    """Beregn signerte Poisson-deviance-residualer på skadeantallskala."""
    observed = np.asarray(observed_claims, dtype=float)
    expected = np.asarray(expected_claims, dtype=float)
    if (expected <= 0).any():
        raise ValueError("Forventet skadeantall må være positivt")
    contribution = expected.copy()
    positive = observed > 0
    contribution[positive] = (
        observed[positive] * np.log(observed[positive] / expected[positive])
        - observed[positive]
        + expected[positive]
    )
    return np.sign(observed - expected) * np.sqrt(2 * np.clip(contribution, 0, None))


def plot_oof_residuals_against_fitted(oof_frequency, residuals):
    """Tegn OOF deviance-residualer mot OOF-predikert skadefrekvens."""
    fig, axis = plt.subplots(figsize=(8.2, 4.2))
    _plot_residuals_with_lowess(
        axis,
        oof_frequency,
        residuals,
        "OOF-predikert skadefrekvens per eksponeringsår",
    )
    axis.set_title("OOF deviance-residualer mot predikert frekvens")
    fig.tight_layout()
    return fig


def plot_oof_residuals_by_continuous_predictor(data, predictors, residuals):
    """Tegn OOF deviance-residualer mot originale kontinuerlige prediktorer."""
    available = [predictor for predictor in predictors if predictor in data]
    if not available:
        raise ValueError("Ingen kontinuerlige prediktorer finnes i modelldatasettet")

    columns = min(2, len(available))
    rows = int(np.ceil(len(available) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(11, 3.2 * rows), squeeze=False)
    for axis, predictor in zip(axes.flat, available, strict=False):
        _plot_residuals_with_lowess(
            axis, data[predictor], residuals, _predictor_label(predictor)
        )
        axis.set_title(f"OOF-residualer mot {_predictor_label(predictor).lower()}")
    for axis in axes.flat[len(available) :]:
        axis.set_visible(False)
    fig.tight_layout()
    return fig


def build_rootogram_table(claims, expected_claims, max_count=10):
    """Bygg observert og Poisson-forventet antall poliseår per skadenivå."""
    claims = np.asarray(claims)
    expected_claims = np.asarray(expected_claims, dtype=float)
    counts = np.arange(max_count)
    expected = [poisson.pmf(count, expected_claims).sum() for count in counts]
    observed = [(claims == count).sum() for count in counts]
    return pd.DataFrame(
        {
            "skadeantall": [*map(str, counts), f"{max_count}+"],
            "observed": [*observed, (claims >= max_count).sum()],
            "expected": [
                *expected,
                poisson.sf(max_count - 1, expected_claims).sum(),
            ],
        }
    )


def plot_rootogram(table):
    """Tegn hengende rootogram etter den tidligere frekvensplot-funksjonen."""
    x = np.arange(len(table))
    sqrt_observed = np.sqrt(table["observed"].to_numpy())
    sqrt_expected = np.sqrt(table["expected"].to_numpy())
    fig, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.bar(
        x,
        sqrt_observed,
        bottom=sqrt_expected - sqrt_observed,
        color="#2f6690",
        width=0.65,
        label="Observert (hengende)",
    )
    axis.plot(
        x,
        sqrt_expected,
        color="#d06b32",
        marker="o",
        markersize=4,
        linewidth=1.6,
        label="Poisson-forventet",
    )
    axis.axhline(0, color="#4d4d4d", linewidth=1)
    axis.set(
        title="OOF-rootogram: observert mot forventet skadeantall",
        xlabel="Skadeantall per poliseår",
        ylabel="Kvadratrot av antall poliseår",
        xticks=x,
        xticklabels=table["skadeantall"],
    )
    axis.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def _plot_residuals_with_lowess(axis, x, residuals, xlabel):
    """Tegn kompakt residualspredning med en stabil LOWESS-kurve."""
    x = np.asarray(x, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    order = np.argsort(x)
    span = np.ptp(x)
    smooth = lowess(
        residuals[order], x[order], frac=0.25, it=0, delta=span * 0.01
    )
    axis.scatter(x, residuals, alpha=0.08, s=9, color="#2f6690", linewidths=0)
    axis.plot(smooth[:, 0], smooth[:, 1], color="#d06b32", linewidth=2)
    axis.axhline(0, color="#4d4d4d", linestyle="--", linewidth=1)
    axis.set(xlabel=xlabel, ylabel="Deviance-residual")


def _predictor_label(predictor):
    """Gi lesbare norske etiketter til de kontinuerlige tariffprediktorene."""
    labels = {
        "driver_age": "Føreralder (år)",
        "log_vehicle_value": "Logaritmert kjøretøyverdi",
        "performance_hp_per_tonne": "Ytelse (hk per tonn)",
    }
    return labels.get(predictor, predictor.replace("_", " ").capitalize())
