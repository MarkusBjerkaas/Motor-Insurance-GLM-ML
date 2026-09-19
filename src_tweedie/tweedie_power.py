"""Iterativ Pearson-estimering av Tweedie-power $p$."""

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def solve_pearson_power(result, low=1.01, high=1.99):
    """Løs Pearson-likningen for $p$ gitt tilpassede forventninger.

    $$\\sum_i \\left(\\frac{w_i (y_i-\\mu_i)^2}{\\hat\\phi\\,\\mu_i^{p}} - 1\\right)\\log\\mu_i = 0,
    \\qquad \\hat\\phi = \\frac{1}{n-k}\\sum_i \\frac{w_i (y_i-\\mu_i)^2}{\\mu_i^{p}}$$

    Skrevet eksplisitt fordi ``GLM.estimate_tweedie_power`` i statsmodels
    multipliserer «−1»-leddet med vekten. Likningen har da forventning
    $1-w_i \\neq 0$ og er ikke skalainvariant, så den passer ikke til
    eksponeringsvekter. Her er $w_i$ = ``var_weights`` (eksponering).
    """
    y, mu = np.asarray(result.model.endog), np.asarray(result.mu)
    weight, log_mu = np.asarray(result.model.var_weights), np.log(mu)

    def estimating_equation(power):
        weighted_sq = weight * (y - mu) ** 2 / mu**power
        phi = weighted_sq.sum() / result.df_resid
        return np.mean((weighted_sq / phi - 1) * log_mu)

    return brentq(estimating_equation, low, high)


def estimate_tweedie_power(
    fit_at_power, start=1.5, low=1.01, high=1.99, tol=1e-3, max_iter=20
):
    """Estimer $p$ ved å veksle mellom å fitte modellen og løse Pearson-likningen.

    1. Fit modellen med power ``p`` (første gang ``start``).
    2. Estimer ny ``p`` fra de tilpassede forventningene med
       ``solve_pearson_power`` (``brentq`` på intervallet ``[low, high]``).
    3. Gjenta til endringen i ``p`` er under ``tol``, eller ``max_iter`` er nådd.

    Parameters
    ----------
    fit_at_power : callable
        ``fit_at_power(power)`` returnerer en tilpasset statsmodels
        ``GLMResults`` for den (ferdig låste) modellen med Tweedie-power
        ``power``. Funksjonen velger ikke variabler selv.
    start, low, high, tol, max_iter : float or int
        Startverdi, søkegrenser (innenfor 1 < p < 2), toleranse for endring
        i ``p`` og maksimalt antall iterasjoner.

    Returns
    -------
    dict
        ``power`` (siste estimat), ``n_iter``, ``converged`` (endring < ``tol``)
        og ``history`` (én rad per iterasjon). Konvergens og gyldighet
        kontrolleres separat, med ``assert_power_estimate``.
    """
    power, history, converged = start, [], False
    for iteration in range(1, max_iter + 1):
        result = fit_at_power(power)
        try:
            new_power = solve_pearson_power(result, low, high)
        except ValueError as error:  # brentq: ingen rot i [low, high]
            raise ValueError(
                f"Pearson-likningen har ingen løsning for p i [{low}, {high}] "
                f"(iterasjon {iteration}, p inn = {power:.4f})."
            ) from error
        change = abs(new_power - power)
        history.append(
            {
                "iteration": iteration,
                "power_in": power,
                "power_out": new_power,
                "change": change,
            }
        )
        power = new_power
        if change < tol:
            converged = True
            break
    return {
        "power": power,
        "n_iter": iteration,
        "converged": converged,
        "history": pd.DataFrame(history),
    }
