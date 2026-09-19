"""Diagnostikk for Tweedie-modellen i ``04_tweedie``."""

import statsmodels.api as sm


def tweedie_deviance_residuals(response, prediction, weight, power):
    """Signerte, eksponeringsvektede Tweedie deviance-residualer (statsmodels ``resid_dev``)."""
    family = sm.families.Tweedie(var_power=power, link=sm.families.links.Log())
    return family.resid_dev(response, prediction, var_weights=weight)
