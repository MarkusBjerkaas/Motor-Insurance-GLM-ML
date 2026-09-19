"""Worked example: fra risikoprofil til ren premie med de låste frekvens- og severitymodellene.

Begge GLM-ene har log-link og ingen interaksjoner, så effekten av hver ratingfaktor er
multiplikativ. Eksempelet endrer én faktor om gangen fra en referanserisiko og viser
hvordan frekvens, severity og ren premie bygges opp. Modulen bruker bare de låste
modellene; ingen 2024-data leses.
"""

import numpy as np
import pandas as pd

from src.model_comparison.locked_models import SCORE_YEAR


def _score_profile(frequency_model, severity_model, profile):
    """Frekvens, severity og ren premie for én risikoprofil."""
    frame = pd.DataFrame(
        [
            {
                **profile,
                "log_vehicle_value": np.log(profile["vehicle_value"]),
                "year": SCORE_YEAR,
            }
        ]
    )
    frequency = float(frequency_model(frame).iloc[0])
    severity = float(severity_model(frame).iloc[0])
    return frequency, severity, frequency * severity


def build_worked_example(frequency_model, severity_model, reference, steps):
    """Bygg profilen trinn for trinn og returner én rad per trinn.

    ``steps`` er en liste av (etikett, {kolonne: ny verdi}). Relativitetene er forholdet
    til forrige trinn. Som kontroll beregnes hver endring også alene fra referansen;
    uten interaksjoner må de to gi samme faktor (rekkefølgen spiller ingen rolle).
    """
    profile = dict(reference)
    base = _score_profile(frequency_model, severity_model, profile)
    rows = [("Referanserisiko", np.nan, np.nan, np.nan, *base)]
    previous = base
    for label, change in steps:
        profile.update(change)
        current = _score_profile(frequency_model, severity_model, profile)
        alone = _score_profile(frequency_model, severity_model, {**reference, **change})
        combined = current[2] / previous[2]
        assert np.isclose(combined, alone[2] / base[2]), f"Interaksjon i steg «{label}»"
        rows.append(
            (
                label,
                current[0] / previous[0],
                current[1] / previous[1],
                combined,
                *current,
            )
        )
        previous = current
    return pd.DataFrame(
        rows,
        columns=[
            "Steg",
            "Frekvens ×",
            "Severity ×",
            "Kombinert ×",
            "Frekvens",
            "Severity (EUR)",
            "Ren premie (EUR/år)",
        ],
    )
