"""Rapporttabeller for trinnvis GLM-seleksjon (felles for frekvens, severity og Tweedie)."""

import pandas as pd
from sklearn.metrics import mean_tweedie_deviance

from src_core_glm.model_selection import term_label


def build_forward_selection_table(stage_tables):
    """Én rad per valgt steg i forover-seleksjonen (tabellene fra ``run_round``/``run_stepwise``).

    ``oof_deviance`` er pooled OOF for modellen etter steget. ``snitt_gevinst`` er
    snittet av fold-gevinstene mot foreldremodellen, ``se_gevinst`` standardfeilen,
    og ``gevinst_i_se`` forholdet mellom dem (over ca. 2 er gevinsten tydelig).
    """
    steps = pd.concat(stage_tables)
    chosen = steps.loc[
        steps["valgt"], ["endring", "oof_deviance", "snitt_gevinst", "se_gevinst"]
    ].reset_index(drop=True)
    chosen["gevinst_i_se"] = chosen["snitt_gevinst"] / chosen["se_gevinst"]
    chosen.index = chosen.index + 1
    return chosen.rename_axis("steg")


def label_terms(definition):
    """Termene i en modelldefinisjon som lesbare navn; splines får df i parentes."""
    return [
        f"{term_label(term)} (spline df {definition['splines'][term]})"
        if term in definition["splines"]
        else term_label(term)
        for term in definition["terms"]
    ]


def build_top_models_table(
    cv_results, definitions, chosen_name, response, weight, power, top_n=3
):
    """De ``top_n`` evaluerte modellene med lavest pooled OOF-deviance.

    Indeksen er modellnavnet. ``endring`` viser hva som skiller modellen fra
    ``chosen_name``, og ``delta_mot_valgt`` er OOF-deviance minus den valgtes
    (negativ = lavere).
    """

    def oof_deviance(name):
        return mean_tweedie_deviance(
            response, cv_results[name]["oof"], sample_weight=weight, power=power
        )

    chosen_labels = label_terms(definitions[chosen_name])
    rows = []
    for name, result in cv_results.items():
        if not (result["valid"] and result["oof"].notna().all()):
            continue
        labels = label_terms(definitions[name])
        changes = [f"+ {label}" for label in labels if label not in chosen_labels] + [
            f"− {label}" for label in chosen_labels if label not in labels
        ]
        rows.append(
            {
                "modell": name,
                "endring": " ".join(changes) or "valgt modell",
                "variabler": ", ".join(labels),
                "parametere": int(result["scores"]["n_params"].max()),
                "oof_deviance": oof_deviance(name),
            }
        )
    table = (
        pd.DataFrame(rows).set_index("modell").sort_values("oof_deviance").head(top_n)
    )
    table["delta_mot_valgt"] = table["oof_deviance"] - oof_deviance(chosen_name)
    return table
