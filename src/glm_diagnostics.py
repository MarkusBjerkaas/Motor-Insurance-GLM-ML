"""Tabeller og diagnostikk for GLM-modelleringen i ``glm_pricing_models.py``.

Modellspesifikasjoner, CV-folder og CV-sløyfen ligger i notebooken. Her ligger
bare funksjoner som beskriver data og folder og oppsummerer resultatene.
"""

import re

import numpy as np
import pandas as pd
from sklearn.metrics import mean_tweedie_deviance


def build_data_overview(model_frame, severity_frame):
    """Nøkkeltall for modellrammen og severity-delmengden (seksjon 1.1)."""
    exposure = model_frame["total_exposure"].sum()
    claim_years = int(model_frame["property_claims"].gt(0).sum())
    return pd.Series(
        {
            "Poliseår": len(model_frame),
            "Eksponering": exposure,
            "Skadeantall": model_frame["property_claims"].sum(),
            "Porteføljefrekvens": model_frame["property_claims"].sum() / exposure,
            "Ren premie per eksponeringsår (EUR)": model_frame[
                "property_incurred"
            ].sum()
            / exposure,
            "Skadeår i severity": len(severity_frame),
            "Skadeår utelatt fra severity (incurred ≤ 0,01)": claim_years
            - len(severity_frame),
            "Snittskade, vektet med antall (EUR)": severity_frame[
                "property_incurred"
            ].sum()
            / severity_frame["property_claims"].sum(),
        },
        name="Verdi",
    )


def build_fold_summary(frame, folds):
    """Beskriv valideringsdelen i hver fold: volum, nivå og sammensetning.

    Brukes til å sjekke at foldene er sammenlignbare, særlig andelen
    kansellerte poliser og andelen 2023-rader, som begge flytter frekvensnivået.
    """
    rows = []
    for fold in folds:
        part = frame.loc[fold["val_index"]]
        rows.append(
            {
                "fold": fold["fold"],
                "poliseår": len(part),
                "unike_poliser": part["insured_id"].nunique(),
                "eksponering": part["total_exposure"].sum(),
                "skadeantall": part["property_claims"].sum(),
                "frekvens": part["property_claims"].sum()
                / part["total_exposure"].sum(),
                "ren_premie": part["property_incurred"].sum()
                / part["total_exposure"].sum(),
                "andel_kansellert_prosent": part["policy_status"].eq("C").mean() * 100,
                "andel_2023_prosent": part["year"].eq(2023).mean() * 100,
            }
        )
    return pd.DataFrame(rows).set_index("fold").round(3)


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
            "familie": type(spec["family"]).__name__,
            "rader": len(data),
            "parametere": len(result.params),
            "deviance": deviance(prediction),
            "d2": 1 - deviance(prediction) / deviance(null_prediction),
            "balanse": (prediction * weight).sum() / (response * weight).sum(),
        },
        name=spec["name"],
    )


def build_relativity_table(spec, result):
    """Relativitetstabell i tariffstil: exp(β) med 95 %-KI per variabel og nivå.

    Patsy-navn som ``C(policy_type, Treatment('COMP_E'))[T.COMP_N]`` deles opp i
    variabel og nivå. Basisnivåene legges til med relativitet 1, slik at hver
    variabel vises komplett. KI-ene bruker kovariansen modellen ble estimert med
    (cluster-robust når ``fit_glm`` fikk ``cluster_groups``).
    """
    confidence = np.exp(result.conf_int())
    rows = []
    for term, coefficient in result.params.items():
        match = re.fullmatch(r"C\((\w+).*\)\[T\.(.+)\]", term)
        variable, level = match.groups() if match else (term, "")
        rows.append(
            {
                "variabel": variable,
                "nivå": level,
                "koeffisient": coefficient,
                "standardfeil": result.bse[term],
                "relativitet": np.exp(coefficient),
                "ki_lav": confidence.loc[term, 0],
                "ki_høy": confidence.loc[term, 1],
            }
        )
    for variable, base in spec["base_levels"].items():
        rows.append(
            {"variabel": variable, "nivå": f"{base} (basis)", "relativitet": 1.0}
        )

    variable_order = {name: i for i, name in enumerate(["Intercept", *spec["x"]])}
    table = pd.DataFrame(rows)
    table["_order"] = table["variabel"].map(variable_order)
    table["_base_last"] = ~table["nivå"].str.endswith("(basis)")
    return (
        table.sort_values(["_order", "_base_last", "nivå"], kind="stable")
        .drop(columns=["_order", "_base_last"])
        .set_index(["variabel", "nivå"])
    )


def summarize_cv_scores(fold_scores):
    """Oppsummer fold-scorene per modell.

    ``fold_scores`` har én rad per (modell, fold), som returnert av
    ``cross_validate_glm``. Standardfeilen er fold-standardavviket delt på
    roten av antall folder. Med fem folder er den et grovt mål.
    """
    grouped = fold_scores.groupby("model", sort=False)
    n_folds = grouped.size()
    return pd.DataFrame(
        {
            "folder": n_folds,
            "train_deviance": grouped["train_deviance"].mean(),
            "oof_deviance": grouped["val_deviance"].mean(),
            "oof_deviance_se": grouped["val_deviance"].std(ddof=1) / np.sqrt(n_folds),
            "oof_d2": grouped["val_d2"].mean(),
            "oof_balanse": grouped["val_balance"].mean(),
        }
    )
