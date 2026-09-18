"""Folddefinisjon og foldvise uttrekk for severity-fasen.

Foldene er de samme som i frekvensfasen: ``GroupKFold(n_splits=5,
shuffle=True, random_state=100)` på hele modellpopulasjonens ``insured_id``
(B-06). De defineres som indeksmengder, slik at nøyaktig de samme foldene
kan brukes både på hele modellrammen og på severity-delmengden.

``make_full_population_hook`` er et ``fold_hook`` til
``src.glm_core.cross_validate_glm``. Severity-modellen trenes og scores på
foldens positive skadeår, men hooket predikerer i tillegg **alle** poliseår i
valideringsfolden, slik at en senere frekvens–severity-sammenkobling får
komplette OOF-prediksjoner. Prediksjonene påvirker ikke severity-scoren.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.glm_core import apply_derived_columns, prepare_design_frame

N_FOLDS = 5
SEED = 100


def build_group_folds(model_frame, n_splits=N_FOLDS, seed=SEED):
    """Fem faste gruppefolder på ``insured_id`` over hele modellpopulasjonen.

    Foldene bygges på ``model_frame`` (alle poliseår), ikke på
    severity-utvalget, slik at begge poliseårene til samme person havner i
    samme fold uansett om de har skade eller ikke.
    """
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        {
            "fold": f"gruppe_{number + 1}",
            "train_index": model_frame.index[train_positions],
            "val_index": model_frame.index[val_positions],
        }
        for number, (train_positions, val_positions) in enumerate(
            splitter.split(model_frame, groups=model_frame["insured_id"])
        )
    ]


def build_fold_table(folds, model_frame, severity_frame):
    """Foldstørrelser i både modellpopulasjonen og severity-utvalget.

    Kolonnene ``skadeår_trening``/``skadeår_validering`` er tallene planen
    låser (4563/1135, 4527/1171, 4593/1105, 4535/1163, 4574/1124) og er det
    som faktisk kontrolleres. ``personer_overlapp`` skal være 0 i alle
    folder: ingen ``insured_id`` kan opptre i både trening og validering.
    """
    rows = []
    for fold in folds:
        train_index, val_index = fold["train_index"], fold["val_index"]
        severity_train = severity_frame.index.intersection(train_index)
        severity_val = severity_frame.index.intersection(val_index)
        rows.append(
            {
                "fold": fold["fold"],
                "poliseår_trening": len(train_index),
                "poliseår_validering": len(val_index),
                "skadeår_trening": len(severity_train),
                "skadeår_validering": len(severity_val),
                "skader_validering": int(
                    severity_frame.loc[severity_val, "property_claims"].sum()
                ),
                "personer_validering": model_frame.loc[
                    val_index, "insured_id"
                ].nunique(),
                "personer_overlapp": len(
                    set(model_frame.loc[train_index, "insured_id"])
                    & set(model_frame.loc[val_index, "insured_id"])
                ),
            }
        )
    return pd.DataFrame(rows)


def make_full_population_hook(model_frame, severity_frame, folds):
    """Lag et ``fold_hook`` som predikerer alle poliseår i valideringsfolden.

    Hooket gjenbruker foldens estimerte modell og den avledede tilstanden som
    ble lært på severity-treningen, og bygger designrammen for hele
    valideringsfolden med nøyaktig samme imputasjonsregler
    (``prepare_design_frame`` lært på foldens severity-trening).

    Kategorinivåer som bare finnes blant skadefrie poliseår har per definisjon
    ingen severity-treningsdata. ``prepare_design_frame`` kaster da
    ``ValueError``. Det er ikke en feil i severity-modellen, så hooket fanger
    det og markerer folden med ``status``-kolonnen i stedet for å stoppe
    CV-kjøringen. Selve severity-scoren er uberørt uansett.
    """
    train_by_fold = {fold["fold"]: fold["train_index"] for fold in folds}
    val_by_fold = {fold["fold"]: fold["val_index"] for fold in folds}

    def fold_hook(spec, fold_name, result, train_design, val_design, derived_state):
        severity_train = severity_frame.loc[
            severity_frame.index.intersection(train_by_fold[fold_name])
        ]
        all_val = model_frame.loc[
            model_frame.index.intersection(val_by_fold[fold_name])
        ]
        try:
            design = prepare_design_frame(
                severity_train, all_val, spec["required_columns"]
            )
            prediction = result.predict(
                apply_derived_columns(spec, design, derived_state)
            )
            status = "ok"
        except ValueError as error:  # usett nivå blant skadefrie poliseår
            prediction = pd.Series(np.nan, index=all_val.index)
            status = f"avvist: {error}".splitlines()[0]
        return {
            "oof_alle_poliseår": pd.DataFrame(
                {
                    "row_index": all_val.index,
                    "fold": fold_name,
                    "model": spec["name"],
                    "predicted_severity": prediction.to_numpy(),
                    "status": status,
                }
            )
        }

    return fold_hook
