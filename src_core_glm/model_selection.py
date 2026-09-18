"""Gjenbrukbare regler for trinnvis GLM-seleksjon på utviklingsdata."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_tweedie_deviance


SELECTION_CRITERIA = ["lavere_oof", "bedre_4_av_5", "gyldig_i_alle_folder"]


def select_candidate_stage(
    cv_results, parent_name, candidate_names, response, sample_weight, power
):
    """Velg beste kandidat som oppfyller den faste treledds CV-regelen."""
    parent = cv_results[parent_name]
    parent_scores = parent["scores"].set_index("fold")
    parent_oof = mean_tweedie_deviance(
        response, parent["oof"], sample_weight=sample_weight, power=power
    )
    rows = []
    for name in candidate_names:
        candidate = cv_results[name]
        is_valid = candidate["valid"] and candidate["oof"].notna().all()
        candidate_scores = candidate["scores"]
        gain = (
            parent_scores["val_deviance"]
            - candidate_scores.set_index("fold")["val_deviance"]
            if "val_deviance" in candidate_scores
            else pd.Series(dtype=float)
        )
        oof_deviance = (
            mean_tweedie_deviance(
                response, candidate["oof"], sample_weight=sample_weight, power=power
            )
            if is_valid
            else np.nan
        )
        rows.append(
            {
                "modell": name,
                "oof_deviance": oof_deviance,
                "lavere_oof": bool(is_valid and oof_deviance < parent_oof),
                "bedre_4_av_5": gain.gt(0).sum() >= 4,
                "gyldig_i_alle_folder": is_valid,
                "cv_feil": candidate["error"] or "",
            }
        )
    table = pd.DataFrame(rows).set_index("modell")
    eligible = table.loc[table[SELECTION_CRITERIA].all(axis=1)]
    selected = eligible["oof_deviance"].idxmin() if len(eligible) else parent_name
    return selected, table


def run_backward_ablation(
    start_name,
    specifications,
    cv_results,
    protected_predictors,
    build_candidate,
    evaluate_candidate,
    select_stage,
):
    """Fjern én kvalifisert råprediktor om gangen og returner beslutningstabellene."""
    current_name = start_name
    removed = []
    tables = []
    while True:
        current_predictors = specifications[current_name]["required_columns"]
        removable = [
            column for column in current_predictors if column not in protected_predictors
        ]
        if not removable:
            return current_name, removed, tables
        candidates = {
            f"{current_name}_uten_{column}": build_candidate(current_name, column)
            for column in removable
        }
        specifications.update(candidates)
        cv_results.update(
            {
                name: evaluate_candidate(specification)
                for name, specification in candidates.items()
            }
        )
        next_name, table = select_stage(current_name, list(candidates))
        table["fjernet_variabel"] = [
            name.rsplit("_uten_", 1)[1] for name in table.index
        ]
        tables.append(table)
        if next_name == current_name:
            return current_name, removed, tables
        removed.append(next_name.rsplit("_uten_", 1)[1])
        current_name = next_name
