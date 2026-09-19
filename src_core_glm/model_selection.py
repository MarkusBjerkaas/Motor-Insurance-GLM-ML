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


# --- Trinnvis seleksjon på modelldefinisjoner --------------------------------
# En modelldefinisjon er {"terms": [...], "splines": {kolonne: df}}. En term er
# en kolonne (str) eller en interaksjon mellom to kolonner (tuple). Låste
# variabler, kjerne og tillegg er bare lister med termer i notebooken.


def term_label(term):
    """Kort navn på en term, brukt i modellnavn og tabeller."""
    return term if isinstance(term, str) else "*".join(term)


def can_add(definition, term):
    """En term legges bare til én gang, og en interaksjon krever begge hovedeffektene."""
    if term in definition["terms"]:
        return False
    return isinstance(term, str) or all(part in definition["terms"] for part in term)


def can_remove(definition, term, protected):
    """Låste termer og hovedeffekter som en gjenværende interaksjon bruker kan ikke fjernes."""
    used_by_interaction = any(
        isinstance(other, tuple) and term in other for other in definition["terms"]
    )
    return term not in protected and not used_by_interaction


def remove_term(definition, term):
    """Definisjonen uten ``term`` (og uten termens spline)."""
    terms = [other for other in definition["terms"] if other != term]
    splines = {col: df for col, df in definition["splines"].items() if col != term}
    return {"terms": terms, "splines": splines}


def propose_additions(candidate_terms):
    """Forslagsfunksjon for forover-seleksjon: legg til én kandidatterm om gangen."""

    def propose(definition):
        return {
            f"+{term_label(term)}": {
                **definition,
                "terms": [*definition["terms"], term],
            }
            for term in candidate_terms
            if can_add(definition, term)
        }

    return propose


def propose_removals(protected):
    """Forslagsfunksjon for bakover-ablasjon: fjern én term om gangen (unntatt låste)."""

    def propose(definition):
        return {
            f"-{term_label(term)}": remove_term(definition, term)
            for term in definition["terms"]
            if can_remove(definition, term, protected)
        }

    return propose


def run_round(current, proposals, definitions, evaluate_model, select_stage):
    """Evaluer alle forslag mot ``current`` og velg beste som oppfyller treleddsregelen.

    ``proposals`` er {endring: modelldefinisjon}. Returnerer valgt modellnavn
    (``current`` hvis ingen består) og beslutningstabellen.
    """
    names = {}
    for change, definition in proposals.items():
        names[f"{current}{change}"] = change
        definitions[f"{current}{change}"] = definition
        evaluate_model(f"{current}{change}")
    selected, table = select_stage(current, list(names))
    table["endring"] = table.index.map(names)
    return selected, table


def run_stepwise(start_name, propose, definitions, evaluate_model, select_stage):
    """Gjenta ``run_round`` fra den nye modellen til ingen forslag består regelen.

    Forover-seleksjon (``propose_additions``) og bakover-ablasjon
    (``propose_removals``) bruker samme løkke og samme regel.
    Returnerer sluttmodellens navn og én tabell med kolonnen ``runde``.
    """
    current, tables = start_name, []
    while proposals := propose(definitions[current]):
        selected, table = run_round(
            current, proposals, definitions, evaluate_model, select_stage
        )
        tables.append(table.assign(runde=len(tables) + 1))
        if selected == current:
            break
        current = selected
    return current, pd.concat(tables) if tables else pd.DataFrame()


def summarize_selection_path(path, cv_results, response, sample_weight, power):
    """Én rad per valgt modell i stigen: pooled OOF-deviance etter hvert steg."""
    return pd.DataFrame(
        [
            {
                "steg": step,
                "modell": name,
                "oof_deviance": mean_tweedie_deviance(
                    response, cv_results[name]["oof"], sample_weight=sample_weight, power=power
                ),
            }
            for step, name in path
        ]
    )
