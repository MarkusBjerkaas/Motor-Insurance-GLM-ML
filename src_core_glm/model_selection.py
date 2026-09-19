"""Gjenbrukbare regler for trinnvis GLM-seleksjon på utviklingsdata."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_tweedie_deviance

from src_core_glm.glm_core import glm_spec

SELECTION_CRITERIA = ["lavere_oof", "bedre_4_av_5", "gyldig_i_alle_folder"]


def select_candidate_stage(
    cv_results,
    parent_name,
    candidate_names,
    response,
    sample_weight,
    power,
    se_multiplier=None,
):
    """Velg beste kandidat som oppfyller den faste treledds CV-regelen.

    Med ``se_multiplier`` (f.eks. 1.0) kommer et fjerde krav: snittet av
    fold-gevinstene (forelder minus kandidat, deviance per fold) må være større
    enn ``se_multiplier`` standardfeil, med SE = std / sqrt(antall folder).
    Uten ``se_multiplier`` er regelen uendret.
    """
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
        row = {
            "modell": name,
            "oof_deviance": oof_deviance,
            "lavere_oof": bool(is_valid and oof_deviance < parent_oof),
            "bedre_4_av_5": gain.gt(0).sum() >= 4,
            "gyldig_i_alle_folder": is_valid,
        }
        if se_multiplier is not None:
            standard_error = gain.std(ddof=1) / np.sqrt(len(gain))
            row["snitt_gevinst"] = gain.mean()
            row["se_gevinst"] = standard_error
            # NaN (ugyldig kandidat eller for få folder) gir False
            row["gevinst_over_se"] = bool(gain.mean() > se_multiplier * standard_error)
        row["cv_feil"] = candidate["error"] or ""
        rows.append(row)
    table = pd.DataFrame(rows).set_index("modell")
    criteria = [
        *SELECTION_CRITERIA,
        *(["gevinst_over_se"] if se_multiplier is not None else []),
    ]
    eligible = table.loc[table[criteria].all(axis=1)]
    selected = eligible["oof_deviance"].idxmin() if len(eligible) else parent_name
    return selected, table


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


def interaction_factor(column, data):
    """Faktor i et interaksjonsledd; kategoriske kolonner får samme referansenivå som hovedeffekten."""
    if pd.api.types.is_numeric_dtype(data[column]) and column != "year":
        return column
    base = data.groupby(column)["total_exposure"].sum().idxmax()
    return f"C({column}, Treatment({base!r}))"


def formula_term(term, splines, data):
    """Ett formelledd: interaksjon, spline eller vanlig kolonne."""
    if isinstance(term, tuple):
        return ":".join(interaction_factor(part, data) for part in term)
    if term in splines:
        return f"cr({term}, df={splines[term]}, constraints='center')"
    return term


def build_specification(name, definition, data, settings):
    """Bygg GLM-spesifikasjonen for en modelldefinisjon.

    ``settings`` er argumentene til ``glm_spec`` som er felles for modellfamilien:
    ``y``, ``family``, ``weight``, ``power`` og eventuelt ``base_level_overrides``.
    """
    terms = [
        formula_term(term, definition["splines"], data) for term in definition["terms"]
    ]
    columns = [term for term in definition["terms"] if isinstance(term, str)]
    return glm_spec(name, terms, data=data, required_columns=columns, **settings)


def evaluate_models_parallel(evaluate_model, specifications, cv_results, n_jobs, names):
    """Kjør ``evaluate_model(navn) -> (spesifikasjon, cv_resultat)`` parallelt for flere kandidater.

    Kandidatene er uavhengige, så resultatene er de samme som ved sekvensiell kjøring.
    Resultatene lagres i ``specifications`` og ``cv_results``.
    """
    evaluated = Parallel(n_jobs=n_jobs)(delayed(evaluate_model)(name) for name in names)
    for name, (specification, result) in zip(names, evaluated):
        specifications[name], cv_results[name] = specification, result


def propose_additions(candidate_terms, spline_alternatives=None):
    """Forslagsfunksjon for forover-seleksjon: legg til én kandidatterm om gangen.

    ``spline_alternatives`` er {kolonne: df}. Kolonnen testes da også som spline
    (``+kolonne_df3``) ved siden av den lineære formen.
    """
    spline_alternatives = spline_alternatives or {}

    def propose(definition):
        proposals = {}
        for term in candidate_terms:
            if not can_add(definition, term):
                continue
            terms = [*definition["terms"], term]
            proposals[f"+{term_label(term)}"] = {**definition, "terms": terms}
            if term in spline_alternatives:
                splines = {**definition["splines"], term: spline_alternatives[term]}
                proposals[f"+{term}_df{spline_alternatives[term]}"] = {
                    "terms": terms,
                    "splines": splines,
                }
        return proposals

    return propose


def propose_spline_forms(definition, column, label, degrees_of_freedom):
    """Spline-alternativer for én kolonne som allerede er i modellen (``+label_df3``)."""
    return {
        f"+{label}_df{df}": {
            **definition,
            "splines": {**definition["splines"], column: df},
        }
        for df in degrees_of_freedom
    }


def propose_geography(definition, added, replaced):
    """To geografialternativer: ``added`` i tillegg til, eller i stedet for, ``replaced``."""
    return {
        f"+{added}": {**definition, "terms": [*definition["terms"], added]},
        f"_bytt_til_{added}": {
            **definition,
            "terms": [
                added if term == replaced else term for term in definition["terms"]
            ],
        },
    }


def propose_removals(protected):
    """Forslagsfunksjon for bakover-ablasjon: fjern én term om gangen (unntatt låste)."""

    def propose(definition):
        return {
            f"-{term_label(term)}": remove_term(definition, term)
            for term in definition["terms"]
            if can_remove(definition, term, protected)
        }

    return propose


def run_round(current, proposals, definitions, evaluate_models, select_stage):
    """Evaluer alle forslag mot ``current`` og velg beste som oppfyller seleksjonsregelen.

    ``proposals`` er {endring: modelldefinisjon}. ``evaluate_models(navn_liste)``
    kjører CV for alle forslagene (kandidatene er uavhengige, så den kan kjøre dem
    parallelt). Returnerer valgt modellnavn (``current`` hvis ingen består) og
    beslutningstabellen, der kolonnen ``valgt`` markerer den valgte kandidaten.
    """
    names = {}
    for change, definition in proposals.items():
        names[f"{current}{change}"] = change
        definitions[f"{current}{change}"] = definition
    evaluate_models(list(names))
    selected, table = select_stage(current, list(names))
    table["endring"] = table.index.map(names)
    table["valgt"] = table.index == selected
    return selected, table


def run_stepwise(start_name, propose, definitions, evaluate_models, select_stage):
    """Gjenta ``run_round`` fra den nye modellen til ingen forslag består regelen.

    Forover-seleksjon (``propose_additions``) og bakover-ablasjon
    (``propose_removals``) bruker samme løkke og samme regel.
    Returnerer sluttmodellens navn og én tabell med kolonnen ``runde``.
    """
    current, tables = start_name, []
    while proposals := propose(definitions[current]):
        selected, table = run_round(
            current, proposals, definitions, evaluate_models, select_stage
        )
        tables.append(table.assign(runde=len(tables) + 1))
        if selected == current:
            break
        current = selected
    return current, pd.concat(tables) if tables else pd.DataFrame()
