"""Fold-plumbing for GLM-kryssvalideringen i ``glm_pricing_models.py``.

De sentrale GLM-byggesteinene (``glm_spec``, ``fit_glm``, ``run_glm``,
``cross_validate_glm``) og ``prepare_design_frame`` (imputasjon, B-09) ligger i
notebooken, siden de er kjernen i metodikken. Denne modulen holder bare
hjelpefunksjonene ``cross_validate_glm`` kaller i hver fold: å bygge de
avledede kolonnene (``derived``, B-28) og å kontrollere at fold-fitten er gyldig
(plan 3.11 pkt. 6). ``prepare_design_frame`` sendes inn som parameter for å
unngå en sirkulær import mot notebooken.
"""

import numpy as np


def prepare_fold_frames(spec, train, frames, prepare_design_frame):
    """Imputer (B-09) og legg til foldvis avledede kolonner, lært på ``train``.

    Parameters
    ----------
    spec : dict
        Modellspesifikasjon fra ``glm_spec``. Brukes for ``required_columns``
        og ``derived``.
    train : pandas.DataFrame
        Treningsdelen av folden. Imputasjonsreglene og de avledede kolonnenes
        tilstand (f.eks. et senter eller en merkeliste) læres herfra.
    frames : list of pandas.DataFrame
        Rammene (typisk trening og validering) som skal imputeres og få de
        avledede kolonnene lagt til, i samme rekkefølge som de sendes inn.
    prepare_design_frame : callable
        ``prepare_design_frame(train, apply, required_columns)`` fra
        notebooken (seksjon 2.6), som fyller manglende verdier lært på
        ``train`` inn i ``apply``.

    Returns
    -------
    prepared : list of pandas.DataFrame
        ``frames`` etter imputasjon og avledede kolonner, i samme rekkefølge.
    derived_state : dict
        Tilstanden hvert element i ``spec["derived"]`` lærte på
        treningsdelen, indeksert på kolonnenavn. Gjenbrukes av
        ``apply_derived_columns`` og av foldkallere som trenger senteret
        eller listen en avledet kolonne ble lært med.
    """
    train_design = prepare_design_frame(train, train, spec["required_columns"])
    derived_state = {
        item["name"]: item["learn"](train_design) for item in spec["derived"]
    }
    prepared = []
    for frame in frames:
        design = prepare_design_frame(train, frame, spec["required_columns"])
        prepared.append(apply_derived_columns(spec, design, derived_state))
    return prepared, derived_state


def apply_derived_columns(spec, frame, derived_state):
    """Legg foldvis avledede kolonner til ``frame`` med tilstand lært i trening.

    Parameters
    ----------
    spec : dict
        Modellspesifikasjon fra ``glm_spec``. ``spec["derived"]`` er en liste
        med dict-er med ``name``, ``learn`` og ``apply``.
    frame : pandas.DataFrame
        Rammen (trenings- eller valideringsdel) de avledede kolonnene skal
        legges til i.
    derived_state : dict
        Tilstanden hvert element i ``spec["derived"]`` lærte på
        treningsdelen, som returnert av ``prepare_fold_frames``.

    Returns
    -------
    pandas.DataFrame
        ``frame`` med de avledede kolonnene lagt til.
    """
    for item in spec["derived"]:
        frame = item["apply"](frame, derived_state[item["name"]])
    return frame


def check_fold_fit(spec, result, train, val, train_prediction, val_prediction):
    """Foldkontrollene fra plan 3.11 pkt. 6.

    En kandidat er bare gyldig når *alle* folder passerer disse kontrollene:
    full rang i designmatrisen, konvergens, samme antall rader inn og ut,
    endelige og positive prediksjoner, endelige parametere og at hvert
    kategorinivå i formelen har positiv respons i treningsfolden.

    Parameters
    ----------
    spec : dict
        Modellspesifikasjon fra ``glm_spec``. Brukes for ``y``, ``weight`` og
        ``base_levels`` (kategoriske prediktorer med basisnivå).
    result : statsmodels GLMResults
        Den estimerte modellen for denne folden.
    train : pandas.DataFrame
        Den imputerte treningsdelen (etter ``prepare_fold_frames``).
    val : pandas.DataFrame
        Den imputerte valideringsdelen (etter ``prepare_fold_frames``).
    train_prediction : array-like
        Predikert respons på ``train``.
    val_prediction : array-like
        Predikert respons på ``val``.

    Returns
    -------
    problems : list of str
        Beskrivelse av hver kontroll som feiler. Tom liste betyr at folden er
        gyldig.
    rank : int
        Rangen til designmatrisen (``result.model.exog``), lagret uavhengig
        av om kontrollen feiler, for diagnostikk.
    """
    problems = []
    exog = result.model.exog
    rank = np.linalg.matrix_rank(exog)
    if rank < exog.shape[1]:
        problems.append(f"rangsvikt ({rank} av {exog.shape[1]} kolonner)")
    if not result.converged:
        problems.append("konvergerte ikke")
    if len(result.model.endog) != len(train) or len(val_prediction) != len(val):
        problems.append("radantall endret i designmatrisen")
    for label, prediction in [
        ("trening", train_prediction),
        ("validering", val_prediction),
    ]:
        values = np.asarray(prediction, dtype=float)
        if not (np.isfinite(values).all() and (values > 0).all()):
            problems.append(f"ikke-endelige eller ikke-positive prediksjoner ({label})")
    if not np.isfinite(result.params).all():
        problems.append("ikke-endelige parametere")
    # Kategoristøtte: hvert nivå i formelen må ha positiv respons i treningsdelen
    weighted_response = train[spec["y"]] * train[spec["weight"]]
    for column in spec["base_levels"]:
        response_by_level = weighted_response.groupby(train[column]).sum()
        empty_levels = response_by_level.index[response_by_level.le(0)].tolist()
        if empty_levels:
            problems.append(
                f"{column} har nivåer uten respons i trening: {empty_levels}"
            )
    return problems, rank
