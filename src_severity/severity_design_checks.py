"""Foldvis designmatrisekontroll av severity-kandidatregisteret (S-10).

Bygger designmatrisen for hver kandidat i hver fold, FØR første kandidatfit,
og kontrollerer kolonneantall (inkl. intercept), rang og minste
kategoristøtte mot planens kandidatregister (``plans/Severity_plan.md``,
"Kandidatregister"). Avvik rapporteres, de rettes ikke: verken parametertall
eller kategorinivåer skal justeres for å få en kandidat til å passe.

Valideringsnivåer uten treningsdata gir ``ValueError`` i
``src.glm_core.prepare_design_frame``; det fanges her per fold og folden
markeres avvist i stedet for å stoppe hele kontrollen.
"""

import numpy as np
import pandas as pd
from patsy import dmatrix


def check_candidate_fold_design(
    specs, severity_frame, folds, prepare_fold_frames, prepare_design_frame
):
    """Bygg og kontroller designmatrisen for hver (kandidat, fold)-kombinasjon.

    Parameters
    ----------
    specs : dict
        Kandidatregisteret, navn -> spesifikasjon fra ``glm_spec``.
    severity_frame : pandas.DataFrame
        Severity-utvalget (positive skadeår), med ``insured_id``.
    folds : list of dict
        Foldene fra ``build_group_folds``, hver med ``train_index``/``val_index``
        over hele modellpopulasjonen.
    prepare_fold_frames : callable
        ``src.glm_core.prepare_fold_frames``, injisert for å unngå duplisering.
    prepare_design_frame : callable
        ``src.glm_core.prepare_design_frame``, sendt videre til
        ``prepare_fold_frames``.

    Returns
    -------
    pandas.DataFrame
        Én rad per kandidat x fold: ``kandidat``, ``fold``, ``kolonner``
        (inkl. intercept), ``rang``, ``full_rang`` (bool), som svarer til
        planens tabell ``PLAN_PARAMETERS``.
    """
    rows = []
    for name, spec in specs.items():
        for fold in folds:
            train = severity_frame.loc[
                severity_frame.index.intersection(fold["train_index"])
            ]
            val = severity_frame.loc[
                severity_frame.index.intersection(fold["val_index"])
            ]
            try:
                (train_design, _val_design), _derived_state = prepare_fold_frames(
                    spec, train, [train, val], prepare_design_frame
                )
                design_matrix = dmatrix(
                    spec["formula"].split("~")[1], train_design, return_type="dataframe"
                )
                n_columns = design_matrix.shape[1]
                rank = int(np.linalg.matrix_rank(design_matrix.to_numpy()))
                # Minste antall unike insured_id blant nivåene i kandidatens
                # kategoriske prediktorer (spec["base_levels"]) — det tallet
                # 50-personerskravet i S-07 gjelder.
                level_supports = [
                    train_design.groupby(column)["insured_id"].nunique().min()
                    for column in spec["base_levels"]
                ]
                min_support = int(min(level_supports)) if level_supports else np.nan
                status = "ok"
            except ValueError as error:
                n_columns = rank = min_support = np.nan
                status = f"avvist: {error}".splitlines()[0]
            rows.append(
                {
                    "kandidat": name,
                    "fold": fold["fold"],
                    "kolonner": n_columns,
                    "rang": rank,
                    "full_rang": bool(pd.notna(rank) and rank == n_columns),
                    "minste_kategoristøtte": min_support,
                    "støtte_ok": bool(pd.notna(min_support) and min_support >= 50),
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def summarize_candidate_design(design_checks):
    """Aggreger den foldvise kontrollen til én rad per kandidat.

    Kolonnene er kolonneantall og minste rang/kategoristøtte over alle folder
    (skal være konstant for kolonneantallet på en gyldig kandidat), samt om
    alle fem folder har full rang og tilstrekkelig kategoristøtte.
    """
    return (
        design_checks.groupby("kandidat", sort=False)
        .agg(
            kolonner=("kolonner", "max"),
            min_rang=("rang", "min"),
            min_kategoristøtte=("minste_kategoristøtte", "min"),
            alle_folder_full_rang=("full_rang", "all"),
            alle_folder_støtte_ok=("støtte_ok", "all"),
            antall_avviste_folder=("status", lambda s: s.ne("ok").sum()),
        )
        .reset_index()
    )
