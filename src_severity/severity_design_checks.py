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
from patsy import PatsyError, dmatrix


def check_candidate_fold_design(
    specs,
    severity_frame,
    folds,
    prepare_fold_frames,
    prepare_design_frame,
    expected_parameters=None,
    locked_references=None,
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
    expected_parameters : dict or None, optional
        Forventet antall designkolonner per kandidat, inkludert intercept.
        Når oppgitt, rapporteres det også om hvert folddesign har dette
        parametertallet.
    locked_references : dict or None, optional
        Låste referansenivåer som skal samsvare med ``spec["base_levels"]``.

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
            reference_mismatches = []
            reference_present = references_match = parameter_count_ok = False
            expected = (expected_parameters or {}).get(name)
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

                # Kontroller både at referansenivået finnes i treningsfolden
                # og at formelen bruker samme nivå som spesifikasjonen.
                for column, reference in spec["base_levels"].items():
                    if column not in train_design:
                        reference_mismatches.append(f"{column} mangler")
                        continue
                    if reference not in set(train_design[column].dropna()):
                        reference_mismatches.append(
                            f"{column} mangler referanse {reference!r} i trening"
                        )
                    expected_term = f"C({column}, Treatment({reference!r}))"
                    if expected_term not in spec["formula"]:
                        reference_mismatches.append(
                            f"{column} bruker ikke spec-referanse {reference!r}"
                        )
                    if locked_references is not None and (
                        column not in locked_references
                        or locked_references[column] != reference
                    ):
                        reference_mismatches.append(
                            f"{column} avviker fra låst referanse "
                            f"{locked_references.get(column)!r}"
                        )

                reference_present = not any(
                    "mangler" in mismatch for mismatch in reference_mismatches
                )
                references_match = reference_present and not any(
                    "bruker ikke" in mismatch or "avviker" in mismatch
                    for mismatch in reference_mismatches
                )
                parameter_count_ok = expected is None or n_columns == expected
                status = "ok"
            except (KeyError, ValueError, np.linalg.LinAlgError, PatsyError) as error:
                n_columns = rank = min_support = np.nan
                reference_present = references_match = parameter_count_ok = False
                status = f"avvist: {error}".splitlines()[0]
            full_rank = bool(pd.notna(rank) and rank == n_columns)
            support_ok = bool(not spec["base_levels"] or (
                pd.notna(min_support) and min_support >= 50
            ))
            if status == "ok" and not full_rank:
                status = f"avvist: rangsvikt ({rank} av {n_columns} kolonner)"
            if status == "ok" and not support_ok:
                status = f"avvist: kategoristøtte under 50 ({min_support})"
            if status == "ok" and not reference_present:
                status = "avvist: låst referansenivå mangler i trening"
            if status == "ok" and not references_match:
                status = "avvist: låst referansenivå samsvarer ikke med spec"
            if status == "ok" and not parameter_count_ok:
                status = (
                    f"avvist: {n_columns} kolonner, forventet "
                    f"{expected}"
                )
            rows.append(
                {
                    "kandidat": name,
                    "fold": fold["fold"],
                    "kolonner": n_columns,
                    "rang": rank,
                    "full_rang": full_rank,
                    "minste_kategoristøtte": min_support,
                    "støtte_ok": support_ok,
                    "referansenivåer_tilstede": reference_present,
                    "referansenivåer_samsvarer": references_match,
                    "referanseavvik": "; ".join(reference_mismatches),
                    "forventede_kolonner": expected,
                    "parametertall_ok": parameter_count_ok,
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
    rows = []
    for candidate, group in design_checks.groupby("kandidat", sort=False):
        column_values = group["kolonner"]
        columns_constant = (
            column_values.notna().all() and column_values.nunique(dropna=False) == 1
        )
        rows.append(
            {
                "kandidat": candidate,
                "antall_folder": len(group),
                "fem_folder": len(group) == 5,
                "kolonner": column_values.max(),
                "kolonner_min": column_values.min(),
                "kolonner_maks": column_values.max(),
                "kolonneantall_konstant": bool(columns_constant),
                "min_rang": group["rang"].min(),
                "min_kategoristøtte": group["minste_kategoristøtte"].min(),
                "alle_folder_full_rang": bool(group["full_rang"].all()),
                "alle_folder_støtte_ok": bool(group["støtte_ok"].all()),
                "alle_folder_referansenivåer_tilstede": bool(
                    group["referansenivåer_tilstede"].all()
                ),
                "alle_folder_referansenivåer_samsvarer": bool(
                    group["referansenivåer_samsvarer"].all()
                ),
                "alle_folder_parametertall_ok": bool(group["parametertall_ok"].all()),
                "antall_avviste_folder": int(group["status"].ne("ok").sum()),
            }
        )
    return pd.DataFrame(rows)
