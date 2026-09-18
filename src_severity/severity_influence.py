"""Innflytelsesdiagnostikk: den låste topp-fem-personer-sensitiviteten (S-04).

Se ``plans/Severity_plan.md``, seksjonen "Etter seleksjon: EUR-kalibrering og
stabilitet", avsnittet om innflytelse. Planens krav ordrett: i hver
treningsfold fjernes de fem personene med høyest samlet registrert kostnad,
S0 og finalisten refittes med **uendret** spesifikasjon og predikerer den
**opprinnelige** valideringsfolden. Dette er maksimalt ti diagnostiske
refittinger (2 modeller × 5 folder); ingen av dem erstatter hovedmodellen.

Alle funksjoner er rene: de tar imot ferdige ``severity_frame``, ``folds`` og
spesifikasjoner som argumenter og laster ingen data selv. Ingen funksjon her
leser eller rører 2024-data — ``severity_frame`` skal alltid komme fra
``src_severity.severity_data.build_severity_inputs`` bygget på
``src.model_data.build_development_frames()`` (kun 2022-2023).

Fit-mekanikken (``prepare_fold_frames`` + ``fit_glm``) er identisk med
``src.glm_core.cross_validate_glm``, bare med et redusert treningsutvalg og
en fast (ikke ny) valideringsfold — ingen ny CV-ramme skrives her.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.glm_core import fit_glm
from src.own_damage_descriptives import ACCENT, CONTEXT, INK_MUTED, SURFACE
from src_severity.severity_calibration import _LINE_COLORS, _style_axes
from src_severity.severity_cv import pooled_deviance

N_PERSONS = 5  # låst i planen
STOP_TOTAL = 0.05  # planens stoppregel: pooled forventet kostnad totalt
STOP_PRODUCT = 0.10  # planens stoppregel: pooled forventet kostnad per produkt


def identify_top_cost_persons(severity_frame, fold, n_persons=N_PERSONS):
    """De ``n_persons`` personene med høyest samlet registrert kostnad i foldens trening.

    "Samlet registrert kostnad" er ``sum(property_incurred)`` per
    ``insured_id``, beregnet **innenfor treningsdelen av folden** (planens
    presisering — ikke over hele severity-utvalget). Likhet brytes
    deterministisk på ``insured_id`` (stigende), slik at resultatet er
    reproduserbart.
    """
    train = severity_frame.loc[severity_frame.index.intersection(fold["train_index"])]
    ranked = (
        train.groupby("insured_id")
        .agg(
            samlet_kostnad=("property_incurred", "sum"),
            antall_skadeår=("property_incurred", "size"),
            skadeantall=("property_claims", "sum"),
        )
        .reset_index()
        .sort_values(["samlet_kostnad", "insured_id"], ascending=[False, True])
    )
    top = ranked.head(n_persons).reset_index(drop=True)
    top.insert(0, "fold", fold["fold"])
    return top


def run_influence_sensitivity(
    specs,
    severity_frame,
    folds,
    fit_kwargs,
    prepare_fold_frames,
    prepare_design_frame,
    n_persons=N_PERSONS,
):
    """Refit hver spesifikasjon i ``specs`` per fold, uten foldens topp-fem-personer.

    Valideringsfolden er uendret — bare treningsdelen mistes de fem
    personene. Spesifikasjonen (``spec["formula"]``, ``spec["base_levels"]``)
    er den samme som hovedmodellen brukte; ingenting reestimeres fra det
    reduserte utvalget utover selve GLM-fitten.

    Returns
    -------
    dict
        ``oof``: modellnavn -> ``pandas.Series`` med sensitivitetens
        prediksjoner, indeksert som ``severity_frame`` (NaN utenfor
        valideringsfoldene). ``log``: én rad per refitting. ``antall_refittinger``:
        faktisk antall GLM-fits (skal være ``len(specs) * len(folds)``, høyst 10).
    """
    oof = {
        name: pd.Series(np.nan, index=severity_frame.index, name=name) for name in specs
    }
    log_rows = []

    for fold in folds:
        train_full = severity_frame.loc[
            severity_frame.index.intersection(fold["train_index"])
        ]
        val = severity_frame.loc[severity_frame.index.intersection(fold["val_index"])]

        # Topp-fem er uavhengig av hvilken modell som fittes -- beregnes én gang per fold.
        top_persons = identify_top_cost_persons(severity_frame, fold, n_persons)
        removed_ids = set(top_persons["insured_id"])
        removed_mask = train_full["insured_id"].isin(removed_ids)
        train_reduced = train_full.loc[~removed_mask]
        assert (
            not set(train_reduced["insured_id"]) & removed_ids
        )  # personene er faktisk borte

        for name, spec in specs.items():
            (train_design, val_design), _ = prepare_fold_frames(
                spec, train_reduced, [train_reduced, val], prepare_design_frame
            )
            result = fit_glm(
                spec, train_design, check_convergence=False, fit_kwargs=fit_kwargs
            )
            prediction = result.predict(val_design)
            oof[name].loc[val_design.index] = prediction.to_numpy()
            log_rows.append(
                {
                    "modell": name,
                    "fold": fold["fold"],
                    "fjernede_personer": ", ".join(
                        str(person_id) for person_id in sorted(removed_ids)
                    ),
                    "rader_fjernet": int(removed_mask.sum()),
                    "skadeantall_fjernet": int(
                        train_full.loc[removed_mask, "property_claims"].sum()
                    ),
                    "kostnad_fjernet": float(
                        train_full.loc[removed_mask, "property_incurred"].sum()
                    ),
                    "konvergerte": bool(result.converged),
                }
            )

    log = pd.DataFrame(log_rows)
    n_refits = len(specs) * len(folds)
    assert len(log) == n_refits, f"Forventet {n_refits} refittinger, fikk {len(log)}"
    assert n_refits <= 10, (
        f"Budsjettet er maksimalt 10 diagnostiske refittinger, fikk {n_refits}"
    )
    return {"oof": oof, "log": log, "antall_refittinger": n_refits}


def _expected_cost_row(
    frame, base_prediction, sensitivity_prediction, model, variable, level
):
    """Forventet kostnad (planens definisjon) og pooled deviance, hoved vs. sensitivitet, for én rad."""
    claims = frame["property_claims"]
    cost_main = float((claims * base_prediction).sum())
    cost_sensitivity = float((claims * sensitivity_prediction).sum())
    return {
        "variabel": variable,
        "nivå": level,
        "modell": model,
        "forventet_kostnad_hoved": cost_main,
        "forventet_kostnad_sensitivitet": cost_sensitivity,
        "relativ_endring": (cost_sensitivity - cost_main) / cost_main,
        "pooled_deviance_hoved": pooled_deviance(frame, base_prediction),
        "pooled_deviance_sensitivitet": pooled_deviance(frame, sensitivity_prediction),
    }


def build_influence_table(severity_frame, base_predictions, sensitivity_predictions):
    """Pooled forventet kostnad, totalt og per produkt, hovedmodell mot sensitivitet.

    Forventet kostnad = ``sum(property_claims * predikert severity)`` (planens
    definisjon). ``relativ_endring`` er en andel (0,03 = 3 %). Sammenligningen
    gjøres på radene der begge modellene har en OOF-prediksjon.
    """
    rows = []
    for name in base_predictions:
        base = base_predictions[name]
        sensitivity = sensitivity_predictions[name]
        common = base.notna() & sensitivity.notna()
        frame = severity_frame.loc[common]
        base_common, sensitivity_common = base[common], sensitivity[common]

        rows.append(
            _expected_cost_row(
                frame, base_common, sensitivity_common, name, "Totalt", "Totalt"
            )
        )
        for level, group in frame.groupby("policy_type"):
            rows.append(
                _expected_cost_row(
                    group,
                    base_common.loc[group.index],
                    sensitivity_common.loc[group.index],
                    name,
                    "policy_type",
                    level,
                )
            )
    return pd.DataFrame(rows)


def evaluate_influence_stop(influence_table):
    """Planens innflytelses-stoppregel: totalt > 5 % eller ett produkt > 10 %."""
    rows = []
    for model, group in influence_table.groupby("modell"):
        total_change = group.loc[group["variabel"] == "Totalt", "relativ_endring"].iloc[
            0
        ]
        rows.append(
            {
                "modell": model,
                "regel": "Totalt > 5 %",
                "utløst": bool(abs(total_change) > STOP_TOTAL),
                "største_endring": total_change,
                "detalj": "Totalt",
            }
        )
        product_rows = group.loc[group["variabel"] == "policy_type"]
        worst = product_rows.loc[product_rows["relativ_endring"].abs().idxmax()]
        rows.append(
            {
                "modell": model,
                "regel": "Produkt > 10 %",
                "utløst": bool(abs(worst["relativ_endring"]) > STOP_PRODUCT),
                "største_endring": worst["relativ_endring"],
                "detalj": f"policy_type={worst['nivå']}",
            }
        )
    return pd.DataFrame(rows)


def build_top_persons_table(severity_frame, folds, n_persons=N_PERSONS):
    """De fjernede personene for alle folder, med deres andel av foldens treningskostnad.

    Hjelpefunksjon som samler ``identify_top_cost_persons`` på tvers av
    foldene og legger til hvor stor del av foldens totale severity-
    treningskostnad de fem personene faktisk utgjør -- input til
    ``plot_influence``, panel (b).
    """
    tables = []
    for fold in folds:
        train = severity_frame.loc[
            severity_frame.index.intersection(fold["train_index"])
        ]
        top = identify_top_cost_persons(severity_frame, fold, n_persons)
        train_cost = train["property_incurred"].sum()
        top["andel_av_treningskostnad"] = top["samlet_kostnad"].sum() / train_cost
        tables.append(top)
    return pd.concat(tables, ignore_index=True)


def plot_influence(influence_table, top_persons_table):
    """Én figur: (a) relativ endring i forventet kostnad mot 5 %/10 %-grensene,
    (b) de fjernede personenes andel av foldens treningskostnad.
    """
    fig, (ax_change, ax_share) = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)

    # (a) Relativ endring i forventet kostnad, totalt og per produkt, per modell.
    models = list(dict.fromkeys(influence_table["modell"]))
    products = sorted(
        influence_table.loc[
            influence_table["variabel"] == "policy_type", "nivå"
        ].unique()
    )
    categories = ["Totalt"] + products
    width = 0.8 / len(models)
    positions = np.arange(len(categories))

    for offset, (model, color) in enumerate(zip(models, _LINE_COLORS)):
        subset = influence_table[influence_table["modell"] == model].set_index("nivå")
        values = [
            subset.loc[category, "relativ_endring"] * 100 for category in categories
        ]
        ax_change.bar(
            positions + offset * width - width * (len(models) - 1) / 2,
            values,
            width=width,
            color=color,
            label=model,
        )
    ax_change.axhline(
        STOP_TOTAL * 100,
        color=ACCENT,
        linestyle="--",
        linewidth=1,
        label="5 %-grense (totalt)",
    )
    ax_change.axhline(-STOP_TOTAL * 100, color=ACCENT, linestyle="--", linewidth=1)
    ax_change.axhline(
        STOP_PRODUCT * 100,
        color=INK_MUTED,
        linestyle=":",
        linewidth=1,
        label="10 %-grense (produkt)",
    )
    ax_change.axhline(-STOP_PRODUCT * 100, color=INK_MUTED, linestyle=":", linewidth=1)
    ax_change.set_xticks(positions)
    ax_change.set_xticklabels(categories)
    _style_axes(
        ax_change,
        "Endring i forventet kostnad, topp-fem-personer fjernet",
        "Relativ endring (%)",
    )
    ax_change.legend(frameon=False, fontsize=8, loc="best")

    # (b) Fjernet kostnad som andel av foldens treningskostnad.
    fold_share = top_persons_table.drop_duplicates("fold")
    ax_share.bar(
        fold_share["fold"], fold_share["andel_av_treningskostnad"] * 100, color=CONTEXT
    )
    ax_share.tick_params(axis="x", rotation=45, labelsize=8)
    _style_axes(ax_share, "De fem personenes andel av treningskostnaden", "Andel (%)")

    fig.tight_layout()
    return fig
