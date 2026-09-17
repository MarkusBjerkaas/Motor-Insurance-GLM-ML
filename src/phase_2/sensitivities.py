"""Fase 1, seksjon 3.11: sensitiviteter rundt den frosne frekvensmodellen F5.

Fire uavhengige sensitiviteter, én funksjon per notebook-celle: (1) uten
kansellerte, (2) fri eksponeringskoeffisient, (4) kjøreerfaring i stedet for
alder, (5) skadetelling/pukkelen. Ingen av dem endrer hovedmodellen (B-30);
alle avhengigheter som er notebook-lokale (spesifikasjon, fit, design, CV-
funksjoner) sendes inn som argumenter for å unngå sirkulær import.
"""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_tweedie_deviance

from src.glm_diagnostics import build_relativity_table
from src.own_damage_descriptives import AXIS, GRIDLINE, SURFACE
from src.phase_2 import frequency_candidates
from src.phase_2.frequency_plots import _MODEL_COLORS, _style_axes, plot_fold_curves
from src.phase_2.frequency_tables import (
    build_candidate_table,
    build_effect_table,
    format_comparison_table,
    paired_deviance_gain,
    report_near_misses,
    summarize_prediction_subsets,
)


def test_without_cancelled(
    frequency_cv,
    model_id,
    final_spec,
    model_frame,
    cv_folds,
    final_design,
    final_fit,
    final_oof,
    cross_validate_glm,
    fit_glm,
    paired_improvement,
):
    """(1) Samme spesifikasjon trent bare på aktive rader i hver foldtrening.

    Returns
    -------
    pandas.DataFrame
        Tabell 1: A/E og prediksjonsratio per delmengde (aktive/kansellerte/alle).
    """
    assert final_spec is frequency_cv[model_id]["spec"]
    cancelled = model_frame["policy_status"].eq("C")
    print(
        "Status i train-poolen: "
        + ", ".join(
            f"{k} {v}" for k, v in model_frame["policy_status"].value_counts().items()
        )
    )
    active_folds = [
        fold
        | {
            "train_index": fold["train_index"][
                ~cancelled.loc[fold["train_index"]].to_numpy()
            ]
        }
        for fold in cv_folds
    ]
    no_cancel_result = cross_validate_glm(final_spec, model_frame, active_folds)
    assert no_cancel_result["valid"], no_cancel_result["error"]
    no_cancel_oof = no_cancel_result["oof"]

    # Parvis per fold på de samme aktive valideringsradene (gevinst > 0: uten kansellerte bedre)
    active_fold_scores = []
    for fold in cv_folds:
        rows = fold["val_index"][~cancelled.loc[fold["val_index"]].to_numpy()]
        part = model_frame.loc[rows]
        for name, rate in (
            ("hovedmodell", final_oof),
            ("uten_kansellerte", no_cancel_oof),
        ):
            active_fold_scores.append(
                {
                    "model": name,
                    "fold": fold["fold"],
                    "val_weight": part["total_exposure"].sum(),
                    "val_deviance": mean_tweedie_deviance(
                        part["claim_frequency"],
                        rate.loc[rows],
                        sample_weight=part["total_exposure"],
                        power=1,
                    ),
                }
            )
    active_paired = paired_improvement(
        pd.DataFrame(active_fold_scores), "hovedmodell", "uten_kansellerte"
    )
    print(
        f"Aktive valideringsrader, uten kansellerte mot hovedmodell: poolet gevinst "
        f"{active_paired['pooled_forbedring']:.6f}, snitt {active_paired['snitt_forbedring']:.6f}, "
        f"SE {active_paired['standardfeil']:.6f}, bedre i {active_paired['folder_med_forbedring']}/5 folder"
    )

    # Relativitetsendring: samme spesifikasjon på aktive rader i hele train-poolen
    active_fit = fit_glm(final_spec, final_design.loc[~cancelled])
    linear_effects = [p for p in final_fit.params.index if not p.startswith("cr(")]
    log_change = (active_fit.params - final_fit.params).loc[linear_effects]
    changed = log_change.drop("Intercept").abs().sort_values(ascending=False).head(3)
    # Nivå måles som eksponeringsvektet prediksjonsratio, ikke intercept: spline-basisen
    # (cr) bærer også nivå, så interceptet alene er ikke tolkbart
    exposure = final_design["total_exposure"]
    level_ratio = np.average(
        active_fit.predict(final_design), weights=exposure
    ) / np.average(final_fit.predict(final_design), weights=exposure)
    print(
        f"Nivå uten kansellerte (in-sample, alle rader): ×{level_ratio:.3f}. Største "
        "relativitetsendringer: "
        + "; ".join(
            f"{p} ×{np.exp(log_change[p]):.3f} ({abs(log_change[p]) / final_fit.bse[p]:.1f} cluster-SE)"
            for p in changed.index
        )
    )
    return (
        summarize_prediction_subsets(
            model_frame,
            {"hovedmodell": final_oof, "uten_kansellerte": no_cancel_oof},
            {
                "aktive": ~cancelled,
                "kansellerte": cancelled,
                "alle": cancelled | ~cancelled,
            },
        )
        .assign(
            prediksjon_ratio=lambda t: t["AE_hovedmodell"] / t["AE_uten_kansellerte"]
        )
        .round(4)
    )


def test_free_exposure_coefficient(final_spec, final_design, final_fit, cluster_groups):
    """(2) Antallsmodell med offset log(e) og et fritt ledd δ·log(e), cluster-KI."""
    exposure_design = final_design.assign(
        log_exposure=np.log(final_design["total_exposure"])
    )
    free_exposure_fit = smf.glm(
        "property_claims ~" + final_spec["formula"].split("~")[1] + " + log_exposure",
        data=exposure_design,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=exposure_design["log_exposure"],
    ).fit(cov_type="cluster", cov_kwds={"groups": cluster_groups})
    delta = free_exposure_fit.params["log_exposure"]
    delta_low, delta_high = free_exposure_fit.conf_int().loc["log_exposure"]
    linear_effects = [p for p in final_fit.params.index if not p.startswith("cr(")]
    relativity_shift = (
        (free_exposure_fit.params.drop("log_exposure") - final_fit.params)
        .loc[linear_effects]
        .drop("Intercept")
    )
    print(
        f"δ = {delta:.3f} (95 % cluster-KI {delta_low:.3f} til {delta_high:.3f}); "
        f"total eksponeringskoeffisient 1+δ = {1 + delta:.3f}. "
        f"Andel poliseår med e < 1: {final_design['total_exposure'].lt(1).mean():.1%}. "
        f"In-sample deviance-endring (antallsskala) {final_fit.deviance - free_exposure_fit.deviance:.1f}. "
        f"Største relativitetsendring: {relativity_shift.abs().idxmax()} "
        f"×{np.exp(relativity_shift[relativity_shift.abs().idxmax()]):.3f}"
    )


def test_experience_vs_age(
    frequency_cv, model_id, final_spec, model_frame, cross_validate_candidate
):
    """(4) Erstatter alder med kjøreerfaring (lineær/df3/df4) i den frosne modellen.

    Returns
    -------
    comparison_table : pandas.DataFrame
        Tabell 2: sammenligning mot den frosne modellen per form.
    figure : matplotlib.figure.Figure
        Plott 1: erfaringskurvene ved siden av alderskurven i F5.
    """
    experience_ids = []
    for form in frequency_candidates.CONTINUOUS_FORMS:
        forms = {
            column: current
            for column, current in final_spec["forms"].items()
            if column != "driver_age"
        } | {"driving_experience_years": form}
        experience_id = f"S4_exp-{frequency_candidates.FORM_LABELS[form]}"
        frequency_cv[experience_id] = cross_validate_candidate(
            frequency_candidates.build_frequency_candidate(
                experience_id,
                "S4",
                [
                    "experience" if b == "driver" else b
                    for b in final_spec["feature_blocks"]
                ],
                forms=forms,
                parent_id=model_id,
            )
        )
        experience_ids.append(experience_id)

    experience_comparisons = pd.DataFrame(
        [
            frequency_candidates.compare_models(frequency_cv, model_id, experience_id)
            for experience_id in experience_ids
        ]
    )
    report_near_misses(experience_comparisons)
    comparison_table = (
        format_comparison_table(experience_comparisons)
        .drop(columns=["Retning", "Innenfor_1SE", "Bestar_B08", "near_miss_3of5"])
        .join(
            build_candidate_table(frequency_cv, experience_ids)[
                ["former", "parametere", "oof_deviance", "oof_d2"]
            ],
            on="Kandidat",
        )
        .set_index(["Referanse", "Kandidat"])
    )

    figure = plot_fold_curves(
        pd.concat(
            [frequency_cv[m]["curves"] for m in experience_ids]
            + [
                frequency_candidates.fit_full_curves(frequency_cv[m]["spec"])["curves"]
                for m in experience_ids
            ]
            + [
                frequency_cv[model_id]["curves"],
                frequency_candidates.fit_full_curves(final_spec)["curves"],
            ],
            ignore_index=True,
        ).query("feature in ['driving_experience_years', 'driver_age']"),
        features=["driving_experience_years", "driver_age"],
        support_frame=model_frame,
        feature_labels={
            "driving_experience_years": "Kjøreerfaring (år)",
            "driver_age": "Førers alder (F5)",
        },
        title="Sensitivitet: kjøreerfaring (lineær/df3/df4) mot alder i F5 (±2 SE)",
    )
    return comparison_table, figure


def test_claim_count_sensitivity(
    model_id,
    model_frame,
    final_spec,
    final_design,
    final_fit,
    final_oof,
    fit_glm,
    cluster_groups,
    cross_validate_glm,
    cv_folds,
    time_fold,
    time_val,
    time_predictions,
    without_year,
    effect_table,
    final_state,
    paired_improvement,
):
    """(5) Skadetelling/pukkelen: fullt antall, kappet ved 3, og skade ja/nei.

    SENSITIVITET: F5 holdes fast (B-30); bare responsen byttes i kopier av
    spesifikasjonen, og ingen modell endres.

    Returns
    -------
    matplotlib.figure.Figure
        Plott 2: relativiteter med tre skadetellinger, punkt + 95 % cluster-KI.
    """
    RESPONSE_LABELS = {
        "main": "fullt antall",
        "cap3": "kappet ved 3",
        "claimant": "skade ja/nei",
    }
    RATE_COLUMNS = {
        "main": "claim_frequency",
        "cap3": "claim_frequency_cap3",
        "claimant": "claim_frequency_claimant",
    }
    capped_claims = model_frame["property_claims"].clip(upper=3)
    claimant = model_frame["property_claims"].gt(0).astype(float)
    count_frame = model_frame.assign(
        claims_cap3=capped_claims,
        claim_frequency_cap3=capped_claims / model_frame["total_exposure"],
        claim_frequency_claimant=claimant / model_frame["total_exposure"],
    )

    def with_response(spec, name):
        """Kopi av ``spec`` med respons ``name``; høyresiden (F5) er uendret."""
        if name == "main":
            return spec
        y = RATE_COLUMNS[name]
        return spec | {
            "name": f"{spec['name']}_{name}",
            "model_id": f"{spec['model_id']}_{name}",
            "y": y,
            "formula": f"{y} ~" + spec["formula"].split("~")[1],
        }

    # --- In-sample relativiteter med cluster-KI på hele train-poolen ---
    response_design = final_design.assign(
        **{column: count_frame[column] for column in RATE_COLUMNS.values()}
    )
    count_fits = {
        name: fit_glm(
            with_response(final_spec, name),
            response_design,
            cluster_groups=cluster_groups,
        )
        for name in RATE_COLUMNS
    }
    assert np.allclose(count_fits["main"].params, final_fit.params, rtol=1e-8)

    def response_effects(name):
        """Relativiteter (uten basisnivåer) og alder ved gridendene mot referansen, med cluster-KI."""
        spec, fit = with_response(final_spec, name), count_fits[name]
        table = build_effect_table(
            build_relativity_table(spec, fit),
            linear_scales={"log_vehicle_value": (np.log(1.1), "+10 % kjøretøyverdi")},
        ).dropna(subset=["standardfeil"])
        table.index = [
            change if variable == "log_vehicle_value" else f"{variable} {level}"
            for (variable, level), change in zip(table.index, table["endring"])
        ]
        # Alder som i 3.10: deltametode med uskalert cluster-kovarians (B-23)
        view = SimpleNamespace(
            model=fit.model,
            params=fit.params,
            cov_params=fit.cov_params,
            pearson_chi2=1.0,
            df_resid=1.0,
            scale=1.0,
        )
        ends = (
            frequency_candidates.frequency_curve_hook(
                spec, "full", view, response_design, response_design, final_state
            )["curves"]
            .query("feature == 'driver_age'")
            .sort_values("x")
            .iloc[[0, -1]]
        )
        reference = frequency_candidates.CURVE_GRIDS["driver_age"]["reference"]
        ages = pd.DataFrame(
            {
                "koeffisient": np.log(ends["relative"].to_numpy()),
                "standardfeil": ends["se_log"].to_numpy(),
            },
            index=[f"alder {x:.0f} mot {reference:.0f} år" for x in ends["x"]],
        )
        ages["relativitet"] = np.exp(ages["koeffisient"])
        ages["ki_lav"] = np.exp(ages["koeffisient"] - 1.96 * ages["standardfeil"])
        ages["ki_høy"] = np.exp(ages["koeffisient"] + 1.96 * ages["standardfeil"])
        columns = ["koeffisient", "standardfeil", "relativitet", "ki_lav", "ki_høy"]
        return pd.concat([table[columns], ages[columns]])

    count_effects = {name: response_effects(name) for name in RATE_COLUMNS}
    main_effects = count_effects["main"]
    # Kontroll mot 3.10: samme relativiteter for fullt antall
    assert np.allclose(
        main_effects.loc["policy_type COMP_N", "relativitet"],
        effect_table.loc[("policy_type", "COMP_N"), "relativitet"],
    )
    print(
        "In-sample relativiteter, "
        + "; ".join(
            f"{RESPONSE_LABELS[name]}: COMP_N ×{effects.loc['policy_type COMP_N', 'relativitet']:.3f}, "
            f"P ×{effects.loc['business_type P', 'relativitet']:.3f}"
            for name, effects in count_effects.items()
        )
    )
    for name in ["cap3", "claimant"]:
        effects = count_effects[name]
        # Skift i koeffisient målt i hovedmodellens cluster-SE; SE-forhold mot hovedmodellen
        shift = (effects["koeffisient"] - main_effects["koeffisient"]) / main_effects[
            "standardfeil"
        ]
        se_ratio = effects["standardfeil"] / main_effects["standardfeil"]
        print(
            f"{RESPONSE_LABELS[name]} mot fullt antall: største skift "
            + "; ".join(
                f"{label} ×{effects.loc[label, 'relativitet']:.3f} ({shift[label]:+.1f} SE)"
                for label in shift.abs().sort_values(ascending=False).index[:3]
            )
            + ". Største endring i cluster-SE: "
            + "; ".join(
                f"{label} ×{se_ratio[label]:.2f}"
                for label in np.log(se_ratio)
                .abs()
                .sort_values(ascending=False)
                .index[:3]
            )
        )

    # Plott 2: punkt + 95 % cluster-KI per relativitet, én farge per respons
    labels = main_effects.index
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(labels) + 1.4), facecolor=SURFACE)
    positions = np.arange(len(labels))
    ax.axvline(1, color=AXIS, linewidth=1)
    for offset, name, color in zip([-0.22, 0.0, 0.22], RATE_COLUMNS, _MODEL_COLORS):
        effects = count_effects[name].loc[labels]
        ax.errorbar(
            effects["relativitet"],
            positions + offset,
            xerr=[
                effects["relativitet"] - effects["ki_lav"],
                effects["ki_høy"] - effects["relativitet"],
            ],
            fmt="o",
            markersize=4,
            capsize=2,
            linewidth=1,
            color=color,
            label=RESPONSE_LABELS[name],
        )
    ax.set_xscale("log")
    ticks = [0.8, 1, 1.25, 1.5, 2, 3, 4]
    ax.set_xticks(ticks, [f"{tick:g}".replace(".", ",") for tick in ticks])
    ax.minorticks_off()
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    _style_axes(
        ax,
        f"Sensitivitet: relativiteter i {model_id} med tre skadetellinger",
        "",
        "Relativitet (log-skala), 95 % cluster-KI",
    )
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    plt.close(fig)

    # --- Rangering: gruppe-CV og tidsfold med nivå rekalibrert i treningsfolden ---
    def calibration_hook(
        spec, fold_name, result, train_design, val_design, derived_state
    ):
        """Faktorer i treningsfolden: faktisk antall (fullt og kappet) / forventet antall."""
        expected = (result.predict(train_design) * train_design["total_exposure"]).sum()
        return {
            "calibration": pd.DataFrame(
                {
                    "fold": [fold_name],
                    "to_main": [train_design["property_claims"].sum() / expected],
                    "to_cap3": [train_design["claims_cap3"].sum() / expected],
                }
            )
        }

    def rescaled_predictions(result, folds, target):
        """Valideringsrater skalert med faktoren fra treningsfolden til respons ``target``."""
        fold_of_row = pd.concat(
            [pd.Series(fold["fold"], index=fold["val_index"]) for fold in folds]
        )
        factors = result["calibration"].set_index("fold")[f"to_{target}"]
        return result["oof"].loc[fold_of_row.index] * fold_of_row.map(factors)

    def ranking_comparison(results, folds, baseline, candidate, target):
        """Parvis gevinst (basis − kandidat) på respons ``target``; > 0 betyr kandidat bedre."""
        rate = RATE_COLUMNS[target]
        predictions = {
            name: rescaled_predictions(results[name], folds, target)
            for name in (baseline, candidate)
        }
        rows = predictions[baseline].index
        cluster = paired_deviance_gain(
            count_frame.loc[rows, rate],
            count_frame.loc[rows, "total_exposure"],
            predictions[baseline],
            predictions[candidate].loc[rows],
            count_frame.loc[rows, "insured_id"],
        )
        row = {
            "baseline": baseline,
            "candidate": candidate,
            "target": target,
            "pooled_gain": cluster["gevinst"],
            "se_cluster": cluster["SE_cluster"],
            "z_cluster": cluster["z"],
        }
        if len(folds) > 1:  # fold-SE og folder bedre, som paired_improvement i (1)
            fold_scores = [
                {
                    "model": name,
                    "fold": fold["fold"],
                    "val_weight": part["total_exposure"].sum(),
                    "val_deviance": mean_tweedie_deviance(
                        part[rate],
                        predictions[name].loc[part.index],
                        sample_weight=part["total_exposure"],
                        power=1,
                    ),
                }
                for fold in folds
                for part in [count_frame.loc[fold["val_index"]]]
                for name in predictions
            ]
            paired = paired_improvement(pd.DataFrame(fold_scores), baseline, candidate)
            assert np.isclose(
                paired["pooled_forbedring"], cluster["gevinst"], rtol=1e-6
            )
            row |= {
                "mean_gain": paired["snitt_forbedring"],
                "se_gain": paired["standardfeil"],
                "folds_improved": paired["folder_med_forbedring"],
                "near_miss_3of5": bool(
                    paired["pooled_forbedring"] > 0
                    and paired["snitt_forbedring"] > paired["standardfeil"]
                    and paired["folder_med_forbedring"] == 3
                ),
            }
        return row

    time_spec = without_year(model_id)
    count_cv, count_time = {}, {}
    for name in RATE_COLUMNS:
        count_cv[name] = cross_validate_glm(
            with_response(final_spec, name), count_frame, cv_folds, calibration_hook
        )
        count_time[name] = cross_validate_glm(
            with_response(time_spec, name), count_frame, time_fold, calibration_hook
        )
        for result in (count_cv[name], count_time[name]):
            assert result["valid"], result["error"]
    # Kontroller: fullt antall gjenskaper hovedstigen, og Poisson-balansen gir faktor 1
    assert np.allclose(count_cv["main"]["oof"], final_oof, rtol=1e-8)
    assert np.allclose(
        count_time["main"]["oof"].loc[time_val.index],
        time_predictions[model_id],
        rtol=1e-8,
    )
    assert np.allclose(count_cv["main"]["calibration"]["to_main"], 1, atol=1e-6)
    print(
        "Rekalibreringsfaktor til fullt antall (gruppefolder; tidsfold): "
        + "; ".join(
            f"{RESPONSE_LABELS[name]} {count_cv[name]['calibration']['to_main'].min():.3f}–"
            f"{count_cv[name]['calibration']['to_main'].max():.3f}; "
            f"{count_time[name]['calibration']['to_main'].iloc[0]:.3f}"
            for name in ["cap3", "claimant"]
        )
    )

    COUNT_COMPARISONS = [
        ("main", "cap3", "main"),
        ("main", "claimant", "main"),
        ("cap3", "main", "cap3"),  # motsatt retning: hovedmodellen på kappet antall
    ]
    group_rankings = pd.DataFrame(
        [ranking_comparison(count_cv, cv_folds, *c) for c in COUNT_COMPARISONS]
    )
    time_rankings = pd.DataFrame(
        [ranking_comparison(count_time, time_fold, *c) for c in COUNT_COMPARISONS]
    )
    report_near_misses(group_rankings)
    for group, in_time in zip(group_rankings.itertuples(), time_rankings.itertuples()):
        print(
            f"{RESPONSE_LABELS[group.candidate]} mot {RESPONSE_LABELS[group.baseline]}, "
            f"scoret på {RESPONSE_LABELS[group.target]} (gevinst > 0: kandidat bedre). "
            f"Gruppe-CV: poolet {group.pooled_gain:.6f}, snitt {group.mean_gain:.6f}, "
            f"fold-SE {group.se_gain:.6f}, cluster-SE {group.se_cluster:.6f} "
            f"(z {group.z_cluster:+.1f}), bedre i {group.folds_improved}/5 folder. "
            f"Tidsfold 2022→2023: {in_time.pooled_gain:.6f}, cluster-SE "
            f"{in_time.se_cluster:.6f} (z {in_time.z_cluster:+.1f})"
        )
    return fig
