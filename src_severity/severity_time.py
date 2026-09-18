"""Årskontroll (2022 -> 2023) og miksstandardisering for severity-finalisten.

Se ``plans/Severity_plan.md``, seksjonen "Avgrenset tidskontroll og modenhet".
Alle funksjoner er rene: de tar imot et ferdig ``severity_frame``/``model_frame``
(fra ``src_severity.severity_data.build_severity_inputs``, som selv bare kan
bygges fra ``src.model_data.build_development_frames()`` — 2022-2023) og laster
ingen data selv. Ingen funksjon her åpner et nytt kandidatsøk: S0 er den låste
referansen, og finalistens termer (``candidate_terms``) sendes inn utenfra.

**Hard begrensning:** ingen funksjon her leser eller berører 2024. Tidskontrollen
er nøyaktig kalenderovergangen 2022 -> 2023.

I tidskontrollen fittes begge spesifikasjonene UTEN årsledd
(``drop_year_term``), og all preprocessing (imputasjon, evt. splineknuter i
S0s aldersspline) læres bare fra 2022-delen, via
``src.glm_core.prepare_fold_frames``. ``prepare_fold_frames`` og
``prepare_design_frame`` injiseres som parametere (samme mønster som
``src_severity.severity_design_checks.check_candidate_fold_design``), slik at
denne modulen ikke importerer eller dupliserer dem.

Planens forbehold gjelder alle funksjonene her: dette er ÉN kalenderovergang.
At tidsstoppkriteriet ikke utløses, betyr «ingen forverring påvist i denne
tidsdelingen», ikke dokumentert tidsstabilitet. Miksstandardiseringen
(``run_mix_standardization``) er forklarende diagnostikk, ikke et
valideringsresultat.
"""

import re
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_tweedie_deviance

from src.glm_core import fit_glm, glm_spec
from src.own_damage_descriptives import (
    ACCENT,
    AXIS,
    CONTEXT,
    CURRENCY_ZERO_TOLERANCE,
    INK_MUTED,
    INK_PRIMARY,
    SURFACE,
)
from src_severity.severity_calibration import _LINE_COLORS, _style_axes
from src_severity.severity_scoring import (
    GAMMA_FAMILY,
    cluster_bootstrap_ci,
    paired_gamma_gain,
)

# Referansespesifikasjonen S0 (med årsledd — fjernes eksplisitt av
# `drop_year_term` før fitting her). Referansens termer og de låste
# referansenivåene sendes inn fra notebooken (`reference_terms`/`base_levels`)
# i stedet for å dupliseres her — den låste spesifikasjonen skal finnes ett
# sted, og en kopi her ville kunne drive fra notebooken uten at noe feiler.

FIT_FAMILY = sm.families.Gamma(sm.families.links.Log())  # log-link for fitting
GAMMA_POWER = 2

# Kandidatblokkene fra planens register som IKKE inngår i S0 eller C1 (brukes
# av `plot_oof_residuals` til å sjekke om et utelatt mønster henger igjen).
CANDIDATE_BLOCK_COLUMNS = [
    "municipality_type",
    "circulation_area",
    "performance_hp_per_tonne",
    "seat_category",
]


def drop_year_term(terms):
    """Fjern årsleddet fra en termliste. Trivielt, men gjør intensjonen eksplisitt."""
    return [term for term in terms if term != "year"]


def _raw_columns(terms):
    """Råkolonnene bak formelleddene; ``cr(x, ...)`` bidrar med kolonnen ``x``."""
    return [
        re.match(r"cr\((\w+)", term)[1] if term.startswith("cr(") else term
        for term in terms
    ]


def _coverage_share(train, reference, column):
    """Andel av ``reference`` utenfor ``train``s observerte kovariatområde.

    Numerisk kolonne: andel utenfor ``[min, max]`` observert i ``train``.
    Kategorisk kolonne: andel med et nivå som ikke finnes i ``train``.
    """
    if pd.api.types.is_numeric_dtype(train[column]):
        low, high = train[column].min(), train[column].max()
        outside = ~reference[column].between(low, high)
    else:
        seen_levels = set(train[column])
        outside = ~reference[column].isin(seen_levels)
    return outside.mean()


def run_year_control(
    candidate_terms,
    reference_terms,
    base_levels,
    severity_frame,
    fit_kwargs,
    prepare_fold_frames,
    prepare_design_frame,
):
    """Fit S0 og finalisten (uten årsledd) på 2022 og prediker begge år.

    Preprocessing læres KUN fra 2022-delen (``prepare_fold_frames`` lærer på
    ``train`` og anvender reglene på ``[train, val]``). Ingen kandidatseleksjon
    gjentas her — S0 og ``candidate_terms`` er låst utenfra.

    Returns
    -------
    dict
        ``predictions`` (modellnavn -> ``pandas.Series`` over 2023-radene,
        beholdt for bakoverkompatibilitet), ``predictions_by_year``
        (modellnavn -> år -> ``pandas.Series``; samme 2022-fit brukes for
        både 2022 og 2023),
        ``scores`` (DataFrame: modell, parametere, pooled_deviance_2023,
        konvergerte) og ``modeller`` (dict modellnavn -> statsmodels GLMResults,
        fittet på 2022).
    """
    train = severity_frame.loc[severity_frame["year"] == 2022]
    val = severity_frame.loc[severity_frame["year"] == 2023]
    assert len(train) + len(val) == len(severity_frame)  # ingen andre år lekker inn

    term_lists = {
        "S0": drop_year_term(reference_terms),
        "C1": drop_year_term(candidate_terms),
    }
    predictions, predictions_by_year, rows, models = {}, {}, [], {}
    for name, terms in term_lists.items():
        assert "year" not in terms  # eksplisitt: intet årsledd i tidskontrollen
        spec = glm_spec(
            name,
            terms,
            "average_severity",
            train,
            FIT_FAMILY,
            "property_claims",
            GAMMA_POWER,
            required_columns=_raw_columns(terms),
            base_level_overrides=base_levels,
        )
        (train_design, val_design), _derived_state = prepare_fold_frames(
            spec, train, [train, val], prepare_design_frame
        )
        result = fit_glm(
            spec, train_design, check_convergence=False, fit_kwargs=fit_kwargs
        )
        # Viktig: dette er én og samme 2022-fit.  2022-prediksjonen er
        # in-sample, mens 2023-prediksjonen er out-of-time.
        prediction_train = result.predict(train_design)
        prediction_val = result.predict(val_design)
        predictions[name] = prediction_val  # gammel API: kun 2023
        predictions_by_year[name] = {2022: prediction_train, 2023: prediction_val}
        models[name] = result
        rows.append(
            {
                "modell": name,
                "parametere": result.model.exog.shape[1],
                "pooled_deviance_2023": mean_tweedie_deviance(
                    val_design["average_severity"],
                    prediction_val,
                    sample_weight=val_design["property_claims"],
                    power=GAMMA_POWER,
                ),
                "konvergerte": bool(result.converged),
            }
        )
    return {
        "predictions": predictions,
        "predictions_by_year": predictions_by_year,
        "scores": pd.DataFrame(rows),
        "modeller": models,
    }


def build_level_shift_table(severity_frame_2023, predictions, severity_frame_2022):
    """Globalt nivåskift (A/E) per modell, samt faktisk severity 2022 mot 2023.

    A/E = sum(property_incurred) / sum(property_claims * predikert severity),
    beregnet på 2023-delen for hver modell i ``predictions``.

    OBS avvik fra oppgavens signatur: ``severity_frame_2022`` er lagt til som
    eget argument. Uten den finnes det ingen måte å beregne det faktiske
    nivåskiftet mellom årene på — modellene i tidskontrollen har intet
    årsledd og genererer derfor ingen egen 2022-vs-2023-prediksjon.
    """
    actual_cost_2023 = severity_frame_2023["property_incurred"].sum()
    rows = []
    for model, prediction in predictions.items():
        claims = severity_frame_2023.loc[prediction.index, "property_claims"]
        expected_cost = (claims * prediction).sum()
        rows.append(
            {
                "modell": model,
                "faktisk_kostnad_2023": actual_cost_2023,
                "forventet_kostnad_2023": expected_cost,
                "A_E": actual_cost_2023 / expected_cost,
            }
        )
    table = pd.DataFrame(rows)

    severity_2022 = (
        severity_frame_2022["property_incurred"].sum()
        / severity_frame_2022["property_claims"].sum()
    )
    severity_2023 = actual_cost_2023 / severity_frame_2023["property_claims"].sum()
    table["severity_2022"] = severity_2022
    table["severity_2023"] = severity_2023
    table["forhold_2023_2022"] = severity_2023 / severity_2022
    return table


def _year_predictions(severity_frame, predictions):
    """Normaliser gamle og nye prediksjonsstrukturer til ``år -> modell``.

    Den gamle tidskontroll-API-en har ``modell -> Series`` (kun 2023).
    ``run_year_control`` returnerer i tillegg ``modell -> år -> Series``.
    Denne adapteren gjør at eldre notebook-kall fortsatt virker, samtidig som
    segmentdriften kan bruke de faste prediksjonene for begge år.
    """
    if not predictions:
        return {}
    first = next(iter(predictions.values()))
    if isinstance(first, Mapping):
        years = sorted(
            {int(year) for by_year in predictions.values() for year in by_year}
        )
        return {
            year: {
                model: by_year[year]
                for model, by_year in predictions.items()
                if year in by_year
            }
            for year in years
        }

    years = sorted(severity_frame["year"].unique())
    if len(years) != 1:
        raise ValueError(
            "Prediksjoner for flere år må gis som modell -> år -> Series."
        )
    return {int(years[0]): predictions}


def _total_ae_by_year(frame, year_predictions):
    """Beregn total A/E per år og modell fra faste prediksjoner."""
    totals = {}
    for year, model_predictions in year_predictions.items():
        year_frame = frame.loc[frame["year"].eq(year)]
        totals[year] = {}
        for model, prediction in model_predictions.items():
            part = year_frame.loc[prediction.index]
            totals[year][model] = part["property_incurred"].sum() / (
                part["property_claims"] * prediction
            ).sum()
    return totals


def build_segment_ae_table(severity_frame_2023, predictions, total_ae):
    """A/E per år, modell og segment, normalisert mot samme års total A/E.

    ``predictions`` kan enten være det gamle ``modell -> Series``-formatet
    (én årgang, typisk 2023), eller ``modell -> år -> Series`` fra
    ``run_year_control["predictions_by_year"]``. I det siste tilfellet bygges
    den låste år × modell × segment-tabellen med ``policy_type`` og
    ``skadeantallsgruppe``. ``A_E_total`` er total A/E i samme år og modell;
    ``A_E_relativ`` skiller nivåskift fra segmentdrift.

    ``total_ae`` beholdes som argument for bakoverkompatibilitet. For den nye
    flerårsvarianten kan det være ``None`` eller en dict på formen
    ``{år: {modell: total_A_E}}``; verdier beregnes fra prediksjonene dersom de
    ikke sendes inn.
    """
    frame = severity_frame_2023.copy()
    frame["skadeantallsgruppe"] = np.select(
        [frame["property_claims"].eq(1), frame["property_claims"].between(2, 3)],
        ["N=1", "N=2-3"],
        default="N>=4",
    )
    year_predictions = _year_predictions(frame, predictions)
    computed_totals = _total_ae_by_year(frame, year_predictions)

    def total_for(year, model):
        if total_ae is None:
            return computed_totals[year][model]
        if isinstance(total_ae, pd.DataFrame):
            match = total_ae.loc[
                total_ae["modell"].eq(model) & total_ae["year"].eq(year), "A_E"
            ]
            return float(match.iloc[0])
        if year in total_ae and isinstance(total_ae[year], Mapping):
            return total_ae[year][model]
        # Gammelt format: modell -> total A/E for den ene årgangen.
        return total_ae[model]

    rows = []
    for year, model_predictions in year_predictions.items():
        year_frame = frame.loc[frame["year"].eq(year)]
        for model, prediction in model_predictions.items():
            segment_frame = year_frame.loc[prediction.index].assign(_predicted=prediction)
            total_ae_same_year = total_for(year, model)
            for variable in ["policy_type", "skadeantallsgruppe"]:
                for level, part in segment_frame.groupby(variable):
                    actual = part["property_incurred"].sum()
                    expected = (part["property_claims"] * part["_predicted"]).sum()
                    ae = actual / expected
                    rows.append(
                        {
                            "year": year,
                            "variabel": variable,
                            "nivå": level,
                            "modell": model,
                            "unike_personer": part["insured_id"].nunique(),
                            "antall_skadeår": len(part),
                            "skadeantall": int(part["property_claims"].sum()),
                            "A_E": ae,
                            "A_E_total": total_ae_same_year,
                            "A_E_relativ": ae / total_ae_same_year,
                        }
                    )
    return pd.DataFrame(rows)


def evaluate_segment_drift_stop(segment_drift_table, min_persons=100):
    """Evaluer det låste stoppkriteriet for segmentdrift over tid.

    Stopp krever samtidig minst ``min_persons`` personer i hvert år, et strengt
    ``Q > 1,20`` eller ``Q < 0,80``, og et bootstrapintervall som utelukker
    1. Tabellen returneres med ``utløst`` og en kort, etterprøvbar begrunnelse.
    """
    table = segment_drift_table.copy()
    support = table["unike_personer_2022"].ge(min_persons) & table[
        "unike_personer_2023"
    ].ge(min_persons)
    material = (table["Q"] > 1.20) | (table["Q"] < 0.80)
    excludes_one = (table["Q_lav"] > 1.0) | (table["Q_høy"] < 1.0)
    table["utløst"] = support & material & excludes_one
    table["begrunnelse"] = np.select(
        [table["utløst"], ~support, ~material, ~excludes_one],
        [
            "Segmentdrift-stopp: støtte, materiell endring og CI som utelukker 1.",
            f"Ingen stopp: færre enn {min_persons} personer i minst ett år.",
            "Ingen stopp: Q ligger innenfor eller på ±20 %-grensen.",
            "Ingen stopp: bootstrapintervallet inkluderer 1.",
        ],
        default="Ingen stopp.",
    )
    return table


def build_segment_drift_table(
    severity_frame,
    predictions_by_year,
    min_persons=100,
    n_draws=2000,
    seed=410,
):
    """Bygg segmentdrift med Q og delt cluster-bootstrap-CI.

    ``predictions_by_year`` må være faste prediksjoner fra samme 2022-fit,
    som ``run_year_control["predictions_by_year"]``. Bootstrapen bruker hele
    den kombinerte 2022--2023-rammen som statistikkgrunnlag og trekker
    ``insured_id``-clustere med tilbakelegging. Dermed trekkes en person
    samlet på tvers av begge år, samme trekk brukes for begge modeller, og
    ingen modell refittes.
    """
    years = set(severity_frame["year"].unique())
    if years != {2022, 2023}:
        raise ValueError("Segmentdrift krever nøyaktig utviklingsårene 2022 og 2023.")
    frame = severity_frame.copy()
    frame["skadeantallsgruppe"] = np.select(
        [frame["property_claims"].eq(1), frame["property_claims"].between(2, 3)],
        ["N=1", "N=2-3"],
        default="N>=4",
    )
    frame = frame.sort_index()
    index = frame.index
    actual = frame["property_incurred"].to_numpy(float)
    weight = frame["property_claims"].to_numpy(float)
    years_array = frame["year"].to_numpy()
    clusters = frame["insured_id"].to_numpy()
    segment_values = {
        variable: frame[variable].to_numpy()
        for variable in ["policy_type", "skadeantallsgruppe"]
    }

    rows = []
    for model, by_year in predictions_by_year.items():
        prediction = pd.concat(
            [by_year[2022], by_year[2023]], axis=0
        ).reindex(index)
        if prediction.isna().any():
            raise ValueError(f"Mangler fast prediksjon for {model} i 2022 eller 2023.")
        mu = prediction.to_numpy(float)
        for variable, values in segment_values.items():
            for level in pd.unique(values):
                level_mask = values == level
                support = {
                    year: int(np.unique(clusters[level_mask & (years_array == year)]).size)
                    for year in (2022, 2023)
                }
                year_masks = {
                    year: level_mask & (years_array == year) for year in (2022, 2023)
                }

                def ae(row_positions, mask, mu=mu):
                    selected = row_positions[mask[row_positions]]
                    return actual[selected].sum() / (
                        weight[selected] * mu[selected]
                    ).sum()

                def q_statistic(row_positions, year_masks=year_masks):
                    total = {
                        year: ae(row_positions, years_array == year)
                        for year in (2022, 2023)
                    }
                    segment = {
                        year: ae(row_positions, year_masks[year])
                        for year in (2022, 2023)
                    }
                    return (segment[2023] / total[2023]) / (
                        segment[2022] / total[2022]
                    )

                point_totals = {
                    year: ae(np.arange(len(frame)), years_array == year)
                    for year in (2022, 2023)
                }
                point_segments = {
                    year: ae(np.arange(len(frame)), year_masks[year])
                    for year in (2022, 2023)
                }
                relative = {
                    year: point_segments[year] / point_totals[year]
                    for year in (2022, 2023)
                }
                q = relative[2023] / relative[2022]
                # Små segmenter skal rapporteres, men skal ikke få et
                # ustabilt 2 000-trekksintervall eller kunne utløse stopp.
                if min(support.values()) >= min_persons:
                    q_ci = cluster_bootstrap_ci(
                        q_statistic, clusters, n_draws=n_draws, seed=seed
                    )
                else:
                    q_ci = {"lav": np.nan, "høy": np.nan, "n_draws": 0}
                rows.append(
                    {
                        "variabel": variable,
                        "nivå": level,
                        "modell": model,
                        "unike_personer_2022": support[2022],
                        "unike_personer_2023": support[2023],
                        "A_E_relativ_2022": relative[2022],
                        "A_E_relativ_2023": relative[2023],
                        "Q": q,
                        "Q_lav": q_ci["lav"],
                        "Q_høy": q_ci["høy"],
                        "n_draws": q_ci["n_draws"],
                    }
                )
    return evaluate_segment_drift_stop(pd.DataFrame(rows), min_persons=min_persons)


def build_year_cost_table(model_frame, severity_frame):
    """Årsvise null-/nærnullkostnader og kostnadsnivåer (B-15-utvalget per år).

    ``model_frame`` er hele utviklingspopulasjonen (alle poliseår, også
    skadefrie); ``severity_frame`` er B-15-utvalget (registrert skade med
    materiell kostnad). Viser om nivåskiftet mellom 2022 og 2023 skyldes få,
    store observasjoner eller en jevn endring.
    """
    rows = []
    for year, year_model in model_frame.groupby("year"):
        year_severity = severity_frame.loc[severity_frame["year"] == year]
        near_zero_mask = year_model["property_claims"].gt(0) & year_model[
            "property_incurred"
        ].le(CURRENCY_ZERO_TOLERANCE)
        rows.append(
            {
                "year": year,
                "poliseår": len(year_model),
                "med_registrert_skade": int(year_model["property_claims"].gt(0).sum()),
                "utelatte_nær_null": int(near_zero_mask.sum()),
                "skadevektet_severity": year_severity["property_incurred"].sum()
                / year_severity["property_claims"].sum(),
                "median_gjennomsnittsskade": year_severity["average_severity"].median(),
                "samlet_kostnad": year_severity["property_incurred"].sum(),
            }
        )
    return pd.DataFrame(rows)


def run_mix_standardization(
    candidate_terms,
    reference_terms,
    base_levels,
    severity_frame,
    fit_kwargs,
    prepare_fold_frames,
    prepare_design_frame,
):
    """Miksstandardisering: S0 og finalisten fittet separat per år, uten årsledd.

    Begge årstilpasningene ("trent på 2022" og "trent på 2023") predikerer den
    SAMME referansepopulasjonen — hele ``severity_frame`` (begge år) — med de
    samme skadevektene. Forskjeller mellom de to årstilpasningene isolerer da
    hvordan treningsåret alene endrer den predikerte severityen.

    **Dette er forklarende diagnostikk, ikke et valideringsresultat**: en
    modell trent på 2023 og evaluert på hele populasjonen (inkludert 2023
    selv) er ikke en uavhengig test — den beskriver bare hvor mye
    årstilpasningen beveger seg, ikke hvor godt den generaliserer.

    Returns
    -------
    dict
        ``"kostnad"``: DataFrame med modell, treningsår,
        forventet_kostnad_på_referansepopulasjon, gjennomsnitt_predikert_severity
        og forhold_2023_til_2022 (samme for begge rader til en modell).
        ``"dekning"``: DataFrame med treningsår, variabel, andel_utenfor_område
        — hvor stor andel av referansepopulasjonen som ligger utenfor det
        aktuelle treningsårets observerte kovariatområde.
    """
    reference = severity_frame  # samme referansepopulasjon og skadevekter begge år
    term_lists = {
        "S0": drop_year_term(reference_terms),
        "C1": drop_year_term(candidate_terms),
    }

    fit_rows = []
    for model, terms in term_lists.items():
        for training_year in (2022, 2023):
            train = severity_frame.loc[severity_frame["year"] == training_year]
            spec = glm_spec(
                model,
                terms,
                "average_severity",
                train,
                FIT_FAMILY,
                "property_claims",
                GAMMA_POWER,
                required_columns=_raw_columns(terms),
                base_level_overrides=base_levels,
            )
            (train_design, reference_design), _derived_state = prepare_fold_frames(
                spec, train, [train, reference], prepare_design_frame
            )
            result = fit_glm(
                spec, train_design, check_convergence=False, fit_kwargs=fit_kwargs
            )
            prediction = result.predict(reference_design)
            fit_rows.append(
                {
                    "modell": model,
                    "treningsår": training_year,
                    "forventet_kostnad_på_referansepopulasjon": (
                        reference_design["property_claims"] * prediction
                    ).sum(),
                    "gjennomsnitt_predikert_severity": prediction.mean(),
                }
            )

    fit_table = pd.DataFrame(fit_rows)
    pivoted = fit_table.pivot(
        index="modell", columns="treningsår", values="gjennomsnitt_predikert_severity"
    )
    ratio = (pivoted[2023] / pivoted[2022]).rename("forhold_2023_til_2022")
    fit_table = fit_table.merge(ratio, on="modell")

    coverage_columns = sorted(
        set(_raw_columns(term_lists["S0"]) + _raw_columns(term_lists["C1"]))
    )
    coverage_rows = [
        {
            "treningsår": training_year,
            "variabel": column,
            "andel_utenfor_område": _coverage_share(
                severity_frame.loc[severity_frame["year"] == training_year],
                reference,
                column,
            ),
        }
        for training_year in (2022, 2023)
        for column in coverage_columns
    ]
    return {"kostnad": fit_table, "dekning": pd.DataFrame(coverage_rows)}


def evaluate_time_stop(
    severity_frame_2023, predictions, reference="S0", candidate="C1"
):
    """Planens tidsstoppregel: taper finalisten mot referansen i tidsfolden?

    Kriteriet («Tidsmessig forverring» i stoppregisteret) utløses bare når
    ``candidate`` er dårligere enn ``reference`` med MER ENN BÅDE én
    cluster-SE OG 0,5 % av referansens deviance. Bruker
    ``paired_gamma_gain`` med clustere på ``insured_id``.

    Uansett utfall: at kriteriet IKKE utløses skal rapporteres som «ingen
    forverring påvist i denne tidsdelingen», ikke som dokumentert
    tidsstabilitet (planen dekker bare én kalenderovergang).
    """
    baseline_prediction = predictions[reference]
    candidate_prediction = predictions[candidate]
    rows = baseline_prediction.index  # samme radnøkler for begge modeller
    actual = severity_frame_2023.loc[rows, "average_severity"]
    weight = severity_frame_2023.loc[rows, "property_claims"]
    clusters = severity_frame_2023.loc[rows, "insured_id"]

    gain = paired_gamma_gain(
        actual, weight, baseline_prediction, candidate_prediction, clusters
    )
    d_reference = mean_tweedie_deviance(
        actual, baseline_prediction, sample_weight=weight, power=GAMMA_POWER
    )
    threshold_relative = 0.005 * d_reference

    # gevinst = D_referanse - D_kandidat (S-04-konvensjonen): negativ gevinst
    # betyr at kandidaten er DÅRLIGERE enn referansen.
    loss = -gain["gevinst"]
    triggered = bool(loss > gain["SE_cluster"] and loss > threshold_relative)

    if triggered:
        reasoning = (
            f"Tidsstoppregelen er utløst: {candidate} taper mot {reference} med "
            f"mer enn både én cluster-SE ({loss:.6f} > {gain['SE_cluster']:.6f}) "
            f"og 0,5 % av D_{reference} ({loss:.6f} > {threshold_relative:.6f})."
        )
    else:
        reasoning = "Ingen forverring påvist i denne tidsdelingen."

    return {
        "gevinst": gain["gevinst"],
        "SE_cluster": gain["SE_cluster"],
        "terskel_relativ": threshold_relative,
        "utløst": triggered,
        "begrunnelse": reasoning,
    }


def plot_year_control(segment_ae_table, level_shift_table):
    """(a) A/E per segment for begge modeller, med total-A/E som referanselinje.

    (b) Normalisert A/E_relativ per segment, med planens ±20 %-segmentdriftbånd
    (0,80-1,20) markert. Bare visualisering av tabeller de andre funksjonene i
    denne modulen har bygget.
    """
    fig, (ax_ae, ax_relative) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE)
    models = list(dict.fromkeys(segment_ae_table["modell"]))
    total_ae = dict(zip(level_shift_table["modell"], level_shift_table["A_E"]))
    segment_ae_table = segment_ae_table.assign(
        segment=segment_ae_table["variabel"]
        + ": "
        + segment_ae_table["nivå"].astype(str)
    )

    for model_name, color in zip(models, _LINE_COLORS):
        subset = segment_ae_table[segment_ae_table["modell"] == model_name]
        ax_ae.scatter(
            subset["segment"], subset["A_E"], color=color, label=model_name, zorder=3
        )
        ax_ae.axhline(total_ae[model_name], color=color, linestyle=":", linewidth=1)
        ax_relative.scatter(
            subset["segment"],
            subset["A_E_relativ"],
            color=color,
            label=model_name,
            zorder=3,
        )

    ax_relative.axhspan(0.8, 1.2, color=CONTEXT, alpha=0.3, zorder=0)
    ax_relative.axhline(1.0, color=INK_MUTED, linewidth=1, zorder=1)

    _style_axes(ax_ae, "A/E per segment (stiplet = total A/E per modell)", "A/E")
    _style_axes(
        ax_relative, "Normalisert A/E (segmentdriftbånd 0,80-1,20)", "A_E_relativ"
    )
    for ax in (ax_ae, ax_relative):
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def compute_oof_residuals(model_id, severity_frame_2023, predictions):
    """Deviance-residualer for ``model_id``s 2023-prediksjon fra ``run_year_control``.

    "OOF" i denne tidsdelingen betyr «utenfor treningsutvalget»: modellen er
    fittet på 2022 og evaluert på 2023, som er nøyaktig radene residualene
    beregnes for her. Deviance-residualen gjenbruker
    ``severity_scoring.GAMMA_FAMILY.resid_dev`` — samme deviansformel som
    scoringen, ikke en ny implementasjon.
    """
    prediction = predictions[model_id]
    rows = severity_frame_2023.loc[prediction.index].copy()
    rows["_predicted_severity"] = prediction
    rows["_residual"] = GAMMA_FAMILY.resid_dev(rows["average_severity"], prediction)
    return rows


def plot_oof_residuals(model_id, terms, severity_frame_2023, predictions):
    """OOF-residualdiagnostikk for én modell i tidskontrollen (deviance-residualer).

    Ett samlet figur med delpaneler, i rekkefølge: residual mot OOF-predikert
    severity, mot hver kontinuerlig variabel i ``terms`` (spredningsplott), mot
    hvert nivå i hver kategorisk variabel i ``terms`` (boksplott), og mot de
    kandidatblokkene fra planens register som IKKE inngår i denne modellen
    (``CANDIDATE_BLOCK_COLUMNS``) — for å se om et utelatt mønster fortsatt
    henger igjen i residualene.

    For å diagnostisere en annen modell: sett ``model_id`` til en annen nøkkel
    i ``predictions`` og ``terms`` til dens (år-frie) termliste, f.eks.
    ``plot_oof_residuals("S0", drop_year_term(S0_TERMS), severity_frame_2023,
    predictions)`` eller tilsvarende for finalisten ``"C1"``.

    Rent diagnostisk: ingen ny modell velges eller forkastes ut fra dette
    plottet — planens kandidatrom er låst før tidskontrollen kjøres.
    """
    data = compute_oof_residuals(model_id, severity_frame_2023, predictions)
    raw_terms = [column for column in _raw_columns(terms) if column != "year"]
    numeric_columns = [c for c in raw_terms if pd.api.types.is_numeric_dtype(data[c])]
    categorical_columns = [c for c in raw_terms if c not in numeric_columns]
    excluded_columns = [
        c for c in CANDIDATE_BLOCK_COLUMNS if c not in raw_terms and c in data.columns
    ]
    # Numerisk/kategorisk avgjøres av dtypen, ikke av om variabelen er i
    # modellen — ellers havner en numerisk utelatt kandidat (f.eks.
    # `performance_hp_per_tonne`) feilaktig i boksplott-grenen.
    numeric_panel_columns = {"_predicted_severity", *numeric_columns} | {
        c for c in excluded_columns if pd.api.types.is_numeric_dtype(data[c])
    }

    panels = [("_predicted_severity", "OOF-predikert severity")]
    panels += [(c, c) for c in numeric_columns]
    panels += [(c, f"{c} (i modellen)") for c in categorical_columns]
    panels += [(c, f"{c} (utelatt kandidat)") for c in excluded_columns]

    n_cols = 3
    n_rows = -(-len(panels) // n_cols)  # rund opp
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.3 * n_cols, 3.6 * n_rows), facecolor=SURFACE
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (column, label) in zip(axes, panels):
        if column in numeric_panel_columns:
            ax.scatter(data[column], data["_residual"], s=10, alpha=0.5, color=ACCENT)
        else:
            levels = sorted(data[column].dropna().astype(str).unique())
            groups = [
                data.loc[data[column].astype(str).eq(level), "_residual"]
                for level in levels
            ]
            boxes = ax.boxplot(
                groups,
                tick_labels=levels,
                patch_artist=True,
                medianprops={"color": INK_PRIMARY},
            )
            for patch in boxes["boxes"]:
                patch.set_facecolor(CONTEXT)
                patch.set_edgecolor(AXIS)
            ax.tick_params(axis="x", rotation=45)
        ax.axhline(0, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)
        _style_axes(ax, label, "Deviance-residual")
    for ax in axes[len(panels) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"OOF-residualdiagnostikk: {model_id} (2023, fittet på 2022)", color=INK_PRIMARY
    )
    fig.tight_layout()
    return fig
