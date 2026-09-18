"""Cluster-bootstrap-usikkerhet for severity-fasens EUR-kalibrering (S0 mot C1).

Se ``plans/Severity_plan.md``, seksjonen "Etter seleksjon: EUR-kalibrering og
stabilitet". Seleksjonen er avsluttet (finalisten er C1, referansen S0) --
funksjonene her REFITTER INGEN modeller og GJENTAR IKKE seleksjonen. De tar
imot ferdige ``severity_frame`` og faste OOF-prediksjoner (``predictions``, en
dict modellnavn -> ``pandas.Series`` indeksert som ``severity_frame``, typisk
``{"S0": severity_results["S0"]["oof"], "C1": severity_results["C1"]["oof"]}``)
og laster ingen data selv -- alt datagrunnlag kommer fra
``src.model_data.build_development_frames()`` (kun 2022-2023). Ingen funksjon
her leser eller berører 2024/testsettet.

Planens krav: 2000 cluster-bootstrap-trekk på ``insured_id``, seed 410, med
percentile-intervaller på 95 %. "Samme trekk for begge modeller" oppfylles
automatisk av ``severity_scoring.cluster_bootstrap_ci`` (tynn adapter over
``scipy.stats.bootstrap``): samme ``seed`` og samme ``clusters``-array gir
identiske resamplinger på tvers av kall. Derfor sendes ALLTID hele
utvalgets ``insured_id`` inn som ``clusters`` her -- også når statistikken
gjelder ett enkelt segment -- og segmentutvalget gjøres i stedet inne i
statistikkfunksjonen, på de radposisjonene bootstrapen allerede har trukket.
Det er dette som gjør trekningene delte på tvers av modeller OG segmenter.

Ytelse: numpy-arrays (respons, vekt, prediksjon, segmentmasker) bygges
UTENFOR bootstrap-løkka, og statistikkfunksjonene bruker bare numpy-indeksering
-- ingen pandas-operasjoner per trekk.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_tweedie_deviance

from src_severity.severity_scoring import (
    GAMMA_FAMILY,
    cluster_bootstrap_ci,
    paired_gamma_gain,
)

GAMMA_POWER = 2  # Gamma-deviance i mean_tweedie_deviance/GAMMA_FAMILY


def _seat_category(seats):
    """Setekategori ``<5`` / ``=5`` / ``>5``, samme faste inndeling som ellers i severity-fasen."""
    return pd.Series(
        np.select([seats.lt(5), seats.eq(5)], ["<5", "=5"], default=">5"),
        index=seats.index,
    )


def _claim_count_group(claims):
    """Skadeantallsgruppe ``N=1`` / ``N=2-3`` / ``N>=4`` (rekkefølgen i ``np.select`` er avgjørende)."""
    return pd.Series(
        np.select([claims.eq(1), claims.le(3)], ["N=1", "N=2-3"], default="N>=4"),
        index=claims.index,
    )


def bootstrap_deviance_ci(severity_frame, predictions, n_draws=2000, seed=410):
    """95 %-percentilintervall for pooled Gamma-deviance, én rad per modell.

    Speiler ``severity_cv.pooled_deviance`` (samme ``mean_tweedie_deviance``-
    kall med ``power=2``), men regnet på cluster-bootstrap-trekk av
    ``insured_id`` i stedet for på hele utvalget direkte. Rader uten
    OOF-prediksjon ekskluderes per modell, akkurat som ``pooled_deviance``
    gjør -- for de to låste modellene (S0, C1) skal dette være alle radene.

    Returnerer en DataFrame med kolonnene modell, punkt, lav, høy, n_draws.
    """
    rows = []
    for model_name, prediction in predictions.items():
        prediction = prediction.reindex(severity_frame.index)
        usable = prediction.notna()
        response = severity_frame.loc[usable, "average_severity"].to_numpy(float)
        weight = severity_frame.loc[usable, "property_claims"].to_numpy(float)
        mu = prediction.loc[usable].to_numpy(float)
        clusters = severity_frame.loc[usable, "insured_id"].to_numpy()

        def statistic(row_positions, response=response, weight=weight, mu=mu):
            return mean_tweedie_deviance(
                response[row_positions],
                mu[row_positions],
                sample_weight=weight[row_positions],
                power=GAMMA_POWER,
            )

        ci = cluster_bootstrap_ci(statistic, clusters, n_draws=n_draws, seed=seed)
        rows.append({"modell": model_name, **ci})
    return pd.DataFrame(rows)


def bootstrap_gain_ci(
    severity_frame, predictions, reference, candidate, n_draws=2000, seed=410
):
    """95 %-percentilintervall for den PARVISE gevinsten D_referanse - D_kandidat.

    Begge modellenes enhetsdevianser beregnes på nøyaktig de samme
    bootstrap-radene i hvert trekk (samme ``row_positions``), slik at
    intervallet er parvis -- ikke en differanse av to uavhengige
    deviance-intervaller, som ville vært for bredt fordi det da også ville
    fanget opp variasjon som er felles for begge modellene (samme fold,
    samme vanskelige poliseår) og som kansellerer i en parvis sammenligning.

    Returnerer et dict med punkt/lav/høy fra bootstrapen, samt ``SE_cluster``
    og ``z`` fra ``paired_gamma_gain`` til sammenligning. OBS: intervallet og
    SE-en måler samme størrelse, men er ikke identiske -- SE-en er analytisk
    med endelig-antall-korreksjon (n_clustere/(n_clustere-1)), mens
    bootstrapintervallet er percentilbasert og ikke bruker den korreksjonen.
    De to skal derfor forventes å ligge nær hverandre, ikke være like.
    """
    reference_prediction = predictions[reference].reindex(severity_frame.index)
    candidate_prediction = predictions[candidate].reindex(severity_frame.index)
    shared = reference_prediction.notna() & candidate_prediction.notna()
    part = severity_frame.loc[shared]

    response = part["average_severity"].to_numpy(float)
    weight = part["property_claims"].to_numpy(float)
    mu_reference = reference_prediction.loc[shared].to_numpy(float)
    mu_candidate = candidate_prediction.loc[shared].to_numpy(float)
    clusters = part["insured_id"].to_numpy()

    def statistic(row_positions):
        response_s, weight_s = response[row_positions], weight[row_positions]
        deviance_reference = (
            GAMMA_FAMILY.resid_dev(response_s, mu_reference[row_positions]) ** 2
        )
        deviance_candidate = (
            GAMMA_FAMILY.resid_dev(response_s, mu_candidate[row_positions]) ** 2
        )
        return np.sum(weight_s * (deviance_reference - deviance_candidate)) / np.sum(
            weight_s
        )

    ci = cluster_bootstrap_ci(statistic, clusters, n_draws=n_draws, seed=seed)
    point = paired_gamma_gain(response, weight, mu_reference, mu_candidate, clusters)
    return {
        "punkt": ci["punkt"],
        "lav": ci["lav"],
        "høy": ci["høy"],
        "SE_cluster": point["SE_cluster"],
        "z": point["z"],
    }


def bootstrap_ae_ci(
    severity_frame,
    predictions,
    min_persons=100,
    n_draws=2000,
    seed=410,
    extra_segments=None,
):
    """95 %-percentilintervall for A/E, totalt og per segment, for hver modell.

    A/E = sum(faktisk kostnad) / sum(skadeantall x predikert severity)
    = ``property_incurred.sum() / (property_claims * predikert).sum()``.

    Faste segmenter: ``policy_type``, ``year``, ``municipality_type``,
    ``circulation_area``, setekategori (avledet fra ``seats``) og
    skadeantallsgruppe (avledet fra ``property_claims``). ``extra_segments``
    er en valgfri dict kolonnenavn -> Series (f.eks. prediksjonsdesiler) som
    legges til som ekstra segmentvariabler på samme måte.

    Alle segmentnivåer -- og «Totalt» -- bootstrappes med NØYAKTIG samme
    clustertrekk: ``clusters`` er alltid hele utvalgets ``insured_id``, og
    segmentutvalget gjøres inne i statistikkfunksjonen ved å maskere de
    radposisjonene bootstrapen allerede har trukket (se modulens docstring).
    Det er IKKE egne trekninger per segment.

    Segmentnivåer med færre enn ``min_persons`` unike personer i det
    OBSERVERTE utvalget hoppes over (ingen bootstrap kjøres for dem) -- de er
    uansett for små til å kunne utløse EUR-stoppregel 2, og å kjøre 2000 trekk
    på dem gir bare et ustabilt intervall. «Totalt» er alltid med, uavhengig
    av ``min_persons``.

    Returnerer en DataFrame med kolonnene variabel, nivå, modell,
    unike_personer, A_E, lav, høy, utelukker_1 (bool: intervallet inneholder
    ikke 1,0).
    """
    frame = severity_frame.copy()
    frame["seat_category"] = _seat_category(frame["seats"])
    frame["skadeantallsgruppe"] = _claim_count_group(frame["property_claims"])

    segment_values = {
        "policy_type": frame["policy_type"].to_numpy(),
        "year": frame["year"].to_numpy(),
        "municipality_type": frame["municipality_type"].to_numpy(),
        "circulation_area": frame["circulation_area"].to_numpy(),
        "seat_category": frame["seat_category"].to_numpy(),
        "skadeantallsgruppe": frame["skadeantallsgruppe"].to_numpy(),
    }
    for name, values in (extra_segments or {}).items():
        segment_values[name] = pd.Series(values).reindex(frame.index).to_numpy()

    actual_cost = frame["property_incurred"].to_numpy(float)
    weight = frame["property_claims"].to_numpy(float)
    clusters = frame["insured_id"].to_numpy()

    # (variabel, nivå, maske) -- maske=None betyr «Totalt» (ingen restriksjon).
    levels = [("Totalt", "Totalt", None)]
    for variable, values in segment_values.items():
        for level in pd.unique(values):
            mask = values == level
            unique_persons = pd.Series(clusters[mask]).nunique()
            if unique_persons >= min_persons:
                levels.append((variable, level, mask))

    def make_statistic(mu, mask):
        def statistic(row_positions):
            if mask is not None:
                row_positions = row_positions[mask[row_positions]]
            return (
                actual_cost[row_positions].sum()
                / (weight[row_positions] * mu[row_positions]).sum()
            )

        return statistic

    rows = []
    for model_name, prediction in predictions.items():
        mu = prediction.reindex(frame.index).to_numpy(float)
        for variable, level, mask in levels:
            unique_persons = (
                len(np.unique(clusters))
                if mask is None
                else pd.Series(clusters[mask]).nunique()
            )
            statistic = make_statistic(mu, mask)
            ci = cluster_bootstrap_ci(statistic, clusters, n_draws=n_draws, seed=seed)
            rows.append(
                {
                    "variabel": variable,
                    "nivå": level,
                    "modell": model_name,
                    "unike_personer": int(unique_persons),
                    "A_E": ci["punkt"],
                    "lav": ci["lav"],
                    "høy": ci["høy"],
                    "utelukker_1": not (ci["lav"] <= 1.0 <= ci["høy"]),
                }
            )
    return pd.DataFrame(rows)


def evaluate_stop_criteria(ae_ci_table):
    """Evaluer planens to EUR-stoppregler for hver modell i ``ae_ci_table``.

    Regel 1 (totalnivå): Totalt A/E utenfor [0,90, 1,10] OG
    bootstrapintervallet utelukker 1.

    Regel 2 (segmentnivå): produkt- (``policy_type``) eller
    prediksjonsdesil-A/E (segmentnavn som inneholder "desil") utenfor
    [0,80, 1,20], med minst 100 unike personer, OG intervallet utelukker 1.

    For BEGGE regler må de to betingelsene være oppfylt SAMTIDIG for at
    regelen telles som utløst: et A/E langt fra 1,0 med et bredt intervall
    som fortsatt dekker 1,0 utløser IKKE regelen (usikkert, ikke påvist), og
    et smalt intervall som utelukker 1,0 men med A/E innenfor båndet utløser
    heller ikke regelen (statistisk signifikant, men ikke materielt).

    Returnerer en DataFrame med kolonnene modell, regel, utløst (bool), detalj.
    """
    rows = []
    for model_name, model_rows in ae_ci_table.groupby("modell"):
        total = model_rows.loc[model_rows["variabel"] == "Totalt"].iloc[0]
        rule1_out_of_band = not (0.90 <= total["A_E"] <= 1.10)
        rows.append(
            {
                "modell": model_name,
                "regel": "1: Totalt A/E i [0,90, 1,10]",
                "utløst": bool(rule1_out_of_band and total["utelukker_1"]),
                "detalj": (
                    f"Totalt A/E={total['A_E']:.4f}, "
                    f"intervall=[{total['lav']:.4f}, {total['høy']:.4f}]"
                ),
            }
        )

        is_segment = (model_rows["variabel"] == "policy_type") | model_rows[
            "variabel"
        ].str.contains("desil", case=False, na=False)
        candidates = model_rows.loc[is_segment & (model_rows["unike_personer"] >= 100)]
        out_of_band = ~candidates["A_E"].between(0.80, 1.20)
        offenders = candidates.loc[out_of_band & candidates["utelukker_1"]]
        detail = (
            "; ".join(
                f"{row.variabel}={row.nivå} (A/E={row.A_E:.4f}, n={row.unike_personer})"
                for row in offenders.itertuples()
            )
            or "ingen segmenter utløser regelen"
        )
        rows.append(
            {
                "modell": model_name,
                "regel": "2: Produkt-/desil-A/E i [0,80, 1,20] (>=100 personer)",
                "utløst": bool(len(offenders) > 0),
                "detalj": detail,
            }
        )
    return pd.DataFrame(rows)
