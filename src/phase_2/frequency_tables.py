"""Tabeller for fase 1 (frekvens): kandidatoversikt, sammenligninger, A/E og rootogram.

Alle funksjoner tar imot objekter som følger datakontrakten i
``plans/glm_pricing_models_plan.md`` (CV-resultat, sammenligningsrad) og
returnerer ferdig formaterte ``pandas``-tabeller. Ingen modellering skjer her.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_tweedie_deviance

from src.glm_diagnostics import summarize_cv_scores

# Norske korttekster for de kontinuerlige blokkene som inngår i "forms".
# Ukjente nøkler faller tilbake til råkolonnenavnet.
_FEATURE_LABELS = {
    "driver_age": "alder",
    "log_vehicle_value": "verdi",
    "performance_hp_per_tonne": "ytelse",
}
_FORM_LABELS = {"linear": "lineær"}  # "df3"/"df4" vises uendret


def _format_forms(forms):
    """Kompakt tekst for kontinuerlige funksjonsformer, f.eks. "alder=df3, verdi=lineær"."""
    if not forms:
        return ""
    parts = []
    for feature, form in forms.items():
        label = _FEATURE_LABELS.get(feature, feature)
        form_label = _FORM_LABELS.get(form, form)
        parts.append(f"{label}={form_label}")
    return ", ".join(parts)


def build_candidate_table(cv_results, model_ids=None):
    """Én rad per kandidat: spesifikasjon, gyldighet og OOF-scorer.

    ``cv_results`` er ``{model_id: cv_result}`` etter kontrakten (nøkler
    ``spec``, ``scores``, ``valid``, ``error``). Ugyldige kandidater
    (``valid=False``) vises med NaN-scorer og feiltekst i stedet for å utelates,
    slik at hele kandidatrommet er synlig i tabellen.
    """
    ids = list(cv_results) if model_ids is None else list(model_ids)

    # OOF-scorer via summarize_cv_scores, bare for de gyldige kandidatene.
    valid_scores = {
        mid: cv_results[mid]["scores"]
        for mid in ids
        if cv_results[mid].get("valid", False)
    }
    summary = pd.DataFrame()
    if valid_scores:
        pooled_scores = pd.concat(
            [scores.assign(model=mid) for mid, scores in valid_scores.items()],
            ignore_index=True,
        )
        summary = summarize_cv_scores(pooled_scores)

    rows = []
    for mid in ids:
        result = cv_results[mid]
        spec = result["spec"]
        valid = result.get("valid", False)
        row = {
            "model_id": mid,
            "stage": spec.get("stage"),
            "parent_id": spec.get("parent_id"),
            "former": _format_forms(spec.get("forms")),
            "parametere": spec.get("n_parameters"),
            "gyldig": valid,
            "feil": result.get("error"),
            "oof_deviance": np.nan,
            "oof_d2": np.nan,
            "oof_balanse": np.nan,
            "train_oof_gap": np.nan,
        }
        if valid and mid in summary.index:
            s = summary.loc[mid]
            row["oof_deviance"] = s["oof_deviance"]
            row["oof_d2"] = s["oof_d2"]
            row["oof_balanse"] = s["oof_balanse"]
            row["train_oof_gap"] = s["train_deviance"] - s["oof_deviance"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("model_id")


_COMPARISON_COLUMNS = {
    "baseline": "Referanse",
    "candidate": "Kandidat",
    "direction": "Retning",
    "pooled_gain": "Poolet_gevinst",
    "mean_gain": "Snitt_gevinst",
    "se_gain": "SE_gevinst",
    "folds_improved": "Folder_forbedret",
    "passes_1se": "Innenfor_1SE",
    "passes_b08": "Bestar_B08",
}
_PRECISE_COLUMNS = ["Poolet_gevinst", "Snitt_gevinst", "SE_gevinst"]


def format_comparison_table(comparisons):
    """Lesbar sammenligningstabell med norske kolonnenavn og en samlet merknadskolonne.

    Gevinstkolonnene rundes IKKE til få desimaler (deviance-gevinster er ofte i
    størrelsesorden 1e-4) — de vises med 6 desimaler. ``merknad`` viser
    NÆR-TREFF-teksten for kandidater som består B-08 i bare 3/5 folder, og
    videreformidler ``note``/``stability`` når de inneholder "UAVKLART".
    """

    def merknad(row):
        parts = []
        if row.get("near_miss_3of5", False):
            folds = row.get("folds_improved")
            parts.append(f"NÆR-TREFF: bedre i bare {folds}/5 folder")
        for key in ("stability", "note"):
            value = row.get(key)
            if isinstance(value, str) and "UAVKLART" in value:
                parts.append(value)
        return "; ".join(parts)

    table = comparisons.copy()
    table["merknad"] = table.apply(merknad, axis=1)
    table = table.rename(columns=_COMPARISON_COLUMNS)
    for col in _PRECISE_COLUMNS:
        if col in table:
            table[col] = table[col].round(6)
    ordered = [
        c for c in list(_COMPARISON_COLUMNS.values()) + ["merknad"] if c in table
    ]
    remaining = [c for c in table.columns if c not in ordered]
    return table[ordered + remaining]


def report_near_misses(comparisons):
    """Printer én linje per nær-treff (B-08 bestått i bare 3/5 folder). Returnerer antallet."""
    near_misses = comparisons[comparisons.get("near_miss_3of5", pd.Series(dtype=bool))]
    if len(near_misses) == 0:
        print("Ingen nær-treff (3/5) i denne sammenligningen.")
        return 0
    for _, row in near_misses.iterrows():
        print(
            f"NÆR-TREFF: {row['candidate']} vs {row['baseline']} — "
            f"bedre i bare {row['folds_improved']}/5 folder "
            f"(snittgevinst {row['mean_gain']:.6f}, SE {row['se_gain']:.6f})"
        )
    return len(near_misses)


def _weighted_quantile_edges(values, weights, n_bins):
    """Bin-grenser for eksponeringsvektede kvantiler. Kan gi færre bins ved mange like verdier."""
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cum_weight = np.cumsum(w)
    total = cum_weight[-1]
    edges = [v[0]]
    for q in np.linspace(0, 1, n_bins + 1)[1:-1]:
        idx = min(np.searchsorted(cum_weight, q * total), len(v) - 1)
        edges.append(v[idx])
    edges.append(v[-1])
    return sorted(set(edges))


def _bucket_numeric(values, weights, n_bins):
    """Eksponeringsvektede kvantilbøtter for en numerisk serie. Returnerer tekstnivåer."""
    # Kvantilgrensene læres bare på observerte verdier; manglende får eget nivå
    observed = values.notna()
    edges = _weighted_quantile_edges(
        values[observed].to_numpy(), weights[observed].to_numpy(), n_bins
    )
    if len(edges) < 2:
        return pd.Series("all", index=values.index)
    binned = pd.cut(values, bins=edges, include_lowest=True, duplicates="drop")
    # Korte, lesbare etiketter (3 signifikante siffer) i stedet for pd.cut sin
    # fulle flyttallspresisjon; kategoriene beholder sin naturlige rekkefølge.
    labels = [f"({iv.left:.3g}, {iv.right:.3g}]" for iv in binned.cat.categories]
    binned = binned.cat.rename_categories(labels)
    if not observed.all():
        binned = binned.cat.add_categories("MISSING").fillna("MISSING")
    return binned


def build_actual_expected_table(
    frame,
    rate,
    segments,
    n_quantiles=5,
    exposure_col="total_exposure",
    claims_col="property_claims",
    decile_label="prediksjonsdesil",
    categorical=("year",),
    dispersion=1.0,
):
    """A/E (observert/forventet) per nivå, for flere segmenter, i langt format.

    ``rate`` er predikert frekvens (skader/eksponeringsår) på ``frame.index``.
    Forventet antall = eksponering * rate. Kolonner i ``categorical`` grupperes
    per nivå selv om de er numeriske (f.eks. `year`); øvrige ikke-numeriske
    kolonner grupperes per nivå (manglende vises som "MISSING"); resterende
    numeriske kolonner deles i ``n_quantiles`` eksponeringsvektede bøtter med
    intervalltekst som nivå. Står ``decile_label`` i ``segments``, lages i
    tillegg 10 eksponeringsvektede desiler av selve ``rate``.

    ``ae_se = sqrt(dispersion / expected)`` er en grov Poisson-tilnærmet
    standardfeil for A/E (skalert med ``dispersion``); den IGNORERER
    korrelasjon innen samme ``insured_id`` over år og er dermed noe optimistisk
    (for lav). ``flag`` markerer nivåer der |A/E - 1| > 2*ae_se.

    OBS: ``ae`` her er observert/forventet — MOTSATT brøk av ``val_balance`` i
    CV-scorene (som er predikert/observert).
    """
    exposure = frame[exposure_col]
    claims = frame[claims_col]
    expected_all = exposure * rate

    rows = []
    for segment in segments:
        if segment == decile_label:
            level = _bucket_numeric(rate, exposure, 10)
        elif segment in categorical or not pd.api.types.is_numeric_dtype(
            frame[segment]
        ):
            level = frame[segment].astype("object").fillna("MISSING")
        else:
            level = _bucket_numeric(frame[segment], exposure, n_quantiles)

        grouped = (
            pd.DataFrame(
                {
                    "level": level,
                    "exposure": exposure,
                    "claims": claims,
                    "expected": expected_all,
                }
            )
            .groupby("level", observed=True, sort=True)
            .sum()
        )
        grouped["ae"] = grouped["claims"] / grouped["expected"]
        grouped["ae_se"] = np.sqrt(dispersion / grouped["expected"])
        grouped["flag"] = (grouped["ae"] - 1).abs() > 2 * grouped["ae_se"]
        grouped.insert(0, "segment", segment)
        rows.append(grouped.reset_index())

    return pd.concat(rows, ignore_index=True)[
        ["segment", "level", "exposure", "claims", "expected", "ae", "ae_se", "flag"]
    ]


def build_rootogram_table(claims, mean_counts, alpha=None, max_count=10, label="Alle"):
    """Observert vs. forventet antall poliseår med N=k, for hengende rootogram.

    Forventet_k = sum_i P(N_i=k | m_i): Poisson når ``alpha`` er None, ellers
    NB2 (Var = m + alpha*m^2, ``alpha`` skalar eller array). k går 0..max_count-1
    pluss en halekategori ``f"{max_count}+"`` (halemasse = 1 - CDF(max_count-1)).
    """
    claims = np.asarray(claims)
    mean_counts = np.asarray(mean_counts, dtype=float)
    ks = np.arange(max_count)

    if alpha is None:
        pmf = np.array([stats.poisson.pmf(k, mu=mean_counts) for k in ks])
        cdf_last = stats.poisson.cdf(max_count - 1, mu=mean_counts)
    else:
        alpha_arr = np.broadcast_to(np.asarray(alpha, dtype=float), mean_counts.shape)
        r = 1.0 / alpha_arr
        p = r / (r + mean_counts)
        pmf = np.array([stats.nbinom.pmf(k, n=r, p=p) for k in ks])
        cdf_last = stats.nbinom.cdf(max_count - 1, n=r, p=p)

    expected = pmf.sum(axis=1)
    expected_tail = np.sum(1 - cdf_last)
    observed = np.array([(claims == k).sum() for k in ks])
    observed_tail = (claims >= max_count).sum()

    k_labels = [str(k) for k in ks] + [f"{max_count}+"]
    return pd.DataFrame(
        {
            "label": label,
            "k": k_labels,
            "observed": list(observed) + [observed_tail],
            "expected": list(expected) + [expected_tail],
        }
    )


def build_finalist_table(
    cv_results, model_ids, comparisons, time_results, time_baseline_id
):
    """Finalisttabell (3.9): OOF-score, gevinst mot basis, foldscorer og tidsfold.

    ``comparisons`` er sammenligningsrader mot felles basis (``candidate`` som
    nøkkel). ``time_results`` er ``{model_id: cv_result}`` fra tidsfolden, der
    ``time_baseline_id`` er tidsbasisen. ``oof_balanse`` og ``tid_balanse`` er
    predikert/observert (motsatt brøk av A/E).
    """
    table = build_candidate_table(cv_results, model_ids)[
        ["parametere", "oof_deviance", "oof_d2", "oof_balanse", "train_oof_gap"]
    ]
    gains = comparisons.set_index("candidate")
    table["gevinst_mot_basis"] = gains["pooled_gain"]
    table["SE_gevinst"] = gains["se_gain"]
    table["folder_bedre"] = gains["folds_improved"]
    # Én kolonne per fold med valideringsdeviance
    fold_deviance = pd.DataFrame(
        {
            m: cv_results[m]["scores"].set_index("fold")["val_deviance"]
            for m in model_ids
        }
    ).T
    table = table.join(fold_deviance.add_prefix("dev_"))
    table["pearson_phi"] = [
        cv_results[m]["scores"]["pearson_phi"].mean() for m in model_ids
    ]
    time = pd.DataFrame({m: time_results[m]["scores"].iloc[0] for m in model_ids}).T
    table["tid_deviance"] = time["val_deviance"].astype(float)
    table["tid_d2"] = time["val_d2"].astype(float)
    table["tid_balanse"] = time["val_balance"].astype(float)
    table["tid_gevinst_mot_tidsbasis"] = (
        table.loc[time_baseline_id, "tid_deviance"] - table["tid_deviance"]
    )
    # Hvor stor del av segmenteringsgevinsten i gruppe-CV som holder fremover i tid
    table["tid_gevinst_andel_av_oof"] = (
        table["tid_gevinst_mot_tidsbasis"] / table["gevinst_mot_basis"]
    )
    return table


def build_effect_table(relativities, linear_scales=None, exposure_by_variable=None):
    """Effekttabell (3.10) fra ``build_relativity_table``, uten intercept og splinebaser.

    ``linear_scales`` er ``{parameter: (skala, tekst)}``: lineære ledd vises som
    exp(skala·β) med KI exp(skala·(β ± 1,96·SE)), f.eks. +10 % verdi. Splineledd
    vises som kurver, ikke som exp(β) for basiskoeffisienter.
    ``exposure_by_variable`` er ``{variabel: Series med eksponering per nivå}``.
    """
    linear_scales = linear_scales or {}
    table = relativities.reset_index()
    keep = ~(
        table["variabel"].eq("Intercept") | table["variabel"].str.startswith("cr(")
    )
    table = table.loc[keep].copy()
    table["endring"] = np.where(table["nivå"].eq(""), "per enhet", "mot basis")
    for parameter, (scale, label) in linear_scales.items():
        row = table["variabel"].eq(parameter)
        coefficient = table.loc[row, "koeffisient"] * scale
        half_width = 1.96 * table.loc[row, "standardfeil"] * abs(scale)
        table.loc[row, "endring"] = label
        table.loc[row, "relativitet"] = np.exp(coefficient)
        table.loc[row, "ki_lav"] = np.exp(coefficient - half_width)
        table.loc[row, "ki_høy"] = np.exp(coefficient + half_width)
    if exposure_by_variable:
        table["eksponering"] = [
            exposure_by_variable[variable]
            .rename(index=str)
            .get(level.removesuffix(" (basis)"), np.nan)
            if variable in exposure_by_variable
            else np.nan
            for variable, level in zip(table["variabel"], table["nivå"])
        ]
    table["ki_utenfor_1"] = (table["ki_lav"] > 1) | (table["ki_høy"] < 1)
    return table.set_index(["variabel", "nivå"])


def summarize_prediction_subsets(frame, rates, subsets, power=1):
    """Deviance og A/E per delmengde for flere prediksjoner på de samme radene (3.11).

    ``rates`` er ``{navn: predikert rate}``, ``subsets`` er ``{navn: boolsk maske}``.
    Deviance er eksponeringsvektet snitt; A/E er observert/forventet antall.
    """
    rows = []
    for subset, mask in subsets.items():
        part = frame.loc[mask]
        row = {
            "delmengde": subset,
            "rader": len(part),
            "eksponering": part["total_exposure"].sum(),
            "skader": part["property_claims"].sum(),
        }
        for name, rate in rates.items():
            prediction = rate.loc[part.index]
            row[f"deviance_{name}"] = mean_tweedie_deviance(
                part["claim_frequency"],
                prediction,
                sample_weight=part["total_exposure"],
                power=power,
            )
            row[f"AE_{name}"] = (
                part["property_claims"].sum()
                / (prediction * part["total_exposure"]).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("delmengde")


def poisson_unit_deviance(actual, predicted):
    """Poisson enhetsdeviance per rad, 2·(y·log(y/μ) − (y − μ)), med 0·log 0 = 0."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    log_ratio = np.log(np.where(actual > 0, actual, 1.0) / predicted)
    return 2 * (np.where(actual > 0, actual * log_ratio, 0.0) - (actual - predicted))


def paired_deviance_gain(actual, weight, baseline, candidate, clusters):
    """Parvis deviancegevinst (basis − kandidat) med cluster-robust SE.

    Gevinsten er Σ w·(d_basis − d_kandidat) / Σ w, altså nøyaktig differansen
    mellom de vektede scorene. SE er for et vektet snitt over clustere:
    u_g = G_g − gevinst·W_g, Var = C/(C−1)·Σ u_g² / (Σ W)², der G_g og W_g er
    summen av vektet differanse og vekt i cluster g (f.eks. ``insured_id``).
    """
    weight = np.asarray(weight, dtype=float)
    difference = weight * (
        poisson_unit_deviance(actual, baseline)
        - poisson_unit_deviance(actual, candidate)
    )
    sums = (
        pd.DataFrame({"g": difference, "w": weight}).groupby(np.asarray(clusters)).sum()
    )
    gain = sums["g"].sum() / sums["w"].sum()
    residual = sums["g"] - gain * sums["w"]
    n_clusters = len(sums)
    se = np.sqrt(n_clusters / (n_clusters - 1) * (residual**2).sum()) / sums["w"].sum()
    return {"gevinst": gain, "SE_cluster": se, "z": gain / se, "clustere": n_clusters}


def decompose_paired_gain(frame, cells, response, weight, predictions):
    """Bidrag per celle til parvis gevinst (første modell i ``predictions`` er basis).

    ``cells`` er en liste med kolonner/Series å gruppere på. ``bidrag`` summerer
    til total gevinst (vektet med total eksponering); A/E = observert/forventet.
    """
    (base_name, base), (cand_name, cand) = predictions.items()
    actual, exposure = frame[response], frame[weight]
    parts = pd.DataFrame(
        {
            "eksponering": exposure,
            "skader": actual * exposure,
            "bidrag": exposure
            * (
                poisson_unit_deviance(actual, base)
                - poisson_unit_deviance(actual, cand)
            )
            / exposure.sum(),
            f"forventet_{base_name}": base * exposure,
            f"forventet_{cand_name}": cand * exposure,
        },
        index=frame.index,
    )
    table = parts.groupby(cells, observed=True).sum()
    for name in (base_name, cand_name):
        table[f"AE_{name}"] = table["skader"] / table.pop(f"forventet_{name}")
    return table
