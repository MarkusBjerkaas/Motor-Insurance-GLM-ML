"""Det låste Gamma-løpet: CV-kjøring, pooled score og parvis sammenligning (S-09).

Modulen holder løkkene som kjører de samme stegene for hver kandidat, slik at
notebooken bare inneholder spesifikasjonene, kallene og konklusjonene.

Scoren beregnes på **pooled OOF-prediksjoner** over alle fem foldene, ikke som
et snitt av foldscorer: ``sklearn.metrics.mean_tweedie_deviance(..., power=2)``
vektet med ``property_claims`` er nøyaktig planens

    D = sum_i N_i d(y_i, mu_i) / sum_i N_i.

Parvis gevinst og cluster-SE kommer fra
``src_severity.severity_scoring.paired_gamma_gain``.

Alle kandidater kjøres med identiske konvergensinnstillinger (S-07), som sendes
inn som ``fit_kwargs``. Da er en scoreforskjell en modellforskjell, ikke en
optimeringsforskjell.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_tweedie_deviance

from src.glm_core import cross_validate_glm
from src_severity.severity_scoring import paired_gamma_gain

RESPONSE = "average_severity"
WEIGHT = "property_claims"
CLUSTER = "insured_id"


def run_candidates(specs, severity_frame, folds, fit_kwargs, fold_hook=None):
    """Kjør femfolds CV for hver kandidat i ``specs`` (dict ID -> spec).

    Returnerer ``(results, budget)``: ``results`` er en dict ID ->
    ``cross_validate_glm``-resultat, og ``budget`` er antall hovedtilpasninger
    som faktisk ble brukt (kandidater × folder). Kalleren kan sende inn et
    før-fit-filtrert register; protokollmaksimumet for 14 faste kandidater er
    70, og én kombinert kandidat kan øke dette til maksimalt 75.
    """
    results = {
        name: cross_validate_glm(
            spec, severity_frame, folds, fold_hook=fold_hook, fit_kwargs=fit_kwargs
        )
        for name, spec in specs.items()
    }
    return results, len(specs) * len(folds)


def pooled_deviance(severity_frame, prediction):
    """Pooled, skadeantallsvektet Gamma-deviance på radene med endelig prediksjon."""
    usable = prediction.notna()
    return mean_tweedie_deviance(
        severity_frame.loc[usable, RESPONSE],
        prediction[usable],
        sample_weight=severity_frame.loc[usable, WEIGHT],
        power=2,
    )


def build_score_table(results, severity_frame, plan_parameters=None):
    """Én rad per kandidat: pooled OOF-score, gyldighet, parametertall, spredning.

    ``gyldig`` er ``cross_validate_glm``s egen foldkontroll (S-07). ``rader_oof``
    viser hvor mange av severity-radene som faktisk fikk en OOF-prediksjon;
    er den lavere enn utvalget, har minst én fold falt ut og kandidaten kan
    ikke sammenlignes parvis på hele utvalget.
    """
    rows = []
    for name, result in results.items():
        oof = result["oof"]
        fold_deviance = result["scores"]["val_deviance"]
        rows.append(
            {
                "ID": name,
                "parametere": int(result["scores"]["n_params"].max()),
                "parametere_plan": (plan_parameters or {}).get(name),
                "pooled_deviance": pooled_deviance(severity_frame, oof),
                "fold_min": fold_deviance.min(),
                "fold_maks": fold_deviance.max(),
                "rader_oof": int(oof.notna().sum()),
                "alle_konvergerte": bool(result["scores"]["converged"].all()),
                "gyldig": bool(result["valid"]),
                "feil": result["error"],
            }
        )
    return pd.DataFrame(rows).sort_values("pooled_deviance").reset_index(drop=True)


def compare_pair(results, severity_frame, reference, candidate):
    """Parvis sammenligning av to kandidater på nøyaktig de samme radene.

    Returnerer ``paired_gamma_gain``-dicten utvidet med ``folder_forbedret``
    (antall folder der kandidaten har lavere valideringsdeviance),
    ``n_folder``, ``rader`` og ``D_referanse``/``D_kandidat`` beregnet på det
    **felles** radutvalget. Rader der én av modellene mangler OOF-prediksjon
    utelates fra begge, slik at sammenligningen aldri blander ulike utvalg;
    ``rader`` gjør et slikt bortfall synlig.
    """
    reference_oof = results[reference]["oof"]
    candidate_oof = results[candidate]["oof"]
    shared = reference_oof.notna() & candidate_oof.notna()
    part = severity_frame.loc[shared]

    gain = paired_gamma_gain(
        part[RESPONSE],
        part[WEIGHT],
        reference_oof[shared],
        candidate_oof[shared],
        part[CLUSTER],
    )
    reference_folds = results[reference]["scores"].set_index("fold")["val_deviance"]
    candidate_folds = results[candidate]["scores"].set_index("fold")["val_deviance"]
    improved = (candidate_folds < reference_folds.reindex(candidate_folds.index)).sum()
    return {
        **gain,
        "referanse": reference,
        "kandidat": candidate,
        "D_referanse": pooled_deviance(part, reference_oof[shared]),
        "D_kandidat": pooled_deviance(part, candidate_oof[shared]),
        "folder_forbedret": int(improved),
        "n_folder": len(candidate_folds),
        "rader": int(shared.sum()),
    }


def qualify_against(
    results, severity_frame, reference, candidates, parameters, qualifies_fn
):
    """Kjør ``compare_pair`` + seleksjonsregelen for hver kandidat mot ``reference``.

    ``parameters`` er en dict ID -> antall parametere, og ``qualifies_fn`` er
    ``severity_selection.qualifies``. Returnerer en liste av poster i den
    formen ``severity_selection.build_selection_table`` forventer, med
    ``comparison`` lagt ved for full sporbarhet.
    """
    records = []
    for name in candidates:
        comparison = compare_pair(results, severity_frame, reference, name)
        verdict = qualifies_fn(
            gain=comparison["gevinst"],
            se_pair=comparison["SE_cluster"],
            reference_deviance=comparison["D_referanse"],
            folds_improved=comparison["folder_forbedret"],
            n_folds=comparison["n_folder"],
            candidate_parameters=parameters[name],
            reference_parameters=parameters[reference],
        )
        records.append(
            {
                "id": name,
                "parameters": parameters[name],
                "deviance": comparison["D_kandidat"],
                "qualifies_result": verdict,
                "comparison": comparison,
            }
        )
    return records


def build_fold_deviance_table(results):
    """Foldvis valideringsdeviance, én kolonne per kandidat.

    Vises alltid, også når pooled score kvalifiserer en forenkling som taper i
    flere folder (S-08): de foldvise forskjellene skal ikke skjules.
    """
    return pd.DataFrame(
        {
            name: result["scores"].set_index("fold")["val_deviance"]
            for name, result in results.items()
        }
    )


def near_tie_inputs(results, severity_frame, best, candidates, parameters):
    """Bygg radene ``severity_selection.choose_near_tie`` trenger.

    ``se_pair`` er den parvise cluster-SE-en **mot beste kandidat**, som er
    den regelen sammenligner mot. Beste kandidat får ``NaN`` (avstanden til
    seg selv er degenerert) og behandles som nesten lik seg selv.

    Forutsetter at bare kandidater som har bestått gyldighetskontrollene
    (S-07) sendes inn; de har da OOF-prediksjon på hele utvalget, slik at
    alle scorene her er beregnet på de samme radene.
    """
    rows = []
    for name in candidates:
        if name == best:
            deviance = pooled_deviance(severity_frame, results[name]["oof"])
            se = np.nan
        else:
            comparison = compare_pair(results, severity_frame, best, name)
            deviance, se = comparison["D_kandidat"], comparison["SE_cluster"]
        rows.append(
            {
                "id": name,
                "deviance": deviance,
                "se_pair": se,
                "parameters": parameters[name],
            }
        )
    return rows


def near_tie_choice(results, severity_frame, options, parameters, choose_fn):
    """Nesten-like-valget (S-08) anvendt på en gruppe kandidater.

    Regelen sammenligner mot kandidaten med lavest pooled deviance, så den
    finnes først; deretter bygges de parvise SE-ene mot den med
    ``near_tie_inputs``, og ``choose_fn`` (``severity_selection.
    choose_near_tie``) gjør selve valget. Brukes både til blokkvalget og til
    finalistvalget, slik at de to stegene garantert bruker samme mekanikk.
    """
    deviances = {
        name: pooled_deviance(severity_frame, results[name]["oof"]) for name in options
    }
    best = min(deviances, key=deviances.get)
    return choose_fn(near_tie_inputs(results, severity_frame, best, options, parameters))
