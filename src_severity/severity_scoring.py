"""Scorings- og usikkerhetsfunksjoner for Gamma-severity.

Bruker etablerte biblioteker der de dekker behovet:
``sklearn.metrics.mean_tweedie_deviance(..., power=2)`` ER pooled,
skadeantallsvektet Gamma-deviance og kalles direkte fra notebooken — det er
derfor ingen egen ``pooled_gamma_deviance``-funksjon her. Gamma-enhetsdeviansen
hentes fra ``statsmodels`` (``sm.families.Gamma().resid_dev(y, mu) ** 2``),
verifisert lik formelen ``2*(y/mu - 1 - log(y/mu))`` til rtol=1e-12.

Det eneste som IKKE finnes ferdig i sklearn/statsmodels/scipy er den parvise,
cluster-robuste standardfeilen på scoreforskjeller mellom to modeller
(``paired_gamma_gain``) — den speiler
``src/phase_2/frequency_tables.paired_deviance_gain`` (Poisson-varianten),
men med Gamma-enhetsdeviansen fra statsmodels. Bootstrap-usikkerheten bruker
``scipy.stats.bootstrap`` direkte via en tynn adapter
(``cluster_bootstrap_ci``). Se ``plans/Severity_plan.md``, seksjonen
"Primærscore og usikkerhet".
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Gjenbrukt Gamma-familieobjekt slik at det ikke bygges på nytt per kall.
GAMMA_FAMILY = sm.families.Gamma()


def paired_gamma_gain(actual, weight, baseline, candidate, clusters):
    """Parvis Gamma-deviancegevinst (basis - kandidat) med cluster-robust SE.

    Nøyaktig samme cluster-SE-estimator som
    ``src/phase_2/frequency_tables.paired_deviance_gain``, men enhetsdeviansen
    hentes fra statsmodels (``GAMMA_FAMILY.resid_dev(y, mu) ** 2``) i stedet
    for Poisson-formelen: differanse_i = w_i*(d_basis_i - d_kandidat_i),
    summert per cluster g (typisk ``insured_id``) til G_g (differanse) og
    W_g (vekt). Gevinst = sum(G_g) / sum(W_g). Sentrerte clusterbidrag
    u_g = G_g - gevinst*W_g gir endelig-antall-korrigert varians
    Var = C/(C-1) * sum(u_g^2) / sum(W)^2, SE = sqrt(Var).

    OBS: cluster-SE beskriver KUN usikkerheten i de OBSERVERTE
    scoreforskjellene på dette utvalget — den korrigerer verken for
    seleksjonsoptimisme (kandidaten kan være valgt blant flere alternativer)
    eller for variasjonen som kommer fra selve modelltreningen
    (parameterusikkerhet). Se ``plans/Severity_plan.md``.

    Degenerert tilfelle: er ``baseline`` og ``candidate`` identiske blir alle
    differanser 0, og dermed ``gevinst == 0`` og ``SE_cluster == 0``; ``z``
    blir da ``0/0`` og vises som NaN (dokumentert valg, ikke en feil).
    """
    weight = np.asarray(weight, dtype=float)
    unit_deviance_baseline = GAMMA_FAMILY.resid_dev(actual, baseline) ** 2
    unit_deviance_candidate = GAMMA_FAMILY.resid_dev(actual, candidate) ** 2
    difference = weight * (unit_deviance_baseline - unit_deviance_candidate)
    sums = (
        pd.DataFrame({"g": difference, "w": weight}).groupby(np.asarray(clusters)).sum()
    )
    total_weight = sums["w"].sum()
    gain = sums["g"].sum() / total_weight
    residual = sums["g"] - gain * sums["w"]
    n_clusters = len(sums)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(n_clusters / (n_clusters - 1) * (residual**2).sum()) / total_weight
        z = gain / se
    return {"gevinst": gain, "SE_cluster": se, "z": z, "clustere": n_clusters}


def cluster_bootstrap_ci(statistic_fn, clusters, n_draws=2000, seed=410, alpha=0.05):
    """Cluster-bootstrap percentilintervall via ``scipy.stats.bootstrap``.

    Tynn adapter: bygger mappingen cluster -> radposisjoner én gang, og lar
    ``scipy.stats.bootstrap`` trekke clustere med tilbakelegging (ingen egen
    resamplingsløkke). ``statistic_fn`` tar imot en array med radposisjoner
    (indekser inn i de samme radene som ``clusters``) og returnerer ett tall.

    Determinisme: ``scipy.stats.bootstrap`` med samme ``random_state`` (seed)
    og samme ``clusters``-array gir identiske trekninger på tvers av kall —
    to statistikker (f.eks. to modeller) evaluert med samme ``clusters`` og
    ``seed`` sammenlignes dermed automatisk på samme bootstrap-utvalg, uten at
    trekningene må lagres eller sendes inn eksplisitt.

    Returnerer ``{"punkt", "lav", "høy", "n_draws"}``, der "punkt" er
    statistikken på hele det opprinnelige utvalget og "lav"/"høy" er et
    (1-alpha) percentilintervall.
    """
    clusters = np.asarray(clusters)
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    positions_by_cluster = {
        cluster: np.flatnonzero(inverse == i)
        for i, cluster in enumerate(unique_clusters)
    }

    def statistic(cluster_sample, axis=None):
        row_positions = np.concatenate(
            [positions_by_cluster[c] for c in cluster_sample]
        )
        return statistic_fn(row_positions)

    point = statistic_fn(np.arange(len(clusters)))
    result = stats.bootstrap(
        (unique_clusters,),
        statistic,
        n_resamples=n_draws,
        method="percentile",
        confidence_level=1 - alpha,
        vectorized=False,
        random_state=seed,
    )
    return {
        "punkt": point,
        "lav": result.confidence_interval.low,
        "høy": result.confidence_interval.high,
        "n_draws": n_draws,
    }
