"""Kandidatregister og seleksjonsregler for fase 1 (frekvens).

De mest sentrale GLM-byggesteinene (``glm_spec``, ``fit_glm``, ``run_glm``,
``cross_validate_glm``, ``pooled_oof_deviance``, ``sign_stability``,
``spline_stability`` og ``cross_validate_candidate``) ligger i notebooken.
Denne modulen holder resten av verktøyene fra seksjon 3.1: blokkregisteret,
kandidatbyggeren, kurvene med deltametode-SE, stabilitetsvurderingen per
blokk og den parvise sammenligningen/seleksjonen (B-08, planens 3.4).

Modulen kan ikke importere notebooken (det ville gitt en sirkulær import), så
funksjonene som trenger notebook-definerte byggeklosser (``glm_spec``,
``fit_glm``, ``prepare_design_frame``, ``apply_derived_columns``,
``paired_improvement``, ``sign_stability``, ``spline_stability``,
``pooled_oof_deviance``) får dem satt én gang via :func:`configure`, kalt fra
notebooken før seksjon 3.1 brukes.
"""

import hashlib
import inspect
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import patsy

import src.glm_core
from src.glm_core import prepare_fold_frames

# Disk-cache av CV-resultater (se cross_validate_candidate), overlever
# kjerneomstart. Git-ignorert.
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cv_cache"

# Byggeklosser fra notebooken, satt via configure(). Alle brukes uendret,
# dependency injection unngår en sirkulær import mot glm_pricing_models.py.
_DEPENDENCIES = {}


def configure(
    glm_spec,
    fit_glm,
    run_glm,
    cross_validate_glm,
    prepare_design_frame,
    apply_derived_columns,
    paired_improvement,
    sign_stability,
    spline_stability,
    pooled_oof_deviance,
    targets,
    model_frame,
):
    """Sett byggeklossene fra notebooken denne modulen trenger.

    Kalles én gang, rett før seksjon 3.1 bruker modulen. Bygger også
    ``CURVE_GRIDS`` (kurvegridet per kontinuerlig råkolonne) fra
    ``model_frame``, siden gridet må bygges etter at train-poolen er ferdig,
    og ``CODE_FINGERPRINT`` (til disk-cachen i ``cross_validate_candidate``),
    slik at en endring i CV-logikken usynliggjør gamle cache-treff.

    Parameters
    ----------
    glm_spec : callable
        ``glm_spec`` fra seksjon 2.7.
    fit_glm : callable
        ``fit_glm`` fra seksjon 2.7.
    run_glm : callable
        ``run_glm`` fra seksjon 2.7. Kun brukt til ``CODE_FINGERPRINT``.
    cross_validate_glm : callable
        ``cross_validate_glm`` fra seksjon 2.7. Kun brukt til
        ``CODE_FINGERPRINT``.
    prepare_design_frame : callable
        ``prepare_design_frame`` fra seksjon 2.6 (imputasjon, B-09).
    apply_derived_columns : callable
        ``apply_derived_columns`` fra ``src.glm_core``.
    paired_improvement : callable
        ``paired_improvement`` fra seksjon 2.8, brukt av ``compare_models``.
    sign_stability : callable
        ``sign_stability`` fra seksjon 3.1 (holdt i notebooken).
    spline_stability : callable
        ``spline_stability`` fra seksjon 3.1 (holdt i notebooken).
    pooled_oof_deviance : callable
        ``pooled_oof_deviance`` fra seksjon 3.1 (holdt i notebooken).
    targets : dict
        ``TARGETS`` fra seksjon 2.7.
    model_frame : pandas.DataFrame
        Train-poolen (2022-2023) etter fase-1-kolonnene (``vehicle_brand``,
        ``seats_group``) er lagt til.
    """
    _DEPENDENCIES.update(
        {
            "glm_spec": glm_spec,
            "fit_glm": fit_glm,
            "prepare_design_frame": prepare_design_frame,
            "apply_derived_columns": apply_derived_columns,
            "paired_improvement": paired_improvement,
            "sign_stability": sign_stability,
            "spline_stability": spline_stability,
            "pooled_oof_deviance": pooled_oof_deviance,
            "targets": targets,
            "model_frame": model_frame,
        }
    )
    global CURVE_GRIDS, CODE_FINGERPRINT
    CURVE_GRIDS = {
        feature: build_curve_grid(model_frame, feature)
        for feature in CONTINUOUS_FEATURES + ["driving_experience_years"]
    }
    CODE_FINGERPRINT = _code_fingerprint(
        functions=[
            glm_spec,
            fit_glm,
            run_glm,
            cross_validate_glm,
            prepare_design_frame,
            apply_derived_columns,
            frequency_curve_hook,
        ],
        modules=[src.glm_core],
    )


def _code_fingerprint(functions, modules):
    """Kort hash av kildekoden til funksjoner og moduler som avgjør CV-resultatet.

    Brukt som del av cache-nøkkelen i ``cross_validate_candidate``, slik at en
    redigering av selve fit/CV-logikken usynliggjør gamle disk-cache-treff i
    stedet for å returnere et resultat fra før endringen.

    Parameters
    ----------
    functions : list of callable
        Funksjoner hvis kildekode (``inspect.getsource``) skal inngå i
        hashen.
    modules : list of module
        Moduler hvis filinnhold skal inngå i hashen (fanger opp alt i
        modulen, ikke bare enkeltfunksjoner).

    Returns
    -------
    str
        Kort heksadesimal SHA-1-hash.
    """
    parts = [inspect.getsource(fn) for fn in functions]
    parts += [Path(module.__file__).read_text() for module in modules]
    return hashlib.sha1("".join(parts).encode()).hexdigest()[:16]


def cache_load(key):
    """Hent et tidligere CV-resultat fra disk-cachen (``CACHE_DIR``).

    Parameters
    ----------
    key : tuple
        Cache-nøkkelen bygget i ``cross_validate_candidate`` (formel,
        familie, kolonner, data- og foldfingerprint, ``CODE_FINGERPRINT``).

    Returns
    -------
    dict or None
        Det lagrede CV-resultatet, eller ``None`` hvis det ikke finnes i
        cachen.
    """
    path = CACHE_DIR / f"{hashlib.sha1(repr(key).encode()).hexdigest()}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as file:
        return pickle.load(file)


def cache_store(key, value):
    """Lagre et CV-resultat i disk-cachen (``CACHE_DIR``), git-ignorert.

    Parameters
    ----------
    key : tuple
        Cache-nøkkelen, se ``cache_load``.
    value : dict
        CV-resultatet fra ``cross_validate_glm``, som skal kunne gjenbrukes
        etter en kjerneomstart.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{hashlib.sha1(repr(key).encode()).hexdigest()}.pkl"
    with path.open("wb") as file:
        pickle.dump(value, file)


CONTINUOUS_FORMS = {
    "linear": "{x}",
    "df3": "cr({x}, df=3, constraints='center')",
    "df4": "cr({x}, df=4, constraints='center')",
}
FORM_DF = {"linear": 1, "df3": 3, "df4": 4}
FORM_LABELS = {"linear": "lin", "df3": "df3", "df4": "df4"}

# Blokkregister: blokknavn → rolle, råkolonne og (for kontinuerlige) kortnavn i ID
FEATURE_BLOCKS = {
    "product": {"role": "kontroll", "column": "policy_type"},
    "year": {"role": "kontroll", "column": "year"},
    "driver": {"role": "kjerne", "column": "driver_age", "label": "age"},
    "value": {"role": "kjerne", "column": "log_vehicle_value", "label": "value"},
    "performance": {
        "role": "kjerne",
        "column": "performance_hp_per_tonne",
        "label": "perf",
    },
    "fuel": {"role": "kjerne", "column": "fuel_type"},
    "circulation": {"role": "kjerne", "column": "circulation_area"},
    "municipality": {"role": "kjerne", "column": "municipality_type"},
    "payment": {"role": "kjerne", "column": "payment_frequency"},
    "business": {"role": "kjerne", "column": "business_type"},
    "brand": {"role": "sekundær", "column": "vehicle_brand_pooled"},
    "seats": {"role": "sekundær", "column": "seats_group"},
    "experience": {
        "role": "sensitivitet",
        "column": "driving_experience_years",
        "label": "exp",
    },
}
CONTROL_BLOCKS = ["product", "year"]
CORE_BLOCKS = [
    name for name, block in FEATURE_BLOCKS.items() if block["role"] == "kjerne"
]
# Tie-break-rekkefølge for form (planens 3.4)
CONTINUOUS_FEATURES = ["driver_age", "log_vehicle_value", "performance_hp_per_tonne"]
MIN_STABILITY_EXPOSURE = 500
STABLE_SIGN_FOLDS = 4
SCORE_TOLERANCE = 1e-10

CURVE_GRIDS = {}  # bygges av configure()
CODE_FINGERPRINT = None  # bygges av configure()
CV_CACHE = {}  # brukes av cross_validate_candidate i notebooken


def form_model_id(stage, forms):
    """Bygg en stabil kandidat-ID fra formvalget per kontinuerlig råkolonne.

    Parameters
    ----------
    stage : str
        Kortnavnet på fasen/steget kandidaten hører til, f.eks. ``"F2"``.
    forms : dict
        Råkolonne → formnavn (``"linear"``, ``"df3"`` eller ``"df4"``).

    Returns
    -------
    str
        ID på formen ``f"{stage}_age-df3_value-lin_perf-df4"``.
    """
    labels = {block["column"]: block.get("label") for block in FEATURE_BLOCKS.values()}
    return (
        stage
        + "_"
        + "_".join(
            f"{labels[column]}-{FORM_LABELS[form]}" for column, form in forms.items()
        )
    )


def build_frequency_candidate(
    model_id,
    stage,
    blocks,
    forms=None,
    parent_id=None,
    eligible_for_selection=True,
    derived=(),
    family="poisson",
    data=None,
):
    """Bygg én frekvenskandidat etter datakontrakten, via ``glm_spec``.

    Parameters
    ----------
    model_id : str
        Kandidatens unike ID, brukt som ``spec["name"]``.
    stage : str
        Fasen/steget kandidaten hører til (planens nummerering, f.eks.
        ``"F2"``).
    blocks : list of str
        Blokknavn fra ``FEATURE_BLOCKS`` som skal være med i modellen.
    forms : dict or None, optional
        Form per kontinuerlig råkolonne (``"linear"``, ``"df3"`` eller
        ``"df4"``). Standard er lineær for alle kontinuerlige blokker som er
        med.
    parent_id : str or None, optional
        ID til modellen kandidaten sammenlignes mot (kun for dokumentasjon i
        resultatet).
    eligible_for_selection : bool, optional
        Om kandidaten kan velges av ``select_from_candidate_set``. Slås
        automatisk av for kandidater med ``"experience"``-blokken (B-11:
        sensitivitet, aldri hovedkandidat).
    derived : tuple of dict, optional
        Foldvis lærte kolonner, se ``glm_spec``.
    family : {"poisson"}, optional
        Bare Poisson er implementert her. NB2 har egen antallsgren i 3.8.
    data : pandas.DataFrame or None, optional
        Datasettet kandidaten bygges mot. Standard er ``model_frame`` satt i
        :func:`configure`.

    Returns
    -------
    dict
        Spesifikasjonen fra ``glm_spec``, utvidet med ``model_id``,
        ``stage``, ``parent_id``, ``feature_blocks``, ``forms``,
        ``n_parameters`` og ``eligible_for_selection``.
    """
    glm_spec = _DEPENDENCIES["glm_spec"]
    prepare_design_frame = _DEPENDENCIES["prepare_design_frame"]
    apply_derived_columns = _DEPENDENCIES["apply_derived_columns"]
    targets = _DEPENDENCIES["targets"]
    data = _DEPENDENCIES["model_frame"] if data is None else data

    if family != "poisson":
        raise NotImplementedError("NB2 implementeres som egen antallsgren i 3.8")
    if {"driver", "experience"} <= set(blocks):
        raise ValueError("B-11: alder og erfaring brukes aldri samtidig")
    forms = dict(forms or {})
    x, required, used_forms = [], [], {}
    for name in blocks:
        column = FEATURE_BLOCKS[name]["column"]
        required.append(column)
        if "label" in FEATURE_BLOCKS[name]:  # kontinuerlig blokk
            used_forms[column] = forms.pop(column, "linear")
            x.append(CONTINUOUS_FORMS[used_forms[column]].format(x=column))
        else:
            x.append(column)
    assert not forms, f"former for blokker som ikke er med: {forms}"
    x += [term for item in derived for term in item["terms"]]
    spec = glm_spec(
        model_id,
        x,
        **{**targets["frequency"], "data": data},
        required_columns=required,
        derived=derived,
    )
    # Parametertall = kolonner i designmatrisen på hele train-poolen, inkl. intercept
    design = prepare_design_frame(data, data, required)
    derived_state = {item["name"]: item["learn"](design) for item in derived}
    design = apply_derived_columns(spec, design, derived_state)
    n_parameters = patsy.dmatrix(spec["formula"].split("~")[1], design).shape[1]
    return spec | {
        "model_id": model_id,
        "stage": stage,
        "parent_id": parent_id,
        "feature_blocks": list(blocks),
        "forms": used_forms,
        "n_parameters": n_parameters,
        "eligible_for_selection": eligible_for_selection and "experience" not in blocks,
    }


def build_curve_grid(frame, feature, n_points=25):
    """Visningsgrid: jevne punkter mellom eksponeringsvektet p2,5 og p97,5.

    Parameters
    ----------
    frame : pandas.DataFrame
        Datasettet gridet bygges fra (typisk hele train-poolen).
    feature : str
        Kolonnen gridet bygges for.
    n_points : int, optional
        Antall punkter i gridet. Standard 25.

    Returns
    -------
    dict
        ``"x"`` (numpy.ndarray med gridpunktene) og ``"reference"``
        (eksponeringsvektet median, brukt som referansepunkt for de
        relative kurvene).
    """
    values = frame[feature].fillna(frame[feature].median()).to_numpy()
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(frame["total_exposure"].to_numpy()[order])
    cumulative /= cumulative[-1]

    # Minste x der kumulativ sortert eksponering når andelen q
    def quantile(q):
        return values[order][np.searchsorted(cumulative, q)]

    return {
        "x": np.linspace(quantile(0.025), quantile(0.975), n_points),
        "reference": quantile(0.5),
    }


def frequency_curve_hook(
    spec, fold_name, result, train_design, val_design, derived_state
):
    """Relative kurver og haleprediksjoner for kontinuerlige ledd i én fold.

    Brukes som ``fold_hook`` i ``cross_validate_glm``/``cross_validate_candidate``.
    Kurven er exp(f(x) − f(ref)) fra foldens egen ``design_info``. SE på
    logskala er deltametoden sqrt(d' Σ d), der Σ er modellbasert kovarians
    skalert med Pearson-φ̂ fra treningsfolden.

    Parameters
    ----------
    spec : dict
        Kandidatspesifikasjonen som ble fittet i denne folden.
    fold_name : str
        Navnet på folden (f.eks. ``"fold_1"`` eller ``"full"``).
    result : statsmodels GLMResults
        Den estimerte modellen for denne folden.
    train_design : pandas.DataFrame
        Den imputerte treningsdelen for denne folden.
    val_design : pandas.DataFrame
        Den imputerte valideringsdelen for denne folden.
    derived_state : dict
        Tilstanden de foldvis avledede kolonnene lærte på treningsdelen.

    Returns
    -------
    dict
        Tom dict hvis kandidaten ikke har kontinuerlige ledd, ellers
        ``"curves"`` (kurvepunkter med relativitet og SE) og ``"tails"``
        (haleprediksjoner og antall valideringsrader utenfor treningsområdet).
    """
    apply_derived_columns = _DEPENDENCIES["apply_derived_columns"]
    design_info = (
        getattr(result.model, "model_spec", None) or result.model.data.model_spec
    )
    pearson_phi = result.pearson_chi2 / result.df_resid
    covariance = (result.cov_params() * pearson_phi / result.scale).to_numpy()
    params = result.params
    curve_rows, tail_rows = [], []
    for feature in spec["forms"]:
        grid = CURVE_GRIDS[feature]
        train_min, train_max = train_design[feature].min(), train_design[feature].max()
        # Rekkefølge: 25 gridpunkter, referanse, treningsmin, treningsmaks
        points = np.concatenate([grid["x"], [grid["reference"], train_min, train_max]])
        for product in spec.get("curve_products", ["ALLE"]):
            frame = train_design.iloc[[0] * len(points)].reset_index(drop=True)
            frame[feature] = points
            if product != "ALLE":
                frame["policy_type"] = product
            frame = apply_derived_columns(spec, frame, derived_state)
            design = patsy.build_design_matrices(
                [design_info], frame, return_type="dataframe"
            )[0]
            difference = (
                design[params.index].to_numpy()
                - design[params.index].to_numpy()[len(grid["x"])]
            )
            log_relative = difference @ params.to_numpy()
            se_log = np.sqrt(
                np.einsum("ij,jk,ik->i", difference, covariance, difference)
            )
            n_curve = len(grid["x"]) + 1  # gridet og referansepunktet
            curve_rows.append(
                pd.DataFrame(
                    {
                        "model_id": spec["model_id"],
                        "fold": fold_name,
                        "feature": feature,
                        "product": product,
                        "x": points[:n_curve],
                        "relative": np.exp(log_relative[:n_curve]),
                        "se_log": se_log[:n_curve],
                    }
                ).sort_values("x")
            )
            tail_rows.append(
                {
                    "model_id": spec["model_id"],
                    "fold": fold_name,
                    "feature": feature,
                    "product": product,
                    "train_min": train_min,
                    "train_max": train_max,
                    "relative_at_min": np.exp(log_relative[-2]),
                    "relative_at_max": np.exp(log_relative[-1]),
                    "n_val_outside": int(
                        (~val_design[feature].between(train_min, train_max)).sum()
                    ),
                }
            )
    if not curve_rows:
        return {}
    return {
        "curves": pd.concat(curve_rows, ignore_index=True),
        "tails": pd.DataFrame(tail_rows),
    }


def fit_full_curves(candidate, data=None):
    """Kurver og haler fra fit på hele train-poolen (``fold="full"``), til visning.

    Parameters
    ----------
    candidate : dict
        Kandidatspesifikasjonen fra ``build_frequency_candidate``.
    data : pandas.DataFrame or None, optional
        Datasettet kandidaten fittes på. Standard er ``model_frame`` satt i
        :func:`configure`.

    Returns
    -------
    dict
        ``"curves"`` og ``"tails"`` fra ``frequency_curve_hook``, evaluert på
        hele treningsdatasettet i stedet for per fold.
    """
    prepare_design_frame = _DEPENDENCIES["prepare_design_frame"]
    fit_glm = _DEPENDENCIES["fit_glm"]
    data = _DEPENDENCIES["model_frame"] if data is None else data
    (design,), derived_state = prepare_fold_frames(
        candidate, data, [data], prepare_design_frame
    )
    result = fit_glm(candidate, design)
    return frequency_curve_hook(
        candidate, "full", result, design, design, derived_state
    )


def data_fingerprint(obj):
    """Kort hash av innholdet i en DataFrame/Series/Index (til cache-nøkkelen).

    Parameters
    ----------
    obj : pandas.DataFrame, pandas.Series or pandas.Index
        Objektet som skal hashes.

    Returns
    -------
    str
        Heksadesimal SHA-1-hash av innholdet.
    """
    import hashlib

    return hashlib.sha1(
        pd.util.hash_pandas_object(obj).to_numpy().tobytes()
    ).hexdigest()


def assess_stability(cv_result, blocks=None):
    """Stabilitet per blokk (planens 3.4 med støyskalerte grenser).

    Parameters
    ----------
    cv_result : dict
        Resultatet fra ``cross_validate_candidate``, med ``spec``, ``params``
        og ``param_se``.
    blocks : list of str or None, optional
        Blokkene som skal vurderes. Standard er alle blokker og
        B-28-interaksjoner i kandidaten.

    Returns
    -------
    status : str
        Samlet status: verste enkeltstatus blant de vurderte effektene, eller
        ``"ikke vurdert"`` hvis ingen effekter ble vurdert (nivåer under
        ``MIN_STABILITY_EXPOSURE`` og ``MISSING`` telles ikke).
    table : pandas.DataFrame
        Én rad per vurdert effekt, med ``block``, ``effect``, ``status`` og
        ``detail``.
    """
    sign_stability = _DEPENDENCIES["sign_stability"]
    spline_stability = _DEPENDENCIES["spline_stability"]
    model_frame = _DEPENDENCIES["model_frame"]

    spec, params, standard_errors = (
        cv_result["spec"],
        cv_result["params"],
        cv_result["param_se"],
    )
    rows = []
    # B-28-interaksjoner er foldvis avledede kolonner med ``slope_feature`` (3.7)
    interactions = {
        item["name"]: item for item in spec["derived"] if "slope_feature" in item
    }
    for name in (
        spec["feature_blocks"] + list(interactions) if blocks is None else blocks
    ):
        if name in interactions:
            item = interactions[name]
            for parameter in item["terms"]:
                estimates = params.loc[parameter]
                rows.append(
                    {
                        "block": name,
                        "effect": parameter,
                        "status": sign_stability(
                            estimates, standard_errors.loc[parameter]
                        ),
                        "detail": f"positive folder {int(estimates.gt(0).sum())}/{len(estimates)}",
                    }
                )
            status, detail = spline_stability(cv_result, item["slope_feature"])
            rows.append(
                {
                    "block": name,
                    "effect": f"{item['slope_feature']} per produkt",
                    "status": status,
                    "detail": detail,
                }
            )
            continue
        column = FEATURE_BLOCKS[name]["column"]
        form = spec["forms"].get(column)
        if form in ("df3", "df4"):
            status, detail = spline_stability(cv_result, column)
            rows.append(
                {"block": name, "effect": column, "status": status, "detail": detail}
            )
            continue
        if form == "linear":
            effects = {column: column}  # parameter → kort effektnavn
        else:
            exposure = (
                model_frame.groupby(column)["total_exposure"].sum().rename(index=str)
            )
            prefix = f"C({column}, "
            levels = {
                parameter: parameter.split("[T.", 1)[1][:-1]
                for parameter in params.index
                if parameter.startswith(prefix)
            }
            effects = {
                parameter: f"{column}={level}"
                for parameter, level in levels.items()
                if exposure.get(level, 0) >= MIN_STABILITY_EXPOSURE
            }
        for parameter, effect in effects.items():
            estimates = params.loc[parameter]
            status = sign_stability(estimates, standard_errors.loc[parameter])
            detail = f"positive folder {int(estimates.gt(0).sum())}/{len(estimates)}"
            rows.append(
                {"block": name, "effect": effect, "status": status, "detail": detail}
            )
    table = pd.DataFrame(rows, columns=["block", "effect", "status", "detail"])
    for status in ("UAVKLART", "nær-null", "stabil"):
        worst = table.loc[table["status"].str.startswith(status), "status"]
        if len(worst):
            return worst.iloc[0], table
    return "ikke vurdert", table


def stability_passes(status):
    """Om en stabilitetsstatus tillater at kandidaten går videre i stigen.

    Parameters
    ----------
    status : str
        Statusen fra ``assess_stability``.

    Returns
    -------
    bool
        ``True`` for ``"stabil"``, ``"nær-null"`` og ``"ikke vurdert"``.
        ``nær-null`` flagges i notatene, men stopper ikke stigen; alt som
        starter med ``"UAVKLART"`` gjør det.
    """
    return status in ("stabil", "nær-null", "ikke vurdert")


def ae_outside_noise(actual, expected, phi):
    """A/E-flagg: |A/E − 1| > 2·SE med SE ≈ sqrt(φ̂ / forventet antall).

    Parameters
    ----------
    actual : array-like
        Observert antall/beløp.
    expected : array-like
        Forventet antall/beløp under modellen.
    phi : float
        Pearson-dispersjon φ̂ fra treningsfolden.

    Returns
    -------
    array-like of bool
        ``True`` der avviket er større enn støygrensen. Ignorerer
        korrelasjon innen ``insured_id``, så grensen er noe optimistisk.
    """
    return (actual / expected - 1).abs() > 2 * np.sqrt(phi / expected)


def differing_blocks(baseline_spec, candidate_spec):
    """Blokker (og B-28-interaksjoner) kandidaten har som baseline mangler eller har med annen form.

    Parameters
    ----------
    baseline_spec : dict
        Spesifikasjonen til baseline-kandidaten.
    candidate_spec : dict
        Spesifikasjonen til kandidaten som sammenlignes mot baseline.

    Returns
    -------
    list of str
        Blokknavn og interaksjonsnavn som er nye i kandidaten eller har
        endret form sammenlignet med baseline. Brukes til å avgjøre hvilke
        blokker ``assess_stability`` skal vurdere i en sammenligning.
    """
    baseline_derived = {item["name"] for item in baseline_spec["derived"]}
    return [
        name
        for name in candidate_spec["feature_blocks"]
        if name not in baseline_spec["feature_blocks"]
        or candidate_spec["forms"].get(FEATURE_BLOCKS[name]["column"])
        != baseline_spec["forms"].get(FEATURE_BLOCKS[name]["column"])
    ] + [
        item["name"]
        for item in candidate_spec["derived"]
        if "slope_feature" in item and item["name"] not in baseline_derived
    ]


def compare_models(cv_results, baseline_id, candidate_id, direction="oppgradering"):
    """Én sammenligningsrad etter datakontrakten (B-08 / forenkling innen 1 SE).

    Parameters
    ----------
    cv_results : dict
        Kandidat-ID → resultat fra ``cross_validate_candidate``.
    baseline_id : str
        ID til modellen som sammenlignes mot.
    candidate_id : str
        ID til kandidaten som testes.
    direction : {"oppgradering", "forenkling"}, optional
        ``"oppgradering"`` tester B-08 (kandidaten må slå baseline).
        ``"forenkling"`` tester at kandidaten (baseline i dette kallet er da
        den rikere modellen) er innen 1 SE av tapet. Standard
        ``"oppgradering"``.

    Returns
    -------
    dict
        Én rad med ``baseline``, ``candidate``, ``direction``,
        ``pooled_gain`` (= $D_{baseline} - D_{candidate}$ pooled),
        ``mean_gain``, ``se_gain``, ``folds_improved``, ``passes_1se``,
        ``passes_b08``, ``near_miss_3of5``, ``stability`` og ``note``.
    """
    paired_improvement = _DEPENDENCIES["paired_improvement"]

    baseline, candidate = cv_results[baseline_id], cv_results[candidate_id]
    row = {"baseline": baseline_id, "candidate": candidate_id, "direction": direction}
    if not (baseline["valid"] and candidate["valid"]):
        errors = "; ".join(r["error"] for r in (baseline, candidate) if r["error"])
        return row | {
            "pooled_gain": np.nan,
            "mean_gain": np.nan,
            "se_gain": np.nan,
            "folds_improved": 0,
            "passes_1se": False,
            "passes_b08": False if direction == "oppgradering" else None,
            "near_miss_3of5": False,
            "stability": "ugyldig",
            "note": f"ugyldig kandidat: {errors}",
        }
    # Parvise sammenligninger krever identiske rader og vekter i hver fold
    columns = ["fold", "n_val", "val_weight"]
    assert baseline["scores"][columns].equals(candidate["scores"][columns])
    paired = paired_improvement(
        pd.concat([baseline["scores"], candidate["scores"]]), baseline_id, candidate_id
    )
    # Stabilitet avgjøres på blokkene som er nye eller har endret form i kandidaten
    changed = differing_blocks(baseline["spec"], candidate["spec"])
    stability = assess_stability(candidate, changed)[0] if changed else "ikke vurdert"
    stable = stability_passes(stability)
    mean_gain, se_gain = paired["snitt_forbedring"], paired["standardfeil"]
    notes = [] if stable or stability == "ikke vurdert" else [stability]
    # Nær-null-effekter i hele kandidaten vises alltid, selv om de ikke stopper valget
    stability_table = assess_stability(candidate)[1]
    near_zero = stability_table.loc[stability_table["status"].eq("nær-null"), "effect"]
    if len(near_zero):
        notes.append("nær-null: " + ", ".join(near_zero))
    if direction == "oppgradering":
        passes_1se = bool(mean_gain > se_gain)
        passes_b08 = bool(paired["passerer_b08"] and stable)
        near_miss = bool(
            paired["pooled_forbedring"] > 0
            and passes_1se
            and stable
            and paired["folder_med_forbedring"] == 3
        )
    else:
        passes_1se = bool(-mean_gain <= se_gain)  # snittap ≤ SE av tapet
        passes_b08, near_miss = None, False
    if near_miss:  # printes med report_near_misses der tabellen vises
        notes.append("NÆR-TREFF: bedre i bare 3/5 folder")
    return row | {
        "pooled_gain": paired["pooled_forbedring"],
        "mean_gain": mean_gain,
        "se_gain": se_gain,
        "folds_improved": paired["folder_med_forbedring"],
        "passes_1se": passes_1se,
        "passes_b08": passes_b08,
        "near_miss_3of5": near_miss,
        "stability": stability,
        "note": "; ".join(notes) or None,
    }


def select_from_candidate_set(cv_results, candidate_ids, anchor_id):
    """Valg i fast kandidatsett (planens 3.4), med anker-kravet etter B-08.

    Parameters
    ----------
    cv_results : dict
        Kandidat-ID → resultat fra ``cross_validate_candidate``, for minst
        alle ID-ene i ``candidate_ids`` og ``anchor_id``.
    candidate_ids : list of str
        Kandidatene valget skal gjøres blant.
    anchor_id : str
        ID til dagens beste/enkleste modell, valget må slå eller forbli.

    Returns
    -------
    dict
        ``selected`` (valgt ID, eller ``None`` hvis ankeret ikke er brukbart),
        ``best`` (lavest pooled OOF-deviance blant brukbare kandidater),
        ``within_1se`` (kandidater innen 1 SE av ``best``), ``table``
        (kandidatoversikt), ``anchor_comparisons`` (sammenligninger mot
        ankeret) og ``reason`` (kort forklaring på valget).
    """
    pooled_oof_deviance = _DEPENDENCIES["pooled_oof_deviance"]

    rows = []
    for model_id in candidate_ids:
        result, spec = cv_results[model_id], cv_results[model_id]["spec"]
        rows.append(
            {
                "model_id": model_id,
                "valid": result["valid"],
                "eligible": spec["eligible_for_selection"],
                "stability": assess_stability(result)[0]
                if result["valid"]
                else "ugyldig",
                "pooled_deviance": pooled_oof_deviance(result)
                if result["valid"]
                else np.nan,
                "n_parameters": spec["n_parameters"],
                "df_order": tuple(
                    FORM_DF.get(spec["forms"].get(f), 0) for f in CONTINUOUS_FEATURES
                ),
            }
        )
    table = pd.DataFrame(rows).set_index("model_id")
    table["usable"] = (
        table["valid"]
        & table["eligible"]
        & ~table["stability"].str.startswith("UAVKLART")
    )
    for model_id, row in table.loc[~table["usable"]].iterrows():
        reason = (
            "ugyldig"
            if not row["valid"]
            else ("ikke valgbar" if not row["eligible"] else row["stability"])
        )
        print(
            f"{'UAVKLART' if reason.startswith('UAVKLART') else 'Utelatt'}: {model_id} ({reason})"
        )
    if not table.loc[anchor_id, "usable"]:
        print(
            f"UAVKLART: ankeret {anchor_id} er ikke brukbart; valget må avklares manuelt."
        )
        return {"selected": None, "table": table, "reason": "anker ikke brukbart"}

    usable = table.loc[table["usable"]]
    best = usable["pooled_deviance"].idxmin()
    within = [
        model_id
        for model_id in usable.index
        if model_id == best
        or compare_models(cv_results, best, model_id, "forenkling")["passes_1se"]
    ]
    # Tie-break: færrest parametere, lavest df fører → verdi → ytelse, alfabetisk ID
    chosen = min(
        within,
        key=lambda m: (usable.loc[m, "n_parameters"], usable.loc[m, "df_order"], m),
    )
    # Mot ankeret: færre parametere er forenkling (innen 1 SE), ellers oppgradering (B-08)
    anchor_parameters = table.loc[anchor_id, "n_parameters"]
    anchor_comparisons = pd.DataFrame(
        [
            compare_models(
                cv_results,
                anchor_id,
                m,
                "forenkling"
                if usable.loc[m, "n_parameters"] < anchor_parameters
                else "oppgradering",
            )
            for m in usable.index
            if m != anchor_id
        ]
    )
    chosen_row = (
        anchor_comparisons.set_index("candidate").loc[chosen]
        if chosen != anchor_id
        else None
    )
    if chosen == anchor_id:
        selected, reason = anchor_id, "ankeret er enklest innen 1 SE av beste kandidat"
    elif chosen_row["direction"] == "forenkling":
        selected, reason = (
            (chosen, f"{chosen} er enklest innen 1 SE og innen 1 SE av ankeret")
            if chosen_row["passes_1se"]
            else (
                anchor_id,
                f"{chosen} er ikke innen 1 SE av ankeret; ankeret beholdes",
            )
        )
    elif chosen_row["passes_b08"]:
        selected, reason = (
            chosen,
            f"{chosen} er enklest innen 1 SE og slår ankeret etter B-08",
        )
    else:
        selected, reason = (
            anchor_id,
            f"{chosen} slår ikke ankeret etter B-08; ankeret beholdes",
        )
    return {
        "selected": selected,
        "best": best,
        "within_1se": within,
        "table": table,
        "anchor_comparisons": anchor_comparisons,
        "reason": reason,
    }


def run_phase1_framework_checks(reference_cv, reference_models):
    """Kontroller fra faseplanens 3.12: sjekker forutsetningene for fase 1.

    Kjøres fra notebooken (etter :func:`configure`) i stedet for å ligge som
    en lang kodecelle der. Feiler høyt (``AssertionError``) hvis én av
    forutsetningene ikke holder:

    1. Pooled score fra samlede OOF-prediksjoner = eksponeringsvektet snitt
       av foldscorene.
    2. Poisson antall + offset $\\log e$ = rate med vekt $e$, også med
       splineledd.
    3. ``cr(..., df=k, constraints='center')`` gir $k$ kolonner og full rang,
       og ``n_parameters`` stemmer med opptelling av nivåer og df.
    4. Prediksjon utenfor treningsområdet krasjer ikke og er lineær på
       logskala.

    Parameters
    ----------
    reference_cv : dict
        CV-resultatene for referansemodellene fra seksjon 2.9. Må inneholde
        nøkkelen ``"frequency_policy_type_year"``.
    reference_models : dict
        De fittede referansemodellene fra seksjon 2.9, samme nøkler som
        ``reference_cv``.

    Returns
    -------
    None
        Funksjonen returnerer ingenting; den kaster ``AssertionError`` hvis
        en kontroll feiler, og printer ingenting hvis alt går bra.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from sklearn.metrics import mean_tweedie_deviance

    prepare_design_frame = _DEPENDENCIES["prepare_design_frame"]
    fit_glm = _DEPENDENCIES["fit_glm"]
    pooled_oof_deviance = _DEPENDENCIES["pooled_oof_deviance"]
    model_frame = _DEPENDENCIES["model_frame"]
    log_link = sm.families.links.Log()

    # 1. Pooled OOF-score = eksponeringsvektet snitt av foldscorene
    reference_result = reference_cv["frequency_policy_type_year"]
    reference_result = reference_result | {
        "spec": reference_models["frequency_policy_type_year"]["spec"]
    }
    assert np.isclose(
        pooled_oof_deviance(reference_result),
        np.average(
            reference_result["scores"]["val_deviance"],
            weights=reference_result["scores"]["val_weight"],
        ),
        rtol=1e-12,
    )

    # 2. Antall + offset = rate + vekt for en fase-1-spesifikasjon med splines
    spline_check = build_frequency_candidate(
        "check_df4",
        "F2",
        CONTROL_BLOCKS + CORE_BLOCKS,
        forms=dict.fromkeys(CONTINUOUS_FEATURES, "df4"),
        data=model_frame,
    )
    spline_design = prepare_design_frame(
        model_frame, model_frame, spline_check["required_columns"]
    )
    rate_fit = fit_glm(spline_check, spline_design)
    count_fit = smf.glm(
        "property_claims ~" + spline_check["formula"].split("~")[1],
        data=spline_design,
        family=sm.families.Poisson(link=log_link),
        offset=np.log(spline_design["total_exposure"]),
    ).fit()
    assert np.allclose(count_fit.params, rate_fit.params, rtol=1e-6, atol=1e-8)
    assert np.isclose(count_fit.deviance, rate_fit.deviance, rtol=1e-8)

    # 3. Sentrerte cr-baser: df kolonner og full rang; n_parameters stemmer med opptelling
    for feature in CONTINUOUS_FEATURES:
        for form in ("df3", "df4"):
            basis = patsy.dmatrix(
                CONTINUOUS_FORMS[form].format(x=feature), spline_design
            )
            assert basis.shape[1] == 1 + FORM_DF[form]  # intercept + df
            assert np.linalg.matrix_rank(basis) == basis.shape[1]
    n_levels = spline_design[
        [
            FEATURE_BLOCKS[b]["column"]
            for b in CONTROL_BLOCKS + CORE_BLOCKS
            if "label" not in FEATURE_BLOCKS[b]
        ]
    ].nunique()
    assert spline_check["n_parameters"] == 1 + (n_levels - 1).sum() + 3 * FORM_DF["df4"]
    assert np.linalg.matrix_rank(rate_fit.model.exog) == spline_check["n_parameters"]

    # 4. Prediksjon utenfor treningsområdet: endelig, positiv og lineær på logskala
    middle_ages = spline_design["driver_age"].between(30, 65)
    range_check = build_frequency_candidate(
        "check_range",
        "F2",
        ["product", "driver"],
        forms={"driver_age": "df4"},
        data=model_frame,
    )
    range_fit = fit_glm(range_check, spline_design.loc[middle_ages])
    outside = spline_design.iloc[[0] * 6].assign(driver_age=[18, 20, 22, 75, 80, 85])
    outside_prediction = range_fit.predict(outside).to_numpy()
    assert np.isfinite(outside_prediction).all() and (outside_prediction > 0).all()
    assert np.allclose(np.diff(np.log(outside_prediction[:3]), 2), 0, atol=1e-8)
    assert np.allclose(np.diff(np.log(outside_prediction[3:]), 2), 0, atol=1e-8)


def check_f0_matches_reference(f0_cv, reference_cv, reference_models):
    """Kontroller at F0 (3.2) gir samme pooled OOF-score som referansemodellen (2.9).

    F0 har bare produkt og år, samme som ``frequency_policy_type_year`` i 2.9,
    så de to spesifikasjonene skal være identiske og gi lik OOF-deviance.
    Flyttet ut av notebooken (seksjon 3.2) på samme måte som kontrollene i
    :func:`run_phase1_framework_checks`.

    Parameters
    ----------
    f0_cv : dict
        CV-resultatet for F0-kandidaten fra ``cross_validate_candidate``.
    reference_cv : dict
        CV-resultatene for referansemodellene fra seksjon 2.9. Må inneholde
        nøkkelen ``"frequency_policy_type_year"``.
    reference_models : dict
        De fittede referansemodellene fra seksjon 2.9, samme nøkler som
        ``reference_cv``. Brukes for ``spec``, som ``reference_cv`` selv ikke
        har.

    Returns
    -------
    None
        Kaster ``AssertionError`` hvis scorene ikke stemmer innenfor
        toleransen; returnerer ellers ingenting.
    """
    pooled_oof_deviance = _DEPENDENCIES["pooled_oof_deviance"]
    reference_result = reference_cv["frequency_policy_type_year"] | {
        "spec": reference_models["frequency_policy_type_year"]["spec"]
    }
    assert np.isclose(
        pooled_oof_deviance(f0_cv),
        pooled_oof_deviance(reference_result),
        rtol=1e-10,
    )
