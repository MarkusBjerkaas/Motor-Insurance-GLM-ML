"""Felles GLM-byggesteiner og fold-plumbing for GLM-modelleringen.

De fire generiske byggesteinene (``prepare_design_frame``, ``glm_spec``,
``fit_glm``, ``cross_validate_glm``) og hjelpefunksjonene
``cross_validate_glm`` kaller i hver fold (``prepare_fold_frames``,
``apply_derived_columns``, ``check_fold_fit``) ligger samlet her, slik at
både ``02_frekvens`` (frekvens/severity/pure premium) og nye
fasenotebooker kan importere dem uten å duplisere logikken. ``run_glm`` er
notebook-spesifikk (bruker ``TARGETS`` og ``display``) og ligger fortsatt i
notebooken.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import PatsyError
from sklearn.metrics import mean_tweedie_deviance


def prepare_design_frame(train, apply, required_columns, categorical=("year",)):
    """B-09: fyll manglende verdier i ``apply`` med regler lært på ``train``.

    Bare ``required_columns`` behandles. Numeriske kolonner får medianen fra
    ``train``. Kategoriske får nivået ``MISSING``, og nivåer i ``apply`` som ikke
    finnes i ``train`` gir ``ValueError`` med antall rader per nivå.
    """
    prepared = apply.copy()
    for column in required_columns:
        if column in categorical or not pd.api.types.is_numeric_dtype(train[column]):
            prepared[column] = prepared[column].fillna("MISSING")
            train_levels = train[column].fillna("MISSING")
            unseen_levels = set(prepared[column]) - set(train_levels)
            if unseen_levels:
                support = pd.DataFrame(
                    {
                        "rader_trening": train_levels.value_counts(),
                        "rader_bruk": prepared[column].value_counts(),
                    }
                ).fillna(0)
                raise ValueError(
                    f"{column} har nivåer uten treningsdata: {sorted(map(str, unseen_levels))}"
                    f"\nStøtte per nivå:\n{support.to_string()}"
                )
        else:
            prepared[column] = prepared[column].fillna(train[column].median())
    if not prepared[list(required_columns)].notna().all().all():
        raise ValueError("Manglende verdier gjenstår etter klargjøring av modellrammen.")
    return prepared


def glm_spec(
    name,
    x,
    y,
    data,
    family,
    weight,
    power,
    categorical=("year",),
    required_columns=None,
    derived=(),
    base_level_overrides=None,
):
    """Bygg patsy-formelen og innstillingene for én GLM.

    Kalles én gang per modell (ikke per fold). Leser dtypen til hver
    prediktor i ``x`` og bygger formelleddet: tekst/kategori (og ``year``,
    B-12) blir ``C(x, Treatment(basis))`` med basisnivået satt til nivået med
    størst eksponering (B-24); numeriske kolonner går inn lineært; alt som
    ikke er en kolonne i ``data`` (f.eks. ``cr(driver_age, df=3)``) sendes
    uendret til patsy.

    Parameters
    ----------
    name : str
        Modellens ID, brukt i tabeller og feilmeldinger.
    x : list of str
        Prediktorene, som kolonnenavn i ``data`` eller ferdige patsy-ledd.
    y : str
        Responskolonnen.
    data : pandas.DataFrame
        Datasettet formelen bygges mot. Brukes bare til å lese dtyper og
        basisnivåer, ikke lagret på spesifikasjonen.
    family : statsmodels family
        GLM-familien (med log-link) responsen skal estimeres med.
    weight : str
        Kolonnen med eksponering/vekt (``var_weights`` i ``fit_glm``).
    power : float
        Tweedie-$p$ scoren beregnes med (1 for Poisson, 2 for Gamma).
    categorical : tuple of str, optional
        Kolonner som alltid skal behandles som kategoriske selv om dtypen er
        numerisk, f.eks. ``year``. Standard ``("year",)``.
    required_columns : list of str or None, optional
        Råkolonnene imputasjonen (B-09) skal behandle. Standard er
        prediktorene i ``x`` som finnes i ``data``; patsy-ledd som ``cr(...)``
        og foldvis avledede kolonner (``derived``) må få råkolonnene sine
        oppgitt eksplisitt her, siden de ellers ikke kan tolkes ut av
        formelteksten.
    derived : tuple of dict, optional
        Foldvis avledede kolonner. Hvert element har ``name``, ``learn``
        (lærer tilstand på treningsdelen) og ``apply`` (bruker tilstanden på
        en vilkårlig ramme). Brukes til interaksjonenes sentrering (B-28) og
        tidsfoldens merkeliste (B-14).
    base_level_overrides : dict or None, optional
        Referansenivåer som settes eksplisitt i stedet for B-24-regelen
        (nivået med størst eksponering). Severity-fasen låser referansene i
        planen (S-06), og for ``year`` faller det låste nivået 2022 ikke
        sammen med nivået med størst eksponering. Referansenivået endrer
        verken tilpasningen eller deviancen, bare hvilke koeffisienter som
        rapporteres relativt til hva.

    Returns
    -------
    dict
        Spesifikasjonen ``cross_validate_glm``, ``fit_glm`` og ``run_glm``
        tar som ``spec``: ``name``, ``x``, ``y``, ``formula``, ``family``
        (navn), ``glm_family`` (statsmodels-objektet), ``weight``, ``power``,
        ``base_levels``, ``required_columns`` og ``derived``.
    """
    terms, base_levels = [], {}
    for predictor in x:
        if predictor not in data:  # ferdig patsy-ledd, f.eks. "cr(driver_age, df=3)"
            terms.append(predictor)
        elif predictor in categorical or not pd.api.types.is_numeric_dtype(
            data[predictor]
        ):
            # B-24: basisnivå = nivået med størst eksponering
            if base_level_overrides and predictor in base_level_overrides:
                base = base_level_overrides[predictor]  # låst referanse, f.eks. S-06
            else:
                base = data.groupby(predictor)["total_exposure"].sum().idxmax()
            base = base.item() if isinstance(base, np.generic) else base
            base_levels[predictor] = base
            terms.append(f"C({predictor}, Treatment({base!r}))")
        else:
            terms.append(predictor)
    if required_columns is None:
        required_columns = [predictor for predictor in x if predictor in data]
    return {
        "name": name,
        "x": list(x),
        "y": y,
        "formula": f"{y} ~ {' + '.join(terms) or '1'}",
        "family": type(family).__name__.lower(),  # "poisson", "gamma", "tweedie"
        "glm_family": family,  # statsmodels-familien som estimeres
        "weight": weight,
        "power": power,
        "base_levels": base_levels,
        "required_columns": list(required_columns),
        "derived": list(derived),
    }


def fit_glm(spec, data, cluster_groups=None, check_convergence=True, fit_kwargs=None):
    """Estimer en ferdig spesifikasjon på ``data``.

    Parameters
    ----------
    spec : dict
        Spesifikasjonen fra ``glm_spec``.
    data : pandas.DataFrame
        Datasettet modellen estimeres på. Må ha fylte manglende verdier
        (``prepare_design_frame``) for kolonnene i ``spec["required_columns"]``.
    cluster_groups : pandas.Series or None, optional
        Klyngevariabelen (typisk ``insured_id``) for cluster-robuste
        standardfeil (B-21). ``None`` gir modellbasert kovarians, brukt i
        hver CV-fold der bare punktestimatene trengs.
    check_convergence : bool, optional
        Om et ikke-konvergert fit skal kaste ``RuntimeError``. Slås av i
        ``cross_validate_glm``, som håndterer ikke-konvergens som en
        foldkontroll i stedet.
    fit_kwargs : dict or None, optional
        Ekstra argumenter til statsmodels' ``.fit()``, typisk
        ``{"maxiter": 200, "tol": 1e-8}``. ``None`` beholder statsmodels'
        standardinnstillinger, slik at eksisterende kall er uendret.

    Returns
    -------
    statsmodels GLMResults
        Den estimerte modellen.
    """
    covariance = (
        {}
        if cluster_groups is None
        else {"cov_type": "cluster", "cov_kwds": {"groups": cluster_groups}}  # B-21
    )
    fit_options = {**covariance, **(fit_kwargs or {})}
    with warnings.catch_warnings():
        # Statsmodels advarer generelt om cov_type + var_weights. Kontrollert:
        # cluster-SE er identiske med antall + offset og med manuell sandwich.
        warnings.filterwarnings("ignore", "cov_type not fully supported")
        result = smf.glm(
            spec["formula"],
            data=data,
            family=spec["glm_family"],
            var_weights=data[spec["weight"]],
        ).fit(**fit_options)
    if check_convergence and not result.converged:
        raise RuntimeError(f"{spec['name']} konvergerte ikke")
    return result


def prepare_fold_frames(spec, train, frames, prepare_design_frame):
    """Imputer (B-09) og legg til foldvis avledede kolonner, lært på ``train``.

    Parameters
    ----------
    spec : dict
        Modellspesifikasjon fra ``glm_spec``. Brukes for ``required_columns``
        og ``derived``.
    train : pandas.DataFrame
        Treningsdelen av folden. Imputasjonsreglene og de avledede kolonnenes
        tilstand (f.eks. et senter eller en merkeliste) læres herfra.
    frames : list of pandas.DataFrame
        Rammene (typisk trening og validering) som skal imputeres og få de
        avledede kolonnene lagt til, i samme rekkefølge som de sendes inn.
    prepare_design_frame : callable
        ``prepare_design_frame(train, apply, required_columns)`` fra
        notebooken (seksjon 2.6), som fyller manglende verdier lært på
        ``train`` inn i ``apply``.

    Returns
    -------
    prepared : list of pandas.DataFrame
        ``frames`` etter imputasjon og avledede kolonner, i samme rekkefølge.
    derived_state : dict
        Tilstanden hvert element i ``spec["derived"]`` lærte på
        treningsdelen, indeksert på kolonnenavn. Gjenbrukes av
        ``apply_derived_columns`` og av foldkallere som trenger senteret
        eller listen en avledet kolonne ble lært med.
    """
    train_design = prepare_design_frame(train, train, spec["required_columns"])
    derived_state = {
        item["name"]: item["learn"](train_design) for item in spec["derived"]
    }
    prepared = []
    for frame in frames:
        design = prepare_design_frame(train, frame, spec["required_columns"])
        prepared.append(apply_derived_columns(spec, design, derived_state))
    return prepared, derived_state


def apply_derived_columns(spec, frame, derived_state):
    """Legg foldvis avledede kolonner til ``frame`` med tilstand lært i trening.

    Parameters
    ----------
    spec : dict
        Modellspesifikasjon fra ``glm_spec``. ``spec["derived"]`` er en liste
        med dict-er med ``name``, ``learn`` og ``apply``.
    frame : pandas.DataFrame
        Rammen (trenings- eller valideringsdel) de avledede kolonnene skal
        legges til i.
    derived_state : dict
        Tilstanden hvert element i ``spec["derived"]`` lærte på
        treningsdelen, som returnert av ``prepare_fold_frames``.

    Returns
    -------
    pandas.DataFrame
        ``frame`` med de avledede kolonnene lagt til.
    """
    for item in spec["derived"]:
        frame = item["apply"](frame, derived_state[item["name"]])
    return frame


def check_fold_fit(spec, result, train, val, train_prediction, val_prediction):
    """Foldkontrollene fra plan 3.11 pkt. 6.

    En kandidat er bare gyldig når *alle* folder passerer disse kontrollene:
    full rang i designmatrisen, konvergens, samme antall rader inn og ut,
    endelige og positive prediksjoner, endelige parametere og at hvert
    kategorinivå i formelen har positiv respons i treningsfolden.

    Parameters
    ----------
    spec : dict
        Modellspesifikasjon fra ``glm_spec``. Brukes for ``y``, ``weight`` og
        ``base_levels`` (kategoriske prediktorer med basisnivå).
    result : statsmodels GLMResults
        Den estimerte modellen for denne folden.
    train : pandas.DataFrame
        Den imputerte treningsdelen (etter ``prepare_fold_frames``).
    val : pandas.DataFrame
        Den imputerte valideringsdelen (etter ``prepare_fold_frames``).
    train_prediction : array-like
        Predikert respons på ``train``.
    val_prediction : array-like
        Predikert respons på ``val``.

    Returns
    -------
    problems : list of str
        Beskrivelse av hver kontroll som feiler. Tom liste betyr at folden er
        gyldig.
    rank : int
        Rangen til designmatrisen (``result.model.exog``), lagret uavhengig
        av om kontrollen feiler, for diagnostikk.
    """
    problems = []
    exog = result.model.exog
    rank = np.linalg.matrix_rank(exog)
    if rank < exog.shape[1]:
        problems.append(f"rangsvikt ({rank} av {exog.shape[1]} kolonner)")
    if not result.converged:
        problems.append("konvergerte ikke")
    if len(result.model.endog) != len(train) or len(val_prediction) != len(val):
        problems.append("radantall endret i designmatrisen")
    for label, prediction in [
        ("trening", train_prediction),
        ("validering", val_prediction),
    ]:
        values = np.asarray(prediction, dtype=float)
        if not (np.isfinite(values).all() and (values > 0).all()):
            problems.append(f"ikke-endelige eller ikke-positive prediksjoner ({label})")
    if not np.isfinite(result.params).all():
        problems.append("ikke-endelige parametere")
    # Kategoristøtte: hvert nivå i formelen må ha positiv respons i treningsdelen
    weighted_response = train[spec["y"]] * train[spec["weight"]]
    for column in spec["base_levels"]:
        response_by_level = weighted_response.groupby(train[column]).sum()
        empty_levels = response_by_level.index[response_by_level.le(0)].tolist()
        if empty_levels:
            problems.append(
                f"{column} har nivåer uten respons i trening: {empty_levels}"
            )
    return problems, rank


def cross_validate_glm(
    spec, data, folds, prediction_data=None, fold_hook=None, fit_kwargs=None
):
    """Estimer og scor én ferdig spesifikasjon out-of-fold.

    I hver fold: manglende-reglene læres på treningsdelen og brukes på begge
    deler (``prepare_fold_frames``, B-09/B-28); modellen estimeres på
    treningsdelen; trenings- og valideringsdelen predikeres og scores med
    vektet Tweedie-deviance; nullmodellen (vektet snitt i treningsdelen)
    scores på valideringsdelen og gir $D^2$; foldkontrollene
    (``check_fold_fit``, plan 3.11 pkt. 6) avgjør om OOF-prediksjonen tas med.

    Parameters
    ----------
    spec : dict
        Spesifikasjonen fra ``glm_spec``.
    data : pandas.DataFrame
        Populasjonen modellen fittes og scores på (f.eks. severity-utvalget).
        Foldene kan indeksere hele modellpopulasjonen; de skjæres mot
        ``data.index``.
    folds : list of dict
        Foldene fra ``build_group_folds``, hver med ``fold`` (navn),
        ``train_index`` og ``val_index``.
    prediction_data : pandas.DataFrame or None, optional
        Populasjonen som skal få OOF-prediksjoner, og som må inneholde alle
        rader i ``data`` (typisk hele modellrammen når ``data`` er en
        delmengde). Manglende-regler og avledede kolonner læres fortsatt bare
        på treningsdelen av ``data``. ``None`` betyr ``data``, altså uendret
        oppførsel.
    fold_hook : callable or None, optional
        ``fold_hook(spec, fold_name, result, train_design, val_design,
        derived_state)``, kalt for hver gyldig fold. Kan returnere en dict
        med DataFrames (f.eks. relative kurver); de samles per nøkkel på
        tvers av foldene og legges til i returverdien. ``None`` betyr ingen
        ekstra uttrekk.
    fit_kwargs : dict or None, optional
        Sendes uendret videre til ``fit_glm`` i hver fold (f.eks.
        ``{"maxiter": 200, "tol": 1e-8}``, som severity-fasen låser).

    Returns
    -------
    dict
        ``oof`` (pandas.Series over ``data`` med OOF-prediksjoner, NaN for
        folder som ikke passerte kontrollene), ``prediction_oof`` (samme, men
        over ``prediction_data``), ``scores`` (scorepopulasjonen) (DataFrame, én rad per fold),
        ``params`` og ``param_se`` (DataFrame, én kolonne per fold, brukt til
        stabilitetssjekk), ``support`` (kategoristøtte per fold og nivå),
        ``valid`` (bool, ``True`` bare når alle folder passerte kontrollene)
        og ``error`` (feilbeskrivelse eller ``None``), samt eventuelle
        nøkler fra ``fold_hook``.
    """
    response, weight = spec["y"], spec["weight"]
    if prediction_data is None:
        prediction_data = data
    elif not data.index.isin(prediction_data.index).all():
        raise ValueError("prediction_data må inneholde alle rader i data.")
    oof_predictions = pd.Series(np.nan, index=data.index, name=spec["name"])
    prediction_oof = pd.Series(np.nan, index=prediction_data.index, name=spec["name"])
    fold_scores, fold_params, fold_se, fold_support, errors = [], [], [], [], []
    extras = {}

    def score(part, prediction):
        return mean_tweedie_deviance(
            part[response], prediction, sample_weight=part[weight], power=spec["power"]
        )

    for fold in folds:
        train = data.loc[data.index.intersection(fold["train_index"])]
        val = data.loc[data.index.intersection(fold["val_index"])]
        # Hele valideringsfolden som skal predikeres (lik val uten prediction_data)
        apply = prediction_data.loc[
            prediction_data.index.intersection(fold["val_index"])
        ]
        try:
            (train_design, val_design, apply_design), derived_state = (
                prepare_fold_frames(
                    spec, train, [train, val, apply], prepare_design_frame
                )
            )
            result = fit_glm(
                spec, train_design, check_convergence=False, fit_kwargs=fit_kwargs
            )
            train_prediction = result.predict(train_design)
            val_prediction = result.predict(val_design)
            apply_prediction = result.predict(apply_design)
        except (ValueError, np.linalg.LinAlgError, PatsyError) as error:
            errors.append(f"{fold['fold']}: {error}")
            continue
        problems, rank = check_fold_fit(
            spec, result, train_design, val_design, train_prediction, val_prediction
        )
        apply_values = np.asarray(apply_prediction, dtype=float)
        if not (np.isfinite(apply_values).all() and (apply_values > 0).all()):
            problems.append("ikke-endelige eller ikke-positive prediksjoner (full validering)")
        errors.extend(f"{fold['fold']}: {problem}" for problem in problems)
        if not problems:  # OOF bare fra folder som passerer kontrollene
            oof_predictions.loc[val.index] = val_prediction
            prediction_oof.loc[apply.index] = apply_prediction

        # Pearson-dispersjon på treningsdelen (antallsskala for frekvens)
        pearson_phi = result.pearson_chi2 / result.df_resid
        null_prediction = np.full(
            len(val), np.average(train[response], weights=train[weight])
        )
        fold_scores.append(
            {
                "model": spec["name"],
                "fold": fold["fold"],
                "n_train": len(train),
                "n_val": len(val),
                "train_weight": train[weight].sum(),
                "val_weight": val[weight].sum(),
                "train_deviance": score(train, train_prediction),
                "val_deviance": score(val, val_prediction),
                "val_null_deviance": score(val, null_prediction),
                "val_actual": (val[response] * val[weight]).sum(),
                "val_predicted": (val_prediction * val[weight]).sum(),
                "val_balance": (val_prediction * val[weight]).sum()
                / (val[response] * val[weight]).sum(),
                "rank": rank,
                "n_params": result.model.exog.shape[1],
                "converged": bool(result.converged),
                "pearson_phi": pearson_phi,
            }
        )
        fold_params.append(result.params.rename(fold["fold"]))
        # SE skalert med Pearson-φ̂ (bse har allerede statsmodels' scale, som er 1 for Poisson)
        fold_se.append(
            (result.bse * np.sqrt(pearson_phi / result.scale)).rename(fold["fold"])
        )
        for column in spec["base_levels"]:
            fold_support.append(
                pd.DataFrame(
                    {
                        "train_weight": train_design.groupby(column)[weight].sum(),
                        "train_response": (
                            train_design[response] * train_design[weight]
                        )
                        .groupby(train_design[column])
                        .sum(),
                    }
                )
                .rename_axis("level")
                .reset_index()
                .assign(fold=fold["fold"], feature=column)
            )
        if fold_hook is not None and not problems:
            hook_output = fold_hook(
                spec, fold["fold"], result, train_design, val_design, derived_state
            )
            for key, frame in hook_output.items():
                extras.setdefault(key, []).append(frame)

    scores = pd.DataFrame(fold_scores)
    if len(scores):
        scores["val_d2"] = 1 - scores["val_deviance"] / scores["val_null_deviance"]
    return {
        "oof": oof_predictions,
        "prediction_oof": prediction_oof,
        "scores": scores,
        "params": pd.concat(fold_params, axis=1) if fold_params else pd.DataFrame(),
        "param_se": pd.concat(fold_se, axis=1) if fold_se else pd.DataFrame(),
        "support": pd.concat(fold_support, ignore_index=True)
        if fold_support
        else pd.DataFrame(),
        **{key: pd.concat(frames, ignore_index=True) for key, frames in extras.items()},
        "valid": not errors and len(fold_scores) == len(folds),
        "error": "; ".join(errors) or None,
    }
