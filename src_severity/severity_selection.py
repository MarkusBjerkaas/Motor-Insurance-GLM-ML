"""Seleksjonsregler for severity-fasen (S-08/S-09), som ren logikk.

Denne modulen implementerer forbedringsregelen, forenklingsregelen, regelen
for nesten like resultater og geografisærregelen slik de er låst i
``plans/Severity_plan.md`` (seksjon "Entydige utvalgsregler" og
"Seleksjonsalgoritmen, utført én gang") og gjengitt i notebooken
``glm_pricing_severity.py`` (2.6-2.7).

Modulen har ingen avhengighet til data, modeller eller andre deler av
prosjektet — den tar imot ferdig beregnede scorer (f.eks. fra
``severity_scoring.paired_gamma_gain``) og avgjør bare om terskelkravene er
oppfylt. All datainnhenting og modellfitting skjer andre steder.

Nøkkelnavn er norske for å være konsistente med resten av ``src_severity/``.
"""

import pandas as pd

# Standard relativ toleranse i alle terskler: 0,5 % av referansens deviance.
DEFAULT_RELATIVE_TOLERANCE = 0.005


def _relative_threshold(reference_deviance, relative_tolerance):
    """0,5 %-komponenten av terskelen: ``relative_tolerance * D_ref``."""
    return relative_tolerance * reference_deviance


def qualifies(
    gain,
    se_pair,
    reference_deviance,
    folds_improved,
    n_folds,
    candidate_parameters,
    reference_parameters,
    min_folds_improved=None,
    relative_tolerance=DEFAULT_RELATIVE_TOLERANCE,
):
    """Avgjør om én kandidat kvalifiserer mot én referanse.

    Regelvalg avgjøres av parametertallet i sammenligningsparet:

    - ``candidate_parameters < reference_parameters`` -> **forenklingsregelen**:
      ``D_kand - D_ref <= min(SE_par, 0,005 * D_ref)``, altså
      ``-gain <= min(se_pair, relative_tolerance * D_ref)``. Ikke-streng
      ulikhet (et tap akkurat lik terskelen kvalifiserer). Ingen krav til
      antall forbedrede folder.
    - ``candidate_parameters >= reference_parameters`` -> **forbedringsregelen**:
      ``gain > max(SE_par, 0,005 * D_ref)`` (streng ulikhet) OG
      ``folds_improved >= min_folds_improved``.

    Parametere
    ----------
    gain : float
        ``D_ref - D_kand``, positiv når kandidaten er bedre. Dette er
        ``"gevinst"`` fra ``severity_scoring.paired_gamma_gain``.
    se_pair : float
        Parvis cluster-robust standardfeil for scoreforskjellen
        (``"SE_cluster"`` fra ``paired_gamma_gain``).
    reference_deviance : float
        Pooled deviance for referansemodellen (``D_ref``).
    folds_improved : int
        Antall av ``n_folds`` folder der kandidaten hadde lavere deviance enn
        referansen. Kun relevant for forbedringsregelen.
    n_folds : int
        Totalt antall folder (5 i denne fasen).
    candidate_parameters, reference_parameters : int
        Antall parametere (med intercept) for hhv. kandidat og referanse.
    min_folds_improved : int, optional
        Eksplisitt foldkrav for forbedringsregelen. Standard er ``None``, som
        løses til ``n_folds - 1`` (4 av 5 i denne fasen) — planens faste krav,
        ikke en gjetning gjort inne i funksjonen uten at det vises utad.
    relative_tolerance : float, optional
        Den relative terskelkomponenten (standard 0,5 % = 0.005).

    Returnerer
    ----------
    dict med norske nøkler, gjennomsiktig for etterprøving:
        "regel": "forbedring" eller "forenkling"
        "gevinst": gain
        "tap": -gain (nyttig visning ved forenkling; negativ ved forbedring)
        "se_par": se_pair
        "terskel_se": se_pair
        "terskel_relativ": relative_tolerance * reference_deviance
        "terskel": terskelen som faktisk brukes (max hhv. min av de to over)
        "bindende_komponent": "SE_par" eller "0,5%_av_referanse"
        "folder_forbedret": folds_improved
        "n_folder": n_folds
        "krav_folder": min_folds_improved ved forbedring, None ved forenkling
        "kvalifiserer": bool
    """
    relative_threshold = _relative_threshold(reference_deviance, relative_tolerance)

    if candidate_parameters < reference_parameters:
        # Forenklingsregelen: ingen foldkrav, ikke-streng ulikhet.
        regel = "forenkling"
        threshold = min(se_pair, relative_threshold)
        bindende = "SE_par" if se_pair <= relative_threshold else "0,5%_av_referanse"
        loss = -gain
        kvalifiserer = loss <= threshold
        krav_folder = None
    else:
        # Forbedringsregelen: streng ulikhet + foldkrav.
        regel = "forbedring"
        threshold = max(se_pair, relative_threshold)
        bindende = "SE_par" if se_pair >= relative_threshold else "0,5%_av_referanse"
        if min_folds_improved is None:
            min_folds_improved = n_folds - 1
        kvalifiserer = (gain > threshold) and (folds_improved >= min_folds_improved)
        krav_folder = min_folds_improved

    return {
        "regel": regel,
        "gevinst": gain,
        "tap": -gain,
        "se_par": se_pair,
        "terskel_se": se_pair,
        "terskel_relativ": relative_threshold,
        "terskel": threshold,
        "bindende_komponent": bindende,
        "folder_forbedret": folds_improved,
        "n_folder": n_folds,
        "krav_folder": krav_folder,
        "kvalifiserer": kvalifiserer,
    }


def choose_near_tie(candidates, relative_tolerance=DEFAULT_RELATIVE_TOLERANCE):
    """Regelen for nesten like resultater blant allerede kvalifiserte kandidater.

    Finner laveste pooled deviance blant ``candidates`` ("beste"), og markerer
    som "nesten like" alle kandidater som ligger innenfor **både** én parvis
    SE mot beste OG ``relative_tolerance`` (0,5 %) av beste score. Blant de
    nesten like velges: færrest parametere -> deretter lavest deviance ->
    deretter alfabetisk ID.

    Parametere
    ----------
    candidates : pandas.DataFrame eller liste av dicts
        Må inneholde kolonnene/nøklene ``id``, ``deviance`` (pooled score),
        ``se_pair`` (parvis SE for denne kandidaten MOT beste kandidat) og
        ``parameters``. Alle kandidater her forutsettes allerede kvalifisert
        (kvalifiseringen gjøres av ``qualifies`` før denne funksjonen kalles).
    relative_tolerance : float, optional
        Standard 0,5 % (0.005), samme konstant som i ``qualifies``.

    ``se_pair`` for beste kandidat selv er degenerert (avstand til seg selv er
    0) og trengs ikke — den behandles alltid som nesten lik seg selv. For
    **enhver annen** kandidat er ``se_pair`` obligatorisk: finnes den ikke
    (NaN/None), reiser funksjonen ``ValueError`` i stedet for å gjette på en
    erstatning (jf. oppgavekravet).

    Returnerer
    ----------
    dict:
        "valgt": id til valgt kandidat
        "nesten_like": liste av id-er som er nesten like beste (inkl. beste selv)
        "begrunnelse": kort norsk forklaring på valget
        "tabell": DataFrame med kolonnene id, parameters, deviance,
            avstand_deviance, avstand_i_se, avstand_i_prosent, nesten_likt
    """
    df = pd.DataFrame(candidates).reset_index(drop=True).copy()
    if df.empty:
        raise ValueError("choose_near_tie mottok en tom kandidatliste.")

    best_position = df["deviance"].idxmin()
    best_deviance = df.loc[best_position, "deviance"]
    relative_threshold = _relative_threshold(best_deviance, relative_tolerance)

    distance = df["deviance"] - best_deviance

    # Kryss av manglende se_pair for alle kandidater UNNTATT beste selv.
    is_best = df.index == best_position
    missing_se = df["se_pair"].isna() & ~is_best
    if missing_se.any():
        missing_ids = df.loc[missing_se, "id"].tolist()
        raise ValueError(
            "se_pair mangler for kandidat(er) "
            f"{missing_ids}; kan ikke vurderes mot beste uten oppgitt parvis SE."
        )

    # Beste kandidat er alltid nesten lik seg selv (avstand 0).
    within_se = is_best | (distance <= df["se_pair"])
    within_relative = is_best | (distance <= relative_threshold)
    near_tie_mask = within_se & within_relative

    distance_in_se = pd.Series(0.0, index=df.index)
    non_best = ~is_best
    distance_in_se[non_best] = distance[non_best] / df.loc[non_best, "se_pair"]
    distance_in_percent = (distance / best_deviance) * 100.0

    table = df.assign(
        avstand_deviance=distance,
        avstand_i_se=distance_in_se,
        avstand_i_prosent=distance_in_percent,
        nesten_likt=near_tie_mask,
    )[
        [
            "id",
            "parameters",
            "deviance",
            "se_pair",
            "avstand_deviance",
            "avstand_i_se",
            "avstand_i_prosent",
            "nesten_likt",
        ]
    ]

    near_tie = table.loc[near_tie_mask].sort_values(
        by=["parameters", "deviance", "id"], ascending=[True, True, True]
    )
    chosen = near_tie.iloc[0]

    n_near_tie = len(near_tie)
    if n_near_tie == 1:
        begrunnelse = (
            f"{chosen['id']} valgt: eneste kandidat innenfor 1 SE og "
            f"{relative_tolerance * 100:.1f}% av beste score ({best_deviance:.6g})."
        )
    else:
        min_params = near_tie["parameters"].min()
        n_min_params = (near_tie["parameters"] == min_params).sum()
        if n_min_params == 1:
            begrunnelse = (
                f"{chosen['id']} valgt blant {n_near_tie} nesten like kandidater: "
                f"færrest parametere ({min_params})."
            )
        else:
            tied_on_params = near_tie.loc[near_tie["parameters"] == min_params]
            min_dev = tied_on_params["deviance"].min()
            n_min_dev = (tied_on_params["deviance"] == min_dev).sum()
            if n_min_dev == 1:
                begrunnelse = (
                    f"{chosen['id']} valgt: likt antall parametere ({min_params}) med "
                    f"{n_min_params - 1} annen/andre, lavest deviance avgjorde."
                )
            else:
                begrunnelse = (
                    f"{chosen['id']} valgt: likt parametertall ({min_params}) og "
                    f"likt score blant {n_min_dev} kandidater, alfabetisk ID avgjorde."
                )

    return {
        "valgt": chosen["id"],
        "nesten_like": near_tie["id"].tolist(),
        "begrunnelse": begrunnelse,
        "tabell": table,
    }


def apply_geography_rule(g2_vs_s0, g2_vs_g1, g1_valid, min_folds_improved=None):
    """Geografisærregelen (S-09, punkt 2): G2 mot både S0 og G1.

    Signatur: i stedet for en lang flat parameterliste tar funksjonen imot to
    ferdigpakkede dicts, én per sammenligning, med nøyaktig de nøklene
    ``qualifies`` krever: ``gain``, ``se_pair``, ``reference_deviance``,
    ``folds_improved``, ``n_folds``, ``candidate_parameters``,
    ``reference_parameters``. Kallende kode henter ``gain`` fra
    ``paired_gamma_gain``-resultatets ``"gevinst"``-nøkkel og ``se_pair`` fra
    ``"SE_cluster"``. Dette holder funksjonen fri for globale avhengigheter
    samtidig som den unngår en uoversiktlig 12+ parameter-signatur.

    Regelen selv (ordrett fra planen): G2 kvalifiserer bare hvis den består
    forbedringsregelen mot **både** S0 og G1, på de samme fem foldene. G1 må
    være gyldig for å være sammenligningsgrunnlag, men trenger ikke selv
    kvalifisere mot S0. Er G1 ugyldig, kvalifiserer ikke G2 uansett utfall mot
    G1 — en gevinst mot G1 kan aldri kompensere for at G2 ikke består kravet
    mot S0.

    Parametere
    ----------
    g2_vs_s0 : dict
        Argumenter til ``qualifies`` for sammenligningen G2 mot S0.
    g2_vs_g1 : dict
        Argumenter til ``qualifies`` for sammenligningen G2 mot G1.
    g1_valid : bool
        Om G1 besto gyldighetskontrollene (S-07) i alle fem folder.
    min_folds_improved : int, optional
        Videreføres til begge ``qualifies``-kall.

    Returnerer
    ----------
    dict:
        "kvalifiserer": bool — sluttresultatet for G2
        "g1_gyldig": g1_valid
        "mot_s0": resultatet fra qualifies(**g2_vs_s0), eller None hvis G1 er
            ugyldig og sammenligningen dermed er uten betydning (se begrunnelse)
        "mot_g1": resultatet fra qualifies(**g2_vs_g1), eller None hvis G1 er ugyldig
        "begrunnelse": kort norsk forklaring
    """
    if not g1_valid:
        return {
            "kvalifiserer": False,
            "g1_gyldig": False,
            "mot_s0": None,
            "mot_g1": None,
            "begrunnelse": (
                "G1 er ugyldig; tilleggsverdien av G2 kan ikke dokumenteres, "
                "så G2 kvalifiserer ikke uavhengig av score."
            ),
        }

    result_vs_s0 = qualifies(min_folds_improved=min_folds_improved, **g2_vs_s0)
    result_vs_g1 = qualifies(min_folds_improved=min_folds_improved, **g2_vs_g1)
    kvalifiserer = result_vs_s0["kvalifiserer"] and result_vs_g1["kvalifiserer"]

    if kvalifiserer:
        begrunnelse = "G2 består forbedringsregelen mot både S0 og G1: kvalifiserer."
    elif not result_vs_s0["kvalifiserer"] and not result_vs_g1["kvalifiserer"]:
        begrunnelse = "G2 består verken kravet mot S0 eller mot G1: kvalifiserer ikke."
    elif not result_vs_s0["kvalifiserer"]:
        begrunnelse = (
            "G2 består kravet mot G1, men ikke mot S0. En gevinst mot G1 kan "
            "ikke kompensere for manglende gevinst mot S0: kvalifiserer ikke."
        )
    else:
        begrunnelse = "G2 består kravet mot S0, men ikke mot G1: kvalifiserer ikke."

    return {
        "kvalifiserer": kvalifiserer,
        "g1_gyldig": True,
        "mot_s0": result_vs_s0,
        "mot_g1": result_vs_g1,
        "begrunnelse": begrunnelse,
    }


def build_selection_table(candidate_records):
    """Bygger én lesbar oppsummeringstabell over kvalifiseringen av kandidater.

    Parametere
    ----------
    candidate_records : liste av dicts
        Hver post beskriver én kandidatsammenligning og må inneholde:
        - "id": kandidat-ID (str)
        - "parameters": antall parametere (int)
        - "deviance": pooled deviance for kandidaten (float)
        - "qualifies_result": dict returnert av ``qualifies`` (eller av
          ``apply_geography_rule``s "mot_s0"/"mot_g1", som har samme form)

    Returnerer
    ----------
    pandas.DataFrame med norske kolonnenavn:
        ID, Parametere, Pooled deviance, Gevinst mot referanse, SE, Terskel,
        Regel, Folder forbedret, Kvalifiserer
    Ingen printing — kallende kode (notebook) viser tabellen selv.
    """
    rows = []
    for record in candidate_records:
        result = record["qualifies_result"]
        rows.append(
            {
                "ID": record["id"],
                "Parametere": record["parameters"],
                "Pooled deviance": record["deviance"],
                "Gevinst mot referanse": result["gevinst"],
                "SE": result["se_par"],
                "Terskel": result["terskel"],
                "Regel": result["regel"],
                "Folder forbedret": result["folder_forbedret"],
                "Kvalifiserer": result["kvalifiserer"],
            }
        )
    return pd.DataFrame(rows)
