"""Kompakte diagnostiske tabeller før train/test-splitt."""

from itertools import combinations

import numpy as np
import pandas as pd

COVERAGES = {
    "Ansvar": ("liability_premium", "liability_claims", "liability_incurred"),
    "Egen skade": ("property_damage_premium", "property_claims", "property_incurred"),
    "Tyveri": ("theft_premium", "theft_claims", "theft_incurred"),
    "Brann": ("fire_premium", "fire_claims", "fire_incurred"),
    "Glass": ("glass_premium", "glass_claims", "glass_incurred"),
    "Rettshjelp": ("legal_protection_premium", "legal_protection_claims", "legal_protection_incurred"),
    "Passasjer": ("occupants_premium", "occupants_claims", "occupants_incurred"),
}

COMPOSITION_COLUMNS = [
    "policy_type",
    "policy_status",
    "business_type",
    "payment_frequency",
    "bonus_score",
    "fuel_type",
    "municipality_type",
    "circulation_area",
]


def build_time_summary(frame):
    """Oppsummer porteføljestørrelse og eksponering per kalenderår."""
    first_year = frame.groupby("insured_id", observed=True)["year"].transform("min")
    labelled = frame.assign(first_observed=frame["year"].eq(first_year))
    summary = labelled.groupby("year", observed=True).agg(
        poliseår=("insured_id", "size"),
        unike_poliser=("insured_id", "nunique"),
        samlet_eksponering=("total_exposure", "sum"),
        gjennomsnittlig_eksponering=("total_exposure", "mean"),
        fullårseksponering=("total_exposure", lambda s: s.eq(1).sum()),
        delårseksponering=("total_exposure", lambda s: s.gt(0).mul(s.lt(1)).sum()),
        null_eksponering=("total_exposure", lambda s: s.eq(0).sum()),
        først_observert_i_datasettet=("first_observed", "sum"),
    )
    summary["tidligere_observert"] = (
        summary["poliseår"] - summary["først_observert_i_datasettet"]
    )
    return summary.reset_index()


def build_panel_summary(frame):
    """Oppsummer observasjonslengde og overlapp mellom påfølgende år."""
    years_per_policy = frame.groupby("insured_id", observed=True).size()
    duration = (
        years_per_policy.value_counts()
        .sort_index()
        .rename_axis("observerte_år")
        .reset_index(name="antall_poliser")
    )
    years = sorted(frame["year"].unique())
    transitions = []
    for from_year, to_year in combinations(years, 2):
        from_ids = set(frame.loc[frame["year"].eq(from_year), "insured_id"])
        to_ids = set(frame.loc[frame["year"].eq(to_year), "insured_id"])
        transitions.append(
            {
                "overgang": f"{from_year} -> {to_year}",
                "poliser_i_begge_år": len(from_ids & to_ids),
                "kun_i_første_år": len(from_ids - to_ids),
                "kun_i_siste_år": len(to_ids - from_ids),
            }
        )
    return duration, pd.DataFrame(transitions)


def build_composition_summary(frame):
    """Oppsummer porteføljesammensetning uten å bruke skadeutfall."""
    summaries = []
    for column in COMPOSITION_COLUMNS:
        grouped = frame.groupby(["year", column], observed=True).agg(
            poliseår=("insured_id", "size"),
            eksponering=("total_exposure", "sum"),
        )
        grouped["andel_poliseår_prosent"] = grouped["poliseår"].div(
            grouped.groupby(level=0)["poliseår"].transform("sum")
        ).mul(100)
        grouped["andel_eksponering_prosent"] = grouped["eksponering"].div(
            grouped.groupby(level=0)["eksponering"].transform("sum")
        ).mul(100)
        summary = grouped.reset_index().rename(columns={column: "nivå"})
        summaries.append(summary.assign(variabel=column))
    return pd.concat(summaries, ignore_index=True)[
        [
            "year",
            "variabel",
            "nivå",
            "poliseår",
            "andel_poliseår_prosent",
            "eksponering",
            "andel_eksponering_prosent",
        ]
    ].round(2)


def build_brand_summary(frame):
    """Oppsummer bredde og konsentrasjon for høy-kardinalitetsvariabelen bilmerke."""
    rows = []
    for year, group in frame.groupby("year", observed=True):
        shares = group["vehicle_brand"].value_counts(normalize=True, dropna=False)
        rows.append(
            {
                "year": year,
                "unike_bilmerker": group["vehicle_brand"].nunique(dropna=True),
                "andel_ti_største_prosent": shares.head(10).sum() * 100,
                "poliseår_i_sjeldne_merker_prosent": shares[shares.lt(0.01)].sum() * 100,
            }
        )
    return pd.DataFrame(rows).round(2)


def build_top_brand_summary(frame):
    """Vis de ti vanligste bilmerkene per år."""
    summaries = []
    for year, group in frame.groupby("year", observed=True):
        top_brands = group["vehicle_brand"].value_counts().head(10)
        summaries.append(
            top_brands.rename_axis("vehicle_brand")
            .reset_index(name="poliseår")
            .assign(year=year)
        )
    return pd.concat(summaries, ignore_index=True)[
        ["year", "vehicle_brand", "poliseår"]
    ]


def _coverage_mask(frame, coverage, premium):
    if coverage == "Ansvar":
        return frame["liability_exposure"].gt(0)
    return frame["total_exposure"].gt(0) & frame[premium].gt(0)


def build_coverage_summary(frame):
    """Definer risikopopulasjon og vis premieavvik per dekning."""
    rows = []
    for coverage, (premium, claims, incurred) in COVERAGES.items():
        active = _coverage_mask(frame, coverage, premium)
        rule = (
            "liability_exposure > 0"
            if coverage == "Ansvar"
            else "total_exposure > 0 og dekningspremie > 0"
        )
        rows.append(
            {
                "dekning": coverage,
                "regel_for_aktiv_dekning": rule,
                "poliseår_i_risiko": int(active.sum()),
                "samlet_eksponering": frame.loc[active, "total_exposure"].sum(),
                "andel_av_porteføljen_prosent": active.mean() * 100,
                "poliseår_med_positiv_premie": int(frame[premium].gt(0).sum()),
                "skade_med_null_premie": int(
                    (frame[claims].gt(0) & frame[premium].eq(0)).sum()
                ),
                "incurred_med_null_premie": int(
                    (frame[incurred].gt(0) & frame[premium].eq(0)).sum()
                ),
            }
        )
    return pd.DataFrame(rows).round(2)


def build_coverage_by_policy_type(frame):
    """Vis hvilke premier som faktisk er positive innen hver produkttype."""
    summaries = []
    for coverage, (premium, _, _) in COVERAGES.items():
        grouped = frame.groupby(["year", "policy_type"], observed=True).agg(
            poliseår=("insured_id", "size"),
            positiv_premie=(premium, lambda s: s.gt(0).sum()),
        )
        grouped["andel_med_positiv_premie_prosent"] = grouped[
            "positiv_premie"
        ].div(grouped["poliseår"]).mul(100)
        summaries.append(grouped.reset_index().assign(dekning=coverage))
    return pd.concat(summaries, ignore_index=True).round(2)


def _model_scope(claim_count):
    if claim_count >= 10_000:
        return "Selvstendig modellering aktuell"
    if claim_count >= 1_000:
        return "Enklere modell eller sammenslåing"
    return "For lite for selvstendig detaljmodell"


def build_response_summary(frame):
    """Oppsummer responsvolum per dekning uten analyse mot risikofaktorer."""
    rows = []
    for coverage, (premium, claims, incurred) in COVERAGES.items():
        active = _coverage_mask(frame, coverage, premium)
        risk_set = frame.loc[active]
        claim_rows = risk_set[claims].gt(0)
        claim_count = risk_set[claims].sum()
        rows.append(
            {
                "dekning": coverage,
                "poliseår_i_risiko": len(risk_set),
                "samlet_eksponering": risk_set["total_exposure"].sum(),
                "skadeantall": claim_count,
                "poliseår_med_skade": int(claim_rows.sum()),
                "poliseår_med_flere_skader": int(risk_set[claims].gt(1).sum()),
                "samlet_incurred": risk_set[incurred].sum(),
                "skaderader_med_null_incurred": int(
                    (claim_rows & risk_set[incurred].eq(0)).sum()
                ),
                "modellomfang": _model_scope(claim_count),
            }
        )
    return pd.DataFrame(rows).round(2)


def build_exposure_summary(frame):
    """Kontroller nivå, fordeling og skadeaktivitet ved null eksponering."""
    rows = []
    for column in ["total_exposure", "liability_exposure"]:
        exposure = frame[column]
        rows.append(
            {
                "eksponering": column,
                "minimum": exposure.min(),
                "p25": exposure.quantile(0.25),
                "median": exposure.median(),
                "p75": exposure.quantile(0.75),
                "maksimum": exposure.max(),
                "null": int(exposure.eq(0).sum()),
                "delår": int(exposure.gt(0).mul(exposure.lt(1)).sum()),
                "fullår": int(exposure.eq(1).sum()),
            }
        )
    comparison = pd.DataFrame(
        {
            "kontroll": [
                "Ulik total- og ansvarseksponering",
                "Skade ved null totaleksponering",
                "Incurred ved null totaleksponering",
            ],
            "antall_poliseår": [
                int((~np.isclose(frame["total_exposure"], frame["liability_exposure"])).sum()),
                int(
                    (frame["total_exposure"].eq(0) & frame["total_claims"].gt(0)).sum()
                ),
                int(
                    (frame["total_exposure"].eq(0) & frame["total_incurred"].gt(0)).sum()
                ),
            ],
        }
    )
    return pd.DataFrame(rows).round(4), comparison


def build_own_damage_scope_validation(frame):
    """Valider modellpopulasjonen for egen-skadedekningen."""
    active = frame["total_exposure"].gt(0) & frame["property_damage_premium"].gt(0)
    labelled = frame.assign(own_damage_scope=active)
    summary = labelled.groupby("policy_type", observed=True).agg(
        poliseår=("insured_id", "size"),
        poliseår_med_positiv_eksponering=("total_exposure", lambda s: s.gt(0).sum()),
        poliseår_i_modell=("own_damage_scope", "sum"),
        skadeantall_i_modell=(
            "property_claims",
            lambda s: s[labelled.loc[s.index, "own_damage_scope"]].sum(),
        ),
    )
    summary["andel_av_positive_eksponeringer_prosent"] = summary[
        "poliseår_i_modell"
    ].div(summary["poliseår_med_positiv_eksponering"]).mul(100)
    outside_loss = ~active & (
        frame["property_claims"].gt(0) | frame["property_incurred"].gt(0)
    )
    exceptions = frame.loc[
        outside_loss,
        [
            "insured_id",
            "year",
            "policy_type",
            "policy_status",
            "total_exposure",
            "property_damage_premium",
            "property_claims",
            "property_incurred",
        ],
    ]
    return summary.reset_index().round(2), exceptions


def build_pre_split_diagnostics(frame):
    """Returner alle tabeller som trengs før train/test-splitt."""
    panel_duration, panel_transitions = build_panel_summary(frame)
    exposure_summary, exposure_checks = build_exposure_summary(frame)
    return {
        "time_summary": build_time_summary(frame),
        "panel_duration": panel_duration,
        "panel_transitions": panel_transitions,
        "composition_summary": build_composition_summary(frame),
        "brand_summary": build_brand_summary(frame),
        "top_brand_summary": build_top_brand_summary(frame),
        "coverage_summary": build_coverage_summary(frame),
        "coverage_by_policy_type": build_coverage_by_policy_type(frame),
        "response_summary": build_response_summary(frame),
        "exposure_summary": exposure_summary,
        "exposure_checks": exposure_checks,
    }
