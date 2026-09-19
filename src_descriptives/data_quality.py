"""Datakvalitet, variabeldokumentasjon og integritetsdiagnostikk.

Notebooken bruker dette modulet som et tynt presentasjonslag: beregningene og
reglene ligger her, mens tabellene vises i ``analysis.py``.
"""

import numpy as np
import pandas as pd

_VARIABLE_ROWS = [
    ("insured_id", "Polise og kontrakt", "Anonym, sekvensiell identifikator for en polise. Samme ID kan forekomme i flere kalenderår.", "ID; ingen måleenhet", "Nøkkel, skal ikke brukes direkte som prediktor", "Int32", "Nei"),
    ("year", "Polise og kontrakt", "Kalenderåret som poliseobservasjonen gjelder.", "2022, 2023 eller 2024", "Tidsindeks", "Int16", "Nei"),
    ("policy_type", "Polise og kontrakt", "Overordnet produkt-/dekningsstruktur for polisen.", "TP=ansvar; TPG=ansvar+glass; CC=ansvar+minst to tilleggsdekninger; COMP_E=kasko med egenandel; COMP_N=kasko uten egenandel", "Kategorisk risikofaktor", "category", "Nei"),
    ("policy_status", "Polise og kontrakt", "Status for polisen i det aktuelle kalenderåret.", "A=aktiv; C=kansellert/opphørt", "Kategorisk kontraktsinformasjon", "category", "Nei"),
    ("business_type", "Polise og kontrakt", "Angir om observasjonen er nytegnet forretning eller eksisterende portefølje.", "NB=nytegning; P=portefølje/fornyelse", "Kategorisk kontraktsinformasjon", "category", "Nei"),
    ("payment_frequency", "Polise og kontrakt", "Avtalt betalingsfrekvens for premien.", "A=årlig; S=halvårlig; Q=kvartalsvis", "Kategorisk kontraktsinformasjon", "category", "Nei"),
    ("bonus_score", "Polise og kontrakt", "Grov klassifisering av forsikringstakerens tidligere skadeerfaring.", "G=god; N=nøytral; B=dårlig skadehistorikk", "Kategorisk/ordinal risikofaktor", "category", "Nei"),
    ("driver_age", "Fører og kjøretøy", "Alder på hovedføreren som er registrert på polisen.", "Hele år", "Numerisk risikofaktor", "Int16", "Nei"),
    ("vehicle_age", "Fører og kjøretøy", "Kildens dokumentasjon kaller feltet kjøretøyalder, men dataene er nesten identiske med utledet førerkortansiennitet. Behandles derfor kun som uavklart diagnostisk felt; reell kjøretøyalder er utilgjengelig.", "Hele år ifølge kilden; observert tolkning er førerkortansiennitet", "Uavklart diagnostisk felt", "Int16", "Ja"),
    ("age_driving_licence", "Fører og kjøretøy", "Kildedokumentasjonen er selvmotsigende. Dataene indikerer svært sterkt at feltet er alder ved førerkorterverv, ikke kalenderår eller ansiennitet. Dette er en arbeidshypotese, ikke bevist uten publisert transformasjonskode.", "Alder i hele år (observert tolkning/arbeidshypotese)", "Numerisk risikofaktor", "Int16", "Ja"),
    ("fuel_type", "Fører og kjøretøy", "Drivstofftype for kjøretøyet.", "D=diesel; G=bensin", "Kategorisk risikofaktor", "category", "Ja"),
    ("vehicle_value", "Fører og kjøretøy", "Oppgitt forsikringsverdi for kjøretøyet. Artikkelen bruker euro i premiepresentasjonen, men variabelarket oppgir ikke valuta eksplisitt for dette feltet.", "Beløp; valuta ikke eksplisitt angitt i variabelarket", "Numerisk risikofaktor", "float64", "Ja"),
    ("seats", "Fører og kjøretøy", "Antall registrerte sitteplasser i kjøretøyet.", "Heltallsantall", "Numerisk risikofaktor", "Int16", "Nei"),
    ("power_to_weight_ratio", "Fører og kjøretøy", "Til tross for feltnavnet beskriver kilden dette som kjøretøyets omtrentlige vekt per hestekraft, brukt som ytelsesindikator.", "kg per hk ifølge kilden", "Numerisk risikofaktor", "float64", "Ja"),
    ("vehicle_brand", "Fører og kjøretøy", "Standardisert bilmerke registrert på polisen.", "68 observerte merkenavn", "Kategorisk risikofaktor", "category", "Nei"),
    ("municipality_type", "Fører og kjøretøy", "Geografisk type for kommunen der kjøretøyet er forsikret.", "I=innland; C=kyst; IS=øyer", "Kategorisk geografisk faktor", "category", "Nei"),
    ("circulation_area", "Fører og kjøretøy", "Om kjøretøyet hovedsakelig brukes i urbant eller ruralt område.", "U=urban; R=rural", "Kategorisk geografisk faktor", "category", "Nei"),
]

_PREMIUM_DESCRIPTIONS = {
    "total_premium": "Total nettopremie, lik summen av alle dekningspremiene i raden.",
    "liability_premium": "Nettopremie for ansvarsforsikring.",
    "property_damage_premium": "Nettopremie for egen kjøretøyskade/kaskoskade.",
    "theft_premium": "Nettopremie for tyveridekning.",
    "fire_premium": "Nettopremie for branndekning.",
    "glass_premium": "Nettopremie for glassdekning.",
    "legal_protection_premium": "Nettopremie for rettshjelpsdekning.",
    "occupants_premium": "Nettopremie for personskadedekning for passasjerer.",
}
_CLAIM_DESCRIPTIONS = {
    "total_claims": "Totalt antall meldte skader på tvers av dekninger i eksponeringsperioden.",
    "liability_claims": "Totalt antall ansvarsskader; sum av materielle skader og personskader under ansvar.",
    "liability_property_claims": "Antall ansvarsskader med materiell skade på tredjepart.",
    "liability_injury_claims": "Antall ansvarsskader med personskade på tredjepart.",
    "property_claims": "Antall skader på eget kjøretøy under kaskodekning.",
    "theft_claims": "Antall tyveriskader.",
    "fire_claims": "Antall brannskader.",
    "glass_claims": "Antall glasskader.",
    "legal_protection_claims": "Antall rettshjelpsskader/-saker.",
    "occupants_claims": "Antall skader under passasjerdekningen.",
}
_INCURRED_DESCRIPTIONS = {
    "total_incurred": "Total påløpt skadekostnad, lik summen av incurred-beløpene på dekningsnivå.",
    "liability_incurred": "Påløpt kostnad for ansvarsskader; sum av materiell skade og personskade under ansvar.",
    "liability_property_incurred": "Påløpt kostnad for materielle ansvarsskader.",
    "liability_injury_incurred": "Påløpt kostnad for personskader under ansvar.",
    "property_incurred": "Påløpt kostnad for skade på eget kjøretøy.",
    "theft_incurred": "Påløpt kostnad for tyveriskader.",
    "fire_incurred": "Påløpt kostnad for brannskader.",
    "glass_incurred": "Påløpt kostnad for glasskader.",
    "legal_protection_incurred": "Påløpt kostnad for rettshjelpsskader/-saker.",
    "occupants_incurred": "Påløpt kostnad under passasjerdekningen.",
}


def build_variable_dictionary():
    """Bygg og valider den forbedrede variabelordboken."""
    rows = list(_VARIABLE_ROWS)
    for variable, description in _PREMIUM_DESCRIPTIONS.items():
        rows.append((variable, "Premier", description, "Beløp (artikkelen presenterer premier i EUR); null betyr normalt at dekningen ikke er tegnet eller at eksponeringen er null", "Premie", "float64", "Nei"))
    for variable, description in _CLAIM_DESCRIPTIONS.items():
        rows.append((variable, "Skadeantall", description, "Ikke-negativt heltallsantall", "Respons-/utfallsvariabel", "Int16", "Nei"))
    for variable, description in _INCURRED_DESCRIPTIONS.items():
        rows.append((variable, "Påløpt skadekostnad", description, "Beløp (antatt EUR); betalt beløp pluss utestående reserve", "Respons-/utfallsvariabel", "float64", "Nei"))
    rows.extend([
        ("total_exposure", "Eksponering", "Tid polisen var eksponert for risiko i kalenderåret.", "Poliseår i intervallet [0, 1]", "Eksponering/offset", "float64", "Nei"),
        ("liability_exposure", "Eksponering", "Eksponeringstid spesifikt for ansvarsdekningen.", "Poliseår i intervallet [0, 1]", "Eksponering/offset", "float64", "Nei"),
    ])
    dictionary = pd.DataFrame(rows, columns=["variable", "group", "description", "unit_or_codes", "analysis_role", "clean_dtype", "missing_allowed"])
    if dictionary["variable"].duplicated().any():
        raise ValueError("Variabelordboken inneholder duplikate variabelnavn.")
    return dictionary


def find_variables(variable_dictionary, search=None, group=None):
    """Finn variabler etter tekst og/eller eksakt gruppenavn."""
    result = variable_dictionary.copy()
    if group is not None:
        result = result.loc[result["group"].eq(group)]
    if search is not None:
        text = result[["variable", "description", "unit_or_codes", "analysis_role"]].astype("string").agg(" ".join, axis=1)
        result = result.loc[text.str.contains(search, case=False, regex=False, na=False)]
    return result.reset_index(drop=True)


CATEGORICAL_LEVELS = {
    "policy_type": ["TP", "TPG", "CC", "COMP_E", "COMP_N"], "policy_status": ["A", "C"],
    "business_type": ["NB", "P"], "payment_frequency": ["A", "S", "Q"], "bonus_score": ["G", "N", "B"],
    "fuel_type": ["D", "G"], "municipality_type": ["I", "C", "IS"], "circulation_area": ["U", "R"],
}
PREMIUM_COMPONENTS = ["liability_premium", "property_damage_premium", "theft_premium", "fire_premium", "glass_premium", "legal_protection_premium", "occupants_premium"]
CLAIM_COMPONENTS = ["liability_claims", "property_claims", "theft_claims", "fire_claims", "glass_claims", "legal_protection_claims", "occupants_claims"]
INCURRED_COMPONENTS = ["liability_incurred", "property_incurred", "theft_incurred", "fire_incurred", "glass_incurred", "legal_protection_incurred", "occupants_incurred"]
INTEGER_DTYPES = {"insured_id": "Int32", "year": "Int16", "driver_age": "Int16", "vehicle_age": "Int16", "age_driving_licence": "Int16", "seats": "Int16", **{variable: "Int16" for variable in _CLAIM_DESCRIPTIONS}}
FLOAT_COLUMNS = [*_PREMIUM_DESCRIPTIONS, "vehicle_value", "power_to_weight_ratio", *_INCURRED_DESCRIPTIONS, "total_exposure", "liability_exposure"]


def validate_source_schema(frame, variable_dictionary):
    """Stopp før rensing dersom kildefilen har uventet skjema eller koder."""
    expected = tuple(variable_dictionary["variable"])
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        raise ValueError(f"Uventet skjema. Mangler: {missing}; ekstra: {extra}.")
    if tuple(frame.columns) != expected:
        raise ValueError("Kolonnerekkefølgen avviker fra det dokumenterte skjemaet.")
    if frame.columns.duplicated().any():
        raise ValueError("Kilden inneholder duplikate kolonnenavn.")
    duplicate_keys = int(frame.duplicated(["insured_id", "year"]).sum())
    if duplicate_keys:
        raise ValueError(f"Kilden inneholder {duplicate_keys} duplikate poliseår.")
    for column, levels in CATEGORICAL_LEVELS.items():
        observed = set(frame[column].dropna().astype("string").str.strip().unique())
        unexpected = sorted(observed - set(levels))
        if unexpected:
            raise ValueError(f"{column} inneholder ukjente koder: {unexpected}.")
    years = set(pd.to_numeric(frame["year"], errors="raise").unique())
    if not years.issubset({2022, 2023, 2024}):
        raise ValueError(f"Uventede kalenderår: {sorted(years)}.")


def clean_motor_data(frame, variable_dictionary):
    """Utfør konservative og dokumenterte transformasjoner av publisert CSV."""
    validate_source_schema(frame, variable_dictionary)
    cleaned = frame.loc[:, variable_dictionary["variable"]].copy()
    whitespace_changes = 0
    for column in [*CATEGORICAL_LEVELS, "vehicle_brand"]:
        original = cleaned[column].astype("string")
        stripped = original.str.strip()
        whitespace_changes += int((original.notna() & original.ne(stripped)).sum())
        cleaned[column] = stripped
    cleaned["vehicle_brand"] = cleaned["vehicle_brand"].str.upper().astype("category")
    for column, levels in CATEGORICAL_LEVELS.items():
        cleaned[column] = cleaned[column].astype(pd.CategoricalDtype(categories=levels))
    for column, dtype in INTEGER_DTYPES.items():
        values = pd.to_numeric(cleaned[column], errors="raise")
        if values.dropna().mod(1).ne(0).any():
            raise ValueError(f"{column} inneholder verdier som ikke er heltall.")
        cleaned[column] = values.astype(dtype)
    for column in FLOAT_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise").astype("float64")
    invalid_power_ratio = cleaned["power_to_weight_ratio"].le(0)
    cleaned.loc[invalid_power_ratio, "power_to_weight_ratio"] = np.nan
    missing_liability_exposure = (
        cleaned["insured_id"].eq(23744)
        & cleaned["year"].eq(2022)
    )
    # Korreksjonen gjelder én rad i 2022; en ramme uten 2022 (f.eks. bare teståret) har ingen.
    expected_corrections = 1 if cleaned["year"].eq(2022).any() else 0
    if int(missing_liability_exposure.sum()) != expected_corrections:
        raise ValueError(
            "Forventet nøyaktig én avtalt korreksjon av ansvarseksponering."
        )
    if not (
        cleaned.loc[missing_liability_exposure, "liability_claims"].gt(0).all()
        and cleaned.loc[missing_liability_exposure, "liability_exposure"].eq(0).all()
    ):
        raise ValueError(
            "Avtalt eksponeringskorreksjon stemmer ikke med kildeobservasjonen."
        )
    cleaned.loc[
        missing_liability_exposure,
        ["total_exposure", "liability_exposure"],
    ] = 1.0
    log = pd.DataFrame([
        {"step": "Kontroll av skjema, kategorikoder og unik poliseår-nøkkel", "affected_rows": 0, "result": "Bestått"},
        {"step": "Fjerning av omkringliggende mellomrom i tekstfelt", "affected_rows": whitespace_changes, "result": "Standardisert"},
        {"step": "Ikke-positiv power_to_weight_ratio satt til manglende", "affected_rows": int(invalid_power_ratio.sum()), "result": "Standardisert"},
        {"step": "Manglende total- og ansvarseksponering satt til 1 for ansvarsskade", "affected_rows": int(missing_liability_exposure.sum()), "result": "Korrigert"},
        {"step": "Rader slettet eller imputert", "affected_rows": 0, "result": "Ingen"},
    ])
    return cleaned, log


def run_integrity_checks(frame, variable_dictionary):
    """Kjør alle radbaserte integritetskontroller og returner oppsummeringen."""
    required = variable_dictionary.loc[variable_dictionary["missing_allowed"].eq("Nei"), "variable"].tolist()
    numeric = frame.select_dtypes(include="number").columns
    finite = np.isfinite(frame[numeric]) | frame[numeric].isna()
    premium_sum = frame[PREMIUM_COMPONENTS].sum(axis=1)
    claim_sum = frame[CLAIM_COMPONENTS].sum(axis=1)
    incurred_sum = frame[INCURRED_COMPONENTS].sum(axis=1)
    checks = [
        ("Unik (insured_id, year)", "Kritisk", "Ingen duplikate poliseår", frame.duplicated(["insured_id", "year"])),
        ("Manglende i obligatoriske felt", "Kritisk", "Ingen manglende verdier", frame[required].isna().any(axis=1)),
        ("Endelige numeriske verdier", "Kritisk", "Ingen +inf eller -inf", ~finite.all(axis=1)),
        ("Ikke-negative premier", "Kritisk", "Alle premier >= 0", frame[["total_premium", *PREMIUM_COMPONENTS]].lt(0).any(axis=1)),
        ("Ikke-negative skadeantall", "Kritisk", "Alle skadeantall >= 0", frame[["total_claims", *CLAIM_COMPONENTS]].lt(0).any(axis=1)),
        ("Ikke-negative incurred-beløp", "Kritisk", "Alle incurred-beløp >= 0", frame[["total_incurred", *INCURRED_COMPONENTS]].lt(0).any(axis=1)),
        ("Eksponering innenfor gyldig område", "Kritisk", "Begge eksponeringer i [0, 1]", ~frame[["total_exposure", "liability_exposure"]].ge(0).all(axis=1) | ~frame[["total_exposure", "liability_exposure"]].le(1).all(axis=1)),
        ("Totalpremie summerer", "Kritisk", "total_premium = sum dekningspremier", ~np.isclose(frame["total_premium"], premium_sum, atol=1e-8, rtol=0)),
        ("Totalt skadeantall summerer", "Kritisk", "total_claims = sum skadeantall per dekning", frame["total_claims"].ne(claim_sum)),
        ("Ansvarsskadeantall summerer", "Kritisk", "liability_claims = property + injury under ansvar", frame["liability_claims"].ne(frame["liability_property_claims"] + frame["liability_injury_claims"])),
        ("Total incurred summerer", "Kritisk", "total_incurred = sum incurred per dekning", ~np.isclose(frame["total_incurred"], incurred_sum, atol=1e-8, rtol=0)),
        ("Ansvar incurred summerer", "Kritisk", "liability_incurred = property + injury under ansvar", ~np.isclose(frame["liability_incurred"], frame["liability_property_incurred"] + frame["liability_injury_incurred"], atol=1e-8, rtol=0)),
        ("Total- og ansvarseksponering er like", "Kritisk", "total_exposure = liability_exposure", ~np.isclose(frame["total_exposure"], frame["liability_exposure"], atol=1e-12, rtol=0)),
        ("Gyldig kjøretøyverdi", "Kritisk", "vehicle_value > 0 eller manglende", frame["vehicle_value"].notna() & frame["vehicle_value"].le(0)),
        ("Gyldig vekt/effekt-forhold", "Kritisk", "power_to_weight_ratio > 0 eller manglende", frame["power_to_weight_ratio"].notna() & frame["power_to_weight_ratio"].le(0)),
        ("Gyldig antall seter", "Kritisk", "seats er positivt heltall", frame["seats"].le(0)),
        ("Plausibel føreralder", "Kritisk", "driver_age i [18, 100]", ~frame["driver_age"].between(18, 100)),
        ("Ikke-negativ observert førerkortalder", "Kritisk", "age_driving_licence >= 0 eller manglende; tolkningen valideres separat som arbeidshypotese", frame["age_driving_licence"].notna() & frame["age_driving_licence"].lt(0)),
        ("Positiv ansvarseksponering med null ansvarspremie", "Undersøk", "Eksponeringen beholdes; null premie ved positiv totalpremie må avklares før premieanalyse", frame["liability_exposure"].gt(0) & frame["liability_premium"].eq(0) & frame["total_premium"].gt(0)),
        ("Skade eller incurred ved null eksponering", "Undersøk", "Null eksponering forventes normalt uten skadeaktivitet", frame["total_exposure"].eq(0) & (frame["total_claims"].gt(0) | frame["total_incurred"].gt(0))),
        ("Positiv incurred uten registrert skade", "Undersøk", "total_claims = 0 forventes normalt å gi total_incurred = 0", frame["total_claims"].eq(0) & frame["total_incurred"].gt(0)),
    ]
    result = pd.DataFrame([{"check": name, "severity": severity, "expectation": expectation, "violating_rows": int(pd.Series(mask, index=frame.index).sum())} for name, severity, expectation, mask in checks])
    result["status"] = np.where(result["violating_rows"].eq(0), "Bestått", "Undersøk")
    return result


def build_integrity_diagnostics(frame, integrity_checks):
    """Bygg alle øvrige integritetsoppsummeringer som vises i notebooken."""
    warning_masks = {
        "Positiv ansvarseksponering med null ansvarspremie": frame["liability_exposure"].gt(0) & frame["liability_premium"].eq(0) & frame["total_premium"].gt(0),
        "Skade eller incurred ved null eksponering": frame["total_exposure"].eq(0) & (frame["total_claims"].gt(0) | frame["total_incurred"].gt(0)),
    }
    example_columns = ["insured_id", "year", "policy_type", "policy_status", "business_type", "age_driving_licence", "liability_premium", "total_premium", "total_claims", "total_incurred", "total_exposure"]
    samples = [frame.loc[mask, example_columns].head(5).assign(check=name) for name, mask in warning_masks.items() if mask.any()]
    warning_examples = pd.concat(samples, ignore_index=True).loc[:, ["check", *example_columns]] if samples else pd.DataFrame(columns=["check", *example_columns])
    coverage_columns = {
        "Ansvar": ("liability_premium", "liability_claims", "liability_incurred"), "Egen skade": ("property_damage_premium", "property_claims", "property_incurred"),
        "Tyveri": ("theft_premium", "theft_claims", "theft_incurred"), "Brann": ("fire_premium", "fire_claims", "fire_incurred"), "Glass": ("glass_premium", "glass_claims", "glass_incurred"),
        "Rettshjelp": ("legal_protection_premium", "legal_protection_claims", "legal_protection_incurred"), "Passasjer": ("occupants_premium", "occupants_claims", "occupants_incurred"),
    }
    coverage = pd.DataFrame([{"coverage": name, "zero_premium_rows": int(frame[premium].eq(0).sum()), "claim_with_zero_premium": int((frame[claims].gt(0) & frame[premium].eq(0)).sum()), "incurred_with_zero_premium": int((frame[incurred].gt(0) & frame[premium].eq(0)).sum())} for name, (premium, claims, incurred) in coverage_columns.items()])
    temporal = build_temporal_diagnostics(frame)
    critical = integrity_checks.loc[integrity_checks["severity"].eq("Kritisk") & integrity_checks["violating_rows"].gt(0)]
    return {"integrity_checks": integrity_checks, "critical_failures": critical, "warning_examples": warning_examples, "coverage_diagnostics": coverage, "temporal_diagnostics": temporal}


def build_temporal_diagnostics(frame):
    """Oppsummer endringer i aldersfelt mellom observasjoner for samme polise."""
    ordered = frame.sort_values(["insured_id", "year"])
    grouped = ordered.groupby("insured_id", observed=True)
    year_gap = ordered["year"] - grouped["year"].shift()
    rows = []
    for column in ["driver_age", "vehicle_age", "age_driving_licence"]:
        change = ordered[column] - grouped[column].shift()
        comparable = year_gap.notna() & change.notna()
        rows.append({"variable": column, "comparable_transitions": int(comparable.sum()), "unchanged": int((comparable & change.eq(0)).sum()), "decrease": int((comparable & change.lt(0)).sum()), "increase_larger_than_year_gap": int((comparable & change.gt(year_gap)).sum())})
    return pd.DataFrame(rows)
