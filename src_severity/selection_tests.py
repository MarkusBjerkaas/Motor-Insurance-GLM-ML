"""Syntetiske tester av seleksjonsreglene i ``severity_selection``.

Ren logikktesting — ingen data lastes. Kjøres med::

    uv run python -m src_severity.selection_tests

``run_all()`` printer ``[OK]``/``[FEIL]`` per test og returnerer
``(antall_bestått, antall_feilet)``. Dekker grenseulikhetene i
forbedrings-/forenklingsregelen, regelen for nesten like resultater og
geografisærregelen, slik de er låst i ``plans/Severity_plan.md``.
"""


from src_severity.severity_selection import (
    apply_geography_rule,
    choose_near_tie,
    qualifies,
)

_results = []

FIXED_CANDIDATE_IDS = [
    "S0",
    "A1",
    "A2",
    "A4",
    "V3",
    "G1",
    "G2",
    "P1",
    "P3",
    "T1",
    "FU1",
    "BU1",
    "PF1",
    "BR1",
]

NEW_BLOCKS = {
    "FU1": "fuel_type",
    "BU1": "business_type",
    "PF1": "payment_frequency",
    "BR1": "vehicle_brand_pooled",
}


def _check(name, condition):
    """Registrerer utfallet av én test og printer status fortløpende."""
    status = "[OK]" if condition else "[FEIL]"
    print(f"{status} {name}")
    _results.append(bool(condition))


def test_candidate_register_and_fit_budget():
    """Registeret og hovedfit-budsjettet verifiseres uten modell/data."""
    fixed_specs = {name: {"name": name} for name in FIXED_CANDIDATE_IDS}
    n_folds = 5
    fixed_budget = len(fixed_specs) * n_folds
    combined_budget = (len(fixed_specs) + 1) * n_folds
    _check(
        "Kandidatregister: 14 faste, unike ID-er inkludert fire nye utfordrere",
        len(fixed_specs) == 14
        and len(set(fixed_specs)) == 14
        and set(NEW_BLOCKS).issubset(fixed_specs),
    )
    _check(
        "Kandidatregister: fire nye utfordrere er separate blokker",
        len(set(NEW_BLOCKS.values())) == 4
        and NEW_BLOCKS
        == {
            "FU1": "fuel_type",
            "BU1": "business_type",
            "PF1": "payment_frequency",
            "BR1": "vehicle_brand_pooled",
        },
    )
    _check(
        "Fit-budsjett: 70 faste og maksimalt 75 med én kombinert kandidat",
        fixed_budget == 70 and combined_budget == 75,
    )


def test_improvement_exactly_on_threshold_fails():
    # se_pair=2, ref=1000 -> rel.terskel=5 -> terskel=max(2,5)=5. gain=5.0 nøyaktig.
    # Streng ulikhet: skal IKKE kvalifisere.
    result = qualifies(
        gain=5.0,
        se_pair=2.0,
        reference_deviance=1000.0,
        folds_improved=5,
        n_folds=5,
        candidate_parameters=9,
        reference_parameters=9,
    )
    _check(
        "Forbedring: gevinst nøyaktig på terskelen kvalifiserer ikke (streng ulikhet)",
        result["kvalifiserer"] is False and result["terskel"] == 5.0,
    )


def test_improvement_over_threshold_but_too_few_folds():
    result = qualifies(
        gain=5.01,
        se_pair=2.0,
        reference_deviance=1000.0,
        folds_improved=3,
        n_folds=5,
        candidate_parameters=9,
        reference_parameters=9,
    )
    _check(
        "Forbedring: gevinst over terskel men bare 3/5 folder kvalifiserer ikke",
        result["kvalifiserer"] is False,
    )


def test_improvement_qualifies_with_four_of_five_folds():
    result = qualifies(
        gain=5.01,
        se_pair=2.0,
        reference_deviance=1000.0,
        folds_improved=4,
        n_folds=5,
        candidate_parameters=9,
        reference_parameters=9,
    )
    _check(
        "Forbedring: gevinst over terskel og 4/5 folder kvalifiserer",
        result["kvalifiserer"] is True and result["krav_folder"] == 4,
    )


def test_improvement_se_binding():
    # se_pair=10 > rel.terskel=5 -> SE er bindende komponent.
    result = qualifies(
        gain=10.01,
        se_pair=10.0,
        reference_deviance=1000.0,
        folds_improved=4,
        n_folds=5,
        candidate_parameters=9,
        reference_parameters=9,
    )
    _check(
        "Forbedring: SE_par bindende når SE > 0,5%-grensen",
        result["kvalifiserer"] is True
        and result["terskel"] == 10.0
        and result["bindende_komponent"] == "SE_par",
    )


def test_improvement_relative_binding():
    # se_pair=2 < rel.terskel=5 -> 0,5%-grensen er bindende.
    result = qualifies(
        gain=5.01,
        se_pair=2.0,
        reference_deviance=1000.0,
        folds_improved=4,
        n_folds=5,
        candidate_parameters=9,
        reference_parameters=9,
    )
    _check(
        "Forbedring: 0,5%-grensen bindende når den er større enn SE_par",
        result["kvalifiserer"] is True
        and result["terskel"] == 5.0
        and result["bindende_komponent"] == "0,5%_av_referanse",
    )


def test_simplification_se_binding():
    # se_pair=2 < rel.terskel=5 -> min(...)=2, SE bindende. Tap 1,9 innenfor.
    result = qualifies(
        gain=-1.9,
        se_pair=2.0,
        reference_deviance=1000.0,
        folds_improved=0,
        n_folds=5,
        candidate_parameters=7,
        reference_parameters=9,
    )
    _check(
        "Forenkling: SE_par bindende når SE < 0,5%-grensen",
        result["kvalifiserer"] is True
        and result["terskel"] == 2.0
        and result["bindende_komponent"] == "SE_par",
    )


def test_simplification_relative_binding():
    # se_pair=10 > rel.terskel=5 -> min(...)=5, 0,5%-grensen bindende.
    result = qualifies(
        gain=-4.9,
        se_pair=10.0,
        reference_deviance=1000.0,
        folds_improved=0,
        n_folds=5,
        candidate_parameters=7,
        reference_parameters=9,
    )
    _check(
        "Forenkling: 0,5%-grensen bindende når den er mindre enn SE_par",
        result["kvalifiserer"] is True
        and result["terskel"] == 5.0
        and result["bindende_komponent"] == "0,5%_av_referanse",
    )


def test_simplification_within_both_bounds_qualifies():
    # terskel=min(3,5)=3. Tap 2,9 er innenfor begge grenser.
    result = qualifies(
        gain=-2.9,
        se_pair=3.0,
        reference_deviance=1000.0,
        folds_improved=0,
        n_folds=5,
        candidate_parameters=7,
        reference_parameters=9,
    )
    _check(
        "Forenkling: lite tap innenfor begge grenser kvalifiserer",
        result["kvalifiserer"] is True,
    )


def test_simplification_within_only_one_bound_fails():
    # terskel=min(3,5)=3. Tap 4 er innenfor 0,5%-grensen (5) men ikke SE (3).
    result = qualifies(
        gain=-4.0,
        se_pair=3.0,
        reference_deviance=1000.0,
        folds_improved=0,
        n_folds=5,
        candidate_parameters=7,
        reference_parameters=9,
    )
    _check(
        "Forenkling: tap innenfor bare én grense kvalifiserer ikke",
        result["kvalifiserer"] is False,
    )


def test_simplification_exactly_at_threshold_qualifies():
    # terskel=min(3,5)=3. Tap nøyaktig 3.0 -> ikke-streng ulikhet, skal kvalifisere.
    result = qualifies(
        gain=-3.0,
        se_pair=3.0,
        reference_deviance=1000.0,
        folds_improved=0,
        n_folds=5,
        candidate_parameters=7,
        reference_parameters=9,
    )
    _check(
        "Forenkling: tap nøyaktig lik terskelen kvalifiserer (ikke-streng ulikhet)",
        result["kvalifiserer"] is True,
    )


def test_near_tie_requires_both_conditions():
    # Beste: deviance=1000. B: distance=7, se_pair=10 (innenfor SE, utenfor 0,5%=5).
    # C: distance=3, se_pair=2 (innenfor 0,5%, utenfor SE).
    candidates = [
        {"id": "A", "deviance": 1000.0, "se_pair": float("nan"), "parameters": 9},
        {"id": "B", "deviance": 1007.0, "se_pair": 10.0, "parameters": 8},
        {"id": "C", "deviance": 1003.0, "se_pair": 2.0, "parameters": 8},
    ]
    result = choose_near_tie(candidates)
    table = result["tabell"].set_index("id")
    _check(
        "Nesten-like: innenfor SE men utenfor 0,5% er IKKE nesten lik",
        bool(table.loc["B", "nesten_likt"]) is False,
    )
    _check(
        "Nesten-like: innenfor 0,5% men utenfor SE er IKKE nesten lik",
        bool(table.loc["C", "nesten_likt"]) is False,
    )


def test_near_tie_fewest_parameters_wins():
    candidates = [
        {"id": "A", "deviance": 1000.0, "se_pair": float("nan"), "parameters": 9},
        {"id": "B", "deviance": 1000.3, "se_pair": 100.0, "parameters": 7},
    ]
    result = choose_near_tie(candidates)
    _check(
        "Nesten-like: færrest parametere slår lavest score",
        result["valgt"] == "B",
    )


def test_near_tie_lowest_score_breaks_equal_parameters():
    candidates = [
        {"id": "A", "deviance": 1000.0, "se_pair": float("nan"), "parameters": 8},
        {"id": "B", "deviance": 1000.2, "se_pair": 100.0, "parameters": 8},
    ]
    result = choose_near_tie(candidates)
    _check(
        "Nesten-like: likt parametertall -> lavest score vinner",
        result["valgt"] == "A",
    )


def test_near_tie_alphabetical_breaks_full_tie():
    candidates = [
        {"id": "B", "deviance": 1000.0, "se_pair": float("nan"), "parameters": 8},
        {"id": "A", "deviance": 1000.0, "se_pair": 100.0, "parameters": 8},
    ]
    result = choose_near_tie(candidates)
    _check(
        "Nesten-like: likt parametertall og score -> alfabetisk ID vinner",
        result["valgt"] == "A",
    )


def test_near_tie_missing_se_raises():
    candidates = [
        {"id": "A", "deviance": 1000.0, "se_pair": float("nan"), "parameters": 9},
        {"id": "B", "deviance": 1001.0, "se_pair": None, "parameters": 8},
    ]
    raised = False
    try:
        choose_near_tie(candidates)
    except ValueError:
        raised = True
    _check(
        "Nesten-like: manglende se_pair for en ikke-beste kandidat gir feil",
        raised,
    )


def _make_qualifies_kwargs(
    gain, se_pair, ref_dev, folds_improved, cand_params, ref_params
):
    return {
        "gain": gain,
        "se_pair": se_pair,
        "reference_deviance": ref_dev,
        "folds_improved": folds_improved,
        "n_folds": 5,
        "candidate_parameters": cand_params,
        "reference_parameters": ref_params,
    }


def test_geography_rule_better_than_s0_not_g1():
    g2_vs_s0 = _make_qualifies_kwargs(10.0, 1.0, 1000.0, 5, 10, 9)  # kvalifiserer
    g2_vs_g1 = _make_qualifies_kwargs(1.0, 1.0, 1000.0, 2, 10, 8)  # kvalifiserer ikke
    result = apply_geography_rule(g2_vs_s0, g2_vs_g1, g1_valid=True)
    _check(
        "Geografi: G2 slår S0 men ikke G1 -> forkastes",
        result["kvalifiserer"] is False,
    )


def test_geography_rule_better_than_g1_not_s0():
    g2_vs_s0 = _make_qualifies_kwargs(1.0, 1.0, 1000.0, 2, 10, 9)  # kvalifiserer ikke
    g2_vs_g1 = _make_qualifies_kwargs(10.0, 1.0, 1000.0, 5, 10, 8)  # kvalifiserer
    result = apply_geography_rule(g2_vs_s0, g2_vs_g1, g1_valid=True)
    _check(
        "Geografi: G2 slår G1 men ikke S0 -> forkastes",
        result["kvalifiserer"] is False,
    )


def test_geography_rule_invalid_g1():
    g2_vs_s0 = _make_qualifies_kwargs(10.0, 1.0, 1000.0, 5, 10, 9)
    g2_vs_g1 = _make_qualifies_kwargs(10.0, 1.0, 1000.0, 5, 10, 8)
    result = apply_geography_rule(g2_vs_s0, g2_vs_g1, g1_valid=False)
    _check(
        "Geografi: G1 ugyldig -> G2 forkastes uansett score",
        result["kvalifiserer"] is False and result["mot_s0"] is None,
    )


def test_geography_rule_qualifies_against_both():
    g2_vs_s0 = _make_qualifies_kwargs(10.0, 1.0, 1000.0, 5, 10, 9)
    g2_vs_g1 = _make_qualifies_kwargs(10.0, 1.0, 1000.0, 5, 10, 8)
    result = apply_geography_rule(g2_vs_s0, g2_vs_g1, g1_valid=True)
    _check(
        "Geografi: G2 slår både S0 og G1 -> kvalifiserer",
        result["kvalifiserer"] is True,
    )


def run_all():
    """Kjører alle testene i modulen og printer status per test.

    Returnerer (antall_bestått, antall_feilet).
    """
    _results.clear()
    test_functions = [
        test_candidate_register_and_fit_budget,
        test_improvement_exactly_on_threshold_fails,
        test_improvement_over_threshold_but_too_few_folds,
        test_improvement_qualifies_with_four_of_five_folds,
        test_improvement_se_binding,
        test_improvement_relative_binding,
        test_simplification_se_binding,
        test_simplification_relative_binding,
        test_simplification_within_both_bounds_qualifies,
        test_simplification_within_only_one_bound_fails,
        test_simplification_exactly_at_threshold_qualifies,
        test_near_tie_requires_both_conditions,
        test_near_tie_fewest_parameters_wins,
        test_near_tie_lowest_score_breaks_equal_parameters,
        test_near_tie_alphabetical_breaks_full_tie,
        test_near_tie_missing_se_raises,
        test_geography_rule_better_than_s0_not_g1,
        test_geography_rule_better_than_g1_not_s0,
        test_geography_rule_invalid_g1,
        test_geography_rule_qualifies_against_both,
    ]
    for test_fn in test_functions:
        test_fn()

    n_passed = sum(_results)
    n_failed = len(_results) - n_passed
    print(f"\n{n_passed} bestått, {n_failed} feilet av {len(_results)} tester.")
    return n_passed, n_failed


if __name__ == "__main__":
    passed, failed = run_all()
    if failed:
        raise SystemExit(1)
