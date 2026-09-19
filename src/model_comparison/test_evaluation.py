"""Låst klargjøring og presentasjon av modellresultater på teståret.

Ingen funksjon i modulen leser data ved import. ``build_approved_test_frame``
kan bare lese 2024 etter at den eksplisitte godkjenningskoden er sendt inn.
"""

from collections.abc import Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_tweedie_deviance

from src.core_glm.model_data import (
    DATA_PATH,
    TEST_YEAR,
    add_transparent_predictors,
    select_own_damage_scope,
)
from src.descriptives.data_quality import build_variable_dictionary, clean_motor_data

TEST_APPROVAL_CODE = "APPROVED_2024_TEST_EVALUATION"


def _require_test_approval(approval_code):
    """Stopp før filtilgang uten den eksplisitte, avtalte godkjenningen."""
    if approval_code != TEST_APPROVAL_CODE:
        raise PermissionError(
            "2024-testen er låst. Sett kun godkjenningskoden etter eksplisitt godkjenning."
        )


def prepare_scoring_frame(raw_frame, retained_brands):
    """Samme rensing, avgrensning og prediktorer som i utviklingskjeden.

    Brand-poolingen er lært fra utviklingsdata og sendes inn som ``retained_brands``.
    Funksjonen leser ingen filer, så den kan prøves på et utviklingsår (tørrkjøring).
    """
    cleaned, _ = clean_motor_data(raw_frame, build_variable_dictionary())
    scored = select_own_damage_scope(cleaned)
    return add_transparent_predictors(scored, retained_brands)


def build_approved_test_frame(retained_brands, *, approval_code, data_path=DATA_PATH):
    """Les og klargjør kun 2024 etter eksplisitt godkjenning."""
    _require_test_approval(approval_code)
    chunks = pd.read_csv(
        data_path, sep=";", encoding="utf-8", low_memory=False, chunksize=100_000
    )
    raw_test = pd.concat([chunk.loc[chunk["year"].eq(TEST_YEAR)] for chunk in chunks])
    test = prepare_scoring_frame(raw_test, retained_brands)
    if test.empty or not test["year"].eq(TEST_YEAR).all():
        raise ValueError("Testgrunnlaget må bestå av minst én rad, og kun av 2024.")
    return test


def evaluate_pricing_models(test_frame, predictors: Mapping[str, Callable], power):
    """Beregn sammenlignbare rene-premie-mål for hver låste modell.

    Hver prediktor mottar test-rammen og returnerer forventet ren premie per
    eksponeringsår. For en todelt modell er dette frekvens ganger severity.
    """
    if not predictors:
        raise ValueError("Minst én låst modellprediktor må oppgis.")
    observed = test_frame["property_incurred"] / test_frame["total_exposure"]
    exposure = test_frame["total_exposure"]
    rows, predictions = [], {}
    for name, predict in predictors.items():
        prediction = pd.Series(predict(test_frame), index=test_frame.index, dtype=float)
        if (
            prediction.isna().any()
            or (~np.isfinite(prediction)).any()
            or prediction.le(0).any()
        ):
            raise ValueError(
                f"{name}: prediksjoner må være endelige og strengt positive."
            )
        predictions[name] = prediction
        observed_cost = test_frame["property_incurred"].sum()
        predicted_cost = (prediction * exposure).sum()
        rows.append(
            {
                "Modell": name,
                "Tweedie-deviance": mean_tweedie_deviance(
                    observed, prediction, sample_weight=exposure, power=power
                ),
                "Vektet MAE": mean_absolute_error(
                    observed, prediction, sample_weight=exposure
                ),
                "Observert incurred (EUR)": observed_cost,
                "Predikert incurred (EUR)": predicted_cost,
                "Balanse (%)": 100 * (predicted_cost / observed_cost - 1),
            }
        )
    return pd.DataFrame(rows).sort_values("Tweedie-deviance"), predictions


def build_calibration_table(test_frame, predictions, n_bins=10):
    """Lag eksponeringsvektet kalibrering i like store prediksjonsgrupper."""
    exposure = test_frame["total_exposure"]
    observed = test_frame["property_incurred"] / exposure
    tables = []
    for name, prediction in predictions.items():
        groups = pd.qcut(prediction.rank(method="first"), q=n_bins, duplicates="drop")
        frame = pd.DataFrame(
            {
                "prediction": prediction,
                "observed": observed,
                "weight": exposure,
                "group": groups,
            }
        )
        summary = frame.groupby("group", observed=True).apply(
            lambda part: pd.Series(
                {
                    "Predikert ren premie": np.average(
                        part["prediction"], weights=part["weight"]
                    ),
                    "Observert ren premie": np.average(
                        part["observed"], weights=part["weight"]
                    ),
                }
            ),
            include_groups=False,
        )
        summary["Modell"] = name
        summary["Prediksjonsgruppe"] = range(1, len(summary) + 1)
        tables.append(summary.reset_index(drop=True))
    return pd.concat(tables, ignore_index=True)


def build_top_decile_component_summary(test_frame, frequency_model, severity_model):
    """Dekomponer toppdesilens rene premie i frekvens og rapportert severity.

    Desilen defineres av den toleddede modellens predikerte rene premie. Severity
    er kostnad per *rapportert* skade og predikeres vektet med forventet skadeantall.
    """
    exposure = test_frame["total_exposure"]
    observed_claims = test_frame["property_claims"]
    observed_cost = test_frame["property_incurred"]
    frequency = pd.Series(frequency_model(test_frame), index=test_frame.index)
    severity = pd.Series(severity_model(test_frame), index=test_frame.index)
    premium = frequency * severity
    group = pd.qcut(premium.rank(method="first"), q=10, labels=False)
    top = group.eq(group.max())

    expected_claims = (exposure[top] * frequency[top]).sum()
    expected_cost = (exposure[top] * premium[top]).sum()
    actual_claims = observed_claims[top].sum()
    actual_cost = observed_cost[top].sum()
    values = {
        "Frekvens": (
            expected_claims / exposure[top].sum(),
            actual_claims / exposure[top].sum(),
        ),
        "Rapportert severity": (
            expected_cost / expected_claims,
            actual_cost / actual_claims,
        ),
        "Ren premie": (
            expected_cost / exposure[top].sum(),
            actual_cost / exposure[top].sum(),
        ),
    }
    summary = pd.DataFrame.from_dict(
        values, orient="index", columns=["Predikert", "Observert"]
    )
    summary["A/E"] = summary["Observert"] / summary["Predikert"]
    return summary


def build_top_decile_profile(test_frame, premium_prediction):
    """Sammenlign modellens høyeste prediksjonsdesil med resten av porteføljen.

    Profilen bruker bare tydelige, observerbare modellprediktorer og beskriver
    sammensetning; den tolker ikke sammenhengene kausalt.
    """
    prediction = pd.Series(premium_prediction, index=test_frame.index)
    group = pd.qcut(prediction.rank(method="first"), q=10, labels=False)
    top = group.eq(group.max())
    exposure = test_frame["total_exposure"]
    characteristics = {
        "Kasko uten egenandel (COMP_N)": test_frame["policy_type"].eq("COMP_N"),
        "Urban bruk": test_frame["circulation_area"].eq("U"),
        "Fornyet portefølje": test_frame["business_type"].eq("P"),
        "Kvartalsvis betaling": test_frame["payment_frequency"].eq("Q"),
    }
    rows = []
    for label, member in characteristics.items():
        rows.append(
            {
                "Kjennetegn": label,
                "Høyeste desil": exposure[top & member].sum() / exposure[top].sum(),
                "Øvrige 90 %": exposure[~top & member].sum() / exposure[~top].sum(),
            }
        )
    return pd.DataFrame(rows)


def plot_test_comparison(summary, calibration):
    """Vis én kompakt figur: kalibrering og total porteføljebalanse."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, part in calibration.groupby("Modell", observed=True):
        axes[0].plot(
            part["Predikert ren premie"],
            part["Observert ren premie"],
            marker="o",
            label=name,
        )
    limit = max(calibration[["Predikert ren premie", "Observert ren premie"]].max())
    axes[0].plot(
        [0, limit],
        [0, limit],
        "--",
        color="black",
        linewidth=1,
        label="Perfekt kalibrering",
    )
    axes[0].set(
        title="Kalibrering etter prediksjonsgruppe",
        xlabel="Predikert ren premie",
        ylabel="Observert ren premie",
    )
    axes[0].legend(fontsize=8)
    axes[1].bar(summary["Modell"], summary["Balanse (%)"])
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(title="Porteføljebalanse", ylabel="Predikert minus observert (%)")
    axes[1].tick_params(axis="x", rotation=25)
    figure.tight_layout()
    return figure
