"""Låste modeller: refit på hele utviklingssettet, lagring og prediksjon.

Hver modellnotebook fitter sin endelige modell på alle utviklingsdata
(2022–2023) og lagrer den under ``models/``. ``model_results`` laster dem og
predikerer teståret én gang. Modulen leser aldri data selv.

Årseffekten: ``year`` er en låst, kategorisk term, og 2024 finnes ikke i
utviklingsdata. Nye rader skåres derfor med nivået til siste utviklingsår
(``SCORE_YEAR``). Det betyr ingen trend-ekstrapolering: en eventuell
prisendring mellom 2023 og 2024 blir liggende i porteføljebalansen.
"""

import copy
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src_core_glm.glm_core import prepare_design_frame
from src_core_glm.model_data import TRAIN_YEARS, add_seat_category, to_model_frame
from src_ml.catboost_pricing import prepare_catboost_features

LOCKED_MODEL_DIR = Path("models")
SCORE_YEAR = max(TRAIN_YEARS)


@dataclass
class LockedGLM:
    """Refittet GLM som predikerer forventet verdi (frekvens, severity eller ren premie)."""

    name: str
    formula: str
    fit: object  # statsmodels GLMResults fra utviklingssettet
    required_columns: list
    train_design: pd.DataFrame  # kun for å gjenbruke treningens medianer og nivåer

    def __call__(self, frame):
        source_columns = [
            "seats" if c == "seat_category" else c for c in self.required_columns
        ]
        scored = to_model_frame(frame, list(dict.fromkeys(source_columns)))
        scored["year"] = SCORE_YEAR
        if "seat_category" in self.required_columns:
            add_seat_category(scored)
        design = prepare_design_frame(self.train_design, scored, self.required_columns)
        # Splines og lineære ledd ekstrapoleres lineært; klipp til treningsområdet,
        # slik at GLM-ene oppfører seg som trærne (flat utenfor treningsdata).
        for column in self.train_design.select_dtypes("number").columns:
            design[column] = design[column].clip(
                self.train_design[column].min(), self.train_design[column].max()
            )
        return pd.Series(np.asarray(self.fit.predict(design)), index=frame.index)


@dataclass
class LockedCatBoost:
    """Refittet CatBoost-modell for ren premie per eksponeringsår."""

    name: str
    model: object
    predictors: list
    categorical_features: list

    def __call__(self, frame):
        scored = to_model_frame(frame, self.predictors)
        scored["year"] = SCORE_YEAR
        features = prepare_catboost_features(
            scored, self.predictors, self.categorical_features
        )
        return pd.Series(self.model.predict(features), index=frame.index)


def _save_and_verify(locked, expected, frame_2023, directory):
    """Lagre, last på nytt og bekreft at prediksjonene tilsvarer refittet i notebooken."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{locked.name}.joblib"
    joblib.dump(locked, path)
    reloaded = joblib.load(path)
    prediction = reloaded(frame_2023)
    if not np.allclose(prediction.to_numpy(), np.asarray(expected), rtol=1e-8):
        raise ValueError(
            f"{locked.name}: lastet modell gir andre prediksjoner enn refittet."
        )
    if not (np.isfinite(prediction).all() and prediction.gt(0).all()):
        raise ValueError(f"{locked.name}: prediksjoner må være endelige og positive.")
    return pd.DataFrame(
        [
            {
                "modell": locked.name,
                "fil": str(path),
                "størrelse_MB": round(path.stat().st_size / 1e6, 2),
                "rader_kontrollert": len(prediction),
                "snitt_prediksjon": prediction.mean(),
            }
        ]
    )


def lock_glm(name, specification, fit, design, development, directory=LOCKED_MODEL_DIR):
    """Lagre en refittet GLM og verifiser den mot notebookens egen ``fit.predict``.

    ``design`` er rammen modellen ble fittet på (etter ``prepare_design_frame``) og
    ``development`` er utviklingsrammen med rå kolonner. Kontrollen bruker
    2023-radene, fordi lagret modell skårer alle rader med ``SCORE_YEAR``.
    """
    required = specification["required_columns"]
    # Kopi uten treningsdata: prediksjon trenger bare koeffisienter og formel.
    slim_fit = copy.deepcopy(fit)
    slim_fit.remove_data()
    locked = LockedGLM(
        name, specification["formula"], slim_fit, required, design[required].copy()
    )
    rows = design.index[design["year"].eq(SCORE_YEAR)]
    frame_2023 = development.loc[rows]
    return _save_and_verify(
        locked, fit.predict(design.loc[rows]), frame_2023, directory
    )


def lock_catboost(
    name,
    result,
    predictors,
    categorical_features,
    development,
    directory=LOCKED_MODEL_DIR,
):
    """Lagre CatBoost-modellen (refittet på alt i ``fit_catboost_grid``) og verifiser den."""
    model = result["best_estimator"]
    locked = LockedCatBoost(name, model, list(predictors), list(categorical_features))
    features = result["features"]
    rows = features.index[features["year"].astype(str).eq(str(SCORE_YEAR))]
    return _save_and_verify(
        locked, model.predict(features.loc[rows]), development.loc[rows], directory
    )


def load_locked_model(name, directory=LOCKED_MODEL_DIR):
    """Last en lagret modell; brukes bare av ``model_results``."""
    return joblib.load(Path(directory) / f"{name}.joblib")
