"""Nivåkalibrering av den låste CatBoost-modellen (kun utviklingsdata).

Regulariserte trær og Tweedie-loss gir ofte en liten nivåskjevhet. Faktoren
nedenfor er forholdet mellom observert og predikert skadekostnad i
out-of-fold-prediksjoner på 2022–2023, og den bruker aldri teståret.
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from src.ml.catboost_pricing import prepare_catboost_features
from src.tweedie.tweedie_data import build_tweedie_frame


def estimate_level_factor(locked_catboost, development, *, n_splits=5, seed=100):
    """Skaleringsfaktor $c$ slik at $\\sum S = c \\sum e\\,\\widehat\\mu$ på OOF-prediksjoner.

    Foldene er de samme gruppefoldene på ``insured_id`` som i CatBoost-notebooken,
    og hyperparameterne er de låste. ``year`` beholder sin egen verdi her, slik at
    faktoren måler ren nivåskjevhet og ikke forskjellen mellom 2022 og 2023.
    """
    model_frame = build_tweedie_frame(development, locked_catboost.predictors)
    features = prepare_catboost_features(
        model_frame, locked_catboost.predictors, locked_catboost.categorical_features
    )
    response = model_frame["pure_premium"]
    weight = model_frame["total_exposure"]
    oof = np.full(len(features), np.nan)
    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, valid in cv.split(features, response, model_frame["insured_id"]):
        model = clone(locked_catboost.model)
        model.fit(
            features.iloc[train],
            response.iloc[train],
            sample_weight=weight.iloc[train],
            cat_features=list(locked_catboost.categorical_features),
        )
        oof[valid] = model.predict(features.iloc[valid])
    return (weight * response).sum() / (weight * oof).sum()
