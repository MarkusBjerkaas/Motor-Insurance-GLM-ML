"""Felles datagrunnlag for egen-skademodellene.

Modulet samler stegene som gjør den rensede porteføljen om til de rammene
modellene trenger. Stegene ble først utviklet og begrunnet i ``analysis.py``
(seksjon 11, 12 og 17). De ligger her slik at ``analysis.py`` og
``glm_pricing_models.py`` bruker nøyaktig samme populasjon, splitt og
prediktorer:

1. **Avgrensning** (``select_own_damage_scope``): poliseår med kaskoprodukt
   (``COMP_E`` eller ``COMP_N``), positiv ``property_damage_premium`` (dekningen
   er aktiv) og positiv ``total_exposure`` (poliseåret har risiko). CC holdes
   utenfor (B-29): bare 0,8 % av CC-poliseårene har egen-skadepremie, og
   gruppen ser ut til å bestå av produktbyttere. Premien brukes bare som
   dekningsflagg, aldri som prediktor.
2. **Tidssplitt** (``split_by_year``): 2022–2023 er train/CV-pool, 2024 er urørt
   test. Splitten er deterministisk og bruker ingen skadeutfall.
3. **Merke-pooling** (``learn_retained_brands``): merker med minst 500
   eksponeringsår i train_pool beholdes. Regelen læres bare fra eksponering i
   train_pool, aldri fra skader eller testdata.
4. **Transparente prediktorer** (``add_transparent_predictors``): forhåndsbestemte
   transformasjoner uten skadeutfall.

5. **Modellramme** (``to_model_frame``): konverterer nullable dtypes til
   ``float64``/``int64``/``object`` slik at patsy-formler kan brukes.

``build_model_frames`` kjører hele kjeden fra CSV til ferdige rammer. CV-folder,
imputasjon og modellspesifikasjoner ligger bevisst ikke her. De er en del av
modelleringen og defineres i notebooken.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_quality import build_variable_dictionary, clean_motor_data

DATA_PATH = Path("data/Dataset of motor insurance portfolio.csv")
TRAIN_YEARS = (2022, 2023)
TEST_YEAR = 2024
BRAND_MIN_EXPOSURE = 500.0
# Kaskoprodukter i modellpopulasjonen; CC holdes utenfor (B-29)
OWN_DAMAGE_PRODUCTS = ("COMP_E", "COMP_N")


def select_own_damage_scope(data, products=OWN_DAMAGE_PRODUCTS):
    """Avgrens til kaskoprodukter med aktiv egen-skadedekning og positiv eksponering."""
    own_damage_mask = (
        data["policy_type"].isin(products)
        & data["property_damage_premium"].gt(0)
        & data["total_exposure"].gt(0)
    )
    return data.loc[own_damage_mask].copy()


def split_by_year(frame, train_years=TRAIN_YEARS, test_year=TEST_YEAR):
    """Del modellpopulasjonen i en train/CV-pool og en urørt testperiode."""
    train_pool = frame.loc[frame["year"].isin(train_years)].copy()
    test = frame.loc[frame["year"].eq(test_year)].copy()
    return train_pool, test


def learn_retained_brands(train_pool, min_exposure=BRAND_MIN_EXPOSURE):
    """Finn merker med nok eksponering i train_pool til å få eget nivå.

    Returnerer ``(retained_brands, brand_exposure)``: indeksen med beholdte
    merker og eksponeringen per merke. Eksponeringen returneres slik at
    notebooken kan dokumentere hvor mye av porteføljen poolingen dekker.
    """
    brand_exposure = train_pool.groupby("vehicle_brand", observed=True)[
        "total_exposure"
    ].sum()
    retained_brands = brand_exposure.loc[brand_exposure.ge(min_exposure)].index.astype(
        str
    )
    return retained_brands, brand_exposure


def add_transparent_predictors(frame, retained_brand_levels):
    """Legg til kun forhåndsdefinerte, ikke-responsbaserte prediktorer.

    - ``driving_experience_years``: føreralder minus alder ved førerkorterverv
      (arbeidshypotesen fra seksjon 17 i ``analysis.py``).
    - ``log_vehicle_value``: log av bilverdien, som gir en multiplikativ tolkning.
    - ``performance_hp_per_tonne``: ``1000 / power_to_weight_ratio``, slik at
      høyere verdi betyr høyere ytelse. Manglende input forblir manglende.
    - ``vehicle_brand_pooled``: beholdte merker, ellers ``OTHER``. Usette merker
      i nye data får også ``OTHER``.
    """
    transformed = frame.copy()
    transformed["driving_experience_years"] = (
        transformed["driver_age"] - transformed["age_driving_licence"]
    )
    transformed["log_vehicle_value"] = np.log(transformed["vehicle_value"])
    transformed["performance_hp_per_tonne"] = (
        1000 / transformed["power_to_weight_ratio"]
    )
    observed_brand = transformed["vehicle_brand"].astype("string")
    transformed["vehicle_brand_pooled"] = observed_brand.where(
        observed_brand.isin(retained_brand_levels), "OTHER"
    ).astype("category")
    return transformed


def to_model_frame(frame, columns):
    """Velg ``columns`` og konverter pandas' nullable dtypes til dtypes patsy kan lese.

    Formelmotoren patsy feiler på ``Int16``/``category`` med ``pd.NA``. Regelen er:

    - kategoriske og tekstkolonner → ``object`` (manglende blir ``NaN``)
    - heltall uten manglende verdier → ``int64`` (f.eks. ``insured_id``, ``year``)
    - øvrige numeriske kolonner → ``float64`` (manglende blir ``NaN``)

    Ingen verdier endres og ingenting læres fra data, så funksjonen kan brukes
    likt på train og test uten lekkasje.
    """
    model_frame = pd.DataFrame(index=frame.index)
    for column in columns:
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            model_frame[column] = series.astype("object").where(series.notna(), np.nan)
        elif pd.api.types.is_integer_dtype(series) and series.notna().all():
            model_frame[column] = series.astype("int64")
        else:
            model_frame[column] = series.astype("float64")
    return model_frame


def build_model_frames(data_path=DATA_PATH, brand_min_exposure=BRAND_MIN_EXPOSURE):
    """Kjør hele kjeden: CSV → rensing → avgrensning → splitt → prediktorer.

    Returnerer en dict med ``train_pool``, ``test``, ``retained_brands``,
    ``brand_exposure`` og ``cleaning_log``. Merke-poolingen læres på train_pool
    og brukes deretter uendret på test.
    """
    raw_data = pd.read_csv(data_path, sep=";", encoding="utf-8", low_memory=False)
    data, cleaning_log = clean_motor_data(raw_data, build_variable_dictionary())
    train_pool, test = split_by_year(select_own_damage_scope(data))
    retained_brands, brand_exposure = learn_retained_brands(
        train_pool, min_exposure=brand_min_exposure
    )
    return {
        "train_pool": add_transparent_predictors(train_pool, retained_brands),
        "test": add_transparent_predictors(test, retained_brands),
        "retained_brands": retained_brands,
        "brand_exposure": brand_exposure,
        "cleaning_log": cleaning_log,
    }
