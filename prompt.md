Jeg anbefaler CatBoost som en fleksibel omnibus-diagnostikk, men ikke å fitte den direkte på én ferdig OOF-residualserie. Det kan gi kryssfold-lekkasje: noen av GLM-modellene som produserte treningsresidualene, kan ha sett responsene i CatBoosts senere valideringsfold.

## Anbefalt oppsett

For hver ytre `insured_id`-fold:

1. Fit GLM på ytre treningsdata og prediker ytre valideringsdata.
2. Lag indre, gruppebaserte OOF-prediksjoner bare innen ytre treningsdata.
3. Beregn en multiplikativ residualratio:

$$
z_i = \frac{y_i}{\hat{\mu}_{i,\mathrm{inner\ OOF}}}
$$

Ved korrekt middelspesifikasjon skal CatBoost i hovedsak predikere $z_i = 1$.

4. Fit CatBoost på ratioen med vekter

$$
a_i = w_i \hat{\mu}_i^{\,2-p}
$$

der:

* frekvens: $p=1,; w=e_i$, altså $a_i=e_i\hat{\mu}_i$
* severity: $p=2,; w=N_i$, altså $a_i=N_i$
* Tweedie: $1<p<2,; w=e_i$

5. Bruk CatBoosts positive prediksjon som korreksjonsfaktor:

$$
\hat{\mu}_{i,\mathrm{korrigert}}
=
\hat{\mu}_{i,\mathrm{GLM}}\hat{z}_i
$$

6. Evaluer GLM og korrigert GLM på original respons med riktig Poisson-, Gamma- eller Tweedie-deviance. Residual-$R^2$ bør ikke være hovedmålet.

CatBoost kan bruke `loss_function="Poisson"` på den ikke-negative ratioen. Her betyr ikke det at severity antas Poisson-fordelt; loss-funksjonen brukes bare for å estimere en positiv betinget korreksjonsfaktor.

## Hvordan skille ikke-linearitet fra interaksjoner

Kjør to låste CatBoost-varianter med samme features:

* `depth=1`: kan fange additive, ikke-lineære restmønstre.
* `depth=2` eller `depth=3`: åpner også for interaksjoner.

Tolkningen blir:

* Ingen stabil gevinst: lite tegn til manglende struktur.
* Gevinst med `depth=1`: sannsynlig manglende funksjonsform.
* Ytterligere gevinst med `depth=2/3`: mulig interaksjonssignal.

Dette er indikasjon, ikke bevis. Korrelerte variabler kan gjøre skillet uklart.

## Felles kode

Jeg anbefaler en egen modul i core-pakken, [src_core_glm](/workspaces/MotorForsikring/src_core_glm), fremfor å legge CatBoost direkte i [glm_core.py](/workspaces/MotorForsikring/src_core_glm/glm_core.py). Da forblir den eksisterende filen en ren GLM-kjerne, samtidig som diagnostikken kan importeres av alle notebookene:

```python
from src_core_glm.residual_diagnostics import (
    cross_validate_residual_catboost,
    summarize_residual_diagnostic,
    plot_residual_diagnostic,
)
```

Hoved-API-et bør være omtrent:

```python
result = cross_validate_residual_catboost(
    spec=final_specification,
    data=model_frame,
    folds=cv_folds,
    feature_columns=diagnostic_features,
    categorical_columns=categorical_features,
    group_column="insured_id",
    depths=(1, 3),
    model_params={
        "iterations": 300,
        "learning_rate": 0.03,
        "random_seed": 100,
    },
    fit_kwargs=FIT_SETTINGS,
)
```

Returverdien bør inneholde:

* OOF-prediksjon fra GLM
* OOF-korreksjonsfaktor fra CatBoost
* korrigert OOF-prediksjon
* foldvis og samlet deviancegevinst
* balanse/A–E
* antall folder med forbedring
* valideringsbasert permutation importance
* `valid` og eventuelle foldfeil

Første kjøring bør bare bruke råvariablene som allerede inngår i GLM-en. En separat kjøring med flere variabler undersøker også utelatte hovedvirkninger og kan derfor ikke tolkes rent som interaksjonsdiagnostikk. `insured_id`, respons, kostnad/skadeantall og andre target-avledede felt må aldri være features.

## Beslutningsregel

Jeg ville krevd:

* positiv pooled OOF-deviancegevinst
* forbedring i minst fire av fem ytre folder
* gevinst utover en enkel foldvis konstant kalibreringsfaktor
* ingen ugyldige folder
* rimelig stabil valideringsbasert feature-rangering

Deretter oversettes signalet til én eller noen få forståelige GLM-kandidater, eksempelvis en spline eller eksplisitt interaksjon, som testes på nytt. CatBoost-resultatet bør ikke automatisk endre sluttmodellen.

Ingen filer er endret ennå. Dette innebærer en vesentlig metodebeslutning og planene krever at den legges frem før implementering. Jeg anbefaler at jeg implementerer den nestede varianten over i `src_core_glm/residual_diagnostics.py`, med syntetiske verifikasjonstester og uten å lese eller berøre 2024-data.
