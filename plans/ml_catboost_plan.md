# Plan: CatBoost-utfordrer for ren premie

## 1. Formål og avgrensning

Denne fasen skal implementere én ryddig CatBoost-utfordrer for forventet ren
egen-skadepremie. Modellen skal kunne sammenlignes direkte med Tweedie-GLM og
den todelte GLM-modellen når alle spesifikasjoner senere evalueres samlet på
det urørte teståret.

Implementasjonen skal være bevisst enkel:

- én respons: ren premie;
- én algoritme: CatBoost;
- én liten `GridSearchCV`;
- samme gruppebaserte utviklingsfolder som GLM-løpet;
- native behandling av kategoriske variabler;
- én resultattabell og én diagnostikkfigur.

`ml_pricing.py` er et tidlig utkast, ikke en implementeringskontrakt. Eksisterende
kode kan og skal slettes, endres eller skrives over når den er feil, uferdig
eller gjør notebooken unødvendig komplisert. Spesielt skal LightGBM, separate
frekvens-/severitymodeller, one-hot-koding, hardkodet `TWEEDIE_POWER = 1.5` og
det ubrukte modellregisteret fjernes.

### Utenfor omfanget

- Ingen lesing, materialisering, inspeksjon eller evaluering av 2024-data.
- Ingen LightGBM eller sammenligning av flere ML-algoritmer.
- Ingen separate CatBoost-modeller for frekvens og severity.
- Ingen nestet CV; 2024 er den endelige, uavhengige testen etter modellåsing.
- Ingen Bayesian search, random search, early stopping eller SHAP-analyse.
- Ingen automatisk featureseleksjon eller stort interaksjonsregister.
- Ingen generelt ML-rammeverk som skal dekke fremtidige algoritmer.
- Ingen avansert syntetisk lekkasje-/spy-testsuite.

## 2. Låste metodebeslutninger

| ID | Beslutning | Begrunnelse |
|---|---|---|
| ML-01 | Modeller `pure_premium = property_incurred / total_exposure` på alle utviklingsrader. | Samme respons som direkte Tweedie-GLM og et direkte prisingsmål. |
| ML-02 | Bruk `total_exposure` som `sample_weight` i fit og scoring. | Et poliseår med mer eksponering gir mer informasjon, og dette matcher GLM-benchmarken. |
| ML-03 | CatBoost bruker Tweedie-loss med nøyaktig samme låste power som Tweedie-GLM. | Loss og rapportert deviance blir faglig sammenlignbare. |
| ML-04 | Primær score er negativ, eksponeringsvektet mean Tweedie deviance. | `GridSearchCV` maksimerer score; negativ deviance gir lavest deviance som beste modell. |
| ML-05 | Bruk de samme fem `insured_id`-gruppefoldene og seed 100 som GLM-ene. | Hindrer personlekkasje og gjør senere sammenligning konsistent. |
| ML-06 | Bruk vanlig `GridSearchCV`, ikke nestet CV. | Hyperparameterne låses før én endelig 2024-test; nesting ville her gitt mer kode og kjøretid enn beslutningsverdi. |
| ML-07 | CV-scoren omtales som utviklingsscore, ikke uavhengig ytelsesestimat. | Samme folder brukes til hyperparameterseleksjon; endelig generalisering avgjøres senere på 2024. |
| ML-08 | Bruk CatBoosts native kategoribehandling. | Bevarer algoritmens styrke og er enklere enn one-hot-pipeline. |
| ML-09 | Ingen `eval_set` eller early stopping under grid search. | `GridSearchCV` eier valideringen; et ekstra eval-sett gir en unødvendig og uklar foldkontrakt. |
| ML-10 | Etter tuning lages OOF-prediksjoner med de valgte hyperparameterne og de samme fem foldene. | Gir kalibrerings- og residualdiagnostikk, men skal merkes som seleksjonspåvirket utviklingsdiagnostikk. |
| ML-11 | `GridSearchCV(refit=True)` gir sluttmodellen fittet på alle utviklingsdata. | Unngår et ekstra, duplisert fit-kall. |
| ML-12 | Endelig testprotokoll implementeres ikke i denne fasen. | 2024 skal først åpnes etter at alle modeller og sammenligningsregler er godkjent og låst. |

## 3. Prediktorer og tilgjengelighet ved prising

Notebooken skal ha følgende eksplisitte celle tidlig, før data bygges eller
modellen defineres:

```python
PREDICTORS = [
    "policy_type",
    "year",
    "driver_age",
    "log_vehicle_value",
    "performance_hp_per_tonne",
    "fuel_type",
    "circulation_area",
    "municipality_type",
    "payment_frequency",
    "business_type",
    "vehicle_brand_pooled",
    "seats",
]

CATEGORICAL_FEATURES = [
    "policy_type",
    "year",
    "fuel_type",
    "circulation_area",
    "municipality_type",
    "payment_frequency",
    "business_type",
    "vehicle_brand_pooled",
]
```

Alle variablene antas kjent ved prisingstidspunktet. `year` behandles som
kategorisk fordi bare 2022 og 2023 finnes i utviklingsdata, og en lineær
tidseffekt skal ikke påtvinges CatBoost.

Følgende skal ikke inngå:

- `insured_id` (kun gruppevariabel);
- `total_exposure` (kun vekt);
- `property_claims`, `property_incurred`, `pure_premium` eller andre utfall;
- `bonus_score`, fordi eksakt as-of-tid fortsatt er uavklart;
- nye features som ikke allerede er dokumentert i dataløpet.

`vehicle_brand_pooled` gjenbrukes fra det felles datagrunnlaget. Poolingen er
responsuavhengig og den eksisterende `OTHER`-håndteringen kan brukes ved senere
prediksjon. Ikke bygg ny pooling i ML-koden.

## 4. Statistisk definisjon

For poliseår $i$ defineres

$$
R_i = \frac{S_i}{e_i},
$$

der $S_i$ er samlet incurred egen-skadekostnad og $e_i$ er eksponering.
CatBoost estimerer

$$
\widehat\mu_i = f_{\theta}(x_i),
$$

med Tweedie-loss og samme varianspower $p$ som Tweedie-GLM. Hyperparametrene
velges ved å minimere

$$
D_p = \operatorname{mean\_tweedie\_deviance}
\left(R, \widehat\mu;\, w=e,\, p\right)
$$

på de fem utviklingsfoldene.

Notebooken skal ha en kort markdown-celle med disse ligningene umiddelbart før
modelltilpasningen. Den skal også forklare intuitivt at modellen anslår forventet
skadekostnad per eksponeringsår, mens eksponeringen bestemmer observasjonens
vekt.

## 5. Hyperparameter-grid

Bruk et lite, lesbart full-grid:

```python
PARAM_GRID = {
    "depth": [4, 6],
    "learning_rate": [0.03, 0.07],
    "iterations": [300, 600],
    "l2_leaf_reg": [3.0, 10.0],
}
```

Dette gir 16 kombinasjoner og 80 fits i femfolds CV. Gridet dekker de viktigste
kapasitets-, regulariserings- og læringsparameterne uten å bli en omfattende
hyperparameterjakt. Ikke legg til flere parametere uten et konkret observert
problem.

Følgende CatBoost-innstillinger låses utenfor gridet:

```python
CatBoostRegressor(
    loss_function=f"Tweedie:variance_power={TWEEDIE_POWER}",
    random_seed=SEED,
    verbose=False,
    allow_writing_files=False,
    cat_features=CATEGORICAL_FEATURES,
)
```

Angi CPU-/trådinnstillinger bare dersom kjøremiljøet faktisk krever det. Ikke
parallelliser både `GridSearchCV` og CatBoost aggressivt samtidig.

## 6. Filstruktur og ansvar

### `ml_pricing.py` / `ml_pricing.ipynb`

Notebooken skal eie og synliggjøre:

1. formål og eksplisitt beskjed om at 2024 er urørt;
2. konstanter, `PREDICTORS`, `CATEGORICAL_FEATURES` og `PARAM_GRID`;
3. bygging av utviklingsramme og ren-premie-respons;
4. den statistiske definisjonen og valg av score;
5. opprettelse av CatBoost-estimatoren;
6. ett kall som kjører grid search og OOF-diagnostikk;
7. én kort resultattabell;
8. én diagnostikkfigur;
9. en kort konklusjon som presiserer at endelig sammenligning skjer senere på 2024.

En kodecelle skal gjøre én tydelig oppgave og normalt være under 25 linjer.
Notebooken skal ikke inneholde egendefinerte CV-sløyfer, plotting-loops eller
lange valideringsblokker.

### Ny pakke: `src_ml/`

Opprett bare:

- `src_ml/__init__.py`
- `src_ml/catboost_pricing.py`

`catboost_pricing.py` skal inneholde små funksjoner med ett ansvar:

```python
def prepare_catboost_features(frame, predictors, categorical_features):
    """Velg features; kategorier fylles med MISSING og konverteres til tekst."""


def make_weighted_tweedie_scorer(sample_weight, power):
    """Returner sklearn-scorer som bruker valideringsradenes indeks til riktige vekter."""


def fit_catboost_grid(
    features,
    response,
    sample_weight,
    groups,
    categorical_features,
    power,
    param_grid,
    *,
    n_splits=5,
    seed=100,
):
    """Kjør GroupKFold GridSearchCV, refit beste modell og lag OOF-prediksjoner."""


def build_catboost_result_table(result):
    """Oppsummer valgt gridrad, OOF-deviance, null-deviance og OOF-D2 i én rad."""


def plot_catboost_diagnostics(result, *, top_n=10):
    """Lag én figur med kalibrering, residualmønster og deskriptiv feature importance."""
```

Ikke opprett baseklasser, config-klasser, dataclasses eller en generell
modellfabrikk. En vanlig dict som returverdi er tilstrekkelig.

`fit_catboost_grid` skal:

1. bygge `GroupKFold(n_splits=5, shuffle=True, random_state=100)`;
2. lage den vektede scorer-funksjonen;
3. kjøre `GridSearchCV(..., scoring=scorer, refit=True, error_score="raise")`;
4. sende `groups` til CV og `sample_weight` til CatBoost-fit;
5. lage OOF-prediksjoner med en enkel femfoldssløyfe og en clone av den valgte
   estimatoren, fordi foldspesifikke vekter og eksakt indeksplassering skal være
   tydelig og robust;
6. returnere `search`, `best_estimator`, `oof_prediction`, foldscore og de
   grunnleggende seriene som tabell/plot trenger.

OOF-sløyfen er støttekode og skal ikke vises i notebooken. Den er ikke en ny
modellseleksjon: hyperparameterne er allerede valgt av `GridSearchCV`.

### Ny fil: `src_asserts/ml_asserts.py`

Filen skal ha tydelige seksjonsmarkører:

```python
# --- Datagrunnlag og featurekontrakt -----------------------------------------
# --- Gruppefolder ------------------------------------------------------------
# --- Grid search og prediksjoner ---------------------------------------------
```

Offentlig notebook-API begrenses til:

```python
def assert_ml_inputs(
    model_frame,
    predictors,
    categorical_features,
    *,
    response="pure_premium",
    weight="total_exposure",
    group="insured_id",
):
    """Samlet kontroll av utviklingsdata og featurekontrakt."""


def assert_ml_result(result, model_frame):
    """Samlet kontroll av CV-resultat, OOF-dekning og prediksjonsdomene."""
```

Interne hjelpefunksjoner kan brukes for lesbarhet, men notebooken skal bare
kalle disse to funksjonene. Kommentarene `# sanity-sjekk, kan fjernes` brukes
ved kallene, slik at presentasjonsnotebooken enkelt kan renses senere.

Ikke endre eksisterende assert-filer dersom det ikke er nødvendig.

## 7. Målrettede kontroller

### Inputkontroller

- Utviklingsdata inneholder bare `TRAIN_YEARS` og aldri 2024.
- `PREDICTORS` er unik og alle kolonnene finnes.
- `CATEGORICAL_FEATURES` er en delmengde av `PREDICTORS`.
- Featurelisten inneholder ingen av:
  `insured_id`, `total_exposure`, `property_claims`, `property_incurred`,
  `pure_premium`, `bonus_score`.
- Responsen er endelig og ikke-negativ.
- Eksponeringsvekten er endelig og strengt positiv.
- Gruppekolonnen mangler ikke verdier.
- Tweedie-power oppfyller $1 < p < 2$.

### Fold- og resultatkontroller

- Ingen `insured_id` finnes på begge sider av en fold.
- Hver utviklingsrad får nøyaktig én OOF-prediksjon.
- OOF-prediksjoner er endelige og strengt positive.
- Det finnes nøyaktig fem foldscorer, og alle er endelige.
- `best_params_` bruker bare nøkler og verdier fra `PARAM_GRID`.
- Rapportert pooled OOF-deviance stemmer med direkte
  `mean_tweedie_deviance`-beregning.

Dette er tilstrekkelig. Ikke legg til mocks, fit-spioner, syntetiske datasett
eller tester av sklearn/CatBoosts dokumenterte foldoppførsel.

## 8. Resultattabell og diagnostikk

### Én resultattabell

Vis én rad med:

- valgt `depth`, `learning_rate`, `iterations`, `l2_leaf_reg`;
- beste gjennomsnittlige grid-deviance og standardavvik mellom foldene;
- pooled OOF-deviance etter refit med valgte parametere;
- nullmodellens OOF-deviance;
- $D^2 = 1 - D_{model}/D_{null}$.

Tabellen skal tydelig skille `grid_mean_deviance` fra
`pooled_oof_deviance`. Begge er utviklingsresultater og ingen av dem er den
endelige 2024-testen.

Nullprediksjonen i hver OOF-fold skal være det eksponeringsvektede gjennomsnittet
fra foldens treningsdel, ikke et globalt gjennomsnitt.

### Én diagnostikkfigur

Lag én figur med inntil tre paneler:

1. eksponeringsvektet observert mot predikert ren premie i prediksjonsdesiler;
2. Tweedie-devianceresidual mot predikert verdi, gjerne aggregert/binned for
   lesbarhet dersom radplottet blir tett;
3. CatBoosts topp-ti feature importance fra sluttmodellen.

Feature importance skal merkes som deskriptiv, ikke kausal eller held-out.
Ikke legg til SHAP eller permutation importance i denne fasen.

## 9. Tweedie-power som forutsetning

Før implementeringen fitter CatBoost skal agenten kjøre den ferdige
Tweedie-GLM-notebooken på utviklingsdata og hente den låste power-verdien.
Verdien skrives eksplisitt i ML-notebookens konfigurasjonscelle med kommentar
om at den er låst fra Tweedie-GLM. Den skal ikke stå som den gamle, vilkårlige
verdien `1.5` med mindre Tweedie-estimeringen faktisk gir nøyaktig dette.

Ikke lag en ny konfigurasjonspakke eller dupliser Pearson-estimeringen bare for
å dele én låst verdi. Samme verdi skal senere inngå i den endelige
sammenligningsprotokollen.

Hvis Tweedie-spesifikasjonen eller power ennå ikke er låst, skal CatBoost-koden
kun klargjøres; full fit og resultattekst skal vente til power er godkjent. Dette
er en reell modellforutsetning, ikke grunn til å bruke 2024.

## 10. Implementeringsrekkefølge

1. Les denne planen og kontroller arbeidskopien for brukerens eksisterende
   endringer. Ikke overskriv urelaterte eller upubliserte endringer.
2. Bekreft den låste Tweedie-poweren fra utviklingsløpet uten å lese 2024.
3. Opprett `src_ml/__init__.py` og `src_ml/catboost_pricing.py`.
4. Opprett `src_asserts/ml_asserts.py` med bare de målrettede kontrollene i
   seksjon 7.
5. Skriv om `ml_pricing.py` etter notebookstrukturen i seksjon 6. Gjenbruk
   `build_development_frames`, `build_tweedie_frame` og eksisterende
   gruppefoldprinsipper.
6. Kjør raske, utviklingsbaserte funksjonskontroller før hele gridet.
7. Kjør notebooken fullt, vurder tabell og figur, og oppdater konklusjonen med
   de faktiske utviklingsresultatene uten å overtolke dem.
8. Synkroniser paret med:
   `uv run jupytext --sync ml_pricing.py`.
9. Kjør den synkroniserte notebooken på nytt og bekreft at output er
   reproduserbar og faglig plausibel.
10. Oppdater denne planens endringslogg med implementerte valg, avvik og
    resultater. Ikke commit notebookendringene før brukeren har bekreftet det i
    tråd med prosjektets notebook-workflow.

## 11. Verifikasjonskommandoer

Alle Python-kommandoer skal kjøres med `uv`:

```bash
uv run jupytext --sync ml_pricing.py
uv run jupyter nbconvert --to notebook --execute ml_pricing.ipynb \
  --output /tmp/ml_pricing.executed.ipynb \
  --ExecutePreprocessor.timeout=1800
uv run ruff check ml_pricing.py src_ml src_asserts/ml_asserts.py
```

Bruk en eksplisitt midlertidig outputfil ved utførelse slik at en feilet kjøring
ikke overskriver den parede notebooken. Dersom full gridkjøring tar lang tid,
skal agenten først kjøre en liten smoke-test gjennom samme offentlige funksjon;
smoke-parametere skal aldri bli stående i den endelige notebooken.

## 12. Akseptansekriterier

Leveransen er ferdig når:

- 2024 aldri er lest, materialisert eller evaluert;
- ML-notebooken inneholder bare CatBoost for direkte ren premie;
- den eksplisitte predictor- og kategorilisten står i én tydelig celle;
- ingen respons-, skade- eller ID-kolonner finnes blant prediktorene;
- CatBoost bruker native kategorier og samme Tweedie-power som GLM-en;
- `GridSearchCV` bruker fem gruppefolder og vektet Tweedie-deviance;
- det lille 16-kombinasjonsgridet er kjørt uten feil;
- sluttmodellen er `best_estimator_` refittet på alle utviklingsdata;
- OOF-prediksjoner dekker alle utviklingsrader nøyaktig én gang;
- notebooken viser maksimalt én tabell og ett plott per outputseksjon;
- assert-kallene i notebooken er redusert til `assert_ml_inputs` og
  `assert_ml_result`;
- `.py` og `.ipynb` er synkronisert;
- full notebook-kjøring og Ruff-kontroll består;
- resultatteksten omtaler CV som utviklingsresultat og reserverer 2024 for den
  endelige, låste sammenligningen.

## 13. Endringslogg

| Dato | Fase | Vurdering / endring |
|---|---|---|
| 2026-09-19 | Planlegging | Erstattet det brede CatBoost/LightGBM-skjelettet med plan for én direkte CatBoost Tweedie-modell. Låste vanlig femfolds gruppe-`GridSearchCV`, native kategorier, lite fireparameters grid, begrensede asserts og senere engangsevaluering på urørt 2024. Ingen implementasjon eller datakjøring utført i denne fasen. |
| 2026-09-19 | Implementering | Implementerte én direkte CatBoost-Tweedie-utfordrer i `ml_pricing` og `src_ml/`, med native kategorier, fem gruppefolder, eksponeringsvektet deviance og OOF-diagnostikk. Låste $p=1.744$ fra Tweedie-GLM-løpet. Full utviklingskjøring valgte depth 4, 300 iterasjoner, learning rate 0.03 og $l_2$-regularisering 3.0; pooled OOF-deviance var 33.2775 mot 34.0919 for foldvis nullmodell ($D^2=0.0239$). 2024 ble ikke lest. Midlertidige kontrollpunkter ble ikke beholdt i leveransekoden etter brukerønske. |
