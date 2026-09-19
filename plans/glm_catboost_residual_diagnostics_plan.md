# Plan: CatBoost-diagnostikk av GLM-residualstruktur

## 1. Formål og avgrensning

Denne fasen skal bygge én gjenbrukbar diagnostikk for de valgte GLM-ene for
frekvens, severity og ren premie/Tweedie. Diagnostikken skal undersøke om
GLM-ens betingede middelverdi fortsatt etterlater systematisk struktur som en
fleksibel modell kan finne, særlig:

- ikke-lineære sammenhenger blant variablene som allerede inngår i GLM-en;
- interaksjoner mellom de samme variablene;
- ren nivåfeil som kan løses med en konstant rekalibrering, og som derfor ikke
  skal forveksles med manglende funksjonsform eller interaksjoner.

CatBoost er her et diagnostisk verktøy og ikke en automatisk utfordrermodell.
Et funn kan formulere en hypotese om en spline eller en eksplisitt interaksjon,
men skal aldri automatisk endre GLM-spesifikasjonen.

Eksisterende CV og variabelseleksjon i `cross_validate_glm` skal ikke endres.
Residualdiagnostikken kjøres først etter at den aktuelle GLM-finalisten er
valgt og låst i notebooken.

### Utenfor omfanget

- Ingen automatisk featureseleksjon eller generering av nye GLM-ledd.
- Ingen CatBoost-hyperparameteroptimalisering.
- Ingen erstatning av eksisterende residualplott, rootogrammer eller annen
  fordelingsdiagnostikk.
- Ingen vurdering av kausale sammenhenger.
- Ingen lesing, inspeksjon, materialisering eller evaluering av 2024-data.
- Ingen endring av eksisterende GLM-seleksjonsregler.

## 2. Låste metodebeslutninger

| ID | Beslutning | Begrunnelse |
|---|---|---|
| RD-01 | Diagnostikken bruker nestet, gruppebasert cross-fitting. | En ny CV direkte på en global OOF-residualserie kan lekke informasjon mellom foldnivåene. |
| RD-02 | Eksisterende notebookdefinerte `insured_id`-folder er ytre folder. | Bevarer samme generaliseringsmål som GLM-seleksjonen og hindrer personlekkasje. |
| RD-03 | Indre OOF-prediksjoner lages bare innen den aktuelle ytre treningsdelen, med fem nye `GroupKFold`-folder. | CatBoosts treningsmål skal aldri være produsert av en GLM som har sett ytre valideringsutfall. |
| RD-04 | Treningsmålet er residualratio `z = y / mu_inner_oof`, med korrekt spesifikasjon representert ved `z = 1`. | Gir samme, intuitive og positive korreksjonsmål for Poisson, Gamma og Tweedie. |
| RD-05 | CatBoost vektes med `a = w * mu_inner_oof ** (2 - p)`. | Dette er log-linkens inverse-varians-/IRLS-vekt opp til dispersjon og gjør metodikken familieagnostisk. |
| RD-06 | CatBoost bruker `loss_function="Poisson"` på den ikke-negative ratioen. | Loss-funksjonen estimerer en positiv betinget korreksjonsfaktor; den er ikke en fordelingspåstand om severity. |
| RD-07 | Korrigert middel er `mu_corrected = mu_glm * h_hat`, der `h_hat` er CatBoost-prediksjonen på respons-skala. | Korreksjonen forblir positiv uten arbitrær clipping. |
| RD-08 | Tre benchmarker rapporteres: uendret GLM, foldlært konstant korreksjon og CatBoost. | Skiller ren nivåkalibrering fra strukturgevinst. |
| RD-09 | CatBoost kjøres med depth 1 og depth 3, ellers identiske låste innstillinger. | Depth 1 undersøker primært additiv reststruktur; mergevinst fra depth 3 indikerer mulig interaksjonsstruktur. |
| RD-10 | Hovedscore er opprinnelig familie-deviance på ytre valideringsrader. | Diagnosen skal vurderes på samme respons og tapsfunksjon som GLM-en, ikke bare på residual-RMSE. |
| RD-11 | Variabelbetydning beregnes ved permutasjon på ytre valideringsfold og måles som økning i korrekt familie-deviance. | Unngår CatBoosts treningsbaserte importance som hovedbevis. |
| RD-12 | OOF-SHAP beregnes på ytre valideringsrader og brukes kun til form-/retningsdiagnostikk etter at et stabilt signal er påvist. | Holder effektvisualisering adskilt fra signaldeteksjon. |
| RD-13 | Første kjøring bruker bare GLM-ens egne råprediktorer. | Da kan signalet tolkes som manglende funksjonsform/interaksjon, ikke som en blanding med utelatte hovedvirkninger. |
| RD-14 | Det settes ingen materiell gevinstgrense i denne implementeringen. | En datadrevet terskel ville skape et nytt seleksjonsledd. Positiv gevinst beskrives derfor som signalstyrke, ikke bestått test. |

## 3. Statistisk definisjon

For observasjon `i` har den låste GLM-en respons `y_i`, predikert middel
`mu_i`, basisvekt `w_i` og Tweedie-kraft `p` fra modellspesifikasjonen.

CatBoosts treningsmål i en ytre treningsfold er

$$
z_i = \frac{y_i}{\hat\mu_{i,\mathrm{inner\ OOF}}}.
$$

Diagnostikkvekten er

$$
a_i = w_i\hat\mu_{i,\mathrm{inner\ OOF}}^{2-p}.
$$

Dette gir følgende spesialtilfeller:

| Modell | `y` | `w` | `p` | Diagnostikkvekt |
|---|---|---|---:|---|
| Frekvens | skadefrekvens | eksponering | 1 | `exposure * mu` |
| Severity | gjennomsnittsskade | skadeantall | 2 | `claim_count` |
| Tweedie | ren premie | eksponering | valgt `p` | `exposure * mu ** (2 - p)` |

CatBoost estimerer en positiv korreksjonsfaktor `h(x)`. På ytre
valideringsdata evalueres

$$
\hat\mu_{i,\mathrm{CatBoost}}=
\hat\mu_{i,\mathrm{outer\ GLM}}\hat h(x_i).
$$

Nullkorreksjonen er `h(x) = 1`. Den konstante benchmarken læres fra de indre
OOF-ratioene i ytre trening:

$$
\hat c = \frac{\sum_i a_i z_i}{\sum_i a_i},
\qquad
\hat\mu_{i,\mathrm{constant}}=
\hat\mu_{i,\mathrm{outer\ GLM}}\hat c.
$$

Denne konstanten er lært uten ytre valideringsutfall og skiller generell
kalibreringsfeil fra kovariatavhengig reststruktur.

Alle modeller scores med
`mean_tweedie_deviance(y, prediction, sample_weight=w, power=p)` på ytre
valideringsrader. Residual-R² og p-verdier fra fem folder skal ikke brukes som
hovedkonklusjon.

## 4. Lekkasjesikker foldflyt

For hver ytre fold `k` skal implementasjonen gjøre følgende i angitt rekkefølge:

1. Valider at ytre train-/val-indekser finnes, er disjunkte og ikke er tomme.
2. Valider at ingen `insured_id` finnes på begge sider av ytre fold.
3. Klargjør ytre trenings- og valideringsramme med
   `prepare_fold_frames`; all preprocessing som er deklarert i
   `spec["derived"]`, inkludert eventuell sentrering, læres bare på ytre
   trening. Forhåndslagde, responsuavhengige utviklingsfeatures som
   `vehicle_brand_pooled` behandles som en del av den allerede låste
   GLM-spesifikasjonen; diagnostikken skal ikke lære ny pooling.
4. Fit den allerede låste GLM-spesifikasjonen på ytre trening med `fit_glm`.
   Bruk samme `fit_kwargs` i alle ytre og indre GLM-fits. Det skal ikke
   gjennomføres ny GLM-seleksjon.
5. Prediker både `mu_outer_train` og `mu_outer_val`, og kjør de samme
   foldkontrollene som vanlig GLM-CV med `check_fold_fit`. Funksjonen krever
   begge prediksjonene.
6. Opprett fem indre `GroupKFold`-folder kun fra ytre trening, gruppert på
   `insured_id`, med deterministisk seed `seed + outer_fold_number`.
7. Kjør `cross_validate_glm` med samme låste `spec` på ytre treningsramme og
   de indre foldene. Dette gir `mu_inner_oof` for alle rader i ytre trening.
8. Krev full indre OOF-dekning og gyldig GLM-fit i alle indre folder.
9. Beregn `z` og `a` fra `mu_inner_oof`. Krev endelige verdier, `z >= 0` og
   `a > 0`; ikke klipp eller erstatt ugyldige prediksjoner.
10. Klargjør CatBoost-features med tilstand lært bare på ytre trening.
11. Lær den konstante korreksjonen og CatBoost-modellene på ytre trening.
12. Ytre valideringsramme skal aldri sendes som `eval_set`, brukes til early
    stopping eller påvirke hyperparametere/preprocessing.
13. Prediker positive korreksjonsfaktorer på ytre validering med
    `model.predict(..., prediction_type="Exponent")`. Standard/raw
    `RawFormulaVal` er log-korreksjonen og skal ikke multipliseres med GLM-ens
    middelverdi.
14. Score GLM, konstant benchmark, depth 1 og depth 3 på opprinnelig respons i
    ytre validering.
15. Beregn permutation importance og SHAP bare på ytre valideringsfeatures.
16. Skriv foldens resultater til outputserier ved eksakt `.loc[val.index]`.

Etter sløyfen skal ytre valideringsindekser dekke inputindeksen nøyaktig én
gang. Manglende eller dupliserte rader er en hard feil; bruk ikke
`Index.intersection`, fordi det kan skjule feil i folddefinisjonen.

### Hvorfor global OOF-residual-CV ikke brukes

Hvis CatBoost trenes på én global serie med GLM-OOF-residualer, kan en
treningsresidual være laget av en GLM som har sett utfallet i CatBoosts ytre
valideringsfold. Raden selv er riktignok holdt ute fra sin GLM, men de to
modellnivåene er ikke uavhengige. Den nestede flyten over fjerner denne
kryssfoldkoblingen.

## 5. Filstruktur og ansvar

### Ny fil: `src_core_glm/residual_diagnostics.py`

All gjenbrukbar CatBoost-diagnostikk legges her. Modulen skal ikke lese data,
opprette utviklings-/testsplitter, bruke notebook-globale variabler eller kalle
`display()`/`plt.show()`.

Modulen deles i fire tydelige blokker:

1. Input- og foldvalidering.
2. Residualratio, vekter og nested CV.
3. Oppsummering og valideringsbasert importance.
4. Plotfunksjoner.

CatBoost skal importeres i denne modulen, ikke i `glm_core.py`. Eksisterende
`glm_core.py` skal i første implementasjon ikke refaktoreres: diagnostikken
gjenbruker `prepare_fold_frames`, `fit_glm`, `check_fold_fit` og
`cross_validate_glm` direkte. Dette holder dagens GLM-seleksjon atskilt og
reduserer regresjonsrisiko. Et senere uttrekk av en generell `fit_glm_fold`
er eksplisitt utenfor denne fasen.

### Ny fil: `tests/test_residual_diagnostics.py`

Permanente tester skal bruke syntetiske data og standardbibliotekets
`unittest`; prosjektet trenger derfor ikke en ny testavhengighet. Testene skal
aldri importere eller lese prosjektets datakilder.

Opprett også en tom `tests/__init__.py`, slik at den dokumenterte
modulkommandoen for testkjøring fungerer likt i alle miljøer.

### Notebookfiler som integreres senere i samme leveranse

- `glm_pricing_models.py`
- `glm_pricing_severity.py`
- `tweedie.py`

Bare `.py`-speilene redigeres. Tilhørende `.ipynb` synkroniseres med
`uv run jupytext --sync <notebook>.py` etter verifisert kodeendring.

### Filer som ikke skal endres

- `src_core_glm/model_data.py`: diagnostikken mottar ferdig utviklingsramme.
- Eksisterende variabelseleksjonslogikk i `src_core_glm/model_selection.py`.
- Eksisterende frekvensdiagnostikk: residualplott og rootogram svarer på andre
  spørsmål og beholdes.
- Rådata og alle filer som kan materialisere 2024-testsettet.

## 6. Offentlig API

Navn, argumenter og returstruktur nedenfor er implementeringskontrakten. En
billigere implementeringsagent skal ikke endre den uten at planen først
revideres.

### 6.1 Hovedfunksjon

```python
def cross_validate_residual_catboost(
    spec,
    data,
    folds,
    feature_columns,
    *,
    categorical_columns=(),
    group_column="insured_id",
    blocked_feature_columns=(),
    depths=(1, 3),
    inner_splits=5,
    fit_kwargs=None,
    catboost_params=None,
    permutation_repeats=3,
    calculate_shap=True,
    seed=100,
):
    """Kjør lekkasjesikker, nestet CatBoost-diagnostikk av en låst GLM."""
```

Funksjonen skal hente respons, basisvekt og power utelukkende fra
`spec["y"]`, `spec["weight"]` og `spec["power"]`. Notebookene skal ikke kunne
sende motstridende kopier av disse verdiene.

`catboost_params` kan overstyre produksjonsstandardene, men følgende skal
låses av funksjonen og ikke kunne overstyres:

- `loss_function="Poisson"`
- `random_seed=seed`
- `verbose=False`
- `allow_writing_files=False`
- `task_type="CPU"`
- `max_ctr_complexity=1`
- `depth` settes av `depths`-sløyfen

Foreslåtte, eksplisitte produksjonsstandarder:

```python
{
    "iterations": 300,
    "learning_rate": 0.03,
    "l2_leaf_reg": 5.0,
    "random_strength": 1.0,
}
```

Det brukes ikke `eval_set` eller early stopping. Alle innstillinger låses før
ytre resultater foreligger. `max_ctr_complexity=1` hindrer CatBoost i å lage
kombinerte kategoriske CTR-features som kunne gitt depth 1 skjult
interaksjonskapasitet. Depth 1 er fortsatt bare en praktisk additiv benchmark,
ikke et formelt bevis på additivitet.

Tillatte nøkler i `catboost_params` er bare `iterations`, `learning_rate`,
`l2_leaf_reg`, `random_strength` og `thread_count`. Alle andre nøkler avvises,
slik at caller ikke kan overstyre loss, depth, seed, CPU-modus,
CTR-kompleksitet, overfitting detector, eval-sett eller filskriving gjennom
merge-rekkefølgen.

### 6.2 Små beregningsfunksjoner

```python
def residual_ratio(response, prediction):
    """Returner y / mu etter eksplisitt domenevalidering."""


def residual_ratio_weight(base_weight, prediction, power):
    """Returner w * mu ** (2 - p) etter eksplisitt validering."""
```

Disse holdes offentlige fordi de er enkle å teste og dokumentere i notebookens
metodeforklaring. De skal bevare pandas-indeks når input er `Series`.

### 6.3 Tabellfunksjoner

```python
def build_residual_diagnostic_summary(result):
    """Én rad per benchmark/depth med pooled OOF-score og foldstabilitet."""


def build_residual_importance_table(result, *, depth=3, top_n=10):
    """Aggreger held-out permutation importance på tvers av folder."""
```

### 6.4 Plotfunksjoner

```python
def plot_residual_diagnostic_overview(result, *, depth=3, top_n=10):
    """To paneler: foldvis deviancegevinst og held-out importance."""


def plot_residual_dependence(
    result,
    data,
    feature,
    *,
    depth=3,
    color_feature=None,
    n_bins=12,
    feature_labels=None,
):
    """Vis OOF-SHAP mot én feature, eventuelt farget etter en mulig interaksjon."""
```

Plotfunksjonene skal returnere `matplotlib.figure.Figure`, aldri vise eller
lagre figuren selv. Norske standardetiketter kan brukes, men caller skal kunne
sende `feature_labels` uten å endre core-koden.

## 7. Inputkontrakter og fail-fast-kontroller

Før første fit skal hovedfunksjonen validere:

- `data.index` er unik;
- alle train-/val-indekser finnes i `data`;
- hver fold har ikke-tom, disjunkt train og val;
- hver ytre valideringsrad forekommer nøyaktig én gang totalt;
- ingen gruppe forekommer i både train og val i samme fold;
- `inner_splits >= 2` og hver ytre trening har minst så mange unike grupper;
- alle feature-, respons-, vekt- og gruppekolonner finnes;
- `categorical_columns` er et delsett av `feature_columns`;
- `feature_columns` er unike;
- respons, `group_column`, `spec["weight"]` og alle
  `blocked_feature_columns` er utelukket fra features;
- responsen er endelig og ikke-negativ; Gamma-spesifikasjonen identifiseres
  maskinelt med `spec["family"] == "gamma"` og krever strengt positiv respons;
- basisvekten er endelig og strengt positiv;
- `spec["family"] == "poisson"` krever `p == 1`, `"gamma"` krever `p == 2`,
  og `"tweedie"` krever `1 < p < 2`;
- `depths` inneholder nøyaktig de forhåndslåste verdiene `(1, 3)` i første
  implementasjon;
- `catboost_params` inneholder bare de eksplisitt tillatte nøklene i seksjon
  6.1.

Kontraktbrudd i denne listen er precondition-feil og skal gi `ValueError` før
fitting. Feil som først oppstår etter at foldkjøringen har startet, for
eksempel ikke-konvergens eller ugyldig inner-OOF, skal fanges og gi et komplett
resultatobjekt med `valid=False` og en foldmerket `error`. Det skal aldri
returneres et delvis resultat som ser gyldig ut.

Notebooken sender en eksplisitt blokklistet samling med responsavledede felt,
blant annet skadeantall, skadekostnad, eksisterende OOF-prediksjoner og
residualkolonner. Core-funksjonen kan bare kontrollere kolonnenavn den får og
kan ikke alene oppdage semantisk target leakage.

Diagnostikkmodulen skal ikke undersøke årskolonnen for å lete etter 2024: det
ville innebære inspeksjon av et eventuelt feilaktig innsendt testsett. Sikkerhet
mot 2024 håndteres før rammen bygges, gjennom eksisterende utviklingsloader og
notebookens `assert_development_years`. Diagnostikkfunksjonen mottar kun den
allerede avgrensede utviklingsrammen.

## 8. CatBoost-featurebehandling

En privat hjelpefunksjon skal ta ytre trening og ytre validering og returnere
to nye DataFrames med samme indeks og kolonnerekkefølge.

`feature_columns` kan i denne fasen bare inneholde kolonner fra den låste
GLM-spesifikasjonens `required_columns`: rå features, deterministiske
transformasjoner eller allerede dokumenterte responsuavhengige
utviklingsfeatures. Diagnostikken tilbyr ingen generell feature-`learn/apply`
hook. En ny feature som må lære pooling, bin-grenser eller annen tilstand fra
data, er derfor utenfor omfanget og krever planrevisjon.

- Kategoriske kolonner fylles med den faste teksten `MISSING` og konverteres
  til streng. CatBoost håndterer nivåer som bare finnes i validering.
- Numeriske kolonner beholdes numeriske; `NaN` håndteres nativt av CatBoost.
- Positive/negative `inf` avvises.
- Ingen one-hot-koding, skalering eller global medianberegning.
- Ingen dtype- eller kategorinivåinformasjon læres fra ytre validering.
- Inputrammene muteres ikke.
- Kolonneordenen skal være identisk med `feature_columns`.

Native kategorier velges fremfor notebookens foreløpige one-hot-oppsett i
`ml_pricing.py`, fordi dette er en egen residualdiagnostikk med eksplisitt
foldkontrakt. Koden fra `ml_pricing.py` skal ikke importeres.

## 9. Returkontrakt

`cross_validate_residual_catboost` returnerer en dict med følgende nøkler:

```python
{
    "predictions": pandas.DataFrame,
    "fold_scores": pandas.DataFrame,
    "permutation_importance": pandas.DataFrame,
    "shap_values": dict[int, pandas.DataFrame],
    "shap_base_values": dict[int, pandas.Series],
    "valid": bool,
    "error": str | None,
    "config": dict,
}
```

### `predictions`

Samme indeks og rekkefølge som `data`, med kolonnene:

- `fold`
- `observed`
- `base_weight`
- `diagnostic_weight`
- `glm`
- `constant`
- `catboost_depth_1`
- `catboost_depth_3`
- `observed_ratio`
- `constant_factor`
- `catboost_factor_depth_1`
- `catboost_factor_depth_3`

`observed_ratio` bruker ytre OOF-GLM-prediksjon og er bare diagnostisk output;
den brukes ikke som CatBoost-treningsmål.

### `fold_scores`

Langformat med én rad per ytre fold og benchmark:

- `fold`
- `benchmark`: `glm`, `constant`, `catboost_depth_1` eller
  `catboost_depth_3`
- `n_train`, `n_val`, `train_weight`, `val_weight`
- `deviance`
- `delta_vs_glm = D_glm - D_benchmark`
- `delta_vs_constant = D_constant - D_benchmark`
- `delta_vs_depth_1 = D_depth_1 - D_benchmark`
- `relative_gain_vs_glm`
- `actual`
- `predicted`
- `ae = actual / predicted`
- `converged` og `valid`

### `permutation_importance`

Langformat:

- `depth`
- `fold`
- `feature`
- `repeat`
- `deviance_increase`

For hver permutasjon holdes ytre GLM-prediksjon fast, bare den aktuelle
CatBoost-featurekolonnen permuteres, og korrigert prediksjon scores på nytt.
Seed skal avledes deterministisk fra hovedseed, foldens posisjon i den
innsendte foldrekkefølgen, depth, featureposisjon og repeat, for eksempel med
`numpy.random.SeedSequence`. Foldnavn skal ikke parses, og Pythons
prosessavhengige `hash()` skal ikke brukes.

Ordinær og permutert CatBoost-prediksjon skal begge bruke
`prediction_type="Exponent"`. SHAP beregnes derimot på modellens rå/log-skala,
slik CatBoosts additive SHAP-dekomponering krever.

### `shap_values`

Én DataFrame per depth, med samme indeks som `data` og én kolonne per feature.
Verdiene er OOF-SHAP fra den foldmodellen der raden var ytre validering. For
Poisson-loss tolkes de på CatBoosts rå/log-korreksjonsskala. Forventet verdi
lagres separat i `shap_base_values`; ikke bland den inn som en feature.

### Gyldighet

`valid=True` krever:

- alle ytre og indre GLM-fits er gyldige;
- komplett og entydig inner- og outer-OOF-dekning;
- endelige og positive GLM-/korreksjonsprediksjoner;
- endelige scorer;
- komplett permutation importance og, når aktivert, SHAP-dekning.

Ved fold-/fitfeil returneres `valid=False` og en presis foldmerket feilmelding.
Precondition-feil kaster `ValueError` før fitting. Det skal ikke presenteres
pooled konklusjoner fra en delvis gyldig kjøring.

## 10. Oppsummering og fortolkning

`build_residual_diagnostic_summary` beregner pooled deviance direkte over alle
OOF-rader, ikke som uvektet gjennomsnitt av foldscorer. Tabellen skal ha én rad
per benchmark og minst:

- `benchmark`
- `pooled_oof_deviance`
- `delta_vs_glm`
- `relative_gain_vs_glm`
- `delta_vs_constant`
- `delta_vs_depth_1`
- `better_than_glm_folds`
- `better_than_constant_folds`
- `better_than_depth_1_folds`
- `oof_ae`
- `valid`

Metrikkene defineres entydig som:

- `delta_vs_glm = D_glm - D_model`;
- `delta_vs_constant = D_constant - D_model`;
- `delta_vs_depth_1 = D_depth_1 - D_model` for depth 3 og `NaN` for øvrige
  benchmarkrader;
- `relative_gain_vs_glm = (D_glm - D_model) / D_glm`;
- `actual = sum(w * y)`;
- `predicted = sum(w * mu_hat)`;
- `ae = actual / predicted`.

Hvis nevneren i en relativ metrikk er null, returneres `NaN`, ikke uendelig.
Ikke-positive predikerte totaler er allerede en gyldighetsfeil.

Fortolkningsrekkefølgen er låst:

1. Kontroller `valid` og full OOF-dekning.
2. Sammenlign konstant benchmark mot GLM. Gevinst bare her er primært
   nivåkalibrering.
3. Sammenlign depth 1 mot både GLM og konstant. Stabil gevinst tyder på
   manglende additiv funksjonsform.
4. Sammenlign depth 3 mot depth 1. Mergevinst pooled og i minst fire av fem
   folder tyder på mulig interaksjonsstruktur.
5. Se først deretter på permutation importance og OOF-SHAP.

En kandidat omtales som stabilt signal når den forbedrer pooled deviance og
minst fire av fem ytre folder. Uten en forhåndsvalgt materiell terskel skal en
liten positiv forbedring beskrives som svak. Det skal ikke beregnes p-verdi av
de fem foldforskjellene.

Depth 1 er ikke et matematisk bevis på «bare nonlinearitet», og depth 3 er ikke
et bevis på interaksjon. Korrelerte features kan flytte både importance og
SHAP-bidrag. Språket i notebookene skal derfor være «indikerer», «forenlig
med» og «hypotese», ikke «påviser».

## 11. Gjenbrukbare figurer

### 11.1 `plot_residual_diagnostic_overview`

Én figur med to delpaneler:

- Venstre: foldvis `delta_vs_glm` for konstant, depth 1 og depth 3, med tydelig
  nullinje. Dette viser både samlet retning og foldstabilitet.
- Høyre: topp `top_n` features etter gjennomsnittlig held-out
  `deviance_increase`, med foldpunkter og aggregert middel. Negative verdier
  beholdes; de skjules ikke. `depth=3` er standard, og importance skal aldri
  aggregeres på tvers av depth 1 og depth 3.

Figuren må kunne brukes identisk for alle tre GLM-familiene. Tittel kan bruke
`spec["name"]` fra `result["config"]`, men ingen notebookspesifikke navn skal
hardkodes.

### 11.2 `plot_residual_dependence`

Denne figuren brukes bare etter at oversikten viser et stabilt signal.

- Numerisk feature: del observasjonene i vektede kvantiler og vis vektet
  gjennomsnittlig OOF-SHAP med spredning per bin. Ikke bruk et tett råscatter
  som skjuler strukturen.
- Kategorisk feature: vis vektet gjennomsnittlig OOF-SHAP og støtte per nivå;
  samle sjeldne visningsnivåer i `OTHER` uten å endre modellinput.
- `color_feature=None`: ett hovedmønster.
- `color_feature` satt: separate kurver/farger for et lite antall nivåer eller
  vektede grupper av en numerisk variabel. Dette er en eksplorativ visning av
  mulig interaksjon, ikke en validert interaksjonstest.
- Y-aksen merkes «OOF-SHAP på log-korreksjonsskala».
- Binning, middel og støtte vektes med radens ytre diagnostikkvekt
  `a_outer = w * mu_outer_glm ** (2 - p)`, lagret som
  `predictions["diagnostic_weight"]`.
- Funksjonen skal validere at feature og eventuell `color_feature` var med i
  CatBoost-kjøringen.

Det skal ikke bygges en generell dashboard- eller plottingklasse. To små
figurfunksjoner er tilstrekkelig.

## 12. Notebook-integrasjon

Hver notebook skal beholde spesifikasjon, features, seed og fit-kall synlig.
Løkker, nesting, validering, tabellbygging og plotting ligger i core-modulen.
En ny notebookcelle skal gjøre én tydelig oppgave og normalt være under 25
kodelinjer.

### 12.1 Felles import

```python
from src_core_glm.residual_diagnostics import (
    build_residual_diagnostic_summary,
    build_residual_importance_table,
    cross_validate_residual_catboost,
    plot_residual_dependence,
    plot_residual_diagnostic_overview,
)
```

### 12.2 Felles markdown før fitting

Før hver diagnostikk-fit skal notebooken kort vise:

$$
z_i=\frac{y_i}{\hat\mu_{i,\mathrm{inner\ OOF}}},
\qquad
a_i=w_i\hat\mu_{i,\mathrm{inner\ OOF}}^{2-p},
\qquad
\hat\mu_i^*=\hat\mu_{i,\mathrm{GLM}}\hat h(x_i).
$$

Teksten skal forklare at inner CV lager CatBoost-målet, outer CV måler
diagnostisk gevinst, og at CatBoost ikke foretar GLM-variabelseleksjonen.

### 12.3 Featuredeklarasjon

Standardfeatures settes eksplisitt fra den valgte spesifikasjonen:

```python
diagnostic_features = final_specification["required_columns"]
diagnostic_categorical = tuple(final_specification["base_levels"])
```

Hvis en spline eller interaksjon finnes i GLM-formelen, brukes dens råkolonner
fra `required_columns`, ikke Patsy-uttrykket. Eventuelle bredere features skal
senere ligge i en separat, tydelig navngitt kjøring og krever planrevisjon.

Hver notebook deklarerer en blokklistet samling som minst dekker:

- `insured_id`;
- respons- og vektkolonnen;
- rå skadeantall og skadekostnad;
- alle eksisterende OOF-/residual-/prediksjonskolonner.

### 12.4 Frekvens: `glm_pricing_models.py`

- Legg seksjonen etter at `review_specification` er bestemt og før dagens
  visuelle OOF-residualdiagnostikk.
- Bruk `model_frame`, `cv_folds`, `review_specification`, `SEED` og eksisterende
  development-avgrensning.
- Frekvensens eksisterende deviance-residualplott og rootogram beholdes.
- Notebooken skal bare inneholde feature-/blokkliste, ett funksjonskall, én
  oppsummeringstabell og figurkall.

### 12.5 Severity: `glm_pricing_severity.py`

- Legg seksjonen etter at `final_specification` er låst og før det avsluttende
  statsmodels-sammendraget.
- Bruk `severity_frame`, eksisterende `cv_folds`, `FIT_SETTINGS` og `SEED`.
- `property_claims` forblir basisvekt og må aldri være feature.
- Diagnostikken kjøres bare på dagens positive severity-populasjon; den skal
  ikke konstruere eller koble inn rader uten skade.

### 12.6 Tweedie: `tweedie.py`

- Integrasjon skjer først når valgt power og variabelspesifikasjon er låst i
  notebooken.
- Sett eksplisitt
  `final_tweedie_specification = selection_specifications[selected_variable_model]`
  før diagnostikkseksjonen.
- Bruk `model_frame`, eksisterende `cv_folds`, `FIT_SETTINGS` og `SEED`.
- `spec["power"]` styrer både diagnostikkvekt og sluttscore; notebooken sender
  ikke et separat power-argument.
- Hvis Tweedie-planens nåværende stopp fortsatt gjelder når implementasjonen
  starter, implementeres og testes core først, mens notebookintegrasjonen
  utsettes til finalisten faktisk er låst.

### 12.7 Outputseksjoner

Følg prosjektets regel om maksimalt én tabell og ett plott per outputseksjon:

1. **Finnes residualstruktur?**
   `build_residual_diagnostic_summary(result)` og
   `plot_residual_diagnostic_overview(result)`.
2. **Hvordan ser det sterkeste signalet ut?**
   Bare dersom første seksjon viser stabil gevinst:
   `build_residual_importance_table(result)` og ett kall til
   `plot_residual_dependence(...)`.

Notebooken skal ikke automatisk velge `feature` eller `color_feature` basert på
samme tabell. Den valgte visningen dokumenteres eksplisitt som eksplorativ.

## 13. Tester uten prosjektdata

Testene kjøres med:

```bash
uv run python -m unittest tests.test_residual_diagnostics
```

CatBoost-testene bruker små syntetiske rammer, om lag 50 iterasjoner og
`thread_count=1`, slik at testene er raske og reproduserbare.

### 13.1 Rene beregningstester

- Håndregn `z` og `a` for `p=1`, `p=2` og en Tweedie-power mellom 1 og 2.
- Kontroller `y=0` for frekvens/Tweedie.
- Kontroller at pandas-indeks bevares.
- Ugyldig `mu <= 0`, ikke-endelige verdier og ikke-positive vekter skal feile.
- Håndregn konstant `c`, pooled deviance og A/E.
- Bruk en stub med råprediksjon `log(2)` og verifiser at ordinær og permutert
  prediksjon bruker responsfaktoren `2`, mens SHAP beholdes på rå/log-skala.

### 13.2 Fold- og lekkasjetester

- Ikke-sammenhengende og stokket indeks skal bevares nøyaktig i output.
- Overlappende train/val, ukjent indeks, duplisert valideringsrad, manglende
  valideringsrad og gruppeoverlapp skal gi tydelig `ValueError` før fitting.
- En spy/stub rundt fit-steget skal bekrefte at ytre valideringsindekser aldri
  inngår i indre GLM-fit, CatBoost-fit, featuretilpasning eller `eval_set`.
- Endre bare responsen i én ytre valideringsfold og kontroller at foldens
  GLM-/CatBoost-prediksjoner er uendret, mens scoren kan endres.
- Ukjent kategorinivå og kategorisk missing kun i ytre validering skal kunne
  predikeres uten at nivået læres fra valideringen.
- Ufullstendig inner-OOF-dekning skal gjøre hele resultatet ugyldig.

### 13.3 Modellatferd på syntetiske data

Disse er små, deterministiske integrasjons-/smoke-tester, ikke skjøre krav til
tilfeldig CatBoost-atferd. Bruk faste seeds, eksplisitt sterk signalstyrke og
forhåndskodede, romslige numeriske marginer. De harde enhetstestene gjelder
først og fremst indeks-, fold-, skala-, formel- og lekkasjekontraktene.

- Korrekt spesifisert additiv GLM: ingen stabil CatBoost-gevinst. Bruk romslig
  toleranse; tilfeldig støy forventes ikke å gi eksakt null.
- Utelatt ikke-lineær sammenheng: depth 1 skal gi tydelig gevinst og riktig
  feature høyt på held-out importance.
- Ren konstruert interaksjon: depth 3 skal slå depth 1 og de to involverte
  variablene skal rangeres høyt.
- Ren nivåfeil: konstant benchmark skal ta hoveddelen av gevinsten; dette skal
  ikke merkes som strukturgevinst.
- Samme seed skal gi samme OOF-prediksjoner og importance innen stram numerisk
  toleranse.
- Alle CatBoost-faktorer og korrigerte middelverdier skal være positive og
  endelige.

### 13.4 Presentasjonstester

- Oppsummeringsfunksjonen returnerer forventede kolonner og fire benchmarkrader.
- Importance-tabellen sorterer deterministisk og beholder negative verdier.
- Oversiktsplottet returnerer én `Figure` med nøyaktig to akser.
- Dependence-plottet returnerer én `Figure` og håndterer både numerisk og
  kategorisk feature uten å kalle `show()`.

## 14. Implementeringsrekkefølge

### Trinn 1 — Ren beregningskjerne

1. Opprett `src_core_glm/residual_diagnostics.py` med inputvalidering,
   `residual_ratio` og `residual_ratio_weight`.
2. Implementer og test konstant benchmark og pooled scorefunksjon.
3. Kjør de rene syntetiske testene før CatBoost-sløyfen bygges.

### Trinn 2 — Nested foldmotor

1. Implementer ytre foldvalidering og ytre GLM-fit ved gjenbruk av eksisterende
   core-funksjoner.
2. Implementer indre grouped OOF via `cross_validate_glm`.
3. Legg inn eksplisitte deknings- og lekkasjekontroller.
4. Implementer CatBoost depth 1 og 3 med låste parametere.
5. Returner komplett `predictions` og `fold_scores` før importance/SHAP legges
   til.
6. Kjør fold-, lekkasje- og reproduserbarhetstestene.

### Trinn 3 — Diagnostisk forklaring

1. Implementer held-out permutation importance med familie-deviance.
2. Implementer OOF-SHAP og kontroller CatBoosts eksplisitte
   respons-/råprediksjonsskala med en liten stub eller håndtest.
3. Implementer tabellfunksjonene.
4. Implementer de to plotfunksjonene og deres presentasjonstester.

### Trinn 4 — Notebookintegrasjon, én notebook om gangen

1. Frekvens først, siden notebooken allerede har OOF-diagnostikk.
2. Severity etter at frekvenskjøringen og outputkontrakten er verifisert.
3. Tweedie bare dersom finalist og gjeldende Tweedie-plan tillater fitting.
4. Etter hver notebook: synkroniser `.py`/`.ipynb`, kjør notebooken, kontroller
   asserts, full OOF-dekning, én-tabell-/ett-plott-regelen og faglig rimelig
   output.

Ved feil i felles core stoppes notebookintegrasjonen til syntetiske tester er
grønne. Ikke tilpass metode eller terskler til faktiske diagnostikkresultater
uten planrevisjon.

Lag lokale `checkpoint:`-commits etter verifiserte trinn dersom det er et
naturlig sjekkpunkt. Ikke push, og ikke lag avsluttende commit uten brukerens
bekreftelse.

### Trinn 5 — Dokumentasjon og beslutningsregistre

1. Legg en kort henvisning til denne planen i hver berørt levende modellplan.
2. Registrer at CatBoost-resultatet er diagnostisk og ikke en ny automatisk
   seleksjonsport.
3. Dersom diagnostikken foreslår en ny spline/interaksjon, opprett en egen
   planbeslutning før den fittes. Samme utviklingsdata er da brukt til
   hypotesegenerering, så vanlig CV-score vil være adaptivt optimistisk.
4. Hold 2024 låst til alle modellspesifikasjoner og diagnostikkbeslutninger er
   ferdige og brukeren senere gir eksplisitt godkjenning.

## 15. Verifikasjon av notebooks

Notebookene verifiseres med prosjektets eksisterende, sikre
utviklingsdataflyt. Implementeringsagenten skal ikke åpne rådata direkte og
skal aldri kjøre en kodevei som materialiserer 2024.

For hver notebook kontrolleres:

- utviklingsassert består før diagnostikken;
- eksisterende GLM-seleksjonsresultat er uendret;
- alle ytre og indre folder er gyldige;
- `predictions.index.equals(model_frame.index)` eller tilsvarende
  severity-indeks;
- alle fire benchmarkprediksjoner er komplette, endelige og positive;
- samme seed gir reproduserbar output;
- tabeller og figurer har korte, norske etiketter og kan forstås uten å lese
  core-koden;
- notebookceller er korte og hver celle har ett tydelig ansvar;
- ingen output eller tekst omtaler diagnostikken som uavhengig test eller
  formell spesifikasjonstest.

Fordi GLM-finalistene allerede er valgt ved gjentatt bruk av de ytre
utviklingsfoldene, er også nested residualdiagnostikk en utviklingsdiagnose,
ikke en ubiasert evaluering av hele seleksjonsprosedyren. Dette skal stå i
notebookenes begrensningstekst.

## 16. Ferdigkriterier

Implementasjonen er ferdig når:

- alle offentlige API-er og returkontrakter i planen er implementert;
- eksisterende `cross_validate_glm` og variabelseleksjon oppfører seg uendret;
- alle syntetiske tester består via `uv run`;
- leakage-testene dokumenterer full separasjon mellom ytre validering og alle
  indre treningsoperasjoner;
- frekvens og severity har korte, identisk strukturerte diagnostikkseksjoner;
- Tweedie er integrert bare dersom den valgte spesifikasjonen faktisk er låst;
- hver outputseksjon har maksimalt én tabell og ett plott;
- held-out permutation importance og OOF-SHAP brukes, ikke train importance;
- tolkningen skiller nivåkalibrering, additiv reststruktur og mulig
  interaksjonsstruktur;
- ingen diagnostikk fører automatisk til en ny GLM-kandidat;
- alle berørte `.py`/`.ipynb`-par er synkronisert og kjører uten feil;
- levende planer og beslutningsregistre er oppdatert;
- 2024 fortsatt er fullstendig urørt.

## 17. Forventet omfang

Veiledende størrelse, ikke et mål i seg selv:

- `residual_diagnostics.py`: omtrent 220–300 lesbare linjer inkludert
  validering, nested CV, tabeller og plotting;
- syntetiske tester: omtrent 150–220 linjer;
- hver notebook: omtrent 10–20 kodelinjer og en kort markdownforklaring;
- ingen ny runtime-dependency, siden CatBoost allerede finnes i prosjektet.

Hvis implementasjonen blir vesentlig større, skal agenten først vurdere om
SHAP-/plotfunksjoner eller validering er blitt overgeneralisert. Ikke innfør
klasser, konfigurasjonsrammeverk eller caching uten et konkret målt behov.

## Endringslogg

| Dato | Fase | Endring og begrunnelse |
|---|---|---|
| 2026-09-19 | Plan før implementering | Låst gjenbrukbar nested residualratio-diagnostikk for frekvens, severity og Tweedie; eksplisitte lekkasjesperrer, konstant benchmark, depth 1/3, held-out deviance/permutation importance, OOF-SHAP, rene notebookkontrakter og syntetisk verifikasjon. Ingen modellkode eller data er endret. |
