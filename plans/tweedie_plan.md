# Implementeringsplan: felles GLM-validering og klargjøring av Tweedie-modell

## 1. Formål og status

Klargjør Tweedie-modellen som utfordrer til den todelte modellen

$$
\widehat{\mathrm{pure\ premium}}_i
=
\widehat{\mathrm{frequency}}_i
\times
\widehat{\mathrm{severity}}_i,
$$

uten å gjennomføre det endelige Tweedie-kandidatsøket. Infrastruktur, felles
folder, asserts og $p$-utledning skal implementeres og verifiseres. Kandidatregisteret er låst (T-07); den endelige kjøringen gjøres av brukeren, og 2024 er urørt.

## 2. Harde avgrensninger

- Faktiske 2024-data skal aldri leses, lastes, inspiseres eller modelleres.
- Bruk bare den sikre utviklingsflyten for 2022–2023.
- Ikke implementer nested CV eller eksponeringsoffset.
- Ikke velg Tweedie-variabler eller kjør det endelige kandidatsøket.
- Ikke tolk resultater eller kåre en modellvinner.
- Ikke endre den låste seleksjonsregelen uten eksplisitt godkjenning.
- Hold løsningen liten og konkret; ikke bygg et generelt modellrammeverk.
- Bevar eksisterende brukerendringer. Ingen push eller destruktive Git-operasjoner.

## 3. Låste metodiske beslutninger

### 3.1 Respons og eksponering

La $S_i$ være samlet `property_incurred`, $e_i$ være `total_exposure` og
$R_i=S_i/e_i$ være observert pure premium per eksponeringsår. Modellen er

$$
R_i
=
\frac{S_i}{e_i}
\sim
\operatorname{Tweedie}
\left(\mu_i,\frac{\phi}{e_i},p\right),
\qquad 1<p<2,
$$

med

$$
\operatorname{E}[R_i\mid x_i]=\mu_i,
\qquad
\operatorname{Var}(R_i\mid x_i)
=
\frac{\phi}{e_i}\mu_i^p,
$$

og

$$
\log(\mu_i)=\beta_0+\sum_j\beta_jx_{ij}.
$$

Implementasjonen skal bruke:

- `pure_premium = property_incurred / total_exposure` som respons;
- Tweedie-familie med log-link;
- `total_exposure` som `var_weights`;
- ingen offset;
- alle skadefrie poliseår som gyldige nullobservasjoner.

Flere skader på samme poliseårsrad er gyldig fordi responsen er aggregert
årlig skadekostnad. `property_claims` skal ikke brukes som Tweedie-vekt.

### 3.2 Estimering av Tweedie-power

Fjern `POWER_GRID`, CV-sammenligning hvor hver kandidat scores med sin egen
power, og automatisk valg av `selected_power` fra dette gridet. Deviance med
forskjellige powerverdier er ikke direkte sammenlignbar.

Pearson-estimering direkte på Tweedie-modellen ble implementert og testet
syntetisk, men ga ingen gyldig rot i $(1,2)$ på de reelle dataene (se T-05).
Implementer i stedet:

1. Bygg severity-utvalget (B-15) med `build_severity_inputs`.
2. Fit en Gamma-GLM på snittskade (vekt = antall skader, log-link) med den brede
   ankermodellen: låste variabler, kjerne, `municipality_type` og alle tillegg.
3. Les av dispersjonen $\hat\phi_s$ (Pearson-$\chi^2$/df) og sett
   $p = (1+2\hat\phi_s)/(1+\hat\phi_s)$ (`severity_implied_power` i `src_tweedie/tweedie_power.py`).
4. Krev $1<p<2$ og lås $p$ før variabelseleksjonen.

**Limitation (skal stå i notebooken, bør vurderes utvidet):** metoden antar
Gamma-fordelte skader med konstant CV og lik dispersjon for alle poliser, og
estimatet er følsomt for storskadene (ca. 1,74 med alle skader, ca. 1,62 uten
de 1 % største snittskadene). Mulige utvidelser er profilelikelihood over et
$p$-gitter (krever eksakt Tweedie-tetthet) eller sensitivitetsanalyse av
sluttmodellen ved andre $p$.

### 3.3 Senere variabelseleksjon

Tweedie følger samme hovedstruktur
som de andre GLM-ene: kjerne, alternative funksjonsformer, geografi og
kjøretøyopplysninger, forhåndsdefinerte tillegg, eventuelle interaksjoner og
eventuell ablasjon.

Med låst $p$ brukes dagens treleddsregel:

- lavere pooled OOF-deviance enn foreldremodellen;
- bedre i minst fire av fem folder;
- gyldig fit i alle folder.

## 4. Levende planer og beslutningsregister

Før implementering skal agenten lese denne planen samt
`plans/glm_pricing_models_plan.md` og `plans/Severity_plan.md`. Denne filen er
den autoritative Tweedie-planen; ikke opprett konkurrerende planfiler.

Opprett et beslutningsregister for respons, vekt, ingen offset, ingen nested
CV, utledning av $p$, felles folder, kandidatregister og fortsatt
sperre på 2024. Oppdater endringsloggen i alle berørte planer, også dersom en
faserevisjon ikke endrer metoden. Nye vesentlige metodevalg må legges frem for
brukeren før implementering.

## 5. Felles foldarkitektur

Opprett en liten felles funksjon, for eksempel i
`src_core_glm/validation.py`:

```python
build_group_folds(
    frame,
    group_column="insured_id",
    n_splits=5,
    shuffle=True,
    random_state=100,
)
```

De eksplisitte CV-parameterne skal fortsatt være synlige i notebookene.
Foldene bygges alltid på hele modellpopulasjonen. Samme fullpopulasjonsindeks,
gruppekolonne og seed skal gi identiske folder for frekvens, severity og
Tweedie.

Kontroller at train/validation er disjunkte, at ingen person overlapper, at
alle indekser tilhører fullrammen, og at hver rad finnes i nøyaktig én
valideringsfold.

## 6. Felles GLM-støtte

Utvid `cross_validate_glm` bakoverkompatibelt med en separat
prediksjonsramme, for eksempel:

```python
cross_validate_glm(
    spec,
    data,
    folds,
    prediction_data=None,
    fold_hook=None,
    fit_kwargs=None,
)
```

- `data` er populasjonen modellen fittes og scores på.
- `prediction_data` er populasjonen som skal få OOF-prediksjoner.
- `None` skal bevare dagens oppførsel.
- All preprocessing læres bare på foldens treningsutvalg.
- Samme lærte preprocessing brukes ved score og full prediksjon.
- Severity scores bare på severity-utvalget, men predikerer hele
  valideringsfolden.

Behold `result["oof"]` for scorepopulasjonen og legg til et tydelig resultat,
for eksempel `result["prediction_oof"]`, for den fulle prediksjonspopulasjonen.
Ikke endre seleksjonslogikken i `src_core_glm/model_selection.py`.

## 7. Korreksjon av severity-foldene

`glm_pricing_severity.py` skal hente både `model_frame` og `severity_frame`.
Foldene bygges på hele `model_frame`, mens kontrakten per fold er

```text
fit_index   = full_train_index ∩ severity_frame.index
score_index = full_val_index   ∩ severity_frame.index
apply_index = full_val_index
```

Gamma-deviance skal ikke beregnes på skadefrie rader. Disse skal bare få en
severity-prediksjon slik at `oof_frequency * oof_severity` senere kan
beregnes for alle poliseår. Samlet `prediction_oof` skal dekke nøyaktig hele
`model_frame`.

Behold notebookens eksisterende hovedstruktur og kandidatstige. Hvis de nye
foldene teknisk endrer valgt severity-modell, skal agenten rapportere
modell-ID før og etter uten å tolke eller foreta et nytt manuelt valg.

## 8. Assert-arkitektur

Opprett:

```text
src_asserts/
├── __init__.py
├── common_glm_asserts.py
├── frequency_asserts.py
├── severity_asserts.py
└── tweedie_asserts.py
```

Funksjonene skal være små, domenespesifikke og gi tydelige feilmeldinger.

Felles kontroller skal dekke tillatte utviklingsår, unik indeks, positiv og
endelig eksponering, ikke-negative utfall, foldedekning, gruppeadskillelse,
full OOF-dekning og endelige positive prediksjoner.

Frekvenskontroller skal blant annet verifisere
`claim_frequency = property_claims / total_exposure`, fullpopulasjonsindeks
og eksponeringsvekt.

Severity-kontroller skal verifisere delmengdeindeks, låst responsmaske,
`average_severity = property_incurred / property_claims`, positive
skadevekter, foldkontrakten og full `prediction_oof`.

Tweedie-kontroller skal verifisere pure-premium-identiteten, positiv
eksponering, ikke-negativ incurred, beholdte nuller, $1<p<2$, endelig
powerestimat, konvergens, eksponeringsvekt og fravær av offset.

`src_frequency`, `src_severity` og nye `src_tweedie` skal importere relevante
kontroller. Notebookene skal normalt kalle én beskrivende valideringsfunksjon
per seksjon fremfor å inneholde lange assertblokker.

## 9. Ny `src_tweedie`-pakke

Opprett bare det konkrete minimumet:

```text
src_tweedie/
├── __init__.py
├── tweedie_data.py
└── tweedie_power.py
```

`tweedie_data.py` bygger full modellramme, oppretter `pure_premium`, bevarer
indeksen og kaller asserts. Den skal ikke definere kandidatregisteret.

`tweedie_power.py` inneholder `severity_implied_power`: Gamma-GLM på
severity-utvalget og $p$ fra dispersjonen (T-05). Funksjonen returnerer
dispersjon, power og antall skaderader, velger ikke variabler og tolker ikke
resultatet.

Modellspesifikasjon og fitting skal fortsatt være synlig i notebooken.

## 10. Tweedie-notebook

Samme hovedstruktur som de andre GLM-notebookene:

1. Datagrunnlag
2. Validering
3. Modellspesifikasjon (eksplisitte lister `LOCKED`, `CORE`, `GEOGRAPHY`, `ADDITIONAL`, `INTERACTIONS`)
4. Sammenligning og valg: 4.1 låsing av $p$ (med limitation), 4.2–4.3
   funksjonsform, 4.4 geografi, 4.5 tillegg, 4.6 interaksjoner, 4.7 ablasjon,
   4.8 oppsummering

Hver celle skal gjøre én tydelig oppgave og normalt være høyst 25 kodelinjer.
Modellformelen skal stå i markdown før fitting. Løkker, tester og tabellbygging
ligger i støttefunksjoner (`src_core_glm/model_selection.py`, `src_tweedie/`).

Notebooken avslutter med meldingen «Spesifikasjonen er ikke låst før den er
godkjent. 2024 er ikke berørt.» Den endelige kjøringen gjøres av brukeren.

## 11. Frekvens og senere benchmark

Oppdater frekvensnotebooken bare for å bruke felles folder og asserts og for å
verifisere bakoverkompatibilitet. Ikke endre spesifikasjoner eller seleksjon.

Klargjør, men ikke kjør, den senere indekskontrakten:

```python
two_part_oof = frequency_oof * severity_prediction_oof
tweedie_oof = tweedie_result["prediction_oof"]
```

Begge skal senere ha identisk indeks, full dekning og endelige positive
prediksjoner. De skal sammenlignes mot samme `pure_premium`, med samme
eksponeringsvekt og én felles, forhåndslåst score-power. Selve benchmarken er
utenfor denne leveransen.

## 12. Verifikasjon uten resultattolkning

Lag et midlertidig syntetisk script under `src_temp/` som tester:

- deterministiske og gruppeadskilte folder;
- full validerings- og OOF-dekning;
- severity-fit på delmengden og prediksjon på full valideringsfold;
- bakoverkompatibel frekvens-/Tweedie-CV;
- ugyldig eksponering, negativ incurred og manglende OOF-rad;
- ugyldig $p$ og gjenfinning av kjent $p$ fra severity-dispersjonen;
- fravær av offset.

Deretter:

1. Kjør import- og syntakskontroll.
2. Kjør de syntetiske kontrakttestene.
3. Synkroniser endrede notebookpar med
   `uv run jupytext --sync <notebook>.py`.
4. Kjør frekvensnotebooken for regresjonskontroll.
5. Kjør severity-notebooken med fullpopulasjonsfolder.
6. Kjør Tweedie-notebooken frem til den planlagte stoppen.

Rapporter bare pass/fail, antall folder, indeks-/OOF-dekning, assertstatus,
kjørestatus og eventuell teknisk endring av severity-modell-ID. Ikke tolk
koeffisienter eller scorer og ikke velg modell.

## 13. Sekvensiell delegering

Unngå samtidige skrivere i felles filer:

1. **Core og asserts:** `src_core_glm/`, `src_asserts/` og syntetiske tester.
2. **Severity:** `src_severity/` og `glm_pricing_severity.py` etter core.
3. **Frekvens:** `src_frequency/` og `glm_pricing_models.py` etter core.
4. **Tweedie:** `src_tweedie/`, `tweedie.py` og denne planen etter core.
5. **Integrasjon:** primært read-only kontroll etter de øvrige pakkene.

Alle notebookendringer gjøres i `.py`-speilene, aldri direkte i `.ipynb`.

## 14. Ferdigkriterier

Leveransen er ferdig når:

- Tweedie-likningen er korrekt og modellen bruker rate og vekt uten offset;
- ugyldig power-grid-CV er fjernet;
- $p$ utledes fra severity-dispersjonen, syntetisk testet, og limitation står i notebooken;
- alle tre modeller bruker samme fullpopulasjonsfolder;
- severity scores riktig delmengde og predikerer hele valideringsfolden;
- `prediction_oof` dekker hele modellpopulasjonen;
- sentrale kontroller ligger i `src_asserts`;
- notebooks er korte, synkroniserte og kjører uten tekniske feil;
- notebooken kjører kandidatseleksjon uten tekniske feil (endelig kjøring gjøres av brukeren);
- ingen modellresultater er faglig tolket av assistenten;
- berørte levende planer har oppdatert endringslogg;
- 2024 fortsatt er fullstendig urørt.

Lag lokale checkpoint-commits ved naturlige, verifiserte delmilepæler med
prefikset `checkpoint:`. Ikke push og ikke lag en avsluttende commit uten
brukerens bekreftelse.

## Beslutningsregister

| ID | Beslutning | Begrunnelse | Status |
|---|---|---|---|
| T-01 | Respons er `pure_premium = property_incurred / total_exposure`; skadefrie poliseår beholdes som nuller. | Tweedie med $1<p<2$ har masse i null og kontinuerlig positiv del. | Låst |
| T-02 | `var_weights = total_exposure`. | Høy eksponering er en mer presis observasjon av samme $\mu_i$. | Låst |
| T-03 | Ingen offset. | Responsen er allerede en rate; offset og vekt ville telle eksponering to ganger. | Låst |
| T-04 | Ingen nested CV. | $p$ estimeres én gang på ankermodellen og låses før seleksjon; deviance med ulik $p$ er ikke sammenlignbar. | Låst |
| T-05 | $p$ utledes fra severity-dispersjonen: $p=(1+2\hat\phi_s)/(1+\hat\phi_s)$, der $\hat\phi_s$ er Pearson-dispersjonen til en Gamma-GLM på snittskade (vekt = antall skader) med bred ankermodell. Låses før seleksjon. Ca. 1,74 med alle skader (B-15-utvalget). | Iterativ Pearson-estimering på Tweedie-modellen (eksplisitt likning, statsmodels' `estimate_tweedie_power` er ikke skalainvariant for eksponeringsvekter) ble implementert og testet syntetisk, men ga ingen rot i $(1,2)$ på reelle data: estimeringslikningen er negativ for alle $p$ og styres av storskadene. Profilelikelihood krever eksakt Tweedie-tetthet (ca. 40 linjer, numerisk sårbar). **Limitation:** antar Gamma-skader med konstant CV; estimatet er følsomt for storskader (ca. 1,62 uten de 1 % største); bør vurderes utvidet med profilelikelihood eller sensitivitetsanalyse. Avvik fra opprinnelig plantekst i §3.2. | Låst |
| T-06 | Felles gruppefolder (`build_group_folds`, `GroupKFold(shuffle=True, random_state=100)` på `insured_id`) på hele modellpopulasjonen for frekvens, severity og Tweedie. | Gir sammenlignbare OOF-prediksjoner rad for rad. | Låst |
| T-07 | Kandidatregister låst: `LOCKED` = `policy_type`, `year`; `CORE` = `circulation_area`, `log_vehicle_value` (lineær), `driver_age` (lineær); geografi = `municipality_type` i stedet for eller i tillegg til `circulation_area`; splines: alder df 2/3/4, bilverdi df 3; `ADDITIONAL` = `performance_hp_per_tonne`, `fuel_type`, `payment_frequency`, `business_type`, `vehicle_brand_pooled`, `seat_category` (<5, =5, >5); `INTERACTIONS` = alder × `performance_hp_per_tonne`, alder × `policy_type`, alder × `log_vehicle_value` (lineær alder). | Godkjent av bruker 2026-09-19. Bred ankermodell for $p$ = låste + kjerne + geografi + alle tillegg, lineært. | Låst |
| T-08 | Sperre på 2024 gjelder fortsatt. | Krever eksplisitt godkjenning etter at alle modellspesifikasjoner er låst. | Låst |
| T-09 | Forover-seleksjon tillater flere tillegg: `run_stepwise` legger til den beste kandidaten som består treleddsregelen og gjentar til ingen består; ablasjon kjører på alle termer unntatt `LOCKED`. Interaksjoner testes bare når begge hovedeffektene er med; en hovedeffekt kan ikke fjernes mens en interaksjon bruker den. Treleddsregelen er uendret. | Forrige variant tillot bare ett tillegg og var for streng. Algoritmen er liten (gjenbruker `select_candidate_stage`). Restrisiko: seleksjonsoptimisme fra mange sammenligninger på samme folder (dempes av 4/5-regelen); 2024 er den uavhengige kontrollen. Portert til frekvens og severity 2026-09-19 (se T-13); planene er skrevet om og logget. | Låst |
| T-10 | Firleddsregel: `select_candidate_stage(..., se_multiplier=1.0)` krever i tillegg at snittet av fold-gevinstene (forelder minus kandidat, deviance per fold) er større enn én standardfeil, $\bar d > s_d/\sqrt{5}$. Gjelder alle trinn i Tweedie-notebooken, også ablasjon. Parameteren er valgfri (standard `None`), så frekvens og severity er uendret. | 4/5-regelen alene slipper inn støy: en ren støyterm får gevinst i minst 4 av 5 folder med ca. 19 % sannsynlighet hvis foldene var uavhengige, og mange kandidater sammenlignes på samme folder. **Limitation:** grov støyfilter, ikke en formell test (fem folder gir usikkert $s_d$; $t>1$ tilsvarer ca. 19 % ensidig $p$). I ablasjonen betyr regelen at en term bare fjernes når fjerningen forbedrer CV-devianen med mer enn én SE, så ablasjonen fjerner sjelden noe. | Låst |
| T-11 | Diagnostikk av valgt modell (notebook §5–§6): CatBoost-residualdiagnostikk (`cross_validate_residual_catboost`, uendret) og OOF deviance-residualer mot predikert verdi og kontinuerlige prediktorer med LOWESS (statsmodels `resid_dev`, eksponeringsvekt). Rootogram er utelatt. | Rootogrammet teller heltallige skadeantall; Tweedie-responsen er kontinuerlig med punktmasse i null. Kun utviklingsdata; ingen seleksjonsport. 2024 er urørt. | Låst |
| T-12 | Rapportering av valget (notebook §4.8–§4.10): tabell over valgte forover-steg med OOF-deviance, snittgevinst, standardfeil og gevinst/SE; tabell over de tre modellene med lavest pooled OOF-deviance blant alle evaluerte (endring mot valgt, antall parametere, delta); statsmodels-`summary()` for den valgte modellen estimert på alle utviklingsdata med cluster-robuste SE på `insured_id`. Tabellene er beslutningsgrunnlag og endrer ikke seleksjonsregelen. | Gjør det synlig hvilke variabler som tilfører tydelig verdi (gevinst i SE) og om en enklere modell er praktisk talt like god, slik at endelig modellvalg kan begrunnes med kompleksitet. **Limitation:** toppmodellene er rangert på samme folder som seleksjonen og er dermed en utviklingsscore. `run_round` merker nå valgt kandidat i kolonnen `valgt` (påvirker bare Tweedie-notebooken); `summarize_selection_path` er fjernet fordi tabellen i §4.8 erstatter den. | Låst |
| T-13 | Seleksjonsalgoritmen (firleddsregel, parallell evaluering, forover-/bakoverstegene, rapporteringstabellene) er flyttet til felles kode i `src_core_glm/` (`model_selection.py`: `run_round`, `run_stepwise`, `propose_*`, `build_specification`, `evaluate_models_parallel`; `selection_report.py`: forovertabell og topp-3) og brukes av Tweedie-, frekvens- og severity-notebookene. `assert_model_definition` ligger i `src_asserts/common_glm_asserts.py`. Tweedie-notebooken gir identisk utvalg som før (verifisert: samme valgte modell og topp-3). **Åpent punkt:** interaksjonen alder × `policy_type` med spline-alder er rangdefekt i én fold i frekvens (13 av 14 kolonner) og markeres ugyldig; dette er en kodingsdetalj i `interaction_factor`, ikke en modellkonklusjon. Foreslått rettelse: kode interaksjonen slik at referansenivået får én kolonne. Krever ny kjøring av Tweedie og egen godkjenning. | Én implementasjon, tre notebooks. | Låst |

## Endringslogg

| Dato | Fase | Endring og begrunnelse |
|---|---|---|
| 2026-09-19 | Plan før implementering | Låst rate-respons med eksponeringsvekt og uten offset, Pearson-estimering av $p$ uten nested CV, felles fullpopulasjonsfolder, korrigert severity-OOF-kontrakt og separat assert-arkitektur. Reelt Tweedie-kandidatsøk utsettes til bred ankermodell og kandidatregister er godkjent. |
| 2026-09-19 | Implementering | Implementert `build_group_folds`, `prediction_data` i `cross_validate_glm` (bakoverkompatibel), `src_asserts/`, `src_tweedie/` og ren Tweedie-notebook som stopper før kandidatsøk. **Avvik:** Pearson-likningen løses eksplisitt (T-05) fordi `GLM.estimate_tweedie_power` ikke er skalainvariant for eksponeringsvekter; metoden er ellers uendret. Severity-foldene bygges nå på hele modellpopulasjonen (se `plans/Severity_plan.md`). Ingen Tweedie-modell er fittet. |
| 2026-09-19 | CatBoost-diagnostikk klargjort | `cross_validate_residual_catboost` (`src_core_glm/residual_diagnostics.py`) er testet syntetisk for Tweedie ($1<p<2$, vekt $w\mu^{2-p}$) i `src_asserts/residual_diagnostics_asserts.py`, men ikke kjørt i Tweedie-notebooken. Diagnostikken er ikke en seleksjonsport. Når Tweedie-modellen er låst kalles den med `final_tweedie_specification = selection_specifications[selected_variable_model]`, `model_frame`, `cv_folds`, `blocked_feature_columns=("property_incurred", "pure_premium")` og `fit_kwargs` fra fasen. |
| 2026-09-19 | Kandidatregister, forover-seleksjon og $p$ | Kandidatregisteret er låst (T-07) og forover-seleksjon med flere tillegg, interaksjoner og ablasjon er implementert i notebooken (T-09). **Avvik:** Pearson-estimering av $p$ ga ingen gyldig rot på reelle data og er erstattet av severity-implisert $p$ (T-05), med limitation dokumentert i notebooken; Pearson-koden er fjernet. §3.2, §9, §10 og §14 skrevet om. Frekvens og severity er uendret; portering av algoritmen er planlagt senere og krever egen godkjenning. |
| 2026-09-19 | Firleddsregel og diagnostikk | Lagt til 1-SE-krav (T-10) i Tweedie-seleksjonen etter brukers ønske om færre variabelendringer, og residualdiagnostikk i notebooken (T-11). `select_candidate_stage` fikk valgfri `se_multiplier`; `plot_oof_residuals_against_fitted` fikk valgfrie akse- og tittelparametere (standard uendret). `run_round`/`run_stepwise` tar nå `evaluate_models(navn_liste)` i stedet for `evaluate_model(navn)`; notebooken kjører kandidatene parallelt med joblib (`N_JOBS = 8`), og utvalget er identisk med sekvensiell kjøring (verifisert). Frekvens og severity er uendret. |
| 2026-09-19 | Rapportering av valget | Lagt til T-12: forover-seleksjonstabell med SE (§4.8), topp-3 konkurrerende modeller (§4.9) og statsmodels-sammendrag av valgt modell (§4.10). Ingen endring i seleksjonsregelen eller valgt modell. Frekvens og severity er uendret. |
| 2026-09-19 | Portering til frekvens og severity | Algoritmen og strukturen er innført i `glm_pricing_models` og `glm_pricing_severity` (T-13). Modellagnostisk kode er flyttet til `src_core_glm/`; Tweedie-notebooken er uendret i oppførsel (topp-3-tabellen bruker nå `build_top_models_table`, samme innhold). Frekvens: kjerne inkl. `payment_frequency`; severity: kjerne som før og geografitest kommunetype mot kjøresone. Se planene for frekvens og severity. |
