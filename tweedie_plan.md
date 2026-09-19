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
folder, asserts og power-estimering skal implementeres og verifiseres. Bred
ankermodell og kandidatregister må låses av brukeren før reell modellkjøring.

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

Implementer i stedet iterativ Pearson-estimering:

1. Start med $p=1{,}5$.
2. Fit den senere låste ankermodellen.
3. Estimer $p$ ved å løse Pearson-likningen med `brentq` (`solve_pearson_power` i `src_tweedie/tweedie_power.py`; se T-05 for hvorfor statsmodels' egen funksjon ikke brukes).
4. Refit med estimert $p$.
5. Gjenta til endringen er under en dokumentert toleranse eller maksimal
   iterasjon er nådd.
6. Krev $1<p<2$ og lås $p$ før variabelseleksjonen.

Funksjonaliteten skal nå bare testes syntetisk. Den skal ikke kjøres på et
ulåst reelt kandidatsett.

### 3.3 Senere variabelseleksjon

Når kandidatregisteret er godkjent, skal Tweedie følge samme hovedstruktur
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
CV, Pearson-estimering, felles folder, ulåst kandidatregister og fortsatt
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

`tweedie_power.py` implementerer iterativ Pearson-estimering med eksplisitte
grenser, toleranse og maksimumsiterasjoner. Returner power, antall
iterasjoner, konvergensstatus og en liten iterasjonshistorikk. Funksjonen skal
ikke velge variabler eller tolke resultatet.

Modellspesifikasjon og fitting skal fortsatt være synlig i notebooken.

## 10. Ren Tweedie-notebook

Behold samme enkle hovedstruktur som de andre GLM-notebookene:

1. Datagrunnlag
2. Validering
3. Modellspesifikasjon
4. Sammenligning og valg

Hver celle skal gjøre én tydelig oppgave og normalt være høyst 25 kodelinjer.
Modellformelen skal stå i markdown før fitting. Flytt løkker, tester og
tabellbygging til beskrivende støttefunksjoner.

Fjern power-grid, automatisk finalist og dagens ulåste ablasjon. Behold en
fleksibel `build_tweedie_specification` som senere kan ta råprediktorer,
splineledd, interaksjoner og låst $p$.

Notebooken skal avslutte kontrollert med meldingen:

> Kandidatregister og bred ankermodell er ikke låst. Ingen
> Tweedie-kandidater fittes eller velges i denne versjonen.

Den skal kunne kjøres uten feil, men ikke kjøre reell Pearson-estimering eller
kandidatseleksjon. Ikke opprett tomme resultatseksjoner.

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
- ugyldig $p$, Pearson-konvergens og maksimumsstopp;
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
- Pearson-estimering er implementert og syntetisk testet;
- alle tre modeller bruker samme fullpopulasjonsfolder;
- severity scores riktig delmengde og predikerer hele valideringsfolden;
- `prediction_oof` dekker hele modellpopulasjonen;
- sentrale kontroller ligger i `src_asserts`;
- notebooks er korte, synkroniserte og kjører uten tekniske feil;
- ingen ulåst Tweedie-kandidat er antatt, fittet eller valgt;
- ingen modellresultater er faglig tolket;
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
| T-05 | $p$ estimeres iterativt (start 1,5, refit, toleranse $10^{-3}$, maks 20 iterasjoner) fra Pearson-likningen $\sum_i (w_i r_i^2/(\hat\phi\mu_i^p) - 1)\log\mu_i = 0$, løst med `brentq` på $[1{,}01;\,1{,}99]$. | `GLM.estimate_tweedie_power` (statsmodels 0.15.0) multipliserer «−1»-leddet med vekten, så likningen har forventning $1-w_i\neq0$ og er ikke skalainvariant: med rå eksponeringsvekter fant den ingen rot på syntetiske data med kjent $p=1{,}5$. Den eksplisitte likningen er skalainvariant. Avvik fra opprinnelig plantekst i §3.2; metoden (Pearson, iterativ, ingen CV) er uendret. | Låst |
| T-06 | Felles gruppefolder (`build_group_folds`, `GroupKFold(shuffle=True, random_state=100)` på `insured_id`) på hele modellpopulasjonen for frekvens, severity og Tweedie. | Gir sammenlignbare OOF-prediksjoner rad for rad. | Låst |
| T-07 | Kandidatregister og bred ankermodell er ikke låst. | Krever brukerens beslutning om baseline-variabelvalg. Ingen Tweedie-kandidat er fittet eller valgt. | Åpen |
| T-08 | Sperre på 2024 gjelder fortsatt. | Krever eksplisitt godkjenning etter at alle modellspesifikasjoner er låst. | Låst |

## Endringslogg

| Dato | Fase | Endring og begrunnelse |
|---|---|---|
| 2026-09-19 | Plan før implementering | Låst rate-respons med eksponeringsvekt og uten offset, Pearson-estimering av $p$ uten nested CV, felles fullpopulasjonsfolder, korrigert severity-OOF-kontrakt og separat assert-arkitektur. Reelt Tweedie-kandidatsøk utsettes til bred ankermodell og kandidatregister er godkjent. |
| 2026-09-19 | Implementering | Implementert `build_group_folds`, `prediction_data` i `cross_validate_glm` (bakoverkompatibel), `src_asserts/`, `src_tweedie/` og ren Tweedie-notebook som stopper før kandidatsøk. **Avvik:** Pearson-likningen løses eksplisitt (T-05) fordi `GLM.estimate_tweedie_power` ikke er skalainvariant for eksponeringsvekter; metoden er ellers uendret. Severity-foldene bygges nå på hele modellpopulasjonen (se `plans/Severity_plan.md`). Ingen Tweedie-modell er fittet. |
| 2026-09-19 | CatBoost-diagnostikk klargjort | `cross_validate_residual_catboost` (`src_core_glm/residual_diagnostics.py`) er testet syntetisk for Tweedie ($1<p<2$, vekt $w\mu^{2-p}$) i `src_asserts/residual_diagnostics_asserts.py`, men ikke kjørt i Tweedie-notebooken. Diagnostikken er ikke en seleksjonsport. Når Tweedie-modellen er låst kalles den med `final_tweedie_specification = selection_specifications[selected_variable_model]`, `model_frame`, `cv_folds`, `blocked_feature_columns=("property_incurred", "pure_premium")` og `fit_kwargs` fra fasen. |
