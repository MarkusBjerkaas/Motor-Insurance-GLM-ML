# Plan: GLM-benchmark for egen skade – fase 1–4 i `glm_pricing_models`

## Status etter fase 1 og faktarevisjon 2026-09-17

Leveranse 0 er implementert, referansemodellene er kjørt, og **fase 1 (frekvens) er fullført og låst**. Additiv modell `F5_minus-municipality-performance` (13 parametere) er den endelige frekvensmodellen etter B-07-overstyringen i B-30: den datadrevne utfordreren `F6_I1_product-x-age` besto B-08 i gruppe-CV, men tapte i tidsfolden 2022→2023 og ble derfor forkastet. NB2 slår ikke Poisson (B-23). En separat, avgrenset ABESS-diagnostikk (2026-09-17) er kjørt i tillegg og endrer ikke denne konklusjonen; se «Separat ABESS-diagnostikk etter fase 1» nedenfor. Leveranse 2–4 er fortsatt utkast og revurderes ved faseovergang — **denne revisjonen retter kun faktagrunnlaget i eksisterende tekst og skriver ikke fase 2-planen**.

Brukeren har **avklart og låst** tre beslutninger: prosjektet fortsetter som metodebenchmark med timingforbehold (B-02/B-27); føreralder brukes i hovedstigen og erfaring bare som sensitivitet (B-11); de to avgrensede interaksjonene kan kvalifisere til benchmarken (B-08/B-28), og i praksis kvalifiserte ingen etter tidsfoldkontrollen (B-30). Beslutningsregisteret og seksjon 2.8/2.10 i `glm_pricing_models.py` er samordnet med disse valgene. Det gjenstår ingen åpne brukerbeslutninger for fase 1. Neste steg er å revurdere og skrive fase 2 (severity)-planen i en egen omgang.

## Kontekst

Den deskriptive analysen i `analysis.py` er ferdig. Seksjon 23 foreslår et første GLM-løp. Neste steg er en egen modelleringsnotebook som:
1. begrunner metrikkvalget ut fra dette datasettet og litteraturen,
2. gjennomfører fase 1 (frekvens), fase 2 (severity og storskader), fase 3 (todelt modell mot Tweedie) og fase 4 (konsolidert benchmark),
3. dokumenterer beslutningene løpende og låser styrende valg før kandidatmodellene estimeres.

Resultatet er en fryst, transparent GLM-benchmark som senere ML-modeller skal slå.

**Rammer fra brukeren**
- **Planen er levende.** Den lagres i `plans/glm_pricing_models_plan.md` i repoet og skal **revurderes, og om nødvendig skrives om, ved starten av hver ny fase**, basert på resultatene fra fasen før. Se «Revurdering mellom faser» nedenfor. Alt fra leveranse 2 og utover er et utkast som med stor sannsynlighet endres.
- All kode for modellspesifikasjon og CV-definisjon skal ligge i notebooken.
- Diagnostikk, tabeller og plott kan legges i `src/` og importeres.
- Datagrunnlaget flyttes til et script og dokumenteres godt i begge notebooks.
- Implementeringen skjer leveranse for leveranse, med stopp for gjennomgang og commit etter hver.
- `analysis.py`/`.ipynb` redigeres bare via `.py` og `jupytext --sync`.
- All tekst er på norsk, variabelnavn på engelsk.
- Commit-meldinger har prefikset `checkpoint:` og ingen AI-attribusjon. Commit gjøres først etter brukerens bekreftelse, i tråd med prosjektets CLAUDE.md.

## Datafunn fra planleggingen (train_pool 2022–2023) som styrer planen

| Funn | Tall | Konsekvens |
|---|---|---|
| Kansellerte poliser (C) | 13 % av skadene og 13 % av kostnaden på 5 % av eksponeringen. Frekvens 0,71 mot 0,24. Andel med skade 17,9 % mot 9,8 %. | Beholdes (B-01). Uten dem faller nivået med ca. 9 %, og relativitetene for bonus N (27 % av skadene fra C), kvartalsbetaling og P dempes. |
| Offset-proporsjonalitet | Fri log(e)-koeffisient: 1,10 (KI 1,01–1,19) med alle rader, 1,53 uten C | Fast offset brukes. A/E per eksponeringsbøtte er påvirket av utfallet og tolkes forsiktig. |
| Pukkel i skadeantall | N=3: 245, N=4: 418, N=5: 335, mest i COMP_N. Median kostnad per skade faller fra ca. 692 til 497 EUR. | Tellemodellen beholdes, med rootogram og A/E per antall (B-03). Tweedie er en robust kontroll. |
| Motsatt retning for frekvens og severity | COMP_N: frekvens 0,69 / severity 680. COMP_E: 0,18 / 1 020. | Et konkret argument for å teste den todelte modellen mot Tweedie i fase 3. |
| Tyngde i halen | Topp 1 % av skadeårene står for 8,5 % av kostnaden. Kostnad over u på skadeårsnivå: 7,5 % (5 000), 3,8 % (7 500), 2,1 % (10 000). Høyeste skade/bilverdi er 0,92, og bare 8 skader ligger over 0,5. | Moderat og avgrenset hale uten tydelig totalskadeklynge. Storskadeanalysen er beslutningsstøtte, ikke en forhåndsbestemt løsning. |
| Enkeltskader kan identifiseres | 67 % av skadeårene har N=1, men de utgjør bare 38 % av skadene | Terskelen vurderes på skadeår med N=1. Kappingen gjøres på snittskaden. |
| Bonus-balanse | Eksponering G 35 773, N 745, B 608 | Brede intervaller for N og B |
| Manglende verdier i train-poolen | fuel_type 438, vehicle_value/log_vehicle_value 33, age_driving_licence/driving_experience_years 1, performance_hp_per_tonne 0 | Egen strategi (B-09). De 3 ikke-positive rå ytelsesverdiene i renseloggen gjelder hele datasettet, ikke denne train-poolen. |
| Teknisk | patsy feiler på pandas `Int16` og andre nullable dtypes | Designrammen konverteres til float/str før formlene brukes |
| Nullkostnadsskader | 9 skadeår med incurred ≤ 0,01 | Utelates fra severity (B-15) |

## Filer og opprinnelig leveransefordeling

Tabellen beskriver opprinnelig etablering. `src/model_data.py`, `src/glm_diagnostics.py` og modellnotebooken finnes nå. Fase 1 skal bygge videre på dem; ikke gjenta leveranse 0 eller overskrive eksisterende brukerendringer.

| Fil | Endring |
|---|---|
| `src/model_data.py` (ny) | `select_own_damage_scope(data)`, `split_by_year(frame, train_years, test_year)`, `learn_retained_brands(train_pool, min_exposure=500.0)` som returnerer `(retained_brands, brand_exposure)`, `add_transparent_predictors(frame, retained_brands)` (flyttet uendret fra `analysis.py:610`) og `build_model_frames(...)` som setter sammen innlasting → `clean_motor_data` → avgrensning → splitt → transformasjoner. Grundige docstrings. |
| `analysis.py` | Seksjon 11 bruker `select_own_damage_scope`. Seksjon 12 bruker `split_by_year`; `GroupKFold`-sjekken beholdes, og det legges til en merknad om at modell-CV defineres i `glm_pricing_models`. Seksjon 17 importerer `learn_retained_brands` og `add_transparent_predictors`. Asserts og `feature_summary` beholdes. Markdown forklarer hva scriptet gjør og hvorfor. **Output skal være uendret.** |
| `glm_pricing_models.py` / `.ipynb` (ny, jupytext percent) | Hele modelleringsløpet: modellspesifikasjoner, CV-sløyfer, manglende-strategi og beslutningsregister |
| `src/glm_diagnostics.py` (ny) | Tabeller og plott: A/E per desil og per nivå, rootogram, Lorenz/Gini, double lift, koeffisientstabilitet over folder, relativitetstabell og -plott, spline-effektplott, deviance-dekomponering (MCB/DSC/UNC via `IsotonicRegression`), formatering av CV-resultattabell. Gjenbruker fargekonstantene fra `src/own_damage_descriptives.py`. |
| `src/large_claims_diagnostics.py` (ny) | Halekonsentrasjon, andel over terskel u, mean excess-plott og fordeling av skade/bilverdi |
| `.claude/settings.json` | `UserPromptSubmit`-hooken utvides til også å synke `glm_pricing_models.py` fra `.ipynb` |
| `CLAUDE.md` | Filstruktur og notebook-workflow nevner den nye notebooken og `plans/`-mappen, med regelen om at planen revurderes ved hver faseovergang |
| `plans/glm_pricing_models_plan.md` (ny) | Denne planen, kopiert inn i repoet. Den er styringsdokumentet og har en endringslogg nederst. Notebookens seksjon 0 lenker til den. |

Gjenbruk: `clean_motor_data`, `build_variable_dictionary` (`src/data_quality.py`); `build_bonus_lagged_panel` (`src/own_damage_descriptives.py`) til lagget historikk i bonussensitiviteten; konstantene `CLAIMS_COL`, `INCURRED_COL`, `EXPOSURE_COL` og `CURRENCY_ZERO_TOLERANCE`.

`dataviz`-skillen lastes før plottkoden skrives.

## Metrikkanalyse (notebookens seksjon 2)

**Prinsipp.** En tariff skal estimere forventningsverdien E[Y|x]. En metrikk som skal velge mellom modeller må derfor være *strengt konsistent for middelverdien*, altså en Bregman-divergens. Deviance-familien er det (Gneiting 2011; Wüthrich & Merz 2023, kap. 4; Fissler, Lorentzen & Mayer 2023). Alle kandidater for samme størrelse vurderes med **samme** scoringsfunksjon på prediksjonen av middelverdien, uavhengig av hvilken likelihood modellen er estimert med. Det gjelder også NB mot Poisson og lognormal mot Gamma.

**Primærmetrikker, out-of-fold:**
- **Frekvens:** eksponeringsvektet Poisson-deviance på raten N/e (`mean_poisson_deviance`, vekt e) og D² mot nullmodellen.
  - *Datagrunn:* 89,7 % nuller, bare en tredel av radene har fullårseksponering, og deviance er konsistent for middelverdien også ved overspredning (Noll, Salzmann & Wüthrich 2018).
- **Severity:** Gamma-deviance vektet med skadeantall på snittskaden.
  - *Datagrunn:* positiv og høyreskjev respons. Relativ feil passer en multiplikativ tariff (Ohlsson & Johansson 2010).
  - *Begrensning:* deviance er skalafri, så store skader veier lite. Metrikken suppleres derfor med A/E i EUR.
- **Ren premie:** eksponeringsvektet Tweedie-deviance ved p̂. Rangeringen kan avhenge av p, så resultatene rapporteres også for p ∈ {1,2; 1,5; 1,8} og p=1. En konklusjon regnes som robust bare når rangeringen er den samme på tvers av p (Delong, Lindholm & Wüthrich 2021; Fissler et al. 2023).

**Sekundærmetrikker:**
- **Kalibrering:** global balanse Σŷ/Σy, A/E per prediksjonsdesil og per nivå av `policy_type`, år, `business_type` og bonus (Goldburd et al. 2020; autokalibrering, Denuit, Charpentier & Trufin 2021).
- **Diskriminering:** Gini fra ordnet Lorenz-kurve (Frees, Meyers & Cummings 2011). Brukes bare sekundært, fordi Gini ikke er en konsistent score og bare er gyldig mellom autokalibrerte modeller (Wüthrich 2023). I fase 3 brukes også double lift.
- **Dekomponering:** mean deviance = MCB − DSC + UNC, via isotonisk regresjon (Fissler et al. 2023), i fase 3 og 4.
- **Fordeling:** Pearson φ̂, Cameron–Trivedi-test (1990) og rootogram (Kleiber & Zeileis 2016).
- **Inferens og stabilitet:** cluster-robuste standardfeil på `insured_id` (Cameron & Miller 2015) og koeffisientspenn over foldene.

**Forkastet, med datagrunn:**
- MAE estimerer medianen, og medianen er 0.
- RMSE domineres av ekstreme årsrater, som 1,19 mill. EUR fra korte eksponeringer.
- R² har ingen klar tolkning for data med så mange nuller.
- AIC/BIC er bare kontroll ved sammenlignbare likelihoods på samme respons og rader; nøsting er ikke nødvendig. De brukes ikke til variabelvalg eller sammenligning av antalls- og rate-likelihood.

**CV-design (definert i notebooken):**
- `GroupKFold(n_splits=5, shuffle=True, random_state=100)` på `insured_id`. Uten shuffle fordeles like store grupper etter id-rekkefølge, og id-rekkefølgen kan henge sammen med tegningstidspunkt (Roberts et al. 2017).
- Tidsfold 2022 → 2023 som obligatorisk robusthetskontroll, ikke som et nytt optimaliseringssett.
- Imputasjonsmedianer, spline-knuter (patsy stateful `cr()`), NB-α, Tweedie-p og storskadetillegget λ læres **inne i treningsfolden**. B-14s eksponeringsbaserte merkeliste og kategorienes referansenivåer er eksplisitte unntak i gruppe-CV; de bruker ingen respons. I tidsfolden læres eventuell merkeliste fra 2022 alene.
- Fold-assert om disjunkte grupper.

**Seleksjonsregel (presisert i fase 1 nedenfor):**
- En mer kompleks utfordrer må gi positiv pooled OOF-gevinst, parvis gjennomsnittsgevinst > 1 SE og forbedring i minst 4 av 5 folder, i tillegg til stabilitet. Dette er prosjektets konservative heuristikk inspirert av 1-SE-regelen, ikke en signifikanstest eller dokumentasjon på fravær av seleksjonsoptimisme.
- En forhåndsdefinert kjerne estimeres samlet. Funksjonsform undersøkes før kontinuerlige blokker forkastes. Ablasjon viser bidrag betinget på resten; flere svake blokker slettes ikke samtidig uten kontroll av den samlede reduksjonen.
- Lineære ledd vurderes på fortegn, kategorier på støttede relativiteter og splines på kurver i sentrale 95 % av eksponeringen. Enklere modell foretrekkes innen parvis 1 SE. Detaljert sammenligningsretning og tie-break står i 3.4–3.6.
- p-verdier, AIC/BIC og one-way-rater velger ikke variabler. Bare de to avgrensede interaksjonene i 3.7 er tillatt.
- Foldene deler treningsdata, og de samme foldene brukes til flere valg. Standardfeilen er derfor grov, og vinnerens OOF-score er en utviklingsscore. 2024 forblir lukket.

## Revurdering mellom faser (obligatorisk)

Planen for fase 2–4 bygger på antakelser som først testes i fasen før. For eksempel avhenger fase 2 av variabelsettet fra fase 1, fase 3 av storskadebeslutningen fra fase 2, og fase 4 av strukturvalget fra fase 3. Derfor gjelder dette **ved starten av hver ny leveranse, før noen kode skrives**:

1. Les `plans/glm_pricing_models_plan.md` og resultatene og beslutningene fra fasen før i notebooken og beslutningsregisteret.
2. Vurder planen for neste fase:
   - Holder antakelsene?
   - Har resultatene gjort steg overflødige? For eksempel kan en lett hale gjøre deler av storskadeanalysen unødvendig.
   - Har det dukket opp nye problemer som krever nye steg?
   - Må metrikker, terskler eller beslutningsregler endres?
3. Skriv planen om ved behov, og legg til en rad i **Endringslogg** (dato, fase, hva som ble endret og hvorfor). Er det ingen endringer, logges det også.
4. Legg frem de vesentlige endringene for brukeren og få bekreftelse før implementeringen starter. Større problemer løftes som spørsmål.
5. Oppdater markdown-teksten i notebookens seksjon 0 hvis omfanget har endret seg.

## Notebookstruktur og implementering per leveranse

### Leveranse 0 – Grunnmur (stopp etter)
1. Før første endring: spør om de eksisterende ucommittede endringene (`analysis.*`, `CLAUDE.md`, `settings.json`) skal committes separat.
2. Opprett `plans/glm_pricing_models_plan.md` med innholdet i denne planen og en tom endringslogg.
3. Lag `src/model_data.py` og refaktorer `analysis.py` seksjon 11, 12 og 17. Kjør `analysis.ipynb` og bekreft at tallene er uendret: 56 084 rader, 37 904,44 eksponering, 10 126 skader, 17 merker og 88,5 % av eksponeringen.
4. Utvid hooken og oppdater `CLAUDE.md`.
5. Lag `glm_pricing_models.py` med disse seksjonene:
   - **Seksjon 0 – Formål og omfang.** Kun GLM, egen skade og train_pool. 2024 åpnes ikke (B-05). Beslutningsregisteret forklares.
   - **Seksjon 1 – Datagrunnlag.** `build_model_frames` kalles, og markdown dokumenterer hva scriptet gjør. Asserts mot tallene fra analysis. Bare `train_pool` beholdes; `test` slettes eksplisitt. Designrammen konverteres fra nullable dtypes.
   - **Seksjon 2 – Evalueringsrammeverk.** Metrikkanalysen over i markdown. CV-folder, tidsfold og en generisk `cross_validate_glm(spec, frame, folds)`-sløyfe i notebooken som returnerer OOF-prediksjoner og fold-scorer. Manglende-strategien `prepare_design_frame(train, apply)` defineres her (B-09). Nullmodell og `policy_type`-modell tas med som referanse.
   - **Seksjon 7 – Beslutningsregister**, som markdown-tabell.

> Leveranse 1–4 starter hver med **Revurdering mellom faser**. Fase 1 nedenfor er revidert mot EDA, kildebeskrivelsen og den implementerte grunnmuren.

### Leveranse 1 – Fase 1: frekvens (seksjon 3, stopp etter)

#### 3.0 Avklar informasjonstidspunkt og lås kandidatrommet

**Nytt kildefunn.** Artikkelens avsnitt om dataprosessering sier at kontraktsendringer ga flere registreringer, og at bare siste registrering per polise/år ble beholdt. Opplysninger som vanligvis er kjent ved tegning, er derfor ikke nødvendigvis registrert med tegningsverdien i dette uttrekket. Det gjelder også produktet i basismodellen. Dette dokumenterer en tidsrisiko, ikke at alle feltene faktisk lekker skadeutfall. [Datakilden](https://pmc.ncbi.nlm.nih.gov/articles/PMC13234478/)

Lag en liten tabell i notebooken med `variable`, `available_at_pricing`, `evidence`, `timing_status` og `allowed_use`. Skill mellom dokumentert startverdi, plausibel men uverifisert startverdi og utelukket felt. Ikke sett første status ut fra normal forsikringspraksis alene.

- **Brukerbesluttet B-27:** Fortsett som en eksplisitt metodebenchmark på registrerte årsopplysninger, med uavklart timing. Den kan vise modellhåndverk, men skal ikke omtales som dokumentert prediksjonsytelse ved nytegning/fornyelse eller som garantert lekkasjefri. Plausible, men uverifiserte kontrakts-/risikofelt kan inngå under denne avgrensningen; det åpner ikke for bonus, status, premier eller samtidige utfall som prediktorer.
- En **prospektiv tariffbenchmark** krever dokumentasjon på tilgjengelighet ved prisingsdato: historiserte startverdier eller troverdig kildebekreftelse for de aktuelle feltene. Gruppe-CV, tidsfold og sterk prediksjon erstatter ikke slik dokumentasjon.
- **Bonusporten er ikke bestått.** `bonus_score_analysis.py`, seksjon 2 og 5, viser et mønster forenlig med lagget informasjon, men ingen presis as-of-dato. Hovedstigen begynner uten bonus. Eventuell senere dokumentasjon må inn i B-13 før bonus vurderes, ikke etter at bonusmodellens score er sett.
- **Brukerbesluttet B-11:** Føreralder er eneste førervariabel i hovedstigen. Erfaring testes separat som sensitivitet og kan ikke velges inn i hovedbenchmarken uten ny beslutning og avklart definisjon. De brukes aldri samtidig.
- **Brukerbesluttet B-28:** De to avgrensede interaksjonene i 3.7 kan inngå etter at den additive modellen er bestemt, hvis støtte- og seleksjonskravene bestås. Det tidligere generelle interaksjonsforbudet er erstattet av denne begrensede tillatelsen.
- B-01, B-03–B-07, B-12–B-15 og B-20 beholdes: samme populasjon, skadeantall inklusive nuller, fast eksponering, gruppefoldene, tidsfolden og stengt 2024. Ikke filtrer bort kansellerte eller sjeldne/høye skadeantall fra hovedløpet.
- Premieflagget og `total_exposure` gir ikke eksakt dekningshistorikk gjennom året. Ukjent egenandelsbeløp, endringer i dekning og rapportering av skader er begrensninger i hva produktrelativitetene betyr. Modellen gjelder **registrert egen-skadefrekvens under observert dekning**, ikke underliggende ulykkesfrekvens uavhengig av egenandel.

**Hva låses nå?** Kandidatvariabler, transformasjoner, alternative df, avgrensede interaksjoner, foldene, score og stoppregler. **Hva avgjøres etter fitting?** Hvilke av kandidatene som bidrar betinget på resten, splineform, Poisson mot NB2 og endelig formel. Residualplott kan forklare utfallet, men skal ikke brukes til å åpne et ubegrenset søk etter nye knuter eller interaksjoner. Nye hypoteser krever en datert planendring og merkes som nye utviklingsforsøk.

#### 3.1 Evidens og forhåndsdefinerte variabelblokker

Evidensen under kommer fra `analysis.py` seksjon 12.1 og 16–23, lagrede outputs i `analysis.ipynb`, `bonus_score_analysis.py` og `src/model_data.py`/`src/data_quality.py`. Rå frekvenser er kontrollert med eksisterende one-way-funksjoner på bare train-poolen. De er hypotesestøtte, ikke multivariate effekter eller løfter om fortegn.

| Blokk / variabler | Train-funn og faglig begrunnelse | Rolle og koding |
|---|---|---|
| Produkt: `policy_type` | Frekvens COMP_E 0,183; COMP_N 0,692. Egenandel og dekning påvirker hvilke skader som meldes. CC er fjernet fra populasjonen (B-29). | Obligatorisk kategorisk kontroll, COMP_E som basis. |
| Kalender: `year` | Frekvens 0,235 i 2022 mot 0,285 i 2023 (gjeldende populasjon, jf. notebookens 3.0). Referansemodellen (uten årsledd, trent på 2022) underpredikerer 2023-frekvensen med ca. 18 % i tidsfolden (balanse 0,82). | Obligatorisk kategorisk kontroll i gruppe-CV, basis 2023. Utelates i tidsfolden. Ingen lineær trend eller offset. |
| Fører: `driver_age` | Frekvens i seks kvantilbøtter: 0,267; 0,242; 0,255; 0,251; 0,277; 0,312. Aldersspennet er 18–85, p1/p99 omtrent 27/73. | Kjerne. Lineær, sentrert naturlig spline df=3 eller 4. Ingen antatt monoton effekt; liten støtte for de yngste/eldste. |
| Verdi: `log_vehicle_value` | Frekvens fra 0,217 i laveste til 0,350 i høyeste kvantilbøtte; ikke helt monoton mellom disse. Signal finnes også i frekvens, ikke bare severity. | Kjerne. Naturlig log av verdi; lineær effekt på logskala mot spline df=3/4 på samme skala. Rå verdi legges ikke til samtidig. |
| Ytelse: `performance_hp_per_tonne` | Frekvens omtrent 0,236 til 0,329 fra laveste til høyeste bøtte. Korrelasjon med logverdi 0,564. p99 omtrent 150,4, maksimum 318,5 hk/tonn. | Kjerne. `1000 / power_to_weight_ratio`, fordi kilden angir kg/hk. Lineær mot spline df=3/4; ikke ta inn rå forholdstall samtidig. |
| Drivstoff: `fuel_type` | Diesel 0,281 mot bensin 0,227; kan reflektere bruk/kjørelengde og bilmiks. | Egen kategorisk kjerneblokk, inklusive MISSING. Ingen tolkning som kausal drivstoffeffekt. |
| Bruksmiljø: `circulation_area` | Urban 0,298 mot rural 0,220; trafikkmiljø er en plausibel risikofaktor. | Egen kategorisk kjerneblokk. Må ikke forveksles med kommunetypen. |
| Kommune: `municipality_type` | Innland/kyst/øyer 0,280/0,241/0,216. Øyer har omtrent 1 406 eksponeringsår og 303 skader. Samvariasjon med bruksmiljø er beskjeden, Cramér's V 0,108. | Egen kategorisk kjerneblokk. Begge geografifeltene kan inngå; i tillegg vises en felles geografiablasjon. |
| Betaling: `payment_frequency` | Årlig/halvårlig/kvartal 0,260/0,272/0,372. Kvartal har 1 845 eksponeringsår og 687 skader. | Egen kategorisk kjerneblokk. Kontrakts-/seleksjonsfaktor; kontroller følsomhet for kansellering og tidspunkt for avtalen. |
| Forretning: `business_type` | NB/P 0,240/0,338. Kan reflektere seleksjon, historikk og eksponeringsmønster. | Egen kategorisk kjerneblokk. P er eksisterende portefølje, ikke næringsbruk av kjøretøyet. |
| Merke: `vehicle_brand_pooled` | 17 beholdte merker dekker 88,5 % av eksponeringen; merkefrekvenser omtrent 0,183–0,342. Korrelerer med bilens øvrige egenskaper. | Sekundær blokk med 17 merker + OTHER, omtrent 17 ekstra parametere. B-14: ≥500 eksponeringsår på hele train-poolen. Ikke response-basert pooling. |
| Seter: `seats_group` | 83 % av eksponeringen har 5 seter, 9,8 % har 7; 9 seter har bare 94 eksponeringsår. Svak rå separasjon. | Sekundær blokk, deterministisk `<5`, `5`, `>5`, basis `5`. Ingen lineær avstandseffekt, spline eller separat koeffisient for hvert sjeldent seteantall. |
| Alternativ fører: `driving_experience_years` | Korrelasjon med alder 0,879. Definisjonen er en arbeidshypotese. Førerkortalder faller i 14,44 % av årsovergangene; 3 rader har verdi <14. | Bare sensitivitet (B-11). Må aldri kombineres med alder eller rå førerkortalder. Uklar semantikk repareres ikke av bedre CV. |
| Bonus: `bonus_score` | G/N/B har frekvens 0,260/0,396/0,526, men N/B har bare 763/616 eksponeringsår. | Utelatt fra hovedstigen så lenge as-of-porten er lukket. Eventuell sensitivitet holdes utenfor vinnerkåringen. |

Rå kvantilbøtter er laget med `pd.qcut(..., q=6)` i `build_numeric_one_way`; de er **ikke** forhåndsdefinerte tariffkategorier. Korrelasjon 0,564 tilsier kontroll av betinget bidrag, ikke automatisk fjerning av verdi eller ytelse. Ved ablasjon fjernes hele en variabels designblokk, aldri enkeltkolonner i dens spline eller enkelte kategorinivåer.

**Utelukk eksplisitt:** `vehicle_age` (uklar/feil semantikk), rå `age_driving_licence`, `policy_status`, `insured_id`, alle samtidige skade- og kostnadsfelt og alle premier som risikoprediktorer. `insured_id` brukes bare til gruppering/clustering. Eksponering brukes til respons, offset/vekt og diagnostikk, ikke som fritt hovedledd. Ingen target encoding, nye empiriske skadegrupper, kjørelengde, eksakt egenandel eller reell kjøretøyalder skal oppfinnes. Disse siste feltene finnes ikke i modellgrunnlaget.

**Manglende data:** B-09 beholdes, med median fra treningsfolden og MISSING for drivstoff. Fuel mangler på 438 rader, logverdi på 33 og erfaring på 1; ytelse mangler ikke i train-poolen. De 33 manglende verdiradene har høy rå frekvens, men er for få til en ny automatisk missing-indikator. Ingen nye numeriske missing-indikatorer i hovedstigen. Rader skal ikke falle bort mellom kandidater.

#### 3.2 Modell, basis og lineær kjerne

Før kode legges en markdown-celle med modellen og tolkningen:

\[
N_i\mid x_i,e_i\sim\operatorname{Poisson}(m_i),\qquad
m_i=e_i\lambda_i,\qquad
\log\lambda_i=\beta_0+\beta_{p(i)}+\beta_{t(i)}+\sum_j f_j(x_{ij}).
\]

`lambda` er frekvens per eksponeringsår, `m` er forventet skadeantall. I Poisson-koden brukes `claim_frequency = property_claims / total_exposure` og `var_weights=total_exposure`, uten ekstra offset. Dette er ekvivalent med antall + log(e)-offset for koeffisienter og deviance. Det forutsetter proporsjonalitet i eksponering; dette undersøkes som sensitivitet, ikke ved automatisk frislipp av offset.

| ID | Kandidat | Formål |
|---|---|---|
| F0 | `policy_type + C(year)` | Gjenbruk allerede estimerte referansespesifikasjoner og gruppefolder når innholdet er identisk. Ingen ny konstantmodell trengs. |
| F1 | F0 + alder + logverdi + ytelse + drivstoff + begge geografifelt + betalingsfrekvens + forretningstype; alle kontinuerlige ledd lineære | Samlet forhåndsdefinert kjerne. Ingen forward-screening av de enkelte variablene. |
| F2 | Samme kjerne og populasjon, med det avgrensede formrutenettet i 3.3 | Fanger mulige ikke-lineære effekter før endelig variabelreduksjon. |
| F3 | F2 med én blokk fjernet om gangen; dessuten to felles ablasjoner | Viser betingede bidrag og overlapp, uten å slette flere svake blokker på én gang. |
| F4 | F2 med ingen / merke / seter / begge sekundære blokker | Tester om de sekundære blokkene gir betinget informasjon og om de konkurrerer med hverandre. |
| F5 | Kontrollert reduksjon av den valgte additive F4-modellen | Fryser en parsimonisk additiv kandidat med beskyttelse mot samlet informasjonstap. |
| F6 | To støtteavhengige interaksjonstester og eventuelt kombinasjonen (B-28) | Begrensede utfordrere til den additive modellen. |
| F7 | NB2 med nøyaktig samme middelverdistruktur som valgt Poisson | Tester fordelingsvalg etter at middelverdistrukturen er bestemt. |

F1 er et utgangspunkt for justering, ikke en påstand om at alle variablene må beholdes. Produkt og år er obligatoriske kontroller og ablateres ikke for å velge hovedmodell. Dersom ingen utvidelse gir robust gevinst mot F0, beholdes F0 som gyldig enklere kandidat.

Den lineære kjernens prediktorliste i eksisterende `glm_spec` skal være:

```python
core_predictors = [
    "policy_type", "year", "driver_age", "log_vehicle_value",
    "performance_hp_per_tonne", "fuel_type", "circulation_area",
    "municipality_type", "payment_frequency", "business_type",
]
```

`glm_spec` gjør kategorikodingen; ikke behandle tallkodet år som et kontinuerlig ledd. Alle sammenligninger bruker de eksisterende fem gruppefoldene og samme rader/eksponering.

#### 3.3 Funksjonsform: bestem kandidatene nå, la CV velge

For alder, logverdi og ytelse brukes bare følgende former:

```python
feature_forms = {
    feature: [
        feature,
        f"cr({feature}, df=3, constraints='center')",
        f"cr({feature}, df=4, constraints='center')",
    ]
    for feature in [
        "driver_age", "log_vehicle_value", "performance_hp_per_tonne"
    ]
}
```

1. Estimer det faste produktet av formvalgene: **3³ = 27 kjernekandidater**, inklusive F1. De kategoriske blokkene holdes like. Dette er et lite, uttømmende søk over tre formvalg; ingen variabelsubsett eller interaksjoner søkes samtidig. Det unngår at valgt alderkurve bestemmes av rekkefølgen på verdi-/ytelsestestene.
2. Vis først tre lettleste deltabeller der bare ett ledd endres fra F1, og deretter den samlede formtabellen. På denne måten blir hver enkelt fleksibilitetsgevinst synlig, også når vinneren kombinerer flere endringer.
3. Bruk regelen i 3.4 til å velge en stabil, enklere form innen 1 SE. Formrutenettet låser ikke at splines skal vinne. Ikke velg knuter eller klippegrenser etter skader, effektplott eller valideringsresultater.
4. Erfaring får ikke en parallell hovedmodellgren. De tre erfaringstestene kjøres bare som sensitivitet i 3.10.
5. Etter at sekundære blokker og reduksjon er bestemt, gjøres **én** avsluttende formsjekk: med variabelsettet låst testes de samme formene for de gjenværende kontinuerlige variablene, maksimalt 27 kandidater. Ingen nye variabler tas inn. Gjenbruk identiske kandidater. Bruk 3.4 med gjeldende F5 som anker: økt kompleksitet krever B-08, forenkling kan velges innen parvis 1 SE. Oppdatert modell må fortsatt ligge innen parvis 1 SE av den faste F4-fullmodellen fra 3.6; ellers beholdes modellen før denne formsjekken. Deretter låses formene, uten ny søkesyklus.

**Identifiserbarhet og foldvis læring:** Bruk `constraints='center'` sammen med intercept. `df=3/4` betyr henholdsvis 3/4 splinekolonner i tillegg til intercept, ikke antall knuter. Kontroller full rang. Patsy lærer knuter, grenser og sentrering i treningsfolden; prediksjon skal bruke samme `design_info`. Ikke bygg en ny basis på valideringsdata. [Patsys splinedokumentasjon](https://patsy.readthedocs.io/en/latest/spline-regression.html)

Splines er naturlige kubiske regresjonssplines med lineære haler på linkskalaen; dette begrenser ikke hvor høye rater log-linken kan gi utenfor god datastøtte. Vis støtte og haleprediksjoner, særlig for alder og ytelse. Ingen responsstyrt winsorisering. Knuteplasseringen følger Patsys ordinære kvantiler i treningsradene; eksponeringskvantiler brukes til diagnostisk grid, ikke som en skjult alternativ knuteregel.

#### 3.4 Operativ sammenligningsregel og stabilitet

For baseline A og kandidat B i de samme fem foldene: `gain_k = D_A,k - D_B,k`. Positivt betyr at B er bedre. Rapporter pooled gain vektet med foldenes valideringseksponering, uvektet `mean_gain`, `se_gain = std(gain_k, ddof=1) / sqrt(5)` og antall `gain_k > 0`. `paired_improvement` beregner dette allerede, men inneholder ikke stabilitetskontrollen.

- **Robust oppgradering (B-08):** pooled gain > 0, mean gain > se gain, minst 4/5 positive folder og ingen uløst stabilitets-/kalibreringssvikt. Ikke sammenlign kandidatens egen score-SE med baseline; usikkerheten skal gjelde den parvise forskjellen.
- **Forenkling innen 1 SE:** For en enklere S mot rikere R defineres tapet `loss_k = D_S,k - D_R,k`. S ligger innen 1 SE hvis gjennomsnittlig tap ≤ parvis SE. Negativt tap er en forbedring. Rapporter pooled tap i tillegg; ikke fremstille grov 1-SE-likeverd som statistisk ekvivalens.
- **Valg i et fast kandidatsett (former/F4):** Finn laveste pooled deviance blant numerisk gyldige kandidater uten uløst materiell ustabilitet. Finn så kandidatene innen parvis 1 SE av denne. Velg færrest estimerbare parametere; ved likt antall foretrekkes lavere df i rekkefølgen fører → verdi → ytelse, deretter alfabetisk kandidat-ID. En valgt utvidelse må i tillegg slå settets enkleste anker etter B-08; hvis ikke beholdes ankeret. For formrutenettet er ankeret den lineære kjernen; for F4 er det F2. Svake nær-null-ledd i den brede kjernen flagges og går videre til ablasjon/reduksjon; de skal ikke stanse hele stigen før F5 får virke. Numerisk svikt eller materiell ustabilitet som gjør selve sammenligningen upålitelig, må derimot avklares før valget.
- **Endelig kontroll:** F5/F6 må også sammenlignes med F0 og F1. Dersom kompleksiteten ikke er berettiget etter samme regel, foretrekkes den enklere stabile kandidaten. Ikke sammenlign scorer fra ulike målvariabler eller ulike rader.

Stabilitet må beskrives på effektskala, ikke med en p-verdi:

1. Lineære ledd: samme fortegn i minst 4/5 folder. Nær-null-effekter som skifter fortegn er argument for forenkling, ikke datarensing.
2. Kategorier: vis relativitet og støtte i hver fold. «Vesentlig eksponering» betyr her ≥500 eksponeringsår i hele train-poolen. For slike nivåer forventes samme retning mot felles basis i minst 4/5 folder; ellers undersøkes om det er en ubetydelig nær-null-effekt eller en materiell reversering. Små nivåer og MISSING vises, men skal ikke alene styre hovedvalget.
3. Splines: beregn relative kurver mot samme referanseverdi, for eksempel train-poolens eksponeringsvektede median, på et fast grid fra p2,5 til p97,5 av eksponeringen. Dette gridet bruker ikke respons og brukes bare til visning. Sammenlign kurver, aldri rå splinekoeffisienter fra ulike foldbasiser. Vis også ytterhaler og antall observasjoner utenfor treningsfoldens område.
4. Materielle reverseringer i godt støttede deler av kurven, ekstrem haleekstrapolasjon eller ustabil produkteffekt markeres som **uavklart**, ikke som automatisk bestått. Kandidaten kan ikke fryses før forholdet er forklart eller en enklere stabil kandidat er valgt. Unngå å innføre nye tallgrenser etter å ha sett hvilken modell de favoriserer.

Dette er et utviklingsoppsett, ikke nested CV for hele seleksjonsprosessen. Mange valg på samme fem folder gir seleksjonsoptimisme, også med en forhåndsdefinert stige. Det rapporteres eksplisitt. Cluster-KI etter modellvalg er dessuten betinget på valgt spesifikasjon og dekker ikke modellseleksjonsusikkerheten.

#### 3.5 Ablasjon og sekundære blokker

**F3, forklaring før reduksjon:** Med valgt F2-formel som felles anker fjernes én av følgende blokker om gangen: fører, verdi, ytelse, drivstoff, bruksmiljø, kommune, betaling, forretningstype. Knuter og koeffisienter læres på nytt i hver treningsfold for den gjenværende modellen. Ikke gjør ny formseleksjon inne i hver ablasjon. Vis dessuten felles fjerning av verdi + ytelse og av de to geografivariablene. Dette viser overlapp som kan gjøre individuelle bidrag små selv om blokkenes felles informasjon er nyttig.

F3-tabellen inneholder `removed_block`, antall sparte parametere, pooled deviance-tap, parvis tap/SE, foldretninger og stabilitet. Den skal ikke tolkes som kausale bidrag eller brukes til å slette alle individuelt svake variabler samtidig.

**F4, betinget sekundær informasjon:** Fra samme F2 sammenlignes fire forhåndsdefinerte alternativer: ingen ekstra blokk, bare merke, bare seter, begge. Formen på kjernen holdes fast. Bruk 3.4, og vis betinget bidrag fra merke gitt seter og fra seter gitt merke dersom begge velges. At bare kombinasjonen gir robust gevinst er tillatt, men må fremgå av tabellen.

- `seats_group` beregnes uten læring som `<5`, `5`, `>5`. Bruk én hel kategoriblokk og vis støtte per nivå/fold. Ingen nye responsbaserte sammenslåinger.
- Merke-pooling følger B-14 på hele 2022–2023-poolen. Dette er et eksplisitt eksponeringsbasert unntak fra foldvis preprocessing, ikke target encoding. 500-grensen skal ikke feilaktig kreves på nytt i hver fold: et globalt beholdt merke kan ha under 500 i foldtreningen.
- I tidsfolden læres merkelisten på 2022 med samme 500-grense, og brukes på 2023. Da brukes ikke fremtidig porteføljesammensetning i robusthetskontrollen. Antall beholdte merker kan avvike; sammenlign effekter bare for felles nivåer og dokumenter endringen. Dette presiserer B-14 for tidskontrollen, uten å endre gruppe-CV-konvensjonen.
- Ukjente rå merker blir OTHER. Valider at OTHER og alle benyttede kategorinivåer faktisk har støtte i foldtreningen. Pooling ved 250/1 000 er senere sensitivitet, ikke ekstra tuningalternativer i F4.

#### 3.6 Kontrollert forenkling og additiv frysing

Start med valgt F4 og behold denne som **fast fullmodell** for samlet tapskontroll. F3 er hovedtabellen for rekkefølgeuavhengige betingede bidrag; reduksjonen nedenfor er en begrenset bakoverprosedyre, og skal omtales som det.

1. Beregn alle enkeltblokk-ablasjoner fra gjeldende modell, med låste former. Produkt og år er beskyttet.
2. En sletting er kvalifisert bare hvis den enklere modellen ligger innen parvis 1 SE av **både gjeldende modell og den faste fullmodellen**, og ikke innfører en materiell diagnostisk svakhet. Denne doble kontrollen hindrer at mange små tap summeres til et stort tap.
3. Velg kvalifisert sletting med lavest pooled deviance. Ved lik score innen numerisk toleranse `1e-10`, velg størst parameterreduksjon og deretter alfabetisk blokknavn. Slett bare én blokk, logg valget, og beregn de gjenværende enkeltablasjonene på nytt.
4. Stopp når ingen sletting kvalifiserer. Maksimalt én sletting per opprinnelig valgfri blokk; ingen automatisk fremovergjeninnsetting eller uttømmende subsettjakt.
5. Kontroller den endelige reduksjonen mot fullmodellen, F2, F1 og F0. Gjør den ene avsluttende formsjekken fra 3.3 og frys deretter den additive kandidaten. Hvis en diagnostisk vurdering er uavklart, leveres beslutningspunktet før videre modellvalg.

Felles ablasjon av verdi/ytelse og geografi er diagnostikk; de brukes ikke som en snarvei til å fjerne to blokker uten den samlede kontrollen. En variabel som blir overflødig når merke kommer inn, kan falle ut her. Korrelasjon alene avgjør ikke hva som skal beholdes.

#### 3.7 To avgrensede interaksjoner (B-28)

Anbefalingen er to faglig begrunnede utfordrere: produktets egenandels-/meldemønster kan variere med førerprofil og bilverdi. Etter B-29 er COMP_N eneste kontrast mot COMP_E, så **utfordrerne er bare én COMP_N-spesifikk lineær helning per variabel**, lagt til de allerede valgte felles hovedkurvene:

```text
I1: additive_model + C(policy_type, Treatment('COMP_E')):driver_age
I2: additive_model + C(policy_type, Treatment('COMP_E')):log_vehicle_value
```

Disse er semantiske formler. Implementeringen skal lage én eksplisitt numerisk kolonne per interaksjon i notebooken: `is_comp_n * (x - center)` (etter B-29), der `center` er eksponeringsvektet gjennomsnitt av x i treningsfolden etter imputasjon. Gjenbruk dette senteret i validering/prediksjon. Da tilfører hver interaksjon **én** parameter. En rå Patsy-utvidelse kan lage en redundant felles helning når hovedleddet er en spline. Behold hovedleddene (hierarki), bruk COMP_E som produktbasis, og kontroller rang mot den additive modellen.

- Hvis alder eller verdi er fjernet i F5, utgår tilhørende interaksjon; ikke gjeninnfør variabelen bare fordi et etterfølgende residualplott ser interessant ut.
- Før fitting vises støtte etter produkt × tredeler av den aktuelle variabelen. Bruk **felles** grenser for alle produktene, fra foldtreningens eksponeringsvektede 1/3- og 2/3-kvantiler etter imputasjon (minste x der kumulativ sortert eksponering når grensen). Intervallene er `(-inf, q1]`, `(q1, q2]`, `(q2, inf)`. Sammenfallende grenser betyr svikt i støtteporten; ikke lag nye grenser. Som forhåndsdefinert, konservativ **prosjektregel** kreves ≥50 eksponeringsår i hver treningscelle og ≥20 skader per produkt i treningsfolden, i alle fem foldtreningene. Dette er en støtteport, ikke en universell aktuarmessig standard. Hvis porten svikter, utgår hele den aktuelle interaksjonskandidaten; ikke endre grensene eller slå sammen produkter etter resultatene.
- I1 og I2 sammenlignes hver for seg med samme additive F5. Hver må bestå B-08 og kurvestabilitet. Bare dersom begge består, estimeres også kombinasjonen. Kombinasjonen må bestå **hele B-08 mot begge** enkeltutfordrerne. Hvis den ikke gjør det, velges enkeltkandidaten med lavest pooled deviance; innen parets 1 SE og ved likt parametertall foretrekkes I1. Ingen av enkeltkandidatene får omgå kravet om først å slå den additive modellen.
- Vis produktvise effekter på felles, støttede områder. Ingen full produktspesifikk df=4-spline, merkeinteraksjoner, treveisledd eller automatisert søk.
- Hvis ingen interaksjon kvalifiserer, beholdes den additive modellen og eventuell reststruktur rapporteres som input til senere arbeid.

#### 3.8 Fordelingsutfordrer: negativ binomisk NB2

Etter at Poisson-strukturen er låst, prøves én NB2 med samme middelverdiformel. Legg først inn:

\[
E[N_i\mid x_i,e_i]=m_i=e_i\exp(\eta_i),\qquad
\operatorname{Var}(N_i\mid x_i,e_i)=m_i+\alpha m_i^2.
\]

Bruk etablert `smf.negativebinomial(..., loglike_method='nb2', exposure=...)` med **heltallsresponsen `property_claims`**, og estimer alpha i hver treningsfold. Ikke bruk både exposure og log(e)-offset. NB2 som antallsmodell beskriver en annen eksponerings-/variansrelasjon enn en NB-familie på vektede rater. Derfor må B-26 presiseres: Poisson-ekvivalensen er ikke en generell tillatelse til å bytte familie i den eksisterende ratemodellen. [Statsmodels NB2](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.NegativeBinomial.html)

Ved prediksjon på nye rader må valideringseksponeringen gis eksplisitt. Del forventet antall på e for å få frekvens, og bruk samme eksponeringsvektede Poisson-deviance som for Poisson. NB-log-score på **antall** er sekundær fordelingsdiagnostikk; Poisson og NB må i så fall sammenlignes på samme antallsrespons og hver sin fullstendige sannsynlighetsmasse.

NB velges bare når B-08 er oppfylt mot Poisson og stabilitet/kalibrering er akseptabel. Rapportér alpha per fold, konvergens og eventuell grense mot alpha=0. Ved konvergenssvikt eller Poisson-grense beholdes Poisson med dokumentasjon; ikke endre data eller drive optimizersøk etter et ønsket utfall.

**Inferens:** Hoved-KI bruker cluster-robust sandwich på `insured_id`. Pearson-dispersjon rapporteres separat. Ikke multipliser allerede cluster-robuste standardfeil med `sqrt(phi)` en gang til. En eventuell Pearson-skalert modellbasert SE-tabell er en separat sammenligning, ikke en ekstra skalering av sandwich-KI. Den tvetydige eldre teksten i B-23 er presisert i denne revisjonen.

#### 3.9 Diagnostikk og tidsrobusthet

Minimum rapporteres for F0, F1, valgt additiv modell og valgt endelig Poisson/NB2. Ikke skriv fulle statsmodels-sammendrag for alle rutenettkandidatene; vis kompakte sammenligningstabeller.

- **Primærscore:** pooled OOF Poisson-deviance, D² mot foldens treningssnitt, forbedring mot F0, alle fem foldscorer og train–OOF-gap.
- **Kalibrering:** observerte skader / forventede skader (`A/E = sum(N) / sum(e * lambda_hat)`) totalt, per prediksjonsdesil, produkt, år, forretningstype, betaling, begge geografifelt, drivstoff og bonus. Bonus kan være diagnostisk segment selv når det ikke er prediktor. Suppler med alder/verdi/ytelse og de utelatte sekundære blokkene. Vis eksponering, skadeantall og forventet antall ved siden av A/E. `val_balance` i dagens kode er motsatt brøk (E/A); merk dette tydelig.
- **Fordeling:** Pearson-dispersjon beregnes på antall `sum((N-m)^2/m)/(n-rank(X))`, ikke på uvektede rater. Cameron–Trivedi er en supplerende modellkontroll; hvis standardtesten antar uavhengige rader, oppgis dette eller auxiliary-regresjonen gis cluster-robust inferens. Overspredning alene bestemmer ikke NB-valget.
- **Rootogram:** forventet antall rader med N=k er `sum_i P(N_i=k | m_i)` med radens faktiske eksponering. Ikke bruk én Poisson med porteføljesnittet eller eksponeringsvekt observerte radantall. Vis 0–9 og en samlet 10+-hale, og særskilt COMP_N for pukkelen ved 4–5.
- **Utfallskondisjonert diagnostikk:** A/E per observert skadeantall, status og eksponeringsbøtte kan forklare registrerings-/kanselleringsmønster, men er ikke ordinær kalibrering på informasjon kjent ved prising. Ikke krev A/E=1 innen observerte skadeantallsgrupper, og ikke konstruer prediktorer av disse gruppene.
- **Effekter:** exp(beta) med cluster-KI for vanlige kategorier og meningsfulle lineære endringer. Spline- og interaksjonseffekter vises som relative prediksjonskurver/kontraster, ikke exp(beta) for vilkårlige basiskoeffisienter. Foldkurver bruker felles referansepunkt og beholdt designinformasjon.
- **Tidsfold:** refitt F0, F1, additiv finalist og eventuell F6/F7 på 2022 og prediker 2023. Fjern C(year) før fitting, og lær all imputasjon og splinebasis på 2022. Rapporter absolutt score/balanse og forskjell mot tidsbasisen. Ikke korriger prediksjonene med observerte 2023-skader for å forbedre scoren. Samme ID på tvers av denne tidsgrensen er forventet for fornyelser; disjunkt-ID-kravet gjelder gruppe-CV.

**B-07-overstyring:** En svakhet skal være systematisk og materiell, støttet av volum og synlig i de forhåndsdefinerte tabellene/kurvene. Felles global kalenderdrift er ikke alene grunn til å bytte variabler. Dersom tidsdeviance og støttede segmentrelativiteter tydelig svekkes, legg frem kandidatene og dokumenter en eventuell overstyring i registeret. Implementereren skal ikke finne opp en etterfølgende vektet totalscore eller automatisk overstyre med Gini, p-verdi eller et lite segment.

#### 3.10 Avgrensede sensitiviteter og arbeid som utsettes

Hold valgt hovedspesifikasjon fast; sensitiviteter starter ikke modellstigen på nytt.

1. **Uten kansellerte:** refitt bare på aktive rader i hver eksisterende treningsfold. Evaluer hovedmodell og sensitivitetsmodell på samme aktive valideringsrader. Vis dessuten prediksjon fra begge på hele valideringsfolden og på kansellerte separat, med tydelig merking av endret treningspopulasjon. Dette skiller relativitetsendring fra at ulike scorer bare skyldes ulike vurderingsrader. Hovedmodellens populasjon forblir uendret.
2. **Fri log(e)-koeffisient:** bruk antall med enten fritt log(e)-ledd uten offset, eller offset log(e) + et ekstra delta·log(e). I siste form er proporsjonalitet delta=0 og total koeffisient 1+delta. Rapporter cluster-KI. Dette er diagnostikk av eksponering/seleksjon, ikke en kvalifisert tariffmodell med fremtidig faktisk varighet som prediktor.
3. **Merkegrense 250/1 000:** hvis merke er med i additiv finalist, refitt samme modell med de to alternative, eksponeringsbaserte poolingene. Vis score, antall parametere og stabile relativiteter. 500 er hovedregel; sensitivitetsvinneren byttes ikke automatisk inn. Hvis merke ikke kvalifiserer ved 500, dokumenteres utelatelsen og ekstra poolingkjøringer utgår.
4. **Erfaring, bare sensitivitet:** erstatt hele førerleddet i den additive finalisten med erfaring lineært/df3/df4, med øvrige ledd fast. Hvis førerleddet er fjernet, sammenlignes disse tilleggene også med finalist uten førerledd. Samme rader, foldmedian for den ene manglende verdien, ingen alder samtidig. Rapportér score, datadefinisjonsforbehold og kurver; ingen opprykk til hovedbenchmark uten ny B-11-beslutning.
5. **Bonus:** ikke nødvendig for å ferdigstille fase 1. Eventuell én bonusutvidelse av additiv finalist merkes «potensielt lekkende sensitivitet» og vises separat. Den får ikke velge øvrige variabler eller slå ut hovedmodellen. Hvis kildeavklaring faktisk åpner B-13 før kandidatfittene, spesifiseres bonus som én egen kategoriblokk og obligatorisk uten-bonus-sensitivitet før stigen kjøres.
6. **Lagget skadehistorikk utsettes:** eksisterende `build_bonus_lagged_panel` gir bare 2023-rader med 2022-historikk. Det er en selektert fornyelsespopulasjon uten tilsvarende 2022-trening til ordinær tidsfold. Fire nye historikk-/bonusmodeller er derfor ikke obligatoriske i fase 1. Et senere forsøk må ha egen populasjon, sammenlignbar basis, gruppefolder og definert historikktilgjengelighet; manglende historikk er ikke null skader. Fjorårets felt er heller ikke automatisk dokumentert tilgjengelige ved en fornyelse midt i året.
7. **Ridge utsettes:** den foreslåtte stigen har få kontinuerlige frihetsgrader og høyst fire ekstra interaksjonsparametere. Det er ingen dokumentert grunn til å bygge en ny regulariseringspipeline nå. Eksakt rangsvikt skal løses i kodingen. Ved vedvarende nær-kollinearitet eller brede, ustabile kurver som ikke løses ved forenkling, dokumenteres et separat ridge-forslag med foldvis skalering, ustraffet intercept, fast straffeskala og indre gruppe-CV før implementering. Lasso skal ikke velge hovedbenchmarkens variabler.

#### 3.11 Implementeringskontrakt for neste utvikler

Arbeidet skal kunne gjøres uten nye modelleringsvalg underveis. Følg rekkefølgen nedenfor; avklarte brukerbeslutninger står i 3.0.

1. Les oppdatert plan og B-register; kontroller at notebookens seksjon 2.8, 2.10 og B-02/B-08/B-10/B-11/B-23/B-26 samt B-27/B-28 fortsatt samsvarer. Beslutningsteksten er oppdatert i denne planrevisjonen; kandidatkode er ikke lagt inn. Legg seksjon 3 før seksjon 7. Modell-likning og kort intuitiv forklaring kommer før hver ny modellfamilie/utvidelse.
2. Behold eksisterende `glm_spec`, `fit_glm`, `cross_validate_glm`, `paired_improvement` og foldindekser som utgangspunkt. Unngå en ny generell modellplattform. Modellformler, kandidatregister, seleksjonsregler, NB-estimering og CV-definisjoner skal være synlige i notebooken; presentasjonsfunksjoner kan ligge i `src/glm_diagnostics.py`.
3. Gi hver kandidat stabil `model_id`, `stage`, `parent_id`, `feature_blocks`, `forms`, `formula`, `family`, `n_parameters` og `eligible_for_selection`. En separat `required_columns`-liste styrer preprocessing. Ikke tolk råkolonner ut av formeltekst med skjøre strengsøk.
4. Endre `prepare_design_frame` til å behandle bare kandidatens nødvendige prediktorer. Dagens funksjon kontrollerer også ubrukte bonus-/merkefelt; dette skal ikke kunne stanse F0. Numeriske medianer læres på trening. For kategorier skal nye nivåer utløse en tydelig kandidat-/foldfeil med støtteoversikt, ikke stille radtap eller en fiktiv nullkoeffisient. Merke har den forhåndsbestemte OTHER-regelen. Ikke utled target-basert fallback.
5. Legg `seats_group` som deterministisk, dokumentert transformasjon i modellnotebooken. Det er ikke nødvendig å endre felles `src/model_data.py` for denne testen. Hvis felles datagrunnlag likevel endres, må både `analysis` og `glm_pricing_models` verifiseres.
6. Hver fold kontrollerer full designrang, konvergens, samme antall rader som inngangen, endelige positive prediksjoner, endelige parametere og passende kategoristøtte. Kandidater med feil merkes ugyldige; ikke dropp bare den problematiske folden og rapporter gjennomsnitt av fire.
7. Utvid CV-resultatet med foldvis rang/parametertall, referanseverdier og relative kurveprediksjoner, eller behold nødvendige fit-/designobjekter for finalistene. Nåværende `params` alene er utilstrekkelig for splinekontroll. Behold `oof`, `scores` og `params` for kompatibilitet med referansemodellene.
8. Gjenbruk identiske spesifikasjoner i minnet, slik at samme kandidat ikke fittes på nytt ved hver tabell. Cache-nøkkelen må skille formel, familie, relevante data/transformasjoner og folds. Bruk en enkel dict; ingen varig modellcache som kan skjule endrede data.
9. Lag tre oversiktstabeller: alle kandidater/feil, parvise beslutninger med begrunnelse, og finalister med diagnostikk. Vis kontinuerlige bidrag som kurver. Oppdater `build_relativity_table` eller filtrer bruken slik at rå spline-/interaksjonskoeffisienter ikke presenteres som vanlige tariffrelativiteter.
10. NB2 får en tydelig separat antallsgren som returnerer samme scoreformat, fremfor å late som `TARGETS['frequency']['family']` er eneste forskjell. Verifiser eksplisitt at forventet antall = eksponering × predikert rate, også på nye rader.
11. Stopp etter fase 1. Lever valgt formel/familie, OOF-resultater, tidskontroll, følsomhet, beslutningslogg og forbehold. Fase 2 får eget variabel- og formvalg; ikke overfør frekvensens forkastede variabler automatisk til severity.

#### 3.12 Ferdigkriterier for fase 1

- De tre brukerbeslutningene i 3.0 etterleves, register og brødtekst sier det samme, og ingen 2024-utfall er brukt.
- Modellrammen har 55 246 rader, omtrent 37 126,01 eksponeringsår og 9 989 skader etter B-29. Alle nullskadeår og de 9 nullkostnadsskadeårene inngår i frekvens.
- Gruppefolder er disjunkte på ID, hver rad har én OOF-prediksjon per gyldig kandidat, og kandidatpar har identiske rader og vekter. Ingen skjult imputasjon/basislæring fra valideringsdata.
- Pooled score beregnet direkte fra samlede OOF-prediksjoner samsvarer med eksponeringsvektet summering av foldscorene. Poisson antall+offset samsvarer med rate+vekt for en faktisk fase-1-spesifikasjon, også når den har splineledd.
- Formrutenett, ablasjon, sekundære blokker, samlet reduksjonskontroll og NB2-sammenligning er dokumentert. B-28s støtteport avgjør hvilke interaksjonskandidater som kjøres. Avbrutte/utelatte kandidater har eksplisitt årsak.
- In-sample-balansen for ustraffet Poisson med intercept er numerisk nær 1. OOF-balanse og D² er observasjoner, ikke hardkodede suksesskrav. En negativ OOF-D² eller et uventet fortegn skal undersøkes; det skal ikke «rettes» ved å endre data eller modellregler.
- Finalisten har støttediagrammer/kurver, A/E og tidskontroll. Ingen uforklart konvergens- eller rangsvikt, dobbel eksponeringsjustering eller dobbel inflasjon av standardfeil.
- Synkroniser med `uv run jupytext --sync glm_pricing_models.py`, kjør notebooken, og kontroller at forklaringer stemmer med output. Hvis Jupytext gir en tidsstempelfeil for en eksisterende fil, prøv samme `--sync` med absolutt sti og verifiser celleinnholdet. Verifikasjonen gjelder implementeringen; denne planrevisjonen alene krever ingen kandidatfitting.
- Rapportér tolkning, kjente tids-/registreringsbegrensninger og seleksjonsusikkerhet. Før fase 2 revurderes planen mot disse resultatene.

### Leveranse 2 – Fase 2: severity og storskader (seksjon 4, stopp etter)
- **Markdown:** log E[X̄ᵢ] = xᵢᵀγ, X̄ᵢ ~ Gamma med Var = φ·μ²/Nᵢ.
- **4.1 Respons:** X̄ = incurred/N for N>0 og incurred > 0,01, vekt N. De 9 nullkostnadsskadeårene holdes utenfor, og skjevheten i E[N]·E[X] kvantifiseres (B-15).
- **4.2 Gamma-GLM** med log-link og egen forhåndsdefinert kjerne, formvalg og ablasjon etter prinsippene i fase 1. Det konkrete kandidatrommet revurderes ved fasestart; en frekvensforkastet variabel kan være viktig for severity. Scoren er OOF Gamma-deviance vektet med N, pluss A/E i EUR per desil.
- **4.3 Lognormal-utfordrer:** vektet OLS på log X̄ med Duan-smearing. Evalueres med samme Gamma-deviance og A/E (B-16).
- **4.4 Flerskadeår:** A/E per N-bøtte, og en sensitivitetsmodell bare på N=1 som diagnose.
- **4.5 Storskadeanalyse**, med diagnostikk fra `src/large_claims_diagnostics.py` og modellspesifikasjonene i notebooken:
  - a) Mean excess-plott og skade/bilverdi for N=1. Tersklene u ∈ {5 000, 7 500, 10 000} EUR er fastsatt på forhånd (B-17).
  - b) Vesentlighet: andel av kostnaden over u.
  - c) Kappet respons N·min(X̄,u). Gamma-GLM på kappede skader. Endring i relativitetene og i foldstabiliteten mot modellen uten kapping.
  - d) Kan overskridelse predikeres? Binomial GLM for P(X̄>u | skade) med høyst 2–3 parametere, fordi det bare er ca. 60–130 hendelser (Harrell 2015). Sammenlignes med flat sannsynlighet via OOF log-loss.
  - e) Tillegget λ = Σoverskridelse / Σkappet kostnad, estimert i treningsfolden, flatt eller segmentert.
  - f) **Beslutningsregel (B-18):** separat behandling velges bare hvis overskridelsen er vesentlig (> 5 % av kostnaden) **og** kapping pluss tillegg forbedrer OOF Gamma-deviance og A/E i EUR i øverste desil, eller stabiliserer relativitetene tydelig. Ellers brukes Gamma uten kapping. Alle varianter predikerer ukappet E[X] og sammenlignes på samme skala.
- **4.6 Oppsummering og beslutninger.**

### Leveranse 3 – Fase 3: ren premie (seksjon 5, stopp etter)
- **Markdown:** den todelte modellen π̂ = μ̂(e=1)·ŝ, med eventuelt storskadetillegg, og Tweedie-modellen log E[Y/e] = xᵀθ med Var = φ·μᵖ/e.
- **5.1 Todelt OOF-prediksjon** med de valgte modellene fra fase 1 og 2 på samme folder.
- **5.2 Tweedie-GLM:** `sm.families.Tweedie(var_power=p, eql=True)`. p̂ velges med EQL-profil over et rutenett i hver treningsfold (Smyth & Jørgensen 2002) (B-19). Variabelsettet er unionen av fase 1 og 2, med bakoversjekk per blokk.
- **5.3 Sammenligning:**
  - Tweedie-deviance ved p̂ og over p-rutenettet
  - global balanse og A/E per desil
  - double lift
  - Gini
  - MCB/DSC/UNC
  - segment-A/E, særlig COMP_N mot COMP_E
  - pukkelen: den todelte modellen påvirkes, Tweedie ikke
- **5.4 Beslutning:** hvilken struktur som blir benchmarken.

### Leveranse 4 – Fase 4: konsolidert benchmark (seksjon 6 og 7, stopp etter)
- **6.1** Mesterspesifikasjonen fryses som formelstrenger og innstillinger i en dict i notebooken, og refittes på hele train_pool.
- **6.2** Relativitetstabell i tariffstil, exp(β) med cluster-KI, og spline-effektplott.
- **6.3** Samlet OOF-tabell for alle kandidater fra fase 1–3, og resultatene fra tidsfolden.
- **6.4** Sammenligning mot dagens premie: `property_damage_premium` brukes bare som benchmark, aldri som prediktor (B-20). Lorenz/Gini for premie mot modell på OOF, og loss ratio per modelldesil.
- **6.5** Kjente svakheter: A/E-varmekart, gjenværende ikke-linearitet og pukkelen. Dette er input til ML-fasen. Eventuelle allerede kvalifiserte interaksjoner etter B-28 beholdes; ingen nye interaksjoner søkes i konsolideringsfasen.
- **6.6** Prediksjonskonvensjon for bruk fremover: e=1 og årsnivå 2023 (B-22). Dokumenteres, men testes ikke på 2024.
- **Seksjon 7** Beslutningsregisteret oppdateres med utfallet.

## Beslutningsregister (oppdatert før fase 1)

| ID | Beslutning | Status |
|---|---|---|
| B-01 | Alle egen-skade-poliseår inkludert kansellerte, fast log(e)-offset, sensitivitet uten C | Brukerbesluttet |
| B-02 | Periodestart/fornyelse er kravet til en prospektiv tariff. Denne metodebenchmarken tillater plausible, uverifiserte risikofelt med timingforbehold etter B-27; status er aldri prediktor | Brukerbesluttet, presisert 2026-09-16 |
| B-03 | Frekvensrespons = skadeantall; pukkelen dokumenteres; Tweedie som kontroll | Brukerbesluttet |
| B-04 | Datagrunnlaget ligger i `src/model_data.py` og dokumenteres i begge notebooks | Brukerbesluttet |
| B-05 | 2024 åpnes én gang for felles sluttevaluering etter at GLM og ML er frosset | Brukerbesluttet |
| B-06 | GroupKFold(5, shuffle, seed 100) på `insured_id` velger modell; tidsfold 2022→2023 er obligatorisk robusthetskontroll | Brukerbesluttet |
| B-07 | Pooled vektet OOF-deviance er primær; overstyring krever dokumentert materiell og systematisk svikt i forhåndsdefinert diagnostikk | Brukerbesluttet |
| B-08 | Samlet forhåndsdefinert kjerne, formvalg før reduksjon, ablasjon og avgrensede sekundære tester. Oppgradering krever positiv pooled gevinst, parvis gevinst > 1 SE, minst 4/5 folder og stabilitet. Forenkling kontrolleres mot både gjeldende og fast fullmodell; bare B-28-interaksjoner | Revidert faseplan på brukerens oppdrag; terskler beholdt |
| B-09 | `fuel_type` får egen kategori MISSING; numeriske variabler imputeres med median lært i folden | Foreslått |
| B-10 | Lineær mot sentrert naturlig spline df=3/4; fast rutenett for alder, logverdi og ytelse før endelig ablasjon; enkleste form innen parvis 1 SE | Brukerbesluttet regel / Datadrevet resultat; implementering presisert |
| B-11 | `driver_age` er førerkandidaten i hovedstigen; erfaring er bare sensitivitet inntil definisjonen avklares. Aldri begge samtidig | Brukerbesluttet 2026-09-16 |
| B-12 | `year` som kategorisk kontroll i gruppe-CV, utelatt i tidsfolden; 2023 er benchmarknivå, ikke trendestimat | Brukerbesluttet |
| B-13 | Bonus krever dokumentert as-of-dato før OOF-seleksjon; ellers bare potensielt lekkende sensitivitet | Brukerbesluttet adgangsregel / Datadrevet resultat |
| B-14 | Merke-pooling ≥500 eksponeringsår lært på train_pool (bare eksponering) i gruppe-CV; 2022 alene i tidsfolden. Sensitivitet med 250 og 1 000 | Brukerbesluttet hovedregel; tidskontroll presisert |
| B-15 | Nullskadeår inngår i frekvens, ikke severity; 9 nullkostnadsskader beholdes i frekvens/ren premie og utelates fra Gamma-severity | Brukerbesluttet |
| B-16 | Gamma med log-link og vekt N; lognormal med smearing som utfordrer | Foreslått / Datadrevet |
| B-17 | Storskadetersklene 5 000, 7 500 og 10 000 fastsatt på forhånd; kapping på snittskaden | Foreslått |
| B-18 | Beslutningsregel for separat storskadebehandling | Foreslått / Datadrevet |
| B-19 | Tweedie-p fra EQL-profil i treningsfolden | Foreslått |
| B-20 | Premie brukes bare som benchmark i fase 4, aldri som prediktor | Brukerbesluttet |
| B-21 | Cluster-robuste standardfeil på `insured_id` | Foreslått |
| B-22 | Prediksjon ved e=1 og årsnivå 2023 som benchmarknivå, ikke fremtidig trendestimat | Brukerbesluttet |
| B-23 | NB2 med antall og eksponering må slå Poisson etter B-08 på samme OOF Poisson-deviance. Ellers Poisson med cluster-robust inferens; Pearson-dispersjon separat, ingen dobbel SE-skalering | Fase-1-presisering av foreslått fordelingsvalg |
| B-24 | Basisnivå for kategoriske variabler = nivået med størst eksponering (f.eks. `COMP_E`) | Foreslått |
| B-25 | Tidsfolden bruker spesifikasjoner uten `C(year)`; nivåskiftet vises som global balanse | Foreslått |
| B-26 | Poisson bruker rate + eksponeringsvekt, ekvivalent med antall + offset. NB2 er et eksplisitt unntak: antallsfit med exposure, deretter rate til felles scoring | Implementert Poisson-konvensjon; NB2 presisert |
| B-27 | Metodebenchmark med eksplisitt timingforbehold: kilden beholder siste poliseårsregistrering, ikke dokumenterte startverdier. Ingen påstand om garantert fravær av tidslekkasje eller prospektiv tariffytelse | Brukerbesluttet 2026-09-16 |
| B-28 | Bare produkt × alder og produkt × logverdi, med én ekstra kontrasthelning (COMP_N) hver etter B-29, støtteport og B-08. Hierarki beholdes; ingen automatisert interaksjonsjakt | Brukerbesluttet adgang; teknisk avgrensning i faseplan |
| B-29 | Modellpopulasjonen er kaskoprodukter (COMP_E, COMP_N). CC holdes utenfor i train og test: bare 0,8 % av CC-poliseårene har egen-skadepremie (0,47 % i 2022, 0,95 % i 2023), 54 % av disse polisene er kasko i det andre året (trolig produktbyttere under siste-registrering-regelen), og egen-skadepremien per verdi er omtrent halvparten av COMP_E | Brukerbesluttet 2026-09-16 |
| B-30 | Frekvensmodellen i fase 1 er den additive `F5_minus-municipality-performance`. `F6_I1_product-x-age` besto B-08, men overstyres etter B-07: gevinsten er bare 1,1 cluster-SE i gruppe-CV, F6_I1 er dårligere i tidsfolden 2022→2023 (−0,0022, z −2,2), COMP_N-helningen er −0,015 i 2022 mot −0,003 i 2023, og tapet bæres av ca. 600 poliseår med ≥4 skader (B-03). Beslutninger nær terskelen kontrolleres med cluster-SE som dokumentasjon, uten å endre reglene med tilbakevirkende kraft | Brukerbesluttet 2026-09-16 |

## Verifikasjon (per leveranse)
1. `uv run jupytext --sync glm_pricing_models.py`, og for leveranse 0 også `analysis.py`.
2. `uv run jupyter nbconvert --to notebook --execute --inplace glm_pricing_models.ipynb --ExecutePreprocessor.timeout=1800`, og tilsvarende for `analysis.ipynb` i leveranse 0.
3. Leveranse 0: analysistallene er identiske før og etter refaktoreringen (seksjon 11/12/17-tabellene og asserts). Foldene er disjunkte. Nullmodellen reproduserer porteføljefrekvensen 0,267 og ren premie ≈232.
4. Sunnhetssjekker: global balanse ≈1 for ustraffet Poisson med intercept på treningsdata. Undersøk D², train–OOF-gap og støttede relativiteter; ikke hardkod positiv OOF-D², bestemt bonusrekkefølge eller at enhver valideringsfold må være dårligere enn trening. Fase 1 bruker de konkrete ferdigkriteriene i 3.12.
5. Resultatene sammenfattes kort med antakelser og begrensninger. Deretter spørres det om commit (`checkpoint:`). Neste leveranse starter med revurdering av planen.

## Endringslogg

| Dato | Fase | Endring | Begrunnelse |
|---|---|---|---|
| 2026-09-15 | Planlegging | Første versjon | Brukerens valg: behold kansellerte, tellemodell for frekvens, datagrunnlag i `src/`, fase for fase |
| 2026-09-15 | Leveranse 0 | La til B-24 (basisnivå), B-25 (tidsfold uten årsledd) og B-26 (rate med vekt). Referansemodeller for alle tre målvariabler, med `paired_improvement` for B-08, er lagt inn allerede i seksjon 2.9. `src/glm_diagnostics.py` startet med `build_fold_summary` og `summarize_cv_scores`. | Implementeringsvalg som oppsto underveis: `C(year)` kan ikke predikere et usett år, og én felles CV-sløyfe krever én responsform. **Til revurdering før fase 1:** årsleddet for severity passerte 1-SE med forbedring i bare 3 av 5 folder. Vurder å skjerpe B-08, f.eks. forbedring i minst 4 av 5 folder. Tidsfolden viser motsatt drift (frekvens −17 %, severity +10 %), noe som påvirker B-22. |
| 2026-09-16 | Leveranse 0 (opprydding) | `to_model_frame` (dtype-konvertering) flyttet til `src/model_data.py`, `build_data_overview`, `build_glm_summary` og `build_relativity_table` lagt i `src/glm_diagnostics.py`. CV og referansemodeller bygger nå på `glm_spec` → `fit_glm` → `run_glm`/`cross_validate_glm`, der formelen bygges fra dtypes én gang per modell. B-24 brukes også på `year` (basis 2023). Ingen sklearn-`Pipeline`. | Notebooken skal fokusere på modellene. Pipeline ble forkastet fordi statsmodels/patsy-formlene og koeffisientene er det notebooken skal vise, og `prepare_design_frame` allerede lærer alt på treningsfolden. OOF-scorene er uendret. |
| 2026-09-16 | Leveranse 0 (opprydding) | Nullmodellene estimeres ikke lenger som GLM. Nullnivået vises som `oof_null_deviance` (vektet treningssnitt per fold), og parvis sammenligning nullmodell → `policy_type` bruker `val_null_deviance`. | En intercept-GLM med log-link gir nøyaktig det vektede snittet, så modellfittene var redundante. Tallene er uendret (f.eks. frekvens: forbedring 0,10266, SE 0,00527). |
| 2026-09-16 | Revurdering før fase 1 | Låste B-02, B-05–B-08, B-10–B-15, B-20 og B-22. Skjerpet seleksjonen til positiv pooled OOF-gevinst, >1 SE og minst 4/5 forbedrede folder; innførte typepasset stabilitet og enkleste modell innen 1 SE. Bonus fikk en as-of-port. | Referansemodellene viste at 1-SE alene kunne passeres med bare 3/5 forbedrede folder og at kalenderdrift må skilles fra variabelseleksjon. Brukerbeslutningene er samlet i notebookens seksjon 2.10. Planen for fase 1 beholdes ellers uendret. |
| 2026-09-16 | Fase 1, modellstige revidert | Erstattet fremoverseleksjon med samtidig kjerne, avgrenset formrutenett, ablasjon, sekundærtest og kontrollert reduksjon. Lagt til evidens per variabel, implementeringskontrakt, NB2-/SE-presisering og verifikasjonskrav. Ridge og lagget historikk utsettes. | EDA viser ikke-linearitet og korrelerte kandidater; kontinuerlige effekter må få riktig form før forkastelse. Planen skal kunne implementeres uten nye faglige designvalg. Ingen kandidatmodeller fittet i revisjonen. |
| 2026-09-16 | Fase 1, nye brukeravklaringer | B-27: metodebenchmark med timingforbehold. B-11: alder i hovedstigen, erfaring kun sensitivitet. B-28: to begrensede interaksjoner kan kvalifisere. B-02/B-08 og notebookens beslutningstekst samordnet. | Kilden beholder siste årsregistrering; tidligere tekst kunne overdrive as-of-sikkerheten. Erfaring har uavklart semantikk. Brukeren besvarte alle tre spørsmål eksplisitt før implementering. |
| 2026-09-16 | Fase 1, operasjonalisering ved implementering | Faste tallgrenser for «materiell» erstattet med støyskalerte grenser: en foldeffekt er materiell når \|log rel\| > 2·SE (deltametode for kurver, Pearson-φ̂-skalert kovarians), og A/E flagges når \|A/E−1\| > 2·√(φ̂/forventet). Kurvegrid: 25 punkter mellom eksponeringsvektet p2,5 og p97,5 med referanse i vektet median. Et fortegnsskifte uten materiell fold klassifiseres som `nær-null`, flagges og går videre. Materiell reversering, også for CC, blir `UAVKLART`. NB2 regnes som på randen når α̂ < 2·SE(α̂); ved randen i flertallet av foldene beholdes Poisson. Støtteporten for B-28 beholdes, og skader per celle rapporteres uten terskel. Haleekstrapolasjon rapporteres uten terskel og vurderes manuelt. Presentasjonskode ligger i `src/phase_2/`. | Brukeren ba om empirisk begrunnede grenser framfor faste prosentgrenser. Grensene er låst før første kandidat i hovedstigen er estimert. Det ble ikke innført en tallgrense for haler, fordi den ville bli satt etter at røykttesten viste ytelse-df4 med relativ 0,11–0,46 ved treningsmaks. |
| 2026-09-16 | Fase 1, populasjonsendring (B-29) | CC fjernet fra modellpopulasjonen i `select_own_damage_scope`. B-28 reduseres til én COMP_N-helning per variabel. Train-poolen blir 55 246 poliseår, 37 126,01 eksponeringsår og 9 989 skader; merkelisten er uendret (17). Alle CV-resultater i 2.9 og 3.2–3.8 kjøres på nytt med de samme låste reglene, og master-spesifikasjonen legges frem igjen hvis valget endres. | CC med egen-skade er en liten restgruppe med uklar dekning og trolig produktbytte innen året (B-27). Den ga en usikker CC-koeffisient uten tariffmening. Beslutningen bygger på dekning, premiestruktur og polisehistorikk i 2022–2023, ikke på 2024. |
| 2026-09-16 | Fase 1, resultat etter B-29 | Rekjøring med samme låste regler: additiv modell uendret (`F5_minus-municipality-performance`, 13 parametere). Endringer: årsleddet består ikke lenger B-08 for frekvens (3/5, obligatorisk, ingen konsekvens); drivstoff stoppes nå både av 1 SE (1,12 SE) og A/E-flagg; produkt × alder (I1) kvalifiserer (2,8 SE, 5/5) og var forkastet med CC. NB2 forkastes fortsatt. Brukeren godkjente `F6_I1_product-x-age` som frekvensmodell for 3.9–3.12. Evidenstabellen i seksjon 3.1 over har fortsatt tall fra populasjonen med CC; gjeldende tall står i notebookens 3.0. | CC-helningen hadde svært lite data og maskerte en stabil COMP_N-helning; beslutningen følger B-08 og brukerens godkjenning. |
| 2026-09-16 | Fase 1, B-07-overstyring (B-30) | Frekvensmodellen settes til F5 i stedet for F6_I1 etter tidsfold med parvis cluster-SE. 3.10–3.11 bygges på F5; NB2 sammenlignes på nytt mot F5. Beslutninger nær terskelen (drivstoff, alder df3/df4, merke, NB2) kontrolleres med cluster-SE. **Til planrevisjonen før fase 2:** vurder parvis cluster-SE på `insured_id` i stedet for fold-SE i B-08, siden fem folder ga et for optimistisk SE for I1. | Fold-SE fra fem folder er selv svært usikker; tidsfolden er den beste tilgjengelige prospektive kontrollen uten å bruke 2024. |
| 2026-09-16 | Fase 1, omfang i modellvalget | Fase 1 kryssvaliderte 77 spesifikasjoner (74 i valgstigen og 3 sensitiviteter), men bare 16 var datadrevne valg, og 3 endret modellen (alder df3, uten kommunetype og uten ytelse). Ca. 90 % av OOF-gevinsten over F0 kommer fra den forhåndslåste kjernen F1. **Til planrevisjonen før fase 2:** krymp kandidatrommet med formvalg per variabel eller fast df i stedet for fullt rutenett, én reduksjonsrunde i stedet for både ablasjon og trinnvis sletting, og et tak på antall datadrevne valg som låses på forhånd. | Brukeren påpekte at antallet var høyt. Antall kjøringer overdriver optimismen, men designet brukte flere kjøringer enn valgene krevde. Severity har de samme skadene, men mer støy per observasjon, så samme stige vil gi mer seleksjonsoptimisme. |
| 2026-09-16 | Fase 1, pukkelen i skadeantall (B-03) | Sensitivitet (5) er lagt til i 3.11: F5 med antall kappet ved 3 og med skade ja/nei, med rekalibrert nivå, scoret på fullt antall i gruppe-CV og tidsfold. COMP_N-relativiteten er ×3,66 med fullt antall, ×2,98 kappet og ×2,07 per skadepoliseår; øvrige effekter ≤1 SE (P −1,5 SE). Kappet modell taper på fullt antall (z −2,9 CV, −5,1 tid). B-03 står. Datakilden er sjekket manuelt: `property_claims` teller «material and personal damages» for COMP; tellemåten per hendelse er ikke dokumentert, og beløp gjøres opp etter CICOS-avtalen med forhåndsavtalte beløp. **Til planrevisjonen før fase 2:** vurder todelt modell som andel poliseår med skade × kostnad per skadepoliseår, sjekk om beløpene klumper seg (CICOS), og behold Tweedie som kontroll. | Poliseår med N ≥ 4 har ca. 38 % av skadene. Pukkelen var kjent (B-03), men konsekvensen var ikke kvantifisert før modellering. Fordelingen mellom frekvens og snittskade er usikker; totalkostnaden er det ikke. |
| 2026-09-17 | Fase 1, separat ABESS-diagnostikk | La til en avgrenset, gruppet ABESS-kontroll i `src/abess_diagnostics.py`, uten endring i `frequency_cv`, kandidat-ID-er eller B-30. To låste oppsett, størrelser 2–12 og samme fem gruppefolder ble evaluert. Begge oppsett velger ni grupper og taper 0,000531 pooled OOF-deviance mot F5 (cluster-SE 0,000840). | Best-subset kan belyse alternative kombinasjoner, men de samme foldene har allerede påvirket funksjonsform og størrelsesvalg. Resultatet er utviklingsdiagnostikk med seleksjonsoptimisme, ikke en ny hovedmodell eller dokumentert generaliseringsgevinst. |
| 2026-09-17 | Faktarevisjon (korrigering, ingen ny modellering) | **Rettet et fortegnsfeil i raden over og i «Separat ABESS-diagnostikk etter fase 1»:** `gevinst_mot_låst_glm = glm_deviance − abess_deviance` var positiv (0,000531) i den lagrede notebook-tabellen, dvs. F5s deviance (1,121675) er *høyere* enn ABESS' (1,121144) — ABESS har altså marginalt *lavere/bedre* pooled OOF-deviance, ikke «taper» slik forrige versjon skrev. Notebookens egen tekst («Positiv `gevinst_mot_låst_glm` betyr at ABESS er bedre enn F5», glm_pricing_models.py) bekrefter konvensjonen. Konklusjonen endres ikke: 0,63 cluster-SE er langt under B-08s 1-SE-krav, og ABESS bruker ca. 30 parametere mot F5s 13. **Andre rettelser i faktagrunnlaget:** `fuel_type`-mangel 439→438 (to steder); Datafunn-tabellens pukkeltall N=3/4/5 246/431/339→245/418/335 og bonusbalanse G/N/B 36 525/763/616→35 773/745/608 (var pre-B-29/CC-tall, verifisert mot `build_model_frames()` på gjeldende train-pool); seksjon 3.1s årsrad 0,282→0,285 og «17 %»→«18 %» underprediksjon (verifisert mot gjeldende referansemodell-tabell i notebooken, seksjon 2.9). Statusavsnittet øverst er oppdatert til å reflektere at fase 1 faktisk er fullført og låst (B-30). Kansellerte-poliser-raden, COMP_N/COMP_E frekvens/severity-raden (skadevektet, korrekt) og B-30s tidsfoldtall i beslutningsregisteret ble kontrollert og er uendret — de stemte allerede med gjeldende kode/output. **Uavklart:** «Fri log(e)-koeffisient 1,53 uten C» i Datafunn-tabellen kunne ikke reproduseres med en avgrenset kontroll (en enkel spesifikasjon uten F5s kovariater ga 1,25/1,63, ikke direkte sammenlignbart) og er ikke rettet; haletabellens tall (8,5 % topp 1 %, terskelandeler 5 000/7 500/10 000, skade/bilverdi 0,92) finnes ikke igjen i noe kjørt script eller notebook-output og kunne ikke verifiseres — begge bør presiseres eller rekjøres før de brukes i severity-planen. `glm_pricing_models.py` seksjon 3.11A har den samme fortegnsfeilen i markdown-teksten («ABESS taper …») som ble rettet her i planen; notebooken er ikke endret i denne revisjonen. | Brukeren ba om en tall- og konsistensrevisjon av planen før severity-planen skrives; feilen i ABESS-fortegnet var konkret meldt inn og verifisert direkte mot lagret celleoutput i `glm_pricing_models.ipynb`. |

## Utenfor omfang nå
GBM og penaliserte GAM, credibility for merke, ekstremverditeori (GPD) utover mean excess-plottet, dispersjonsmodellering (DGLM), interaksjoner utover B-28, ridge/lasso i denne leveransen, den separate laggede historikkmodellen og all evaluering på 2024.

## Separat ABESS-diagnostikk etter fase 1

ABESS er en eksplorativ kontroll av den gjennomførte frekvensprosessen, ikke en
ny hovedmodell eller et nytt kandidatregister. Den bruker bare train-poolen
2022–2023, samme fem `insured_id`-gruppefolder, responsen `claim_frequency` og
eksponeringsvekt som fase 1. I hver treningsfold læres imputasjon og
Patsy-splinebasis før en separat gruppeseleksjon. Produkt og år er obligatoriske
blokker; interseptet er alltid med, men teller ikke som en gruppe.

Kandidatrommet er låst til to additive oppsett med føreralder som sentrert
naturlig spline (df=3), lineær log(bilverdi), og enten lineær ytelse (A) eller
sentrert naturlig spline for ytelse (df=3, B). De ti øvrige valgbare blokkene
er de tidligere fase-1-blokkene, inkludert forkastede kommune-, merke- og
seteblokker. Det søkes bare over hele Patsy-blokker, aldri enkeltnivåer eller
splinekolonner. Alle størrelser 2–12 evalueres eksplisitt uten ABESS-EBIC eller
intern CV. Etter CV beskrives et fit på hele utviklingssettet, men det brukes
ikke til en ny valideringsscore.

**Resultat (2026-09-17, rettet 2026-09-17 – se endringslogg).** Begge oppsett hadde minimum ved ni grupper, og
1-SE-heuristikken ga samme størrelse. Det diagnostiske subsetet var produkt,
år, føreralder, log(bilverdi), drivstoff, urban/rural, betalingsfrekvens, NB/P
og poolingsdefinert merke; ytelse, kommune og seter ble utelatt. A og B er
derfor identiske i det valgte subsetet, med i snitt 30 parametere (mot F5s 13).
Pooled OOF Poisson-deviance var 1,121144, som er 0,000531 *lavere* (marginalt
bedre) enn den låste GLM-en `F5_minus-municipality-performance` sin 1,121675
(cluster-SE 0,000840; z ≈ 0,63, i ABESS' favør). Gevinsten er dermed langt
under 1-SE-kravet i B-08 og oppnås med mer enn dobbelt så mange parametere, så
den gir ikke grunnlag for å bytte ut den låste modellen — men den er heller
ikke et tap for ABESS, slik en tidligere versjon av dette avsnittet feilaktig
skrev. Dette er uansett ikke dokumentasjon på generaliseringsgevinst for noen
av modellene; både funksjonsformene og ABESS-størrelsen er vurdert på de
samme overlappende foldene som tidligere fase-1-valg. Seleksjonsfrekvenser
over fem treninger er bare beskrivende stabilitet, og den clusterbaserte
usikkerheten dekker ikke seleksjonsusikkerheten. Timingforbeholdet i B-27
gjelder uendret.

## Litteratur (referanseliste i notebookens seksjon 2)
Gneiting (2011, JASA); Wüthrich & Merz (2023, Springer); Fissler, Lorentzen & Mayer (2023, arXiv:2202.12780); Wüthrich (2023, Eur. Actuar. J., Gini under autokalibrering); Frees, Meyers & Cummings (2011, JASA); Goldburd, Khare, Tevet & Guller (2020, CAS Monograph 5); Ohlsson & Johansson (2010, Springer); Noll, Salzmann & Wüthrich (2018, SSRN 3164764); Delong, Lindholm & Wüthrich (2021, Eur. Actuar. J.); Smyth & Jørgensen (2002, ASTIN Bull.); Jørgensen & de Souza (1994, Scand. Actuar. J.); Denuit, Charpentier & Trufin (2021, IME); Kleiber & Zeileis (2016, Am. Stat.); Cameron & Trivedi (1990, J. Econometrics); Cameron & Miller (2015, J. Human Resources); Roberts et al. (2017, Ecography); Hastie, Tibshirani & Friedman (2009); Harrell (2015); Duan (1983, JASA); Embrechts, Klüppelberg & Mikosch (1997).
