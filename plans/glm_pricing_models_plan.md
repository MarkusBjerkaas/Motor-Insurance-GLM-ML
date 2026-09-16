# Plan: GLM-benchmark for egen skade – fase 1–4 i `glm_pricing_models`

## Context

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
| Pukkel i skadeantall | N=3: 246, N=4: 431, N=5: 339, mest i COMP_N. Median kostnad per skade faller fra 690 til 495 EUR. | Tellemodellen beholdes, med rootogram og A/E per antall (B-03). Tweedie er en robust kontroll. |
| Motsatt retning for frekvens og severity | COMP_N: frekvens 0,69 / severity 680. COMP_E: 0,18 / 1 020. | Et konkret argument for å teste den todelte modellen mot Tweedie i fase 3. |
| Tyngde i halen | Topp 1 % av skadeårene står for 8,5 % av kostnaden. Kostnad over u på skadeårsnivå: 7,5 % (5 000), 3,8 % (7 500), 2,1 % (10 000). Høyeste skade/bilverdi er 0,92, og bare 8 skader ligger over 0,5. | Moderat og avgrenset hale uten tydelig totalskadeklynge. Storskadeanalysen er beslutningsstøtte, ikke en forhåndsbestemt løsning. |
| Enkeltskader kan identifiseres | 67 % av skadeårene har N=1, men de utgjør bare 38 % av skadene | Terskelen vurderes på skadeår med N=1. Kappingen gjøres på snittskaden. |
| Bonus-balanse | Eksponering G 36 525, N 763, B 616 | Brede intervaller for N og B |
| Manglende verdier | fuel_type 439, vehicle_value 33, age_driving_licence 1, power_to_weight 3 (satt til NA i rensingen) | Egen strategi (B-09) |
| Teknisk | patsy feiler på pandas `Int16` og andre nullable dtypes | Designrammen konverteres til float/str før formlene brukes |
| Nullkostnadsskader | 9 skadeår med incurred ≤ 0,01 | Utelates fra severity (B-15) |

## Filer

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
- AIC/BIC brukes bare som sjekk av nøstede modeller.

**CV-design (definert i notebooken):**
- `GroupKFold(n_splits=5, shuffle=True, random_state=100)` på `insured_id`. Uten shuffle fordeles like store grupper etter id-rekkefølge, og id-rekkefølgen kan henge sammen med tegningstidspunkt (Roberts et al. 2017).
- Tidsfold 2022 → 2023 som obligatorisk robusthetskontroll, ikke som et nytt optimaliseringssett.
- Alt som avhenger av data læres **inne i treningsfolden**: imputasjonsmedianer, spline-knuter (patsy stateful `cr()`), NB-α, Tweedie-p og storskadetillegget λ.
- Fold-assert om disjunkte grupper.

**Seleksjonsregel (låst før første kandidatfit i fase 1):**
- En blokk beholdes når samlet vektet OOF-gevinst er positiv, gjennomsnittlig parvis forbedring i fold-deviance er større enn én standardfeil (1-SE-regelen, Hastie et al. 2009), og minst 4 av 5 folder forbedres.
- Stabilitet vurderes på riktig skala: fortegn for enkle lineære ledd, eksponeringsstøttede relativiteter for kategorier og predikert kurve i sentrale 95 % for splines. Innen 1 SE velges enkleste modell. Fullmodellen får en bakoversjekk blokk for blokk; interaksjoner inngår ikke.
- p-verdier og one-way-rater er ikke seleksjonskriterier.
- Begrensning: med 5 folder er standardfeilen grov.

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

> Leveranse 1–4 starter hver med **Revurdering mellom faser**. Innholdet under er dagens beste utkast.

### Leveranse 1 – Fase 1: frekvens (seksjon 3, stopp etter)
- **Markdown:** modellen i LaTeX: log μᵢ = log eᵢ + xᵢᵀβ, Nᵢ ~ Poisson(μᵢ), NB2 med Var = μ + αμ².
- **3.1 Poisson-basis** med `policy_type + C(year)`. Pearson φ̂, Cameron–Trivedi og rootogram.
- **3.2 Blokkvis fremoverseleksjon** i forhåndsbestemt rekkefølge:
  - B1 `policy_type + year`
  - B2 `bonus_score`, bare dersom as-of-porten i B-13 er bestått
  - B3 `cr(log_vehicle_value)` og `cr(performance_hp_per_tonne)`
  - B4 `cr(driver_age)` **eller** `cr(driving_experience_years)` (B-11)
  - B5 `fuel_type`, `municipality_type`, `circulation_area`, `payment_frequency`, `business_type`
  - B6 `vehicle_brand_pooled`
  - B7 `seats`

  Til slutt kjøres en bakoversjekk der hver blokk fjernes fra fullmodellen.
- **3.3 Form på kontinuerlige ledd:** lineært mot `cr(df=3)` mot `cr(df=4)`, valgt på OOF-deviance.
- **3.4 NB2** (`smf.negativebinomial` med `exposure`) med samme variabelsett. Sammenlignes med OOF Poisson-deviance på middelverdien og OOF NB-log-score. Beslutningsregel B-23: NB velges bare hvis den forbedrer middelverdi-scoren. Ellers brukes Poisson med Pearson-skalert, cluster-robust inferens.
- **3.5 Sensitiviteter:**
  - dersom bonus består as-of-porten: uten `bonus_score` (obligatorisk)
  - dersom bonus ikke består as-of-porten: hovedmodell uten bonus; eventuell modell med bonus vises bare som tydelig merket, potensielt lekkende sensitivitet
  - lagget skadehistorikk på 2023-rader med observasjon året før, via `build_bonus_lagged_panel`, i fire varianter: basis, +lag, +bonus og +begge
  - uten kansellerte (B-01), med sammenligning av nivå og relativiteter
  - fri log(e)-koeffisient
- **3.6 Diagnostikk:** A/E per nivå, også for variabler som ikke er i modellen og per eksponeringsbøtte (med forbeholdet om utfallsavhengighet); A/E per observert skadeantall (pukkelen); koeffisienter med cluster-KI; koeffisientstabilitet over foldene; tidsfolden.
- **3.7 Oppsummering og beslutninger:** B-23, B-10, B-11, B-13 og det valgte variabelsettet.

### Leveranse 2 – Fase 2: severity og storskader (seksjon 4, stopp etter)
- **Markdown:** log E[X̄ᵢ] = xᵢᵀγ, X̄ᵢ ~ Gamma med Var = φ·μ²/Nᵢ.
- **4.1 Respons:** X̄ = incurred/N for N>0 og incurred > 0,01, vekt N. De 9 nullkostnadsskadeårene holdes utenfor, og skjevheten i E[N]·E[X] kvantifiseres (B-15).
- **4.2 Gamma-GLM** med log-link og egen blokkseleksjon (samme regel og rekkefølge). Scoren er OOF Gamma-deviance vektet med N, pluss A/E i EUR per desil.
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
- **6.5** Kjente svakheter: A/E-varmekart for to kandidatinteraksjoner (for eksempel `policy_type` × alder), gjenværende ikke-linearitet og pukkelen. Dette er input til ML-fasen. Interaksjoner tas ikke inn i mestermodellen.
- **6.6** Prediksjonskonvensjon for bruk fremover: e=1 og årsnivå 2023 (B-22). Dokumenteres, men testes ikke på 2024.
- **Seksjon 7** Beslutningsregisteret oppdateres med utfallet.

## Beslutningsregister (startinnhold; statusene er Brukerbesluttet, Foreslått eller Datadrevet)

| ID | Beslutning | Status |
|---|---|---|
| B-01 | Alle egen-skade-poliseår inkludert kansellerte, fast log(e)-offset, sensitivitet uten C | Brukerbesluttet |
| B-02 | Bare opplysninger kjent ved periodestart/fornyelse er kvalifisert; `policy_status` brukes aldri som prediktor | Brukerbesluttet |
| B-03 | Frekvensrespons = skadeantall; pukkelen dokumenteres; Tweedie som kontroll | Brukerbesluttet |
| B-04 | Datagrunnlaget ligger i `src/model_data.py` og dokumenteres i begge notebooks | Brukerbesluttet |
| B-05 | 2024 åpnes én gang for felles sluttevaluering etter at GLM og ML er frosset | Brukerbesluttet |
| B-06 | GroupKFold(5, shuffle, seed 100) på `insured_id` velger modell; tidsfold 2022→2023 er obligatorisk robusthetskontroll | Brukerbesluttet |
| B-07 | Pooled vektet OOF-deviance er primær; overstyring krever dokumentert materiell og systematisk svikt i forhåndsdefinert diagnostikk | Brukerbesluttet |
| B-08 | Positiv pooled gevinst, parvis gevinst > 1 SE, minst 4/5 folder, typepasset stabilitet, bakoversjekk og enkleste modell innen 1 SE | Brukerbesluttet |
| B-09 | `fuel_type` får egen kategori MISSING; numeriske variabler imputeres med median lært i folden | Foreslått |
| B-10 | Lineær mot naturlige kubiske splines (df 3/4), valgt på OOF; enkleste form innen 1 SE | Brukerbesluttet regel / Datadrevet resultat |
| B-11 | `driver_age` og `driving_experience_years` testes separat; ikke begge; alder vinner innen 1 SE | Brukerbesluttet regel / Datadrevet resultat |
| B-12 | `year` som kategorisk kontroll i gruppe-CV, utelatt i tidsfolden; 2023 er benchmarknivå, ikke trendestimat | Brukerbesluttet |
| B-13 | Bonus krever dokumentert as-of-dato før OOF-seleksjon; ellers bare potensielt lekkende sensitivitet | Brukerbesluttet adgangsregel / Datadrevet resultat |
| B-14 | Merke-pooling ≥500 eksponeringsår lært på train_pool (bare eksponering); sensitivitet med 250 og 1 000 | Brukerbesluttet |
| B-15 | Nullskadeår inngår i frekvens, ikke severity; 9 nullkostnadsskader beholdes i frekvens/ren premie og utelates fra Gamma-severity | Brukerbesluttet |
| B-16 | Gamma med log-link og vekt N; lognormal med smearing som utfordrer | Foreslått / Datadrevet |
| B-17 | Storskadetersklene 5 000, 7 500 og 10 000 fastsatt på forhånd; kapping på snittskaden | Foreslått |
| B-18 | Beslutningsregel for separat storskadebehandling | Foreslått / Datadrevet |
| B-19 | Tweedie-p fra EQL-profil i treningsfolden | Foreslått |
| B-20 | Premie brukes bare som benchmark i fase 4, aldri som prediktor | Brukerbesluttet |
| B-21 | Cluster-robuste standardfeil på `insured_id` | Foreslått |
| B-22 | Prediksjon ved e=1 og årsnivå 2023 som benchmarknivå, ikke fremtidig trendestimat | Brukerbesluttet |
| B-23 | NB velges bare ved bedre OOF-score på middelverdien | Foreslått |
| B-24 | Basisnivå for kategoriske variabler = nivået med størst eksponering (f.eks. `COMP_E`) | Foreslått |
| B-25 | Tidsfolden bruker spesifikasjoner uten `C(year)`; nivåskiftet vises som global balanse | Foreslått |
| B-26 | Alle responser modelleres som rate med `var_weights`; for frekvens identisk med antall + log(e)-offset (kontrollert numerisk) | Foreslått |

## Verifikasjon (per leveranse)
1. `uv run jupytext --sync glm_pricing_models.py`, og for leveranse 0 også `analysis.py`.
2. `uv run jupyter nbconvert --to notebook --execute --inplace glm_pricing_models.ipynb --ExecutePreprocessor.timeout=1800`, og tilsvarende for `analysis.ipynb` i leveranse 0.
3. Leveranse 0: analysistallene er identiske før og etter refaktoreringen (seksjon 11/12/17-tabellene og asserts). Foldene er disjunkte. Nullmodellen reproduserer porteføljefrekvensen 0,267 og ren premie ≈232.
4. Sunnhetssjekker: global balanse ≈1 for Poisson med log-link på treningsdata. D² > 0 mot nullmodellen. Relativitetene har rimelig retning; for eksempel skal COMP_N ha høyere frekvens og bonus B > N > G. OOF-scorene er ikke bedre enn in-sample.
5. Resultatene sammenfattes kort med antakelser og begrensninger. Deretter spørres det om commit (`checkpoint:`). Neste leveranse starter med revurdering av planen.

## Endringslogg

| Dato | Fase | Endring | Begrunnelse |
|---|---|---|---|
| 2026-09-15 | Planlegging | Første versjon | Brukerens valg: behold kansellerte, tellemodell for frekvens, datagrunnlag i `src/`, fase for fase |
| 2026-09-15 | Leveranse 0 | La til B-24 (basisnivå), B-25 (tidsfold uten årsledd) og B-26 (rate med vekt). Referansemodeller for alle tre målvariabler, med `paired_improvement` for B-08, er lagt inn allerede i seksjon 2.9. `src/glm_diagnostics.py` startet med `build_fold_summary` og `summarize_cv_scores`. | Implementeringsvalg som oppsto underveis: `C(year)` kan ikke predikere et usett år, og én felles CV-sløyfe krever én responsform. **Til revurdering før fase 1:** årsleddet for severity passerte 1-SE med forbedring i bare 3 av 5 folder. Vurder å skjerpe B-08, f.eks. forbedring i minst 4 av 5 folder. Tidsfolden viser motsatt drift (frekvens −17 %, severity +10 %), noe som påvirker B-22. |
| 2026-09-16 | Leveranse 0 (opprydding) | `to_model_frame` (dtype-konvertering) flyttet til `src/model_data.py`, `build_data_overview`, `build_glm_summary` og `build_relativity_table` lagt i `src/glm_diagnostics.py`. CV og referansemodeller bygger nå på `glm_spec` → `fit_glm` → `run_glm`/`cross_validate_glm`, der formelen bygges fra dtypes én gang per modell. B-24 brukes også på `year` (basis 2023). Ingen sklearn-`Pipeline`. | Notebooken skal fokusere på modellene. Pipeline ble forkastet fordi statsmodels/patsy-formlene og koeffisientene er det notebooken skal vise, og `prepare_design_frame` allerede lærer alt på treningsfolden. OOF-scorene er uendret. |
| 2026-09-16 | Leveranse 0 (opprydding) | Nullmodellene estimeres ikke lenger som GLM. Nullnivået vises som `oof_null_deviance` (vektet treningssnitt per fold), og parvis sammenligning nullmodell → `policy_type` bruker `val_null_deviance`. | En intercept-GLM med log-link gir nøyaktig det vektede snittet, så modellfittene var redundante. Tallene er uendret (f.eks. frekvens: forbedring 0,10266, SE 0,00527). |
| 2026-09-16 | Revurdering før fase 1 | Låste B-02, B-05–B-08, B-10–B-15, B-20 og B-22. Skjerpet seleksjonen til positiv pooled OOF-gevinst, >1 SE og minst 4/5 forbedrede folder; innførte typepasset stabilitet og enkleste modell innen 1 SE. Bonus fikk en as-of-port. | Referansemodellene viste at 1-SE alene kunne passeres med bare 3/5 forbedrede folder og at kalenderdrift må skilles fra variabelseleksjon. Brukerbeslutningene er samlet i notebookens seksjon 2.10. Planen for fase 1 beholdes ellers uendret. |

## Utenfor omfang nå
GBM og GAM, credibility for merke, ekstremverditeori (GPD) utover mean excess-plottet, dispersjonsmodellering (DGLM), interaksjoner i mestermodellen og all evaluering på 2024.

## Litteratur (referanseliste i notebookens seksjon 2)
Gneiting (2011, JASA); Wüthrich & Merz (2023, Springer); Fissler, Lorentzen & Mayer (2023, arXiv:2202.12780); Wüthrich (2023, Eur. Actuar. J., Gini under autokalibrering); Frees, Meyers & Cummings (2011, JASA); Goldburd, Khare, Tevet & Guller (2020, CAS Monograph 5); Ohlsson & Johansson (2010, Springer); Noll, Salzmann & Wüthrich (2018, SSRN 3164764); Delong, Lindholm & Wüthrich (2021, Eur. Actuar. J.); Smyth & Jørgensen (2002, ASTIN Bull.); Jørgensen & de Souza (1994, Scand. Actuar. J.); Denuit, Charpentier & Trufin (2021, IME); Kleiber & Zeileis (2016, Am. Stat.); Cameron & Trivedi (1990, J. Econometrics); Cameron & Miller (2015, J. Human Resources); Roberts et al. (2017, Ecography); Hastie, Tibshirani & Friedman (2009); Harrell (2015); Duan (1983, JASA); Embrechts, Klüppelberg & Mikosch (1997).
