# Plan: enkel frekvensmodell

## Status

Frekvensfasen er forenklet 2026-09-18 og fikk 2026-09-19 den felles
seleksjonsalgoritmen fra `04_tweedie.py` (forover-/bakoverseleksjon med
firleddsregel). Arbeidet fokuserer bare på en presentasjonsvennlig Poisson-GLM
for egen skade. Severity, ren premie,
fordelingsutfordrere og sensitiviteter tas opp først ved en senere,
eksplisitt faseovergang.

## Modell og data

- Kun kaskoproduktene `COMP_E` og `COMP_N`.
- Kun utviklingsårene 2022–2023. `build_development_frames` filtrerer hver
  data-bit før den materialiseres; 2024 leses derfor aldri i frekvensløpet.
- Respons: `claim_frequency = property_claims / total_exposure`.
- Estimering: Poisson-GLM med log-link og `total_exposure` som vekt.
- Manglende tallverdier erstattes med treningsfoldens median; manglende
  kategorier får nivået `MISSING`.

$$
N_i \sim \operatorname{Poisson}(e_i\lambda_i),
\qquad \log\lambda_i = \beta_0 + \sum_j\beta_j x_{ij}.
$$

## Spesifikasjoner og valg

Fem `GroupKFold`-folder på `insured_id` (seed 100) brukes for alle sammenligninger.
Seleksjonsalgoritmen og notebook-strukturen er identiske med `04_tweedie.py`
(`tweedie_plan.md`, T-09 og T-10); bare responsen (Poisson, $p=1$, vekt
`total_exposure`) og kandidatregisteret er frekvensspesifikke.

**Kandidatregister (låst 2026-09-19):**

- `LOCKED`: `policy_type`, `year` (aldri fjernet).
- `CORE`: `log_vehicle_value`, `circulation_area`, `payment_frequency`, `driver_age` (alle lineære).
- Funksjonsformer: alder som naturlig spline df 2/3/4, bilverdi df 3.
- Geografi: `municipality_type` i tillegg til eller i stedet for `circulation_area`.
- `ADDITIONAL`: `performance_hp_per_tonne`, `fuel_type`, `business_type`,
  `vehicle_brand_pooled`, `seat_category` (`payment_frequency` er flyttet til kjernen).
- `INTERACTIONS`: alder × `performance_hp_per_tonne`, alder × `policy_type`,
  alder × `log_vehicle_value`.

**Prosedyre (notebook §4):** K0 (låste + kjerne) → alder → bilverdi → geografi →
tillegg (forover, flere) → interaksjoner (forover) → ablasjon (bakover). Hvert trinn
kjører kandidatene parallelt (joblib) og velger med firleddsregelen under.

Primærscore er eksponeringsvektet OOF Poisson-deviance. En kandidat erstatter
foreldremodellen bare når alle fire kriterier er oppfylt:

1. Lavere samlet OOF-deviance enn foreldremodellen.
2. Lavere deviance i minst fire av fem folder.
3. Gyldig tilpasning i alle fem folder.
4. Snittet av fold-gevinstene er større enn én standardfeil (`SE_MULTIPLIER = 1.0`).

Interaksjoner testes bare når begge hovedeffektene er med, og en hovedeffekt kan
ikke fjernes mens en interaksjon bruker den. Dette er en bevisst enkel
presentasjonsregel, ikke en uttømmende søkeprosess.

**Resultat (2026-09-19):** K0 + alder (spline df 4) + `business_type`; 12 parametere,
OOF-deviance 1,1222. Forover-steg: `+alder_df4` (1,1269, gevinst/SE 3,01), deretter
`+business_type` (1,1222, 5,92). Ikke valgt: bilverdi-spline, `municipality_type`,
`performance_hp_per_tonne`, `seat_category`, `vehicle_brand_pooled` og alle
interaksjoner; `fuel_type` (1,1216) består ikke 4/5-regelen. Ablasjonen fjerner
ingenting. Interaksjonen alder × `policy_type` gir rangdefekt i én fold og markeres
ugyldig (se `tweedie_plan.md`, åpent punkt).

## Beslutningstabell

Notebooken rapporterer valget med (§4.8) en forover-seleksjonstabell med
OOF-deviance, snittgevinst, standardfeil og gevinst/SE, (§4.9) de tre modellene med
lavest pooled OOF-deviance blant alle evaluerte (endring mot valgt, antall
parametere, delta) og (§4.10) statsmodels-`summary()` for den valgte modellen
estimert på alle utviklingsdata med cluster-robuste SE på `insured_id`. Toppmodellene
er rangert på samme folder som seleksjonen og er en utviklingsscore. OOF-metrikkene
styrer valget; koeffisientene er forklaring.

## Hva som er fjernet

- Tidsfolden 2022 → 2023.
- Stigen F0–F7, spline-grid, NB2, ABESS og sensitiviteter.
- Den gamle stigen (kjernevariabler → enkeltvise funksjonsformer → kjøretøy →
  produkt × alder → ablasjon), `summarize_candidate` og den manuelle
  sammenligningen i §9. Erstattet av den felles algoritmen (T-09/T-10).
- Fase-1-cache og tekniske rammeverkskontroller; de tilhørende
  `src/phase_2`-skriptene og alle interaksjonsplott.

Beholdt og koblet til den nye sluttmodellen: koeffisienttabeller (§5),
koeffisientstabilitet mot de to nærmeste konkurrentene (§6), CatBoost-residualdiagnostikk
(§7), OOF Poisson-residualplott (§8) og rootogram (§9).

## Begrensninger

OOF-resultatene er utviklingsresultater og kan være optimistiske. De viser
prediktiv assosiasjon, ikke kausale effekter. Modellspesifikasjonen låses før
en senere, separat test på 2024.

## Endringslogg

| Dato | Endring |
|---|---|
| 2026-09-18 | Fase 1 skrevet om til en kort CV-stige: kjerne, splines, kjøretøyopplysninger, én direkte spesifisert produkt × alder-interaksjon og iterativ ablasjon. Tids-CV, individuelle seleksjonsregler og diagnoseplott er fjernet. Avsluttes med en beslutningstabell for de tre nærmeste modellene. |
| 2026-09-19 | Faserevisjon før Tweedie-implementering: metoden er uendret. Foldene bygges nå med den felles `build_group_folds` (samme `GroupKFold`, seed 100 og gruppekolonne) og sanity-sjekker ligger i `src_asserts/`. Frekvensnotebooken er kjørt på nytt; output er identisk med før (bare tidsstempel er forskjellig). |
| 2026-09-19 | CatBoost-residualdiagnostikk lagt til som §7 i frekvensnotebooken (se `plans/glm_catboost_residual_diagnostics_plan.md`). Ren diagnose, ikke seleksjonsport: ingen stabil gevinst over GLM-en (pooled OOF-deviance 1,1217 mot 1,1219/1,1227), så F5 og beslutningsregisteret er uendret. Seksjonene etter er renummerert til 8–11. |
| 2026-09-19 | Tweedie-notebooken har fått en forover-/bakover-seleksjon som tillater flere tillegg, interaksjoner og ablasjon (se `tweedie_plan.md`, T-09). Metoden i denne planen (frekvens) er uendret. Portering av algoritmen hit er planlagt først etter at Tweedie fungerer og etter egen godkjenning; planen skrives da om. `add_seat_category` er flyttet til `src_core_glm/model_data.py` og delt med severity. |
| 2026-09-19 | Portering av seleksjonsalgoritmen fra `04_tweedie.py` (siste steg). Ny kjerne (`log_vehicle_value`, `circulation_area`, `payment_frequency`, `driver_age`), `payment_frequency` fjernet fra tilleggene, samme splines, geografitest, tillegg og interaksjoner som Tweedie. Firleddsregel med `SE_MULTIPLIER = 1.0`, parallell evaluering, forover-seleksjonstabell med SE, topp-3 og `summary()`. Modellagnostisk kode ligger i `src_core_glm/model_selection.py` og `selection_report.py`. Valgt modell: K0 + alder (spline df 4) + `business_type`. Diagnostikk (§5–§9) er beholdt og koblet til den nye sluttmodellen; gamle «Tolkning»-avsnitt er fjernet fordi tallene er endret. Metoden i planen er skrevet om (dette er den vesentlige endringen); 2024 er urørt. |
| 2026-09-19 | Klargjøring før 2024-testen (metoden i planen er uendret). Hver modellnotebook refitter nå sin endelige modell på hele 2022–2023 og lagrer den i `models/` via `src_model_comparison/locked_models.py`, med kontroll mot notebookens egen `predict`. Variabelsjekk (`variable_coverage.py`) sikrer at alle ikke-utelatte variabler er testet i hver modell. Lekkasjekontroll fant at `year` er en låst kategorisk term som 2024 ikke kan skåres på: alle modeller skåres derfor med 2023-nivået (`SCORE_YEAR`), og numeriske prediktorer klippes til treningsområdet. Årsdrift blir liggende i porteføljebalansen. `clean_motor_data` feilet uten 2022-rader (rettet), og skåringskjeden er tørrkjørt på 2023. Ingen 2024-data er lest. |
