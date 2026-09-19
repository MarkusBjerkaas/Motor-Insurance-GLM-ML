# Plan: enkel frekvensmodell

## Status

Frekvensfasen er forenklet 2026-09-18. Arbeidet fokuserer bare på en
presentasjonsvennlig Poisson-GLM for egen skade. Severity, ren premie,
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

Fem `GroupKFold`-folder på `insured_id` brukes for alle sammenligninger.

1. `kjernevariabler`: produkt, år, føreralder, log bilverdi, ytelse,
   drivstoff, område, kommunetype, betalingsfrekvens og privat/næring.
2. Funksjonsform: én naturlig spline (df=3) utfordrer om gangen den lineære
   formen for alder, bilverdi og ytelse.
3. Kjøretøyopplysninger: bilmerke og antall seter utfordrer hver for seg den
   valgte additive modellen.
4. `kjerne_med_produkt_alder_interaksjon`: produkt × alder legges direkte til
   den valgte additive GLM-formelen.
5. Ablasjon: én ikke-beskyttet variabel fjernes om gangen fra den valgte
   modellen. Runden gjentas bare når en fjerning kvalifiserer.

Primærscore er eksponeringsvektet OOF Poisson-deviance. En utvidelse velges
bare når alle tre kriterier er oppfylt:

1. Lavere samlet OOF-deviance enn foreldremodellen.
2. Lavere deviance i minst fire av fem folder.
3. Gyldig tilpasning i alle fem folder.

Interaksjonen er bare én ordinær modellspesifikasjon og får ingen egne
diagnostikkplott eller script. Dette er en bevisst enkel presentasjonsregel,
ikke en uttømmende søkeprosess.

I ablasjonen er produkt og år beskyttet. Føreralder er også beskyttet dersom
produkt × alder-interaksjonen er valgt, siden et interaksjonsledd uten
hovedleddet ikke er meningsfullt.

## Beslutningstabell

Notebooken avsluttes med valgt modell og de to nærmeste konkurrentene målt på
OOF-deviance. Tabellen viser OOF-deviance, OOF D², OOF-balanse, antall
parametere og en kompakt oversikt over koeffisientene fra tilpasning på hele
utviklingssettet. OOF-metrikkene styrer valget; koeffisientene er forklaring.

## Hva som er fjernet

- Tidsfolden 2022 → 2023.
- Stigen F0–F7, spline-grid, ablasjon, NB2, ABESS og sensitiviteter.
- Faktorvise seleksjonsregler, fase-1-cache og tekniske
  rammeverkskontroller.
- De tilhørende `src/phase_2`-skriptene og alle interaksjonsplott.

Tidsfolden påvirket tidligere modellvalget ved å forkaste en
interaksjonsmodell. Den er derfor ikke «unødvendig» i den gamle analysen, men
er irrelevant etter at den komplekse kandidatstigen er erstattet av denne
forhåndsdefinerte sammenligningen.

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
