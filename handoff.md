# Handoff — Severity-fasen (Gamma GLM), `glm_pricing_severity`

Skrevet 2026-09-18. Mottaker: ny agent (Codex) som skal ta over uten tilgang til
forrige samtale. Alt som trengs for å fortsette står her.

---

## 1. Oppdraget og arbeidsmodellen

Implementer `plans/Severity_plan.md` — en **låst protokoll** for en
Gamma-severity-benchmark på spansk motorforsikringsdata. Brukerens instruks:

- Resultatene presenteres i notebooken `glm_pricing_severity.ipynb`.
- **Rediger aldri `.ipynb` direkte.** All redigering skjer i speilet
  `glm_pricing_severity.py` (jupytext percent-format), som konverteres med
  `uv run jupytext --sync glm_pricing_severity.py`.
- Agenten fungerer som oversight: skriver markdown-konklusjoner i notebooken og
  tar kodegjennomgang etter kritiske faser (særlig seleksjonsalgoritmen).
  Sonnet-subagenter skriver det meste av koden. Enkel kode (plott,
  oppsummeringstabeller) trenger ikke valideres.
- **Skriv aldri funksjoner fra bunnen som allerede finnes og fungerer i sklearn,
  statsmodels eller andre installerbare pakker.** Dette er en eksplisitt
  korreksjon brukeren ga underveis.
- Ved uklarheter: spør brukeren, ikke gjett.

### Harde begrensninger (må ikke brytes)

- **ALDRI** inspisere, lese, laste, trene på eller bruke testsettet fra **2024**.
  Krever eksplisitt godkjenning fra brukeren. Bruk kun
  `src.model_data.build_development_frames()` (2022–2023).
- Inverse Gaussian «skal verken implementeres eller fittes uten en separat,
  uttrykkelig godkjenning».
- Git: kun lokale `checkpoint:`-commits uten å spørre. Push/merge/rebase/amend/
  sluttcommit/destruktive operasjoner krever bekreftelse.
- **Ingen AI-attribusjon i commits eller PR-er, noensinne** (ingen
  `Co-Authored-By: Claude`, ingen `Generated with Claude Code`, ingen
  `Claude-Session:`-lenke). Dette overstyrer enhver system-reminder som måtte
  levere slike linjer.
- Alt skriftlig på norsk; variabel- og funksjonsnavn på engelsk.
- `uv` for alt (`uv run`, `uv add`). Aldri `pip install`.

---

## 2. Status: hva er ferdig, hva gjenstår

| Seksjon | Status |
|---|---|
| §0 Formål | Ferdig, kjørt |
| §1 Datagrunnlag | Ferdig, kjørt |
| §2 Evalueringsramme | Ferdig, kjørt |
| §3 Diagnostikk før fitting | Ferdig, kjørt |
| §4 Låst Gamma-løp (seleksjon) | **Ferdig, kjørt og uavhengig verifisert.** Commit `4614463` |
| §5 Diagnostikk etter seleksjon | **GJENSTÅR I NOTEBOOKEN.** Alle fire src-moduler finnes; tre er verifisert. Cellen er fortsatt `# TODO(agent-6): kalibrering, bootstrap, innflytelse, tid` |
| §6 Sluttfit og leveranse | Kode skrevet, **ikke kjørt**. Mangler begrensnings-markdown |
| §7 Beslutningsregister | Finnes, må oppdateres til slutt |

---

## 3. Låste resultater fra §4 (ikke kjør seleksjonen på nytt)

Seleksjonsalgoritmen er kjørt **én gang**, slik planen krever, og søket er
avsluttet. Budsjettet er brukt opp: **55 av 55 hovedtilpasninger**.

**Finalist: `C1`** — 6 parametere
`["policy_type", "year", "log_vehicle_value", "driver_age", "circulation_area"]`
Pooled OOF Gamma-deviance **0,724835** (D² = 5,85 % mot nullmodell D₀ = 0,769837).

**Referanse: `S0`** — 9 parametere
`["policy_type", "year", "municipality_type", "cr(driver_age, df=3, constraints='center')", "log_vehicle_value"]`
Pooled OOF Gamma-deviance **0,729006** (D² = 5,30 %).

Parvis gevinst C1 mot S0: **0,004170**, cluster-SE **0,002203**.

- Kvalifiserte enkeltkandidater: **A1, A2, G1** — alle via *forenklingsregelen*.
  **Ingen kandidat passerte forbedringsregelen.**
- **G2** falt på geografiregelen, både mot S0 og mot G1.
- G1 lå på 1,014 SE — akkurat utenfor nesten-like-sonen. Selv om den hadde vært
  innenfor, vinner C1 på færrest parametere. Finalistvalget er robust.

### Låste konstanter (må gjenbrukes ordrett)

```python
GAMMA_POWER  = 2
GAMMA_FAMILY = sm.families.Gamma(sm.families.links.Log())   # MERK: log-link!
FIT_SETTINGS = {"maxiter": 200, "tol": 1e-8}
BASE_LEVELS  = {"policy_type": "COMP_E", "year": 2022, "municipality_type": "I",
                "circulation_area": "U", "seat_category": "=5"}
```

Modell: Gamma GLM med log-link, respons `average_severity =
property_incurred / property_claims`, `var_weights = property_claims`,
**ingen eksponerings-offset**. Folder: `GroupKFold(n_splits=5, shuffle=True,
random_state=100)` på `insured_id` over hele modellpopulasjonen (55 246 rader).

### Kontrolltall — bruk disse som regresjonstest

| Størrelse | Verdi |
|---|---|
| Rader i severity-utvalget | **5 698** |
| Unike `insured_id` | **5 323** |
| Sum `property_claims` | **9 979** |
| Skadevektet severity | 872,645 EUR |
| Pooled deviance S0 | **0,729006** |
| Pooled deviance C1 | **0,724835** |

Reproduserer du ikke disse, er oppsettet feil — **juster ikke tallene**, finn
feilen.

---

## 4. Kritiske fallgruver (alle observert i praksis denne økten)

### 4.1 Gamma-link-fellen — den dyreste feilen så langt

`sm.families.Gamma()` har **invers link** som standard, ikke log.
`src_severity/severity_scoring.py` definerer:

```python
GAMMA_FAMILY = sm.families.Gamma()   # INVERS link
```

Denne er **kun trygg til `resid_dev`** (deviance-residualer er
link-uavhengige), og brukes riktig slik i `severity_calibration.py` og
`severity_time.py`. Men bruker du den til **fitting**, får du stille gale tall:
i en verifikasjonskjøring ga den D(S0) = 0,729861 og D(C1) = 0,725137 i stedet
for 0,729006 / 0,724835 — nær nok til å se riktig ut, galt nok til å ødelegge
alt. **Alltid `sm.families.Gamma(sm.families.links.Log())` ved fitting.**

### 4.2 `build_numeric_correlation_table` returnerer NaN ved NaN i input

`log_vehicle_value` har 3 NaN i `severity_frame` (imputeres først i
`prepare_design_frame`). `np.cov` forplanter NaN gjennom hele matrisen.
Kall funksjonen på `severity_frame.dropna(subset=CORR_COLUMNS)`, eller legg
dropna inn i funksjonen. Verifisert riktig verdi etter dropna:
korr(driver_age, log_vehicle_value) = **−0,012917**.

### 4.3 `build_segment_ae_table` tar A/E per modell, ikke hele tabellen

```python
total_ae = level_shift.set_index("modell")["A_E"]        # riktig
segment_ae = build_segment_ae_table(frame_2023, predictions, total_ae)
```
Sender du hele `level_shift`-DataFrame-en får du `KeyError: 'S0'`.

### 4.4 `jupytext --sync` gir falsk `FileNotFoundError`

`uv run jupytext --sync glm_pricing_severity.py` kan skrive
`FileNotFoundError: The file glm_pricing_severity.ipynb does not exist` selv når
filen finnes og synkingen lykkes. **Verifiser ved å grepe begge filene for nytt
innhold** i stedet for å stole på exit-meldingen. Bruk aldri
`jupytext --to ipynb --output ...`.

### 4.5 Scratch-scripts trenger PYTHONPATH

```bash
PYTHONPATH=/workspaces/MotorForsikring uv run python <script>
```
Legg engangsscripts i scratchpad eller `src_temp/`, aldri i `src_severity/`.

### 4.6 Duplisering mellom `src/glm_core.py` og `glm_pricing_models.py`

`glm_pricing_models.py` definerer fortsatt sine egne `prepare_design_frame`
(≈linje 436), `glm_spec` (569), `fit_glm` (661) og `cross_validate_glm` (748),
mens `src/glm_core.py` nå har utvidede kopier (med `fit_kwargs` og
`base_level_overrides`). Dette følger brukerens tidligere beslutning («Flytt til
`src/glm_core.py`, la frekvens stå»), men kopiene har nå **divergert** og kan
drifte videre. **Dette er ikke rapportert til brukeren ennå — gjør det.**

Endringene i `src/model_data.py` er rent additive (0 slettede linjer), så
`glm_pricing_models.ipynb` er upåvirket.

---

## 5. Modulstatus i `src_severity/`

| Fil | Status |
|---|---|
| `severity_data.py`, `severity_folds.py`, `severity_cv.py`, `severity_scoring.py`, `severity_selection.py`, `selection_tests.py`, `severity_descriptives.py`, `severity_design_checks.py` | Ferdig, brukt i §1–§4, verifisert |
| `severity_calibration.py` | Ferdig. **Verifisert av meg** (52/52 sjekker) |
| `severity_time.py` | Ferdig. **Verifisert av meg**, og refaktorert (se §5.1) |
| `severity_influence.py` | Ferdig. Verifisert av subagenten, som reproduserte 0,729006 / 0,724835 eksakt |
| `severity_bootstrap.py` | **Skrevet, men ALDRI VERIFISERT** — se §7.1 |

Alle filer er `ruff`-rene (`uv run ruff check src_severity/` — grønn).

### 5.1 Refaktorering utført i `severity_time.py` (ukommitert)

Modulen hadde kopiert den låste spesifikasjonen (`S0_TERMS`, `BASE_LEVELS`,
`FIT_SETTINGS`) inn som modulkonstanter. Det bryter med prosjektets regel om at
modellspesifikasjonen skal ligge i notebooken, og kunne drifte stille fra den.
Konstantene er fjernet; `run_year_control` og `run_mix_standardization` tar nå:

```python
run_year_control(candidate_terms, reference_terms, base_levels, severity_frame,
                 fit_kwargs, prepare_fold_frames, prepare_design_frame)
run_mix_standardization(candidate_terms, reference_terms, base_levels, severity_frame,
                        fit_kwargs, prepare_fold_frames, prepare_design_frame)
```

**Bekreftet at alle tall er uendret etter refaktoreringen.** `FIT_FAMILY`,
`GAMMA_POWER` og `CANDIDATE_BLOCK_COLUMNS` er beholdt som modulkonstanter.
`glm_spec` ignorerer ukjente nøkler i `base_level_overrides`, så notebookens
`BASE_LEVELS` (som inkluderer `year`) kan sendes rett inn selv om tidskontrollen
kjører uten årsledd.

`severity_time.py` inneholder også `compute_oof_residuals` og
`plot_oof_residuals`, som ikke var i den opprinnelige bestillingen, men som er
legitim residualdiagnostikk og gjenbruker `GAMMA_FAMILY.resid_dev`. Behold dem.

---

## 6. Verifiserte §5-resultater (klare til å skrives inn i notebooken)

### 6.1 Kalibrering (`severity_calibration.py`) — verifisert

- **A/E totalt:** S0 = 1,000583, C1 = 1,000254. Begge svært nær 1.
- **Desiler** (referansedesiler fra S0, skadeantallsvektet, 996–1000 skader
  hver, monotone i predikert severity):

| Desil | Skadeantall | Predikert | Faktisk | A/E |
|---|---|---|---|---|
| 1 | 997 | 599,41 | 620,01 | 1,034 |
| 2 | 997 | 650,38 | 645,50 | 0,993 |
| 3 | 996 | 685,31 | 678,90 | 0,991 |
| 4 | 999 | 732,64 | 755,83 | 1,032 |
| 5 | 1000 | 843,16 | 923,36 | 1,095 |
| 6 | 998 | 943,73 | 954,27 | 1,011 |
| 7 | 998 | 990,23 | 951,06 | 0,960 |
| 8 | 998 | 1030,53 | 1078,63 | 1,047 |
| 9 | 998 | 1076,65 | 985,21 | 0,915 |
| 10 | 998 | 1168,66 | 1132,82 | 0,969 |

Alle desiler ligger innenfor 0,80–1,20. Ingen systematisk skjevhet i endene.

- **Deviance-bidrag per beløpsklasse (C1):**

| Beløpsklasse | Skadeantall | Deviance-andel | Kostnadsandel | EUR-feil |
|---|---|---|---|---|
| ≤1 | 1 | 0,0028 | 0,0000 | −861 |
| (1, 10] | 2 | 0,0024 | 0,0000 | −2 067 |
| (10, 100] | 276 | 0,1666 | 0,0015 | −253 570 |
| (100, 1000] | 7 660 | 0,3808 | 0,4615 | −2 476 163 |
| (1000, 5000] | 1 899 | 0,2110 | 0,3947 | +1 636 361 |
| >5000 | 141 | 0,2364 | 0,1423 | +1 098 510 |

**Poeng verdt en markdown-konklusjon:** tre skadeår under 10 EUR bidrar med
0,5 % av deviancen og 0,00 % av kostnaden — Gamma-deviancen straffer små beløp
hardt uten at de betyr noe i EUR. Motsatt bærer 141 skadeår over 5 000 EUR
14 % av kostnaden og 24 % av deviancen.

- **Foldstabilitet (C1, gjennomsnitt over 5 folder / relativ spredning):**
  Intercept 4,9394 / 0,207 · `policy_type[COMP_N]` −0,3998 / 0,090 ·
  `year[2023]` −0,0846 / 0,552 · `circulation_area[R]` 0,1040 / 0,598 ·
  `log_vehicle_value` 0,1986 / 0,499. Produkteffekten er klart mest stabil;
  geografi og år beveger seg mest mellom folder.

### 6.2 Tidskontroll 2022 → 2023 (`severity_time.py`) — verifisert

Begge modeller fittet **uten årsledd**, kun på 2022 (1 722 rader), evaluert på
2023 (3 976 rader). All preprocessing læres kun fra 2022.

- Pooled deviance på 2023: **S0 = 0,708886** (8 param), **C1 = 0,709531**
  (5 param). Praktisk talt identisk; C1 taper marginalt.
- **Tidsstoppregelen utløses IKKE.** Tap = 0,000645 < både én cluster-SE
  (0,003315) og 0,5 % av D_S0 (0,003544). Begge betingelser må være oppfylt.
  **Obligatorisk formulering i notebooken: «ingen forverring påvist i denne
  tidsdelingen»** — dette er *ikke* dokumentert tidsstabilitet.
- Nivåskift: faktisk skadevektet severity 939,78 (2022) → 846,17 (2023),
  forhold **0,9004**. A/E på 2023: S0 = 0,9153, C1 = 0,9163 — begge
  overpredikerer 2023, konsistent med det generelle nivåfallet.
- Miksstandardisering (kun diagnostisk): forhold 2023-fit/2022-fit på samme
  referansepopulasjon: S0 = 0,9168, C1 = 0,9172. Nesten identisk mellom
  modellene ⇒ nivåskiftet er en egenskap ved året, ikke ved modellvalget.
- Dekning/ekstrapolasjon: 0–0,12 % av referansepopulasjonen utenfor
  treningsårets område. Ingen praktisk bekymring.
- 375 personer opptrer i severity-utvalget i begge år (rapportert, ikke filtrert).

> **⚠️ ÅPENT FAGLIG SPØRSMÅL — krever brukerens beslutning, se §7.2**
> Segmentdrift-kriteriet i stoppregisteret **treffer**: `skadeantallsgruppe`
> `N=1` har A/E-relativ ≈ **1,33** og `N>=4` ≈ **0,74** — over 20 %-grensen, i
> **begge** modeller.

### 6.3 Innflytelse (`severity_influence.py`) — verifisert av subagent

Planens eneste innflytelsessensitivitet: de fem `insured_id` med høyest
`sum(property_incurred)` fjernes fra hver **treningsfold**; S0 og C1 refittes
med uendret spesifikasjon og predikerer den **uendrede** valideringsfolden.

- Nøyaktig **10 diagnostiske refittinger**, alle konvergerte.
- **7 unike personer** fjernet over de fem foldene (av maks 25 slots) — de
  største kostnadsdriverne går igjen i flere folders treningsdel.
- De fem utgjør **1,67–1,76 %** av foldens treningskostnad.
- Relativ endring i pooled forventet kostnad totalt: **S0 −1,57 %**,
  **C1 −1,58 %**. Største produktendring: `COMP_E` (S0 −1,69 %, C1 −1,72 %).
- **Ingen stoppregler utløses** (5 % totalt / 10 % per produkt).
- Rangeringen er uendret: C1 slår fortsatt S0 (0,7247 mot 0,7286 i
  sensitiviteten). Finalistvalget er robust.

---

## 7. Hva gjenstår — prioritert

### 7.1 Verifiser `severity_bootstrap.py` (HØYEST PRIORITET)

Filen ble skrevet (13 029 bytes, ruff-ren), men agenten døde på ukesgrensen
**før verifikasjonen ble lest**. **Ingen tall fra denne modulen er bekreftet.**

Funksjoner som skal finnes: `bootstrap_deviance_ci`, `bootstrap_gain_ci`,
`bootstrap_ae_ci`, `evaluate_stop_criteria`.

Planens krav: 2 000 cluster-bootstrap-trekk på `insured_id`, **seed 410**,
**faste OOF-prediksjoner**, **samme trekk for begge modeller**,
percentile-intervaller på 95 %. **Bootstrapen refitter ikke modeller og
gjentar ikke seleksjonen.** Bruk
`severity_scoring.cluster_bootstrap_ci` (tynn adapter over
`scipy.stats.bootstrap`, `method="percentile"`); samme seed + samme
`clusters`-array gir identiske trekninger på tvers av kall, så «samme trekk»
oppfylles automatisk — trekningene skal ikke lagres eller sendes rundt manuelt.

Verifiser minst:
1. Utvalget: 5 698 / 5 323 / 9 979.
2. Punktestimatene er lik de direkte beregnede (0,729006 / 0,724835; gevinst
   0,004170, SE 0,002203).
3. **Determinisme:** to kall med samme seed gir identiske intervaller.
4. **Delte trekk:** intervallet for den *parvise* gevinsten er vesentlig
   smalere enn differansen mellom de to modellenes uavhengige
   deviance-intervaller. Dette er beviset på at trekningene faktisk deles.
5. Kjøretid (2 000 trekk × mange segmenter kan bli tregt — forhåndsbygg numpy-
   arrays utenfor statistikkfunksjonen; **ikke reduser antall trekk**, 2 000 er
   låst i planen).

De to EUR-stoppreglene (begge betingelser må gjelde samtidig):
- Totalt A/E utenfor 0,90–1,10 **og** intervallet utelukker 1.
- Produkt- eller desil-A/E utenfor 0,80–1,20 med ≥100 personer **og**
  intervallet utelukker 1.

### 7.2 Legg segmentdriften frem for brukeren

Tidskontrollen treffer stoppregisterets «Segmentdrift over tid»-kriterium på
`skadeantallsgruppe` (`N=1` ≈ 1,33, `N>=4` ≈ 0,74, begge modeller). Avviket er
likt for S0 og C1, altså ikke noe C1 gjør galt, men en generell egenskap ved
severity-fordelingen betinget på skadeantall over kalenderårsskiftet.
**Planen krever faglig stopp/avklaring her — spør brukeren før du går videre.**

Merk samtidig planens obligatoriske `N=1`-tolkning som skal inn i §5:
et skadeår med `N=1` er **«et selektert utvalg og ikke nødvendigvis én fysisk
hendelse»**.

### 7.3 Skriv §5 i notebooken

Erstatt `# TODO(agent-6): kalibrering, bootstrap, innflytelse, tid` med celler
for alle fire delene, hver med markdown-konklusjon. Husk:
- Maks **én tabell og ett plott per output-seksjon** (subplots er lov).
- Celler ≤ 25 kodelinjer, én tydelig oppgave per celle.
- `severity_scoring` og `PREDICTORS` må **legges tilbake i importblokken** —
  de ble fjernet av ruff (F401) under en refaktorering av §4.
- Kollinearitets-/tolkningsdiskusjonen skal med (bruk
  `build_numeric_correlation_table`, husk dropna fra §4.2).

### 7.4 Kjør §6 og skriv begrensnings-markdown

§6-koden er skrevet, men ikke kjørt. Den fitter `FINAL_SPECS = {"C1": ..., "S0": ...}`
på hele `severity_frame`, bygger `coefficient_table` med cluster-robuste SE og
relativiteter `exp(β)` med konfidensintervall, og har allerede en
LaTeX-modellikning over seg.

Planen krever at begrensningsavsnittet eksplisitt dekker:
1. Den tilsiktede preferansen for enklere modeller.
2. At **bare én** kombinert C1 er testet — en bedre delkombinasjon kan være
   uoppdaget.
3. Seleksjonsoptimisme fra å gjenbruke de samme CV-observasjonene.
4. At tidskontrollen dekker **én** kalenderovergang: «ingen forverring påvist i
   denne tidsdelingen» ≠ dokumentert tidsstabilitet.
5. Usikker tolkning av korrelerte variabelblokker og svakt støttede segmenter.

### 7.5 Avslutning

- `uv run jupytext --sync glm_pricing_severity.py` + full kjøring av notebooken.
- Oppdater beslutningsregisteret (§7 i notebooken).
- **Logg faserevisjonen i endringsloggen i `plans/Severity_plan.md`** — CLAUDE.md
  krever logging ved hver revurdering, også når ingenting endres.
- Rapporter `glm_core`-dupliseringen fra §4.6 til brukeren.
- Spør om commit.

---

## 8. Nyttige kommandoer

```bash
# Notebook-workflow (aldri rediger .ipynb direkte)
uv run jupytext --sync glm_pricing_severity.py
uv run jupyter nbconvert --to notebook --execute --inplace glm_pricing_severity.ipynb

# Lint
uv run ruff check src_severity/

# Engangsscript
PYTHONPATH=/workspaces/MotorForsikring uv run python <script>
```

## 9. Git-tilstand ved handoff

Siste commit: `4614463 checkpoint: låst Gamma-løp for severity kjørt, finalist C1 valgt`
Gren: `fase1-frekvens`

Ukommitert:
- `glm_pricing_severity.py` (§6 skrevet, §5 fortsatt TODO)
- `src_severity/severity_calibration.py`, `severity_time.py` (+ refaktorering),
  `severity_influence.py`, `severity_bootstrap.py` — alle nye/utrackede
- `src_severity/selection_tests.py` (ryddet ubrukt `pandas`-import)
- `glm_pricing_models.ipynb` / `.py` (kun jupytext-tidsstempel)
