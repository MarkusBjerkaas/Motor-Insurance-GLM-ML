# Motorforsikring: transparent prisingsbenchmark

> **Status:** Arbeid pågår. Dette dokumentet er en mal; felter markert med `[fyll inn]` oppdateres når analyse- og modelleringsløpet er ferdigstilt.

Et porteføljeprosjekt om prising av egen-skade-dekning i spansk motorforsikring. Prosjektet viser et komplett og reproduserbart modelleringsløp: fra datakvalitet og deskriptiv analyse, via en transparent GLM-benchmark, til sammenligning med maskinlæringsmodeller.

Målet er å estimere en risikodifferensiert ren premie og samtidig dokumentere hvordan modellvalgene kan forklares og brukes i underwriting og prising.

## Innhold

- [Problemstilling og avgrensning](#problemstilling-og-avgrensning)
- [Modelleringsløp](#modelleringsløp)
- [Viktigste resultater](#viktigste-resultater)
- [Modelltolkning og prisingsimplikasjoner](#modelltolkning-og-prisingsimplikasjoner)
- [Out-of-sample-evaluering](#out-of-sample-evaluering)
- [Reproduserbarhet](#reproduserbarhet)
- [Filstruktur](#filstruktur)
- [Datakilde og begrensninger](#datakilde-og-begrensninger)

## Problemstilling og avgrensning

**Problemstilling.** Hvordan kan skadefrekvens og skadekostnad kombineres til en robust, forklarbar og konkurransedyktig prisingsmodell for egen-skade-dekning?

**Målsvariabel.** `[fyll inn: definisjon av ren premie / skadebeløp og eksponeringsenhet]`.

**Portefølje og periode.** `[fyll inn: hvilke observasjoner og perioder som inngår i utviklingsdataene]`.

**Avgrensning.** Analysen gjelder `[fyll inn: dekning og relevante porteføljeavgrensninger]`. Premien som modelleres er en ren premie; administrasjonskostnader, reassuranse, fortjeneste og regulatoriske påslag ligger utenfor modellen.

> **Viktig om testsettet:** Kalenderåret 2024 er holdt helt utenfor utforskning, feature engineering, modellvalg og tuning. Det brukes kun én gang til endelig out-of-sample-evaluering etter at alle modellspesifikasjoner er låst.

## Modelleringsløp

```text
Rådata
  → datakvalitet og avgrensning
  → tidsriktig trening/validering
  → frekvensmodell + severitymodell / Tweedie-benchmark
  → kryssvalidering og modellvalg
  → låst spesifikasjon
  → endelig evaluering på 2024
```

| Steg | Hva gjøres | Hvor dokumenteres det? |
|---|---|---|
| 1. Datagrunnlag | Kontrollerer datakvalitet, eksponering, dekning og avgrensninger. | `analysis.ipynb` |
| 2. Deskriptiv analyse | Beskriver portefølje, skadefrekvens og skadebeløp før modellering. | `analysis.ipynb` |
| 3. GLM-benchmark | Estimerer forklarbare modeller for frekvens og severity, samt samlet ren premie. | `glm_pricing_models.ipynb` |
| 4. Modellvalg | Sammenligner forhåndsdefinerte kandidater med gruppebasert, eksponeringsvektet kryssvalidering. | `glm_pricing_models.ipynb` |
| 5. ML-utfordrer | Vurderer om CatBoost gir dokumentert merverdi utover benchmarken. | `ml_pricing.ipynb` |
| 6. Sluttevaluering | Evaluerer den låste spesifikasjonen på det urørte testsettet fra 2024. | `[fyll inn: notebook/rapport]` |

## Viktigste resultater

Denne seksjonen skal oppdateres med et begrenset utvalg figurer som svarer direkte på problemstillingen. Legg eksporterte figurer i `docs/figures/`, og behold kildekoden som produserer dem i notebookene.

### 1. Porteføljen og skadebildet

![Portefølje og skadebilde — erstatt med eksportert figur](docs/figures/portfolio_overview.png)

*Figur 1. `[fyll inn: hva figuren viser, periode, enhet og viktigste observasjon.]`*

### 2. Kalibrering av valgt modell

![Kalibrering — erstatt med eksportert figur](docs/figures/model_calibration.png)

*Figur 2. `[fyll inn: observert mot predikert ren premie per risikogruppe, og kort tolkning.]`*

### 3. Risikodifferensiering og modellforklaring

![Risikodifferensiering — erstatt med eksportert figur](docs/figures/risk_differentiation.png)

*Figur 3. `[fyll inn: relativ premieeffekt / viktigste drivere og hva dette betyr.]`*

### Funn i korte trekk

1. `[fyll inn: viktigste funn om skadebildet.]`
2. `[fyll inn: viktigste funn fra modellvalget.]`
3. `[fyll inn: viktigste funn om kalibrering eller segmenter.]`

## Modelltolkning og prisingsimplikasjoner

### Valgt modell

`[fyll inn: modellnavn, for eksempel Poisson-Gamma GLM eller Tweedie GLM]`

$$
E[Y_i \mid x_i] = [fyll inn: modell-likning]
$$

`[fyll inn: kort, intuitiv forklaring av hvordan frekvens, severity og/eller eksponering inngår.]`

### Tolkning av sentrale drivere

| Variabel | Modellresultat | Praktisk tolkning | Mulig prisingsimplikasjon |
|---|---:|---|---|
| `[fyll inn]` | `[fyll inn: relativ effekt / SHAP / annen effekt]` | `[fyll inn]` | `[fyll inn]` |
| `[fyll inn]` | `[fyll inn: relativ effekt / SHAP / annen effekt]` | `[fyll inn]` | `[fyll inn]` |
| `[fyll inn]` | `[fyll inn: relativ effekt / SHAP / annen effekt]` | `[fyll inn]` | `[fyll inn]` |

### Implikasjoner for forsikringsprising

- **Segmentering:** `[fyll inn: hvilke risikosegmenter som skiller seg ut, og med hvilken sikkerhet.]`
- **Tariff:** `[fyll inn: hvordan modellen kan omsettes til relativpriser eller tariffledd.]`
- **Kalibrering og kontroll:** `[fyll inn: behov for kalibrering, caps/floors eller porteføljekontroll.]`
- **Forbehold:** `[fyll inn: databegrensninger, stabilitet over tid, fairness/regulatoriske hensyn og behov for faglig overstyring.]`

## Out-of-sample-evaluering

Følgende tabell fylles ut først etter at modellspesifikasjonen er låst og 2024-testsettet er evaluert. Lavere verdi er bedre for tapmålene; kalibrering vurderes opp mot 1,00.

| Modell | Spesifikasjon låst dato | Poisson deviance | Gamma/Tweedie deviance | Gini / rank-mål | Kalibrering (observert / predikert) | Vurdering |
|---|---|---:|---:|---:|---:|---|
| GLM-benchmark | `[fyll inn]` | `[fyll inn]` | `[fyll inn]` | `[fyll inn]` | `[fyll inn]` | `[fyll inn]` |
| ML-utfordrer | `[fyll inn]` | `[fyll inn]` | `[fyll inn]` | `[fyll inn]` | `[fyll inn]` | `[fyll inn]` |
| Valgt modell | `[fyll inn]` | `[fyll inn]` | `[fyll inn]` | `[fyll inn]` | `[fyll inn]` | `[fyll inn]` |

**Konklusjon fra sluttevalueringen:** `[fyll inn: hvilken modell velges, hvordan ytelse og forklarbarhet veies, og eventuelle forbehold.]`

## Reproduserbarhet

Prosjektet bruker Python og `uv` for avhengigheter. Fra rotmappen:

```bash
uv sync
uv run jupyter lab
```

Notebookene er paret med tekstfiler i Jupytext-percent-format. Endre alltid den tilhørende `.py`-filen og synkroniser deretter, for eksempel:

```bash
uv run jupytext --sync analysis.py
```

## Filstruktur

```text
.
├── analysis.ipynb / analysis.py       # Dataforståelse, avgrensning og sentral feature engineering
├── glm_pricing_models.ipynb           # GLM-benchmark, CV og beslutningsregister
├── ml_pricing.ipynb / ml_pricing.py   # ML-utfordrer og sammenligning mot benchmark
├── tweedie.ipynb / tweedie.py          # Tweedie-modellering og diagnostikk
├── bonus_score_analysis.ipynb          # Separat diagnostikk av bonus_score
├── src_core_glm/                       # Felles datagrunnlag, GLM-logikk og validering
├── src_frequency/                      # Støttefunksjoner for frekvens
├── src_severity/                       # Støttefunksjoner for severity
├── src_tweedie/                        # Støttefunksjoner for Tweedie
├── src_ml/                             # Støttefunksjoner for maskinlæring
├── src_descriptives/                   # Datakvalitet og deskriptive visualiseringer
├── src_model_comparison/               # Sluttevaluering og modellsammenligning
├── plans/                              # Levende planer og beslutningsgrunnlag
├── docs/figures/                       # Eksporterte figurer til README
├── data/                               # Lokalt datagrunnlag (ikke nødvendigvis publisert)
├── pyproject.toml                      # Prosjektmetadata og avhengigheter
└── uv.lock                             # Låste avhengighetsversjoner
```

## Datakilde og begrensninger

Dataene er spanske motorforsikringsdata publisert sammen med den faglige beskrivelsen i [PMC-artikkelen](https://pmc.ncbi.nlm.nih.gov/articles/PMC13234478/). `[fyll inn: lisens, eventuell tilgangsinformasjon og dataversjon.]`

Dette er et pedagogisk porteføljeprosjekt. Resultatene er ikke en produksjonsklar tariff og må ikke brukes direkte til kommersiell prising uten ytterligere validering, governance og vurdering av regulatoriske krav.
