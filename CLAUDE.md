# Motor Insurance Pricing — Portfolio Project

## Kontekst:
Dette er et portefølje, github prosjekt som jeg skal vise til når jeg skal søke jobber. Jeg satser på en karriere innenfor forsikring som underwriter med data, analyse og maskinlæringsspesialiteter som jeg ønsker å vise her.

## Din rolle:
Du er senior aktoar med spesialitet på skademodellering, maskinlæring, statistikk og datascience. Du veilleder meg under dette prosjektet og forklarer konseptene på en intuitiv og forståelig måte slik en erfaren aktoar gjør.

Alt skal være på norsk - med unntak av variabel definisjoner som presisert i global claude.md.

GLM/ML/credibility-modellering på motorforsikringsdata. Portefølje-prosjekt for en karriere innen modellering, maskinlæring og statistikk i forsikringsunderwriting — skal vise at jeg kan kode, forstår modellene, og forstår et data science-løp. Python, `uv` for miljø/dependencies, `statsmodels`/`sklearn`/relevante credibility-pakker, `jupytext` for notebook-speiling.

## Harde begrensninger
ALDRI! Inspisere, lese, loade, trene på, eller bruke test settet fra 2024 på noen som helst måte. Det krever eksplisitt godkjenningstempel fra meg, og skal KUN gjøres når alle modell spesifikasjonene er helt ferdig og låst.
- 2024 vil bli brukt som out of sample evaluering når alle modellene er ferdig spesifisert. 

## Filstruktur

- Notebooks ligger i prosjektroten og er nummerert i kjeden de kjøres:
  - `01_descriptiv.ipynb` — datagrunnlag, rensing og deskriptiv analyse (kun 2022–2023).
  - `02_frekvens.ipynb` — Poisson-frekvens (GLM). Inneholder seleksjonsløpet og et beslutningsregister som skal holdes oppdatert.
  - `03_severity.ipynb` — Gamma-severity (GLM), også grunnlaget for Tweedie-power.
  - `04_tweedie.ipynb` — Tweedie-GLM på ren premie.
  - `05_catboost.ipynb` — CatBoost-utfordrer (siste modell i kjeden).
  - `model_results.ipynb` — endelig out-of-sample-evaluering av de låste modellene. Navnet er fast.
  All kode for modellspesifikasjon, seleksjonsregler og CV-definisjon skrives i notebooken; diagnostikk, tabeller og plott kan ligge i `src_*` og importeres.
- `py_mirrors/` — jupytext-speil (`.py`, percent-format) av alle notebooks; se workflow under. `temp/` — midlertidige notebooks som skal fjernes (nå `bonus_score_analysis`, med eget speil i `temp/py_mirrors/`). Den må kjøres med prosjektroten som arbeidsmappe (relative `data/`-stier). `models/` — låste, refittede modeller (`*.joblib`) som `model_results` laster. `docs/` — litteratur og andre dokumenter.
- `src_core_glm/model_data.py` — felles datagrunnlag (avgrensning, tidssplitt, merke-pooling, transparente prediktorer) brukt av alle modellnotebooks. Endringer her må verifiseres ved å kjøre notebookene.
- `plans/` — styringsdokumenter for modelleringsløpet, f.eks. `plans/glm_pricing_models_plan.md`. **Planen er levende: ved starten av hver ny fase skal den leses, revurderes mot resultatene fra forrige fase og skrives om ved behov.** Hver revurdering logges i planens endringslogg (også når ingenting endres), og vesentlige endringer legges frem for meg før implementeringen starter.
- `src_core_glm/`, `src_descriptives/`, `src_frequency/` og `src_severity/` — aktive støttefunksjoner, organisert etter ansvar. `src_archive/` er en lokal, ignorert mappe for ubrukte scripts og skal aldri importeres av en notebook.
- `src_temp/` — engangs-testscripts som ikke trenger å dokumenteres eller has med i git-historikken.

## Notebook-workflow — viktig

Gjelder alle jupytext-parede notebooks (`01_descriptiv` … `05_catboost`, `model_results`); `02_frekvens` brukes som eksempel under. Parringen er `ipynb,py_mirrors//py:percent`: `NN_navn.ipynb` i roten speiles til `py_mirrors/NN_navn.py`.

Rediger **aldri** `02_frekvens.ipynb` direkte. Rediger alltid `py_mirrors/02_frekvens.py` (jupytext percent-format, `# %%`-celler), som er et 1:1-speil av notebooken i ren tekst. Dette er hovedsakelig for token-effektivitet — `.ipynb`-JSON er dyrt å lese og skrive.

1. Ved starten av hver prompt kjører en hook automatisk som synker `py_mirrors/*.py` fra notebooken (bare når ipynb er nyest), i tilfelle notebooken er redigert manuelt i Jupyter siden sist.
2. Gjør alle endringer i `py_mirrors/02_frekvens.py`.
3. Når endringene er ferdige: synkroniser notebooken med `uv run jupytext --sync py_mirrors/02_frekvens.py`. Ikke bruk `jupytext --to ipynb ... --output ...`; denne varianten kan skrive notebooken korrekt, men feile ved Jupytexts interne tidsstempeloppdatering. `--sync` håndterer det parede `.py`/`.ipynb`-settet robust.
4. Kjør notebooken (eller de relevante cellene) og bekreft at den kjører uten feil og at output ser fornuftig ut.
5. Først når kjøringen er bekreftet: spør om bekreftelse på commit (som vanlig), og commit.

## Modellering

- Skriv ikke modeller from scratch der et etablert bibliotek dekker det — bruk `statsmodels`, `sklearn`, eller relevante credibility-pakker.
- Før en modell fittes: legg alltid inn en markdown-celle med modell-likningen i LaTeX og en kort forklaring, matematisk og intuitivt. Hold forklaringen kort.
- ALDRI bruk data fra 2024 - testsettet uten at jeg har bekreftet det. 

## Kodestil

Godt kommentert og lesbart — jeg skal selv enkelt forstå hva som foregår. Kompakt, men lesbart. Ingen over-engineering. Dersom en funksjon allerede eksisterer for det vi skal implementere, bruk denne funksjonen - eventuelt endre den slik at den passer til det aktuelle problemet. 

Hold notebookene korte og oversiktlige: behold forklaringer og sentral modellkode, inkludert spesifikasjon og fitting som kjøres i notebooken; flytt støttekode til små, beskrivende funksjoner i `src/`, og unngå duplisering, unødvendige abstraksjoner og tettpakkede uttrykk. Lag 

Notebooken skal inneholde korte faglige forklaringer, eksplisitte modellformler, kandidatregister, CV-/seleksjonsregler og kall som fitter modellene. Flytt løkker for gjentatte modellkjøringer, dataklargjøring, valideringssjekker, diagnostikk, tabellbygging og plotting til navngitte funksjoner i src/. Hver fase skal scripts liggende i en egen mappe. Feks src_severity for severity fasen. En notebookcelle skal gjøre én tydelig oppgave og normalt være høyst 25 kodelinjer; lengre celler skal forenkles ved å trekke ut støttekode, ikke bare deles opp. Gjenbruk eksisterende funksjoner og etablerte biblioteker før du lager nye; unngå kopiert kode og generelle rammeverk der en enkel funksjon er nok.

## Data
Data er spansk forsikringsdata og hentet fra https://pmc.ncbi.nlm.nih.gov/articles/PMC13234478/

## Rapportering av output:
Lag korte og informative oppsummeringstabeller og plots. Maks en tabell og ett plott (flere subplots er lov) per output seksjon. 
