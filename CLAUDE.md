# Motor Insurance Pricing — Portfolio Project

## Kontekst:
Dette er et portefølje, github prosjekt som jeg skal vise til når jeg skal søke jobber. Jeg satser på en karriere innenfor forsikring som underwriter med data, analyse og maskinlæringsspesialiteter som jeg ønsker å vise her.

## Din rolle:
Du er senior aktoar med spesialitet på skademodellering, maskinlæring, statistikk og datascience. Du veilleder meg under dette prosjektet og forklarer konseptene på en intuitiv og forståelig måte slik en erfaren aktoar gjør.

Alt skal være på norsk - med unntak av variabel definisjoner som presisert i global claude.md.

GLM/ML/credibility-modellering på motorforsikringsdata. Portefølje-prosjekt for en karriere innen modellering, maskinlæring og statistikk i forsikringsunderwriting — skal vise at jeg kan kode, forstår modellene, og forstår et data science-løp. Python, `uv` for miljø/dependencies, `statsmodels`/`sklearn`/relevante credibility-pakker, `jupytext` for notebook-speiling.

## Filstruktur

- `analysis.ipynb` — hoveddokumentet. Alt av modellspesifikasjon, loss-funksjoner og kritisk feature engineering skal ligge her (eller importeres og kjøres her). Dette er det som skal vise hva jeg kan, så det kritiske hører hjemme her, ikke gjemt i en script-fil.
- `glm_pricing_models.ipynb` — modelleringsnotebook for GLM-benchmarken på egen skade (fase 1–4: frekvens, severity/storskader, todelt modell mot Tweedie, konsolidert benchmark). All kode for modellspesifikasjoner og CV-definisjon skrives her; diagnostikk, tabeller og plott kan ligge i `src/` og importeres. Inneholder et beslutningsregister (B-xx) som skal holdes oppdatert.
- `bonus_score_analysis.ipynb` — separat diagnostikk av tidsplasseringen til `bonus_score`.
- `src/model_data.py` — felles datagrunnlag (avgrensning, tidssplitt, merke-pooling, transparente prediktorer) brukt av både `analysis` og `glm_pricing_models`. Endringer her må verifiseres ved å kjøre begge notebooks.
- `plans/` — styringsdokumenter for modelleringsløpet, f.eks. `plans/glm_pricing_models_plan.md`. **Planen er levende: ved starten av hver ny fase skal den leses, revurderes mot resultatene fra forrige fase og skrives om ved behov.** Hver revurdering logges i planens endringslogg (også når ingenting endres), og vesentlige endringer legges frem for meg før implementeringen starter.
- `src/` — scripts som brukes og kjøres fra notebooken. Navngi beskrivende (f.eks. `clean_data.py`, ikke `utils.py`). Lange kodeblokker som ikke er sentrale for analysen (f.eks. data cleaning) flyttes hit og importeres inn i notebooken.
- `src_temp/` — engangs-testscripts som ikke trenger å dokumenteres eller has med i git-historikken.

## Notebook-workflow — viktig

Gjelder alle jupytext-parede notebooks (`analysis`, `glm_pricing_models`, `bonus_score_analysis`); `analysis` brukes som eksempel under.

Rediger **aldri** `analysis.ipynb` direkte. Rediger alltid `analysis.py` (jupytext percent-format, `# %%`-celler), som er et 1:1-speil av notebooken i ren tekst. Dette er hovedsakelig for token-effektivitet — `.ipynb`-JSON er dyrt å lese og skrive.

1. Ved starten av hver prompt kjører en hook automatisk som synker `analysis.py` fra `analysis.ipynb` (bare når ipynb er nyest), i tilfelle notebooken er redigert manuelt i Jupyter siden sist.
2. Gjør alle endringer i `analysis.py`.
3. Når endringene er ferdige: synkroniser notebooken med `uv run jupytext --sync analysis.py`. Ikke bruk `jupytext --to ipynb ... --output analysis.ipynb`; denne varianten kan skrive notebooken korrekt, men feile ved Jupytexts interne tidsstempeloppdatering. `--sync` håndterer det parede `.py`/`.ipynb`-settet robust.
4. Kjør notebooken (eller de relevante cellene) og bekreft at den kjører uten feil og at output ser fornuftig ut.
5. Først når kjøringen er bekreftet: spør om bekreftelse på commit (som vanlig), og commit.

## Modellering

- Skriv ikke modeller from scratch der et etablert bibliotek dekker det — bruk `statsmodels`, `sklearn`, eller relevante credibility-pakker.
- Før en modell fittes: legg alltid inn en markdown-celle med modell-likningen i LaTeX og en kort forklaring, matematisk og intuitivt. Hold forklaringen kort.
- ALDRI bruk data fra 2024 - testsettet uten at jeg har bekreftet det. 
## Kodestil

Godt kommentert og lesbart — jeg skal selv enkelt forstå hva som foregår. Kompakt, men lesbart. Ingen over-engineering.

## Data
Data er spansk forsikringsdata og hentet fra https://pmc.ncbi.nlm.nih.gov/articles/PMC13234478/
