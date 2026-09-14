# Motor Insurance Pricing — Portfolio Project

Alt skal være på norsk - med unntak av variabel definisjoner som presisert i global claude.md.

GLM/ML/credibility-modellering på motorforsikringsdata. Portefølje-prosjekt for en karriere innen modellering, maskinlæring og statistikk i forsikringsunderwriting — skal vise at jeg kan kode, forstår modellene, og forstår et data science-løp. Python, `uv` for miljø/dependencies, `statsmodels`/`sklearn`/relevante credibility-pakker, `jupytext` for notebook-speiling.

## Filstruktur

- `analysis.ipynb` — hoveddokumentet. Alt av modellspesifikasjon, loss-funksjoner og kritisk feature engineering skal ligge her (eller importeres og kjøres her). Dette er det som skal vise hva jeg kan, så det kritiske hører hjemme her, ikke gjemt i en script-fil.
- `src/` — scripts som brukes og kjøres fra notebooken. Navngi beskrivende (f.eks. `clean_data.py`, ikke `utils.py`). Lange kodeblokker som ikke er sentrale for analysen (f.eks. data cleaning) flyttes hit og importeres inn i notebooken.
- `src_temp/` — engangs-testscripts som ikke trenger å dokumenteres eller has med i git-historikken.

## Notebook-workflow — viktig

Rediger **aldri** `analysis.ipynb` direkte. Rediger alltid `analysis.py` (jupytext percent-format, `# %%`-celler), som er et 1:1-speil av notebooken i ren tekst. Dette er hovedsakelig for token-effektivitet — `.ipynb`-JSON er dyrt å lese og skrive.

1. Ved starten av hver prompt kjører en hook automatisk som synker `analysis.py` fra `analysis.ipynb` (bare når ipynb er nyest), i tilfelle notebooken er redigert manuelt i Jupyter siden sist.
2. Gjør alle endringer i `analysis.py`.
3. Når endringene er ferdige: spør om bekreftelse på commit (som vanlig), commit, og kjør deretter konverteringsscriptet som oppdaterer `analysis.ipynb` fra `analysis.py`. Konverter aldri tilbake før commit er gjort.

## Modellering

- Skriv ikke modeller from scratch der et etablert bibliotek dekker det — bruk `statsmodels`, `sklearn`, eller relevante credibility-pakker.
- Før en modell fittes: legg alltid inn en markdown-celle med modell-likningen i LaTeX og en kort forklaring, matematisk og intuitivt. Hold forklaringen kort.

## Kodestil

Godt kommentert og lesbart — jeg skal selv enkelt forstå hva som foregår. Kompakt, men lesbart. Ingen over-engineering.
