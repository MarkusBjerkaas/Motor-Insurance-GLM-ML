# Motorforsikring: transparent prisingsbenchmark

Porteføljeprosjekt for skadefrekvens, severity og ren premie på spanske
motorforsikringsdata. Hovedløpet er en transparent GLM-benchmark med
eksponeringsvektet OOF-evaluering og gruppefold på `insured_id`.

En separat ABESS-diagnostikk i `src/abess_diagnostics.py` undersøker om
gruppert best-subset-seleksjon peker på andre additive frekvensblokker enn den
frosne GLM-prosessen. Den er eksplorativ, bruker bare 2022–2023 og har eget
resultatobjekt; den endrer ikke hovedmodellens kandidatregister eller modell-ID.
Mulige senere utvidelser er credibility for bilmerke og ML-utfordrere, først
etter at benchmarken og felles sluttevaluering er frosset.
