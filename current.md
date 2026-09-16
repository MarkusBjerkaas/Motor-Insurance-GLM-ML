## Codex Respons:

Hva vi må beslutte før neste fase.

1. Hva modellen faktisk skal predikere:
- Er incurred like modne for alle år? Sansynligvis nei, ettersom det vil være mer IBNR, og mer preget av reserver i 2024 sammenlignet med 2023 og 2022. Uttrekket er derimot fra 2026, og vi antar at dette ikke er særlig problematisk ettersom det er kasko skader hvor skader typisk oppstår og rapporteres med en gang.
- Representerer kansellerte poliser faktisk populasjonen? Usikker, må se på egen analyse av dette problemet. 
- Vi antar de ni skadene av null er relle nullskader. Dette må da tas hensyn til i modellen.

2. Hvilken prediktorer er tilgjengelig på prediksjonstidspunkt?
- Bonus score holdes utenfor i starten. Analysen viser derimot at den er basert på tidligere år, men ikke bare skader, og kan derfor være en nyttig prediktor. På den andre siden kan vi ikke garantere lekkasje eller ikke.
- Driver age og driving experience years er sterkt korrelerte og må vurderes ved modell fit. Vi må og vurdere om vi skal bruke den utledede eller ikke. 
- Policy status bør nok også ikke inn som prediktor, fordi det blir vanligvis kjent om den er cancelled på slutten eller ila året, altså ikke på prediksjonstidspunktet.
- Policy type er nok en nyttig prediktor nettopp fordi den sier noe om det er egenandel eller ikke, og er sansynligvis svært viktig for frekvensen. Det som er problematisk er at ikke vi vet egenandelen.

3. Hvordan gruppe CV og tidsvalidering skal brukes når de er uenige?
- Gruppe CV som primærseleksjonsmekasnisme, men krevd at en kandidat ikke vesentlig forverrer tidsbalansen eller relativitetsstabilitet. Global nivådrift bør håndteres som kalibritetsproblem.  


## Respons til codex forslag:


1. Respons, populasjon og kansellerte poliser:
- Enig i forslagene med unntak av følgende:
- Det blir vel feil å inkludere nullskader i frekvens og severity estimeringen? Disse bør holdes utenfor, ettersom vi ikke forsøker å predikere null skader. Hva tenker du?

2. Vi implementerer den betingede adgangsreglen som du nevner.

3. Enig i de valgene du foreslår her.

4. Jeg er enig i seleksjonsregelen og variabel stabilitetsregelen.

5. Jeg er enig i at deviance bør være primærmetrikk, men det må være mulig å la det overstyres av andre metriks dersom resultatene tilsier at deviance feiler.

6. Jeg er enig i dette. Jeg er enig at vi bør teste driver_age og driving_experience_years hver for seg, og gjøre valget da.  

Jeg ønsker at du oppsummerer disse beslutningene i notebooken så er vi klar for neste fase. 