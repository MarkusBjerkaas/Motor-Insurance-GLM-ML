# Severity-modellering: låst kandidatsøk og tydelige stoppkriterier

**Protokollversjon 2.1 (2026-09-18).** Denne versjonen utvider den første,
låste Gamma-seleksjonsrunden med fire forhåndsdefinerte enkeltutfordrere for
drivstoff, forretningstype, betalingsfrekvens og pool'et bilmerke. Ingen andre deler av
kandidatrommet eller seleksjonsgrensene åpnes.

## 1. Formål, beslutninger og faktagrunnlag

Denne planen vedlikeholdes i `plans/Severity_plan.md`. Den eksisterende hovedplanen beholdes som historikk.

Målet er å utvikle en transparent Gamma-benchmark for registrert skadekostnad per registrert skade, og undersøke om avgrensede endringer forbedrer en faglig og EDA-forankret baseline.

**Brukerbeslutninger som ligger fast:**

- Vanlig femfolds gruppe-CV på `insured_id`. Ingen indre CV eller ekstra holdout.
- Baseline og kandidatrommet nedenfor er låst.
- Bare seteinndelingen `<5`, `=5`, `>5` inngår.
- Enklere modeller foretrekkes ved tilnærmet like resultater.
- Storskader undersøkes diagnostisk. Ingen kapping eller halemodell i dette løpet.
- Inverse Gaussian er eneste mulige fordelingsutfordrer, helt til slutt. **Den skal verken implementeres eller fittes uten en separat, uttrykkelig godkjenning.**
- Frekvensmodellen F5 forblir låst.
- Teknisk årssplitt av kildefilen er tillatt. Videre behandling av 2024 er ikke tillatt.

De tallfestede seleksjons- og stoppgrensene i denne planen er faglige styringsforslag som låses med planen. De er ikke litteraturbestemte sannheter eller tidligere godkjente tallgrenser.

**Planen er levende, men seleksjonsreglene er bindende.** Nye funn kan avdekke at datagrunnlaget eller modelloppgaven må revurderes. De gir ikke adgang til å endre features, kategorigrenser, splinegrader, CV-seed, score eller akseptgrenser for å forbedre et sett resultater som allerede er sett.

### Kontrollert nåsituasjon

Følgende er reprodusert på utviklingsårene, uten modellfitting:

| Kontrollpunkt | Resultat |
|---|---:|
| Poliseår i modellpopulasjonen | 55 246 |
| Eksponeringsår | 37 126,014 |
| Registrerte skader | 9 989 |
| Poliseår med registrert skade | 5 707 |
| Positive skadeår til severity | 5 698 |
| Unike `insured_id` i severity-utvalget | 5 323 |
| Registrerte skader i positivt utvalg | 9 979 |
| Utelatte null-/nærnullkostnadsår | 9 år med 10 registrerte skader |
| Skadevektet severity, alle registrerte skader | 871,771 EUR |
| Skadevektet severity, positivt utvalg | 872,645 EUR |

De utelatte årenes samlede kostnad er numerisk tilnærmet null. De skal fortsatt inngå i frekvens og senere ren-premie-evaluering.

Frekvensmodellen er `F5_minus-municipality-performance`: Den beholdt kjøresone og fjernet kommunetype. Dette begrenser ikke det avtalte severity-kandidatrommet.

Lagrede frekvensresultater viser F5-deviance 1,121675 og ABESS-deviance 1,121144. ABESS har altså **0,000531 lavere/betre** utviklingsscore, omtrent 0,63 cluster-SE. Notebookens gjenværende tekst om at ABESS «taper», skal korrigeres uten endring av frekvensmodellen.

Tall fra lagret modelloutput skal merkes som dette inntil reproduksjon er bekreftet. Samsvar mellom notebookens kilde og `.py` beviser ikke at lagret output er produsert av dagens kode og støttefunksjoner.

### Korrigert litteraturgrunnlag

Baselinen skal omtales som **faglig og EDA-forankret, med begrenset direkte kaskolitteratur**. Den er ikke uavhengig av tidligere innsikt fra 2022–2023.

Reiff-studien bruker justert lineær regresjon på log-severity for slovakisk motoransvar. Det er ikke en Gamma-regresjon for kasko. [Reiff mfl., fulltekst](https://economic-policy.pl/index.php/eq/article/download/2058/1960).

| Variabel | Dokumentasjon og justering | Overførbarhet | Form i vårt løp og evidensstatus |
|---|---|---|---|
| Produkt | Deknings-/egenandelsmekanikk og egen EDA; ingen direkte justert studie verifisert i gjennomgangen | Relevant, men rapportering og seleksjon gjør retningen usikker | Kategorisk; arbeidsantakelse |
| År | Egen portefølje og behov for årskontroll | Kontrollerer registrerte årsforskjeller, ikke identifisert inflasjon | Kategorisk; metodisk valg |
| Bilverdi | Reparasjons-/erstatningsverdi og egen EDA; ingen direkte justert kilde verifisert | Faglig plausibel kaskodriver | Log-verdi lineært eller spline df3; begge er arbeidsantakelser |
| Føreralder | Reiff finner justerte forskjeller etter **forsikringstakers alder**, med merkeinteraksjon; ingen generell U-form dokumentert | Indirekte: annet aldersbegrep og annen dekning | Lineær/df2/df3/df4; funksjonsformene er våre valg |
| Kommunetype og kjøresone | Egen EDA; Reiffs distriktseffekt dokumenterer ikke våre geografikategorier eller deres retning | Begrenset litteraturstøtte for de konkrete feltene | Kategoriske alternativer; empirisk spørsmål |
| Ytelse | Reiff finner positiv motoreffekt og negativ vekteffekt, justert og modellert separat | Indirekte støtte; dokumenterer ikke vårt forholdstall | Lineær eller spline df3; egen transformasjon og form |
| Seter | Egen EDA; ingen verifisert kostnadsseverity-kilde | Usikker prediktiv verdi utover bilverdi og ytelse | `<5`, `=5`, `>5`; forhåndsvalgt arbeidsantakelse |

Gamma/log-link har etablert metodisk støtte for severity. Det begrunner en startmodell, ikke at fordelingen nødvendigvis beskriver våre data korrekt. [CAS Monograph 5, andre utgave](https://www.casact.org/sites/default/files/database/monographs_papers_05-goldburd-khare-tevet.pdf).

Oyugi-kilden skal ikke brukes som argument for lognormalfordeling av opprinnelige beløp: Fullteksten beskriver fordelingsfitting på trippel-log-transformerte beløp. Påstanden om K–S/Anderson–Darling som grunnlag for den rapporterte sammenligningen er heller ikke verifisert. [Oyugi, side 13–15](https://actuaries.org/app/uploads/2025/07/ICA2010_ASTIN_22_final_-paper_Oyugi.pdf).

Personskadestudiene dokumenterer en annen respons enn kostnadsseverity og skal ikke brukes til å fastsette retninger her. Manglende litteraturstøtte betyr ikke dokumentert manglende prediktiv verdi.

## 2. Datagrunnlag, modeller og bindende seleksjonsalgoritme

### Sikker utviklingsdataflyt

Dagens innleser behandler 2024 før testrammen fjernes. Denne flyten må erstattes før notebookene kjøres.

- Les kildefilen i biter og behold bare 2022–2023 umiddelbart.
- Forkast øvrige rader før rensing, kvalitetsoppsummeringer, transformasjoner eller modellering. Ikke rapporter testårets antall eller innhold.
- Rens hele utviklingspopulasjonen før produktavgrensning; eksisterende rensekode forventer en bestemt korreksjonsrad fra 2022.
- Behold eksisterende avgrensning: `COMP_E`/`COMP_N`, positiv egen-skadepremie og positiv eksponering. Kansellerte poliser beholdes.
- Bevar radidentitet og rekkefølge slik at frekvens- og severity-foldene kan kobles entydig.
- Kontroller tillatte år ved inngangen til hver analyse- og modelleringsfunksjon.

### Respons og arbeidsmodell

Bruk:

- `average_severity = property_incurred / property_claims`.
- Utvalg: `property_claims > 0` og **samlet** `property_incurred > 0.01`.
- Gamma med log-link og `var_weights=property_claims`.
- Ingen eksponeringsvekt eller eksponeringsoffset i severity-modellen.
- Ingen kapping, winsorisering eller automatisk rekalibrering.

EDA har tidligere anvendt nullgrensen på gjennomsnittsskaden. Definisjonene gir samme utvalg i dagens data, men modellen skal bruke definisjonen over konsekvent.

Skadeantallsvekten retter modellen mot kostnad per registrert skade i aggregatet. Observasjonene er fortsatt poliseår, ikke rekonstruerte enkeltskader. Arbeidsantakelsen om varians proporsjonal med `mu²/N` kan svikte ved avhengige skader og uavklart skadetelling.

### Kandidatregister

Alle kandidater sammenlignes først mot samme faste baseline. Ingen univariat screening, interaksjoner eller generell baklengs variabelseleksjon.

| ID | Spesifikasjon | Regresjonsparametere inkludert intercept |
|---|---|---:|
| S0 | Produkt + år + kommunetype + aldersspline df3 + lineær log-bilverdi | 9 |
| A1 | S0 med lineær alder | 7 |
| A2 | S0 med aldersspline df2 | 8 |
| A4 | S0 med aldersspline df4 | 10 |
| V3 | S0 med spline df3 for log-bilverdi | 11 |
| G1 | S0 med kjøresone **i stedet for** kommunetype | 8 |
| G2 | S0 med både kommunetype og kjøresone | 10 |
| P1 | S0 + lineær ytelse | 10 |
| P3 | S0 + ytelsesspline df3 | 12 |
| T1 | S0 + seter kategorisk: `<5`, `=5`, `>5` | 11 |
| FU1 | S0 + drivstofftype (`D`, `G`, eksplisitt `MISSING`) | 11 |
| BU1 | S0 + forretningstype | 10 |
| PF1 | S0 + betalingsfrekvens | 11 |
| BR1 | S0 + pool'et bilmerke | Fastsettes som 9 + antall ikke-referansenivåer før fitting |

Parametertallene forutsetter de dokumenterte kategorinivåene og full rang. Før første kandidatfit skal implementeringen bygge designmatrisene foldvis og kontrollere kategorinivåer, referansenivåer, kolonneantall og rang mot registeret. Avvik håndteres etter gyldighetsreglene nedenfor; implementeringen skal ikke stille endre parametertall eller utelate nivåer for å få kandidaten til å passe.

Splines bruker `cr(..., df=k, constraints='center')`. Knuter og sentrering læres på treningsfolden og gjenbrukes ved prediksjon. Numerisk imputasjon bruker treningsmedian. Ingen automatisk sletting av rader.

Referansenivåer: produkt `COMP_E`, år 2022, kommune `I`, kjøresone `U`, seter
`=5`, drivstoff `D`, forretningstype `NB` og betalingsfrekvens `A`.
Drivstoff har manglende observasjoner; den allerede låste preprocessingen
behandler disse som det eksplisitte nivået `MISSING`. Dette er derfor et tredje
FU1-nivå, ikke numerisk imputasjon eller radbortfall.
Nivåer og referanser for disse fire feltene skal verifiseres og låses mot
utviklingsdata før første fit. Referansen for `vehicle_brand_pooled` låses til
nivået med størst eksponering. Det eksakte BR1-parametertallet låses fra antall
pool'ede merkenivåer før designkontrollen og skal deretter stemme i alle folder.

Merke, bilalder og erfaring inngår ikke i seleksjonen; eventuell senere
sensitivitet krever eget avgrenset opplegg. Tidligere lekkasjeeksklusjoner
videreføres.

### Fem faste gruppefolder

Bruk eksisterende `GroupKFold(n_splits=5, shuffle=True, random_state=100)` på hele modellpopulasjonens `insured_id`. Severity trenes på foldens positive skadeår.

| Fold | Positive skadeår i trening | Positive skadeår i validering |
|---|---:|---:|
| 1 | 4 563 | 1 135 |
| 2 | 4 527 | 1 171 |
| 3 | 4 593 | 1 105 |
| 4 | 4 535 | 1 163 |
| 5 | 4 574 | 1 124 |

Ingen ny seed, omfordeling etter skadekostnad eller gjentatt CV for å lete etter gunstigere resultater.

Modellen skal også predikere **alle poliseår i valideringsfolden**, slik at senere frekvens–severity-sammenkobling får komplette OOF-prediksjoner. Severity-scoren beregnes bare på det definerte positive utvalget.

### Gyldighet før seleksjon

En kandidat må bestå i alle fem folder:

- Minst 50 unike `insured_id` i hvert kategorinivå i severity-treningen.
- Full rang, ingen uventet radbortfall, endelige parametere og positive, endelige prediksjoner.
- Konvergens med samme innstillinger for alle kandidater: maksimalt 200 iterasjoner og toleranse `1e-8`.
- Ingen usette kategorinivåer ved prediksjon og ingen manglende verdier etter preprocessing.

En valgfri kandidat som ikke består, markeres ugyldig og fittes ikke. Hele
kandidatregisteret og før-fit-designrapporten beholdes for sporbarhet, men
senere kvalifisering og blokkvalg bruker bare fittede kandidater som også består
den foldvise gyldighetskontrollen. Ingen ny pooling, kategorisering eller
regularisering legges til som redningsforsøk. Svikt i S0 utløser stopp for hele
seleksjonsløpet. I den låste utviklingskontrollen er FU1 ugyldig fordi
`fuel_type = MISSING` har 39–43 unike personer i treningsfoldene, og BR1 ugyldig
fordi `CHEVROLET` har 42–50 og fire folder under 50. BU1 og PF1 består.

50-personerskravet gjelder **treningsstøtte**, ikke størrelsen på hvert valideringssegment. Det innføres ingen minstegrense per valideringskategori som kan fjerne observasjoner fra pooled score eller gjøre en ellers gyldig kandidat ugyldig. Rapportér antall skadeår og unike personer per segment, både foldvis og samlet OOF. Små valideringssegmenter merkes som svakt støttet og skal ikke alene begrunne segmentkonklusjoner. De særskilte 100-personerskravene i stoppregisteret gjelder den populasjonen den aktuelle diagnostikken beregnes på: samlet OOF for produkt/desil og hvert enkelt år for segmentdrift.

### Primærscore og usikkerhet

Primærscore er pooled, skadeantallsvektet Gamma-deviance:

`D = sum(N × d(y, mu)) / sum(N)`

med `d(y, mu) = 2 × (y/mu − 1 − log(y/mu))`.

Foldscorene skal ikke gjennomsnittsberegnes uvektet.

Definer positiv gevinst som `D_reference − D_candidate`. Parvis SE beregnes fra de vektede scoreforskjellene, gruppert på `insured_id`, med sentrerte clusterbidrag og endelig-antall-korreksjon. Den eksisterende Poisson-spesifikke hjelpefunksjonen skal ikke brukes uendret.

Cluster-SE beskriver usikkerhet i de observerte scoreforskjellene. Den korrigerer ikke seleksjonsoptimisme eller hele variasjonen fra modelltrening.

### Entydige utvalgsregler

**Forbedringsregel:** En kandidat med like mange eller flere parametere kvalifiserer bare hvis:

- Gevinsten er større enn både én parvis cluster-SE og **0,5 % av referansens deviance**.
- Kandidaten forbedrer scoren i minst fire av fem folder.

**Forenklingsregel:** En kandidat med færre parametere kvalifiserer dersom:

`D_candidate − D_reference ≤ min(SE_pair, 0.005 × D_reference)`.

En forenkling kan dermed aksepteres med en liten forverring, men bare innen begge grensene.

Asymmetrien er tilsiktet: Forbedringsregelen krever gevinst i minst fire av fem folder, mens forenklingsregelen ikke krever et bestemt antall forbedrede folder. Dette uttrykker den avtalte preferansen for enkelhet, ikke en påstand om at forenklingen er bedre i hver fold. Foldvise scoreforskjeller skal fortsatt vises, også når pooled score kvalifiserer en forenkling som taper i flere folder. Ingen ny foldterskel legges til etter at resultatene er sett.

**Regel for nesten like resultater:** Finn laveste score blant kvalifiserte kandidater. Ta med alternativer som ligger innen både én parvis SE og 0,5 % av denne beste scoren. Velg færrest parametere; deretter lavest score; deretter alfabetisk kandidat-ID.

Utfør følgende én gang:

1. Kvalifiser alle enkeltutfordrere som besto før-fit-porten mot S0.
2. G2 må bestå forbedringsregelen mot **både S0 og G1**, på de samme fem foldene. G1 må være gyldig etter støtte- og gyldighetskontrollene, men trenger ikke selv å kvalifisere mot S0 for å være sammenligningsgrunnlag. Hvis G1 er ugyldig, kan tilleggsverdien ikke dokumenteres, og G2 kvalifiserer ikke. En gevinst mot G1 kan aldri kompensere for at G2 ikke består kravet mot S0. Begge geografivariabler må dermed begrunne merverdien utover **hver** enkeltvariabel.
3. Velg ett alternativ i hver blokk: alder, verdi, geografi, ytelse, seter,
   drivstoff, forretningstype, betalingsfrekvens og merke. S0 representerer uendret
   blokk. Bruk regelen for nesten like resultater. De fire nye kategoriske
   variablene er separate blokker og kan, hvis de kvalifiserer og vinner
   blokkens nesten-like-vurdering, inngå i den ene kombinerte kandidaten.
4. Dersom minst to blokker endres, bygg **én** samlet kandidat C1. Ingen andre kombinasjoner tillates.
5. C1 må bestå den relevante forbedrings-/forenklingsregelen mot S0 og mot hver valgt enkeltutfordrer. Reglene anvendes ut fra parametertallet i hvert sammenligningspar.
6. Velg finalist blant S0, kvalifiserte enkeltutfordrere og eventuell kvalifisert C1 med den samme regelen for nesten like resultater.
7. Dersom C1 forkastes, brukes de allerede kvalifiserte enkeltmodellene som fallback. Ingen oppdeling av C1 og ny kombinasjonsrunde.

Maksimalt **14 faste spesifikasjoner og én kombinert kandidat**. Dette gir et
protokollmaksimum på 70 faste hovedtilpasninger og 75 med C1. Faktisk budsjett
er antallet kandidater som består før-fit-porten multiplisert med fem; i denne
kontrollen er det 12 gyldige faste spesifikasjoner, altså 60, og 65 dersom C1
bygges. Diagnostiske refittinger nedenfor føres separat i kjøreloggen og kan
ikke bli nye kandidater.

0,5 %-grensen og 1-SE-regelen er konservative beslutningsheuristikker, ikke signifikanstester. De gjenbrukes ved enkeltkvalifisering, geografisammenligning, blokkvalg, kombinasjonskontroll og finalistvalg på de samme valideringsobservasjonene. Beslutningene er avhengige; tersklene gir ingen kontrollert samlet feilrate og fjerner ikke seleksjonsoptimismen gjennom hele prosedyren. En eventuell gevinst for vinneren kan derfor overvurdere forbedringen på nye data. Det låste kandidatbudsjettet begrenser søket, men opphever ikke denne begrensningen. Søket er avsluttet også når konklusjonen blir at S0 beholdes.

## 3. Diagnostikk, stoppkriterier og revisjon

### Før fitting: skadebeløp og støtte

Lag korte tabeller og plott for:

- Kostnadskonsentrasjon og de største bidragene per poliseår og person.
- Overskridelser ved 5 000, 7 500 og 10 000 EUR, separat for samlet årskostnad og gjennomsnittsskade.
- Antall overskridende skadeår, unike personer, kostnadsandel og **overskytende** kostnadsandel. Disse størrelsene må ikke blandes.
- Null-/nærnullkostnader, beløp under 1 og 10 EUR, og tydelige beløpsklumper.
- Støtte per produkt, år, geografi, setekategori, skadeantallsgruppe,
  `fuel_type`, `business_type`, `payment_frequency` og
  `vehicle_brand_pooled`.

Kontrollen har funnet 136, 68 og 37 positive skadeår med gjennomsnittsskade over de tre tersklene. Dette er ikke antall individuelle storskader. Ved flere registrerte skader kan én stor skade skjules i gjennomsnittet.

Ingen av disse diagnostikkene åpner for nye features eller nye terskler i kandidatsøket.

### Etter seleksjon: EUR-kalibrering og stabilitet

Undersøk både S0 og den valgte finalisten:

- A/E = faktisk kostnad delt på `sum(N × predikert severity)`.
- Totalt, per produkt, år, geografivariabel, setekategori og `N=1`, `N=2–3`, `N≥4`.
- Prediksjonsdesiler definert fra S0s OOF-prediksjoner, med skadeantallsvektede grenser. Bruk samme grupper for begge modeller.
- Foldvise effekter og splinekurver, med synlig observasjonsstøtte.
- Bidrag til deviance og EUR-feil fra små beløp og store kostnader.

**Kollinearitet og fortolkning:** Etter finalistvalget undersøkes sammenhengen mellom log-bilverdi og ytelse dersom begge inngår. Rapportér korrelasjonen mellom de numeriske inputvariablene i severity-treningen og vurder foldstabiliteten til hele effektkurvene/blokkene på et felles, støttet kovariatområde. Skill mellom stabile samlede prediksjoner og usikker fordeling av effekten mellom variablene. VIF for enkelte splinekolonner skal ikke brukes som automatisk grense: Høy VIF kan skyldes basisrepresentasjonen. Diagnostikken er en vurdering av fortolkningsbegrensninger, ikke en ny seleksjons- eller stoppregel; ingen variabel fjernes eller kandidat prøves på nytt på dette grunnlaget. Full rang og øvrige eksisterende gyldighetskrav gjelder fortsatt.

Bruk 2 000 cluster-bootstrap-trekk på `insured_id`, seed 410, med faste OOF-prediksjoner og samme trekk for modellene. Rapportér percentile-intervaller på 95 %. Ikke kopier Poisson-baserte usikkerhetsgrenser fra frekvens.

Bootstrapen refitter ikke modeller og gjentar ikke seleksjonen. Intervallene er derfor betinget på de tilgjengelige OOF-prediksjonene, og gir ingen garanti for haler som ikke er representert.

For innflytelse gjennomføres én fast sensitivitet: I hver treningsfold fjernes de fem personene med høyest samlet registrert kostnad. S0 og finalisten refittes med uendret spesifikasjon og predikerer den opprinnelige valideringsfolden. Dette er maksimalt ti diagnostiske refittinger; ingen av dem kan erstatte hovedmodellen.

`N=1` vurderes mot hovedmodellens kovariatjusterte OOF-forventning. Det er et selektert utvalg og ikke nødvendigvis én fysisk hendelse. Ingen forklaring med flåtestørrelse brukes uten dokumentasjon.

### Avgrenset tidskontroll og modenhet

Etter CV-valget låses finalistens formel. Fit S0 og finalisten på 2022 og evaluer på 2023, uten årsledd og med preprocessing lært bare fra 2022. Ingen kandidatseleksjon gjentas på tidsresultatet.

Rapportér:

- Globalt nivåskift.
- A/E per produkt og skadeantallsgruppe.
- Årsvise null-/nærnullkostnader og kostnadsnivåer.
- Segmentenes A/E relativt til årets totale A/E, slik at global drift skilles fra endrede relativiteter.

For miksstandardisering fit begge spesifikasjoner separat per år uten årsledd. Predikér begge årstilpasningene på samme samlede positive utviklingspopulasjon med samme skadevekter. Vis også hvor mye av referansepopulasjonen som ligger utenfor det enkelte årets observerte kovariatområde. Disse tilpasningene er forklarende diagnostikk, ikke valideringsresultater.

Tidskontrollen tillater samme person i begge år: Den undersøker en annen generaliseringssituasjon enn gruppe-CV.

Det finnes bare **én kalenderovergang**, 2022→2023. Vi kan ha informasjon nok til å oppdage en scoreforskjell innen 2023, men kan ikke vurdere hvordan forskjellen varierer mellom flere fremtidige år. Usikkerhetsintervallene innen denne tidsdelingen dekker ikke variasjonen mellom kalenderoverganger. At tidsstoppkriteriet ikke utløses, skal rapporteres som «ingen forverring påvist i denne tidsdelingen», ikke som dokumentert tidsstabilitet. Dette forbeholdet skal også fremgå av sluttrapporten.

Bruk betegnelsen **årskontroll**, ikke estimert skadeinflasjon. Dataene mangler grunnlag for å skille kalenderutvikling fra reserve-/oppgjørsutvikling. Modellen gjelder registrert incurred, ikke dokumentert ultimate kostnad.

### Stoppregister

Tallgrensene under låses før kandidatfitting. Kalibrerings- og innflytelsesgrenser er praktiske revisjonssignaler, ikke formelle tester med kontrollert samlet feilrate.

| Situasjon | Konkret utløser | Pålagt handling |
|---|---|---|
| Datagrense eller lekkasje | Andre år enn 2022–2023 når analysefunksjoner; overlappende CV-personer; transformasjoner lærer fra valideringen | **Stopp kjøringen.** Rett datatilgangen eller implementeringsfeilen før modeller kjøres |
| Avvik i datagrunnlaget | Kontrolltall, radnøkler eller responsdefinisjon avviker uten forklart årsak; negativ materiell incurred; positiv kostnad uten registrert skade | **Stopp før seleksjon.** Avklar datafeil eller semantikk |
| Numerisk/støttemessig svikt | S0 bryter en gyldighetsregel; eller finalisten ikke kan refittes med den låste spesifikasjonen | **Stopp.** Ikke løs problemet ved å endre modellinnhold automatisk |
| Valgfri kandidat svikter | Kandidaten bryter støtte-/gyldighetskrav eller består ikke seleksjonsregelen | **Forkast kandidaten og fortsett etter algoritmen.** Dette utløser ikke ny modelljakt |
| Materiell EUR-feil | Totalt A/E utenfor 0,90–1,10 og bootstrapintervallet utelukker 1; eller produkt-/prediksjonsdesil-A/E utenfor 0,80–1,20 med minst 100 personer og intervallet utelukker 1 | **Stopp sluttgodkjenningen.** Undersøk feilen; ikke velg neste kandidat til en passer |
| Sterk innflytelse | Den faste topp-fem-personer-sensitiviteten endrer pooled forventet kostnad mer enn 5 %, eller mer enn 10 % i et produkt | **Stopp sluttgodkjenningen.** Vurder om forventningsestimatet har tilstrekkelig støtte |
| Små beløp styrer scoren | Skadeår med gjennomsnittsskade ≤1 EUR står for over 10 % av Gamma-deviancen, men under 0,1 % av kostnaden | **Faglig stopp.** Kontroller beløpsbetydning og målkonflikt; ikke flytt nullgrensen etter score |
| Tidsmessig forverring | Finalisten taper mot S0 i tidsfolden med mer enn både én cluster-SE og 0,5 % deviance | **Stopp finalistens sluttgodkjenning.** Ikke start ny tuning på 2023 |
| Segmentdrift over tid | For et produkt eller en skadeantallsgruppe med minst 100 personer i hvert år endres normalisert A/E med over 20 %, og bootstrapintervallet for forholdet utelukker 1 | **Faglig stopp.** Avklar om stabilitetsforutsetningen holder |
| Nye dokumenterte fakta | Ny kildeinformasjon endrer betydningen av skadeantall, incurred, dekning eller tidspunktet for prediktorene | **Stopp berørte konklusjoner.** Revider problemdefinisjonen før videre arbeid |

Et svakt eller motsatt forventet fortegn er ikke alene et stoppkriterium. Et lite, usikkert segment markeres som svakt støttet; det blir ikke automatisk en ny feature eller interaksjon.

Et globalt tidsnivåskift er heller ikke alene bevis for at segmentrelativitetene er feil. Det skal rapporteres og vurderes før senere prising.

### Hva en faglig stopp innebærer

Agenten leverer et kort stoppnotat i planens endringslogg med:

1. Utløst kriterium og konkrete resultater.
2. Hvilken forutsetning eller konklusjon som er berørt.
3. Hvilke valideringsresultater som allerede er sett.
4. Foreslått avklaring eller ny, avgrenset protokoll.

En faktisk implementeringsfeil kan rettes og det samme låste løpet kjøres på nytt, med sporbar forklaring. En metodisk endring etter at resultatene er sett krever brukerens godkjenning og ny protokollversjon. Resultatene fra den opprinnelige protokollen beholdes.

De samme foldene blir ikke «nye» eller uavhengige ved å bytte seed. Revidert modellering på samme utviklingsdata må omtales som videre utviklingsarbeid.

Storskadeproblemer kan begrunne en separat plan, men denne planen autoriserer **null halebehandlinger**. En eventuell senere behandling må ha egne støttekrav, treningsfoldbasert estimering og evaluering mot ukappet kostnad, med overskytende kostnad inkludert i forventningen.

## 4. Implementering, grensesnitt og verifikasjon

### Ansvarsdeling mellom agenter

Senioransvarlig eier protokollen, avviksvurderingen og den endelige faglige konklusjonen. Implementerende agenter skal ikke overstyre låste regler.

Arbeidet deles i tre avgrensede leveranser:

1. **Data og sikker kjøring:** Utviklingsdataflyt, avstemming, kontroller og kompatible rad-/foldnøkler.
2. **Modellering:** Kandidatregister, Gamma-modeller, CV og den eksakte seleksjonsalgoritmen.
3. **Diagnostikk og uavhengig kontroll:** Tabeller, plott, bootstrap, tidskontroll og etterprøving av implementasjonen.

Agentene får konkrete filområder. Bare én agent redigerer modellnotebooken om gangen. Hver leveranse skal kunne verifiseres før neste avhengige leveranse starter.

### Kodeplassering og nødvendige grensesnitt

- Modellspesifikasjoner, kandidatregister, preprocessingregler, CV-definisjon og seleksjonsalgoritme ligger i `glm_pricing_models.py`.
- Diagnostikk og presentasjon kan legges i beskrivende moduler under `src/`.
- Felles utviklingsinnlesing legges i `src_core_glm/model_data.py`, med en egen utviklingsfunksjon som ikke returnerer eller transformerer en testramme.
- Begge hovednotebookene bruker den sikre utviklingsflyten. Eksisterende celler som inspiserer teståret må ikke kjøres.

CV-resultatet skal minst inneholde:

- Kandidat-ID og full spesifikasjon.
- OOF-severity for alle poliseår, med entydig radnøkkel og fold-ID.
- Markering av radene som inngår i severity-scoren.
- Foldscore, samlet score, parametertall og gyldighetsstatus.
- Årsak til eventuell forkasting og en maskinlesbar seleksjonslogg.
- Treningsfoldens skadeantall i henholdsvis positivt utvalg og alle skadeår.

Til senere frekvens–severity-sammenkobling dokumenteres at F5 inkluderer de ti skadene i utelatte kostnadsår. En eventuell korreksjon må bygge på **skadeantall**, ikke andel rader, og læres i treningsfolden. Fullutvalgets forhold er omtrent `9979/9989 = 0,998999`. Dette er en mulig porteføljekorreksjon, ikke en identifisert individuell sannsynlighet for positiv skade. Selve sammenkoblingen og dens segmentantakelser avgjøres i fase 3.

### Obligatoriske tester

Testene skal særlig beskytte metode og datagrenser:

- Syntetiske 2024-rader med endrede utfall og kovariater påvirker ikke utviklingsramme, renselogger eller transformasjoner.
- Ingen person overlapper mellom trening og validering; alle utviklingsrader får nøyaktig én OOF-prediksjon.
- Endringer i valideringsdata påvirker ikke treningsmedianer, splineknuter eller estimerte parametere.
- Responsmasken, skadevektene og den vektede severityen reproducerer kontrolltallene.
- Seter 4, 5 og 6 havner i riktig kategori. Ingen alternativ binning introduseres.
- Aldersbasisene har riktig antall kolonner og full rang sammen med intercept.
- Kandidatregisterets kategorinivåer, referansenivåer og parametertall kontrolleres i foldvise designmatriser før første kandidatfit.
- Egen beregning av pooled Gamma-deviance samsvarer med bibliotekets score.
- Parvis cluster-SE bruker Gamma-bidrag og korrekt fortegn.
- Seleksjonen testes med syntetiske scorer for likhet, terskelgrenser, færre parametere, ugyldige kandidater, geografisærregelen og forkastet kombinasjon.
- Geografisærregelen testes når G1 er gyldig, men ikke kvalifiserer mot S0; når G1 er ugyldig; og når G2 slår G1, men ikke består forbedringskravet mot S0.
- Forenklingsregelen testes med en kandidat som består pooled tapsgrenser, men taper i tre av fem folder: Den skal ikke forkastes av et utilsiktet krav om foldflertall.
- Små valideringssegmenter beholder observasjonene i pooled score, men rapporteres med støtteforbehold; de særskilte 100-personerskravene kontrolleres på riktig OOF-/årspopulasjon.
- Kandidatbudsjettet overskrides ikke, og en stopp utløser ingen automatisk ny søkerunde.
- OOF-prediksjoner kan kobles til F5 uten manglende eller dupliserte nøkler.

Frekvensens tidligere kontroll av robuste standardfeil for Poisson skal ikke fremstilles som en verifikasjon for Gamma. Eventuell koeffisientinferens krever separat kontroll; den brukes ikke til variabelvalg.

Notebookene redigeres gjennom `.py`, synkroniseres med `uv run jupytext --sync ...` og kjøres med den sikre utviklingsflyten. Endringer i det felles datagrunnlaget verifiseres i begge hovednotebookene. Bevar eksisterende brukerendringer.

Før hver modelltilpasning dokumenteres modell-likningen og antakelsene i en kort markdown-celle. Hver outputseksjon får maksimalt én tabell og ett plott, eventuelt med flere delpaneler.

## 5. Gjennomføringsrekkefølge og ferdigkriterier

**Trinn 1 — Protokoll og datakontroll.** Opprett den separate planen, registrer brukerbeslutningene og de nye tallgrensene, rett dokumentasjonsfeil og etabler sikker utviklingsinnlesing. Logg at vanlig gruppe-CV erstatter den tidligere foreslåtte nestingen.

**Trinn 2 — Diagnostikk før fitting.** Avstem respons, kategoristøtte, små beløp og kostnadskonsentrasjon. Kontroller først kategorinivåer, referansenivåer, parametertall og rang i de foldvise designmatrisene. Avklar eventuelle datastopp før kandidatmodellene kjøres.

**Trinn 3 — Låst Gamma-løp.** Kjør de 14 faste spesifikasjonene og høyst én
kombinert kandidat. Lagre alle resultater, inkludert forkastede kandidater, og
avslutt seleksjonen etter algoritmen.

**Trinn 4 — Diagnostikk etter seleksjon.** Gjennomfør EUR-kalibrering, bootstrap, innflytelse, `N=1`-kontroll, kollinearitets-/fortolkningsdiagnostikk, tidskontroll og miksstandardisering. Utløste faglige stopp behandles før sluttmodellen godkjennes.

**Trinn 5 — Sluttfit og leveranse.** Når kontrollene er avklart, fit den valgte Gamma-spesifikasjonen på alle 5 698 positive skadeår. Behold S0 som dokumentert referanse. Lever formel, preprocessing, OOF-resultater, diagnostikk, beslutningslogg og begrensninger.

Sluttleveransen skal eksplisitt beskrive:

- Den tilsiktede preferansen for enklere modeller, også når de ikke forbedrer alle eller flertallet av foldene.
- At bare én samlet C1 testes. Hvis tre blokker velges, testes ikke alle to-av-tre-kombinasjoner; en bedre delkombinasjon kan derfor forbli uoppdaget. Det hevdes ikke at finalisten er best blant alle mulige kombinasjoner.
- Seleksjonsoptimismen fra gjentatt bruk av samme CV-observasjoner gjennom hele beslutningskjeden. 1-SE-/0,5 %-heuristikken gir ingen samlet feilratekontroll eller uavhengig evaluering av vinneren.
- At tidskontrollen omfatter én kalenderovergang, og at manglende påvist forverring ikke dokumenterer tidsstabilitet.
- Eventuell usikker fortolkning av korrelerte variabelblokker og hvilke valideringssegmenter som er for svakt støttet til egne konklusjoner.

**Trinn 6 — Separat beslutning om Inverse Gaussian.** Etter ferdig Gamma-leveranse kan brukeren godkjenne én IG-utfordrer med samme låste prediktorer og funksjonsformer. Ingen ny featureseleksjon. Ved godkjenning brukes samme gruppefolder, Gamma-score og forbedringsregel mot Gamma-finalisten, etterfulgt av samme relevante diagnostikk. Uten godkjenning slutter arbeidet ved Gamma-leveransen.

Fasen er ferdig når implementeringen er verifisert, seleksjonen kan reproduseres, alle stopp er avklart eller tydelig dokumentert som uavklarte, og konklusjonene samsvarer med resultatene.

**Rapporteringsgrensen er bindende:** Kandidatene er valgt med de samme gruppefoldene som brukes til å rapportere utviklingsscore. Derfor skal gevinsten ikke omtales som en uavhengig evaluering av seleksjonsprosedyren. Tidligere EDA og tidligere dokumentert innsyn i aggregerte testresultater oppheves heller ikke av denne protokollen. Ingen ny 2024-evaluering inngår; den krever senere eksplisitt godkjenning etter at modellspesifikasjonene er låst.

## Endringslogg

| Dato | Fase | Endring og begrunnelse |
|---|---|---|
| 2026-09-19 | CatBoost-residualdiagnostikk (ny §8) | Diagnose av valgt Gamma-modell (`med_forretningstype_uten_driver_age`) med nested, gruppebasert cross-fitting, vekt = skadeantall ($p=2$). Ingen stabil gevinst (pooled OOF-deviance 0,7233 mot 0,7236 dybde 1 og 0,7266 dybde 3; dybde 3 slår ikke GLM-en i mer enn én av fem folder), så modellvalget er uendret. Detaljer i `plans/glm_catboost_residual_diagnostics_plan.md`. |
| 2026-09-19 | Faserevisjon før Tweedie-implementering (foldkorreksjon) | Kandidatrom, seleksjonsregel og preprocessing er uendret. Foldene bygges nå på hele modellpopulasjonen (`build_group_folds`) og skjæres mot severity-delmengden: fit = train ∩ severity, score = val ∩ severity, prediksjon på hele valideringsfolden. `prediction_oof` dekker hele modellpopulasjonen slik at severity kan kombineres med frekvens rad for rad. Foldene er dermed andre enn i tidligere kjøringer, og resultatene er ikke direkte sammenlignbare. Teknisk endring ved ny kjøring av `glm_pricing_severity`: valgt modell-ID gikk fra `kjerne_med_kjøresone` til `med_forretningstype_uten_driver_age`. Endringen er ikke faglig vurdert her; den må vurderes før senere bruk av severity-modellen i benchmarken. |
| 2026-09-18 | Revidert Gamma-løp fullført | Tolv faste spesifikasjoner ble fittet etter at FU1 og BR1 falt i før-fit-støtteporten; C1 ble bygget og valgt som finalist med lineær alder og kjøresone. Faktisk hovedbudsjett var 65 fits. C1-deviance var 0,724835 mot 0,729006 for S0. Ingen småbeløps-, EUR-, innflytelses-, tids- eller segmentdriftstopp ble utløst. Sluttfitten ble derfor gjennomført med låst C1-formel. Resultatene er utviklingsresultater på de samme foldene som ble brukt til seleksjon. |
| 2026-09-18 | Før-fit-kontroll før revidert kandidatfit | Før-fit-porten avviste FU1 (`fuel_type = MISSING`, 39–43 unike personer i treningsfoldene) og BR1 (`CHEVROLET`, 42–50 og fire folder under 50). De to valgfrie kandidatene fittes ikke; BU1 og PF1 består. Register og designrapport beholdes, senere seleksjon bruker bare fittede gyldige kandidater, og ingen redningspooling innføres. Faktisk fast budsjett er derfor 60 (protokollmaksimum 70), og 65 dersom C1 bygges. |
| 2026-09-18 | Implementeringskorreksjon før første kandidatfit | Før-fit-designporten avdekket at FU1 har 11, ikke 10, parametere fordi den låste kategoriske preprocessingen gjør manglende `fuel_type` til et eksplisitt `MISSING`-nivå. Registeret og nivåkontrollen er korrigert uten å endre modellinnhold, preprocessing eller seleksjonsregler. Ingen kandidat ble fittet før avviket ble funnet. |
| 2026-09-18 | Protokollversjon 2.1 før revidert kandidatfit | Brukeren presiserte og godkjente at alle fire kategoriske felt skal være enkeltutfordrere. Lagt til BR1 (`vehicle_brand_pooled`) som egen blokk. Budsjettet er derfor 14 faste spesifikasjoner = 70 obligatoriske hovedtilpasninger og maksimalt 75 med én C1. Merkets referanse og BR1-parametertall låses fra utviklingsdata før fitting; øvrige regler er uendret. |
| 2026-09-18 | Protokollversjon 2.0 før revidert kandidatfit | Utvidet det låste kandidatregisteret med FU1 (`fuel_type`), BU1 (`business_type`) og PF1 (`payment_frequency`) som tre separate blokker. Låst referansenivåene D/NB/A og lagt de tre feltene samt `vehicle_brand_pooled` inn i nivå-, støtte- og designkontroll før fitting. Oppdatert registeret til 13 faste spesifikasjoner, 65 obligatoriske hovedtilpasninger og maksimalt 70 med én kombinert kandidat. Modellfamilie, folder, seed, preprocessing, score, seleksjonsgrenser, geografisærregel og stoppkriterier er uendret. `vehicle_brand_pooled` er kontrollfelt, ikke en fjerde ny kandidat, i samsvar med det eksplisitte 13-spesifikasjonsbudsjettet. |
| 2026-09-18 | Planrevisjon før severity-implementering | Presisert G1s rolle som gyldig, men ikke nødvendigvis kvalifisert referanse for G2; tilsiktet asymmetri i forenklingsregelen; kollinearitetsdiagnostikk uten seleksjonsvirkning; begrensningen ved én kalenderovergang; utestede delkombinasjoner; og seleksjonsoptimisme gjennom hele beslutningskjeden. Flyttet kontroll av parametertall/nivåer eksplisitt foran første fit og avklart treningens støttekrav mot valideringens diagnostiske støtte. Utvidet verifikasjons- og leveransekravene og rettet dokumentets filhenvisning. Kandidatrom, tallgrenser og modellbudsjett er uendret. |
| 2026-09-18 | Faserevisjon etter diagnostikk | Kandidatrom, terskler og modellbudsjett er uendret. §4 ga C1 som finalist. Bootstrap, kalibrering og innflytelse i §5 utløste ingen stopp. Tidskontrollens pooled-regel utløste heller ikke stopp; «ingen forverring påvist i denne tidsdelingen». Korrigert handoff-feil: 2023-normalisert A/E 1,33/0,74 er ikke år-til-år-drift; korrekt Q = (normalisert A/E 2023)/(normalisert A/E 2022) var ca. 1,02–1,07 for skadeantallsgruppene, og alle 95 % cluster-bootstrap-CI inkluderte 1. Segmentdriftstopp utløses derfor ikke. |
