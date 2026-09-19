# Litteraturgjennomgang: severity-drivere i motorforsikring

**Status:** Dette er kun litteraturgjennomgangen beskrevet i `plan_phase_prompt_3.md`,
seksjon "Litteraturforankret baseline". Det er **ikke** en fullstendig severity-plan,
og **ingen** funksjonsform eller variabel her er en vedtatt modellspesifikasjon.
Baseline-variabler og funksjonsformer skal fortsatt godkjennes av brukeren før noe
testes mot CV-foldene.

## Metode og avgrensning

Søket ble delegert til en egen agent ("Luna"-rollen) med instruks om å finne ekte,
verifiserbare kilder (fagfellevurderte artikler, aktuarielle rapporter/monografier)
og eksplisitt flagge alt den ikke kunne verifisere — aldri fylle igjen hull med en
plausibel, men usporet påstand. 31 kildeoppslag ble gjort (WebSearch/WebFetch),
gruppert i fire blokker: kjøretøy, fører, geografi, kontrakt.

Kilder er vurdert etter to akser, i tråd med planens prioritering:

- **Dekningstype**: kasko/comprehensive (vårt datagrunnlag) vs. motoransvar (MTPL/TPL)
  vs. personskade/ulykkesalvorlighet. Dette er en helt annen skademekanisme enn
  materiell egenskade og svekker overførbarheten mer enn geografisk avstand alene.
- **Responsdefinisjon**: kostnad per registrert skade (vår `severity`) vs. skadegrad/
  injury severity vs. skadefrekvens. Flere kilder som dukket opp i søket handler om
  frekvens eller personskadealvorlighet, ikke kostnadsseverity — disse er markert
  tydelig og ikke brukt til å hevde noe om kostnadsseverityens retning.

Der websøket bare fant en sekundærpåstand uten sporbar primærkilde, eller en kilde
der PDF-innhentingen feilet, er dette eksplisitt flagget under tabellen og **ikke**
brukt som grunnlag for noen rad.

Alle kilder som faktisk brukes som grunnlag i tabellen er i ettertid verifisert på
nytt (egen WebFetch/WebSearch-runde, uavhengig av førsteutkastet) mot tittel,
forfattere, tidsskrift/utgiver og — der relevant — det konkrete faktapåstanden
gjelder. Se referanselisten til slutt for direkte lenker og hva som ble bekreftet.
Referansene i tabellen viser til `[1]`–`[5]` i den listen.

Tre nivåer av støtte holdes atskilt gjennom hele dokumentet, slik planen krever:

1. **Relevans** — er variabelen i det hele tatt dokumentert som severity-driver?
2. **Retning** — hvilken vei peker effekten, og under hvilke betingelser?
3. **Funksjonsform** — er den konkrete transformasjonen (log, spline, kategorisering)
   selve litteraturfunnet, eller en generell GLM-konvensjon/arbeidsantakelse vi
   pålegger en variabel litteraturen bare sier er ikke-lineær eller monoton?

Manglende litteraturstøtte for en variabel likestilles ikke med manglende prediktiv
verdi — der det gjelder er det skrevet eksplisitt.

## Evidenstabell

| Variabel | Dokumentert sammenheng / retning | Justert eller marginalt | Dekning & respons i kilde | Kilde | Overførbarhet til våre data | Foreslått funksjonsform | Litteraturstøttet vs. arbeidsantakelse |
|---|---|---|---|---|---|---|---|
| `log_vehicle_value` | Relevans: sterk, men mekanisk (dyrere bil → dyrere reparasjon/totalskade). Retning: positiv. | Ingen studie testet dette justert eller marginalt — sammenhengen er ikke hentet fra en severity-regresjon. | — (ingen direkte kilde) | Ingen severity-spesifikk kilde funnet for funksjonsform | Relevans-overførbarhet høy (samme dekning: kasko); funksjonsform ikke overførbar fra noen kilde | Log-transform (multiplikativ tolkning) | **Relevans**: arbeidsantakelse basert på dekningsmekanikk, ikke sitert funn. **Retning**: arbeidsantakelse. **Form**: ren GLM-konvensjon for positive, høyreskjeve pengevariabler — ikke et litteraturfunn om denne variabelen spesifikt |
| `performance_hp_per_tonne` | Relevans: moderat-god. Retning: positiv (mer ytelse → høyere severity) i justert modell. | **Justert** (multivariat GLM + kontrastanalyse) | MTPL (motoransvar), ikke kasko. Slovakia. | [2] Reiff et al. (2022) | Moderat: annen dekningstype (ansvar vs. kasko) og annet marked, men mekanismen (kraftigere motor → mer skade ved uhell) er plausibelt overførbar | Ingen presis form spesifisert i kilden | **Relevans + retning**: litteraturstøttet (justert), men fra feil dekningstype. **Form**: arbeidsantakelse (prosjektets `1000/power_to_weight_ratio`-transformasjon er egendefinert, ikke fra kilden) |
| `vehicle_age` | Personskade-litteratur finnes, men peker på en **annen** responsvariabel enn vår | Justert (GLMM) i personskadestudien | **Personskadegrad hos ulykkesofre**, ikke materiell kostnadsseverity. Spania (DGT-politidata 2016). | [3] Santolino, Céspedes & Ayuso (2022) | **Lav / potensielt villedende**: mekanismen der (eldre bil → svakere sikkerhetsteknologi → mer alvorlig personskade) er ikke samme mekanisme som materiell egenskade (eldre bil → lavere markedsverdi → typisk lavere utbetaling). Retningen kan være motsatt for kostnadsseverity | Ingen forslag — utilstrekkelig grunnlag | **Ingen litteraturstøtte for retning i vår responsdefinisjon.** Denne kilden må ikke siteres for å hevde noen retning i kostnadsseverity |
| `vehicle_brand` (pooling) | Luksus-/premiummerker signifikant høyere severity | Justert (samme studie som ytelse) | MTPL, Slovakia (samme mismatch som ytelse over) | [2] Reiff et al. (2022) | Moderat (samme forbehold: ansvar, ikke kasko) | Pooling av lavvolummerker til `OTHER` (kredibilitetsmetodikk) | **Relevans**: moderat støttet, feil dekningstype. **Form** (pooling-terskel): ren kredibilitetskonvensjon, ikke fra severity-litteraturen |
| `fuel_type` | Kun forbrukerrettede sammenligningssider (ikke fagfellevurdert) | Ingen justert kilde — kildene selv peker på at effekten trolig er confoundet av bilverdi/motorstørrelse | Uklar/kommersielt innhold, blandet marked | Kommersielle sammenligningssider (UK/India) — lav evidensvekt, ikke sitert som funn | Lav | Kategorisk dummy (irrelevant funksjonsform-spørsmål) | **Ingen brukbar litteraturstøtte.** Ren arbeidsantakelse fra `current.md` |
| `seats` | **Ingen brukbar kilde funnet** for kostnadsseverity | — | — | — | — | — | Ren arbeidsantakelse (størrelsesproksy korrelert med bilverdi) — eksplisitt gap |
| `driver_age` | Bred konsensus om ikke-lineær (U-formet) alderssammenheng, men mesteparten av belegget gjelder **frekvens** eller **personskadegrad**, ikke kostnadsseverity | Blandet; det severity-spesifikke sitatet kunne ikke spores til navngitt kilde | Varierer per kilde; ingen god kasko-severity-match funnet | Ingen severity-spesifikk kilde verifisert | Lav for kostnadsseverity spesifikt, selv om ikke-linearitet i alder er en velkjent generell foisikringssammenheng | Ingen splinegrad e.l. foreslått herfra | **Relevans**: svakt støttet for kostnadsseverity (sterkere for frekvens). **Retning og form**: arbeidsantakelse basert på egen EDA i `current.md`, ikke litteratur |
| `driving_experience_years` / `age_driving_licence` | Ulykkesalvorlighet varierer mellom novise/erfarne førere i én ML-studie | Feature importance (SHAP/CatBoost), ikke justert regresjonskoeffisient | Ukjent marked, ulykkesalvorlighet (personskade), ikke kostnadsseverity | [4] Chen, Shao & Ji (2021) | Lav (feil responsdefinisjon, feature importance ≠ justert effekt) | Ingen | **Ingen litteraturstøtte for kostnadsseverity.** Multikollinearitet med `driver_age` er en datamekanisk realitet, ikke noe som trenger støtte |
| `circulation_area` (urban/rural) | **Ingen fagfellevurdert kilde.** Svak industrikilde antyder urban → høyere severity (reparasjonslønn), motsatt av `current.md`s hypotese (rural → høyere severity via fart) | — | Kommersielt/anekdotisk, blander frekvens- og severity-mekanismer | Spanske ekspatblogger — lav evidensvekt | Lav | Kategorisk | **Åpent empirisk spørsmål.** Ingen kilde avgjør retningen; må presenteres som sådan, ikke som en bekreftet sammenheng i noen retning |
| `municipality_type` | **Fullstendig gap** — ingen litteratur funnet, verken Spania eller andre markeder | — | — | — | — | — | Ren arbeidsantakelse / ren datadrevet kandidat |
| `policy_type` (COMP_E vs. COMP_N, egenandel) | Generell forsikringsøkonomisk teori om egenandeler og moral hazard er veletablert, men ikke motor-severity- eller Spania-spesifikk | Teoretisk/generell, ikke en direkte justert kasko-severity-studie | Hovedsakelig ansvar (BI liability), ikke kasko | Klassisk forsikringsøkonomisk litteratur (generisk, ikke navngitt enkeltstudie i denne runden) | Moderat: mekanismen er kjent, men **mekanisk trunkering** (egenandelen trekkes fra utbetalingen) og **atferdsmessig moral hazard** er to separate forklaringer som dataene ikke kan skille mellom | Kategorisk | **Relevans**: teoretisk støttet. **Retning**: forventet (COMP_N > COMP_E), men den mekaniske og atferdsmessige forklaringen må ikke sammenblandes i teksten |
| `payment_frequency` | **Ingen severity-spesifikk kilde.** Ett sitat om betalingsfrekvens fantes, men gjaldt frekvens og kunne ikke spores til navngitt primærkilde | — | — | Usporet sekundærpåstand — ikke brukt | — | Kategorisk | Ren arbeidsantakelse; `current.md`s skepsis (sannsynligvis ikke viktig for severity) forblir verken bekreftet eller avkreftet |

## Eksplisitte gap (ingen brukbar kilde funnet)

- `seats` × kostnadsseverity
- `municipality_type` × kostnadsseverity
- `payment_frequency` × kostnadsseverity
- Konkret funksjonsform (spline/log/polynom-grad) for **samtlige** variabler — ingen
  kilde i dette søket går utover å dokumentere at en ikke-lineær eller monoton
  sammenheng finnes

## Kilder som krever fulltekstlesning før tung sitering

- **[5] Oyugi (2010)** — riktig, fungerende lenke er funnet og bibliografiske
  detaljer (tittel, forfatter, år, kongress) er bekreftet direkte fra IAAs egen
  ressursside. Selve PDF-en lot seg derimot ikke tekstlesning automatisk (both
  ved førstegangs- og andregangsforsøk — komprimert/binær PDF-strøm). Konklusjonen
  om at lognormal passer bedre enn Gamma/Weibull/eksponential (K-S- og
  Anderson-Darling-tester) er bekreftet av flere uavhengige sekundærkilder
  (IAA-ressurssiden, academia.edu-speilet, sciepub-referanseoppføringen), men
  **er en ubetinget fordelingstest, ikke en regresjonsjustert prediktiv
  sammenligning** — les primærteksten selv før den siteres tungt i modellvalget.
- **IIHS/HLDI-bulletinen** om at 1 hk ekstra per 100 lbs gir ca. 5 % høyere tap
  kunne **ikke** identifiseres som en spesifikk, navngitt bulletin i
  oppfølgingssøket (flere HLDI-bulletiner om hestekrefter/vekt finnes, men ingen
  matchet tallet direkte). Tallet forblir uverifisert og er **ikke** brukt noe
  sted i tabellen over.

## Uverifiserte påstander som bevisst er utelatt

- Et sitat om at "district, Bonus-Malus, merke, betalingsfrekvens og
  poliseansiennitet er de fem viktigste variablene for skadefrekvens" dukket opp
  to ganger i søket, men kunne ikke spores til en navngitt, verifiserbar
  primærkilde. Gjelder uansett frekvens, ikke severity. Ikke brukt.

## Sidemerknad: distribusjonsfamilie (ikke del av prediktor-evidensen)

Dette fremkom underveis i søket og er ikke en del av evidenstabellen for
severity-drivere, men er relevant bakgrunn for en senere, egen diskusjon om
modellfamilie (som plan_phase_prompt_3.md holder åpen):

- **Gamma med log-link** har bred, veletablert konsensus som severity-GLM-standard
  ([1] Goldburd, Khare & Tevet, *CAS Monograph 5*).
- [5] Oyugi (2010, se over) fant lognormal bedre tilpasning enn Gamma i en
  *ubetinget* fordelingstest for kasko i Kenya — dette sier ikke noe om hvilken
  modell som predikerer best justert for kovariater, og bør ikke brukes til å
  forhåndsvelge lognormal fremfor Gamma i regresjon.

## Konklusjon til bruk i severity-planen

Litteraturen gir **relevans- og retningsstøtte** (delvis justert, men fra
dekningstyper som ikke er kasko) for ytelse (`performance_hp_per_tonne`) og
bilmerke. Den gir **ingen** severity-spesifikk funksjonsform for noen variabel —
enhver log-transform, spline eller kategorisering i den kommende baselinen er en
GLM-konvensjon eller arbeidsantakelse fra `current.md`/EDA, ikke et sitert funn,
og må presenteres som det. `circulation_area` er et reelt åpent spørsmål med
motstridende (svake) indikasjoner på retning. `seats`, `municipality_type` og
`payment_frequency` har ingen litteraturstøtte i noen retning, uten at dette betyr
at de mangler prediktiv verdi i våre data. `vehicle_age`-litteraturen som ble
funnet gjelder en annen skademekanisme (personskade) og må ikke brukes til å
hevde en retning for kostnadsseverity.

## Referanseliste (verifisert)

Alle lenker er kontrollert direkte (WebFetch) i en egen, uavhengig
verifiseringsrunde etter at evidenstabellen ble skrevet. "Bekreftet" betyr at
tittel, forfattere og den konkrete faktapåstanden i tabellen over ble lest
direkte fra kilden eller — for [5], der selve PDF-en ikke lot seg tekstlese
automatisk — fra utgiverens egen ressursside pluss flere uavhengige
sekundærkilder.

1. **Goldburd, M., Khare, A. & Tevet, D. (2016).** *Generalized Linear Models
   for Insurance Rating.* CAS Monograph Series, No. 5. Casualty Actuarial
   Society.
   https://www.casact.org/sites/default/files/database/monographs_papers_05-goldburd-khare-tevet.pdf
   — Bekreftet: tittel, forfattere, utgiver, at Gamma/log-link presenteres som
   standardvalg for severity-modellering.

2. **Reiff, M., Šoltés, E., Komara, S., Šoltésová, T. & Zelinová, S. (2022).**
   "Segmentation and estimation of claim severity in motor third-party
   liability insurance through contrast analysis." *Equilibrium. Quarterly
   Journal of Economics and Economic Policy*, 17(3), 803–842.
   DOI: https://doi.org/10.24136/eq.2022.028
   (også tilgjengelig på https://journals.economic-research.pl/eq/article/view/2058)
   — Bekreftet: tittel, forfattere, tidsskrift, volum/hefte/sider, og at
   funnene (vekt, motoreffekt, alder og merke på bil, forsikringstakers alder
   og distrikt som signifikante, justerte drivere av claim severity i MTPL)
   stemmer med det som er sitert i tabellen.

3. **Santolino, M., Céspedes, L. & Ayuso, M. (2022).** "The Impact of Aging
   Drivers and Vehicles on the Injury Severity of Crash Victims."
   *International Journal of Environmental Research and Public Health*,
   19(24), 17097. DOI: https://doi.org/10.3390/ijerph192417097
   PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9778893/
   — Bekreftet: tittel, forfattere, tidsskrift/volum/artikkelnummer, bruk av
   DGT-politidata for Spania (2016), GLMM justert for kjønn, kjøretøytype,
   vei-/siktforhold og ulykkestype, og at bilalderseffekten øker opp til 18 år
   og deretter flater ut. Responsvariabelen er personskadegrad hos
   ulykkesofre — ikke materiell kostnadsseverity — som presisert i tabellen.

4. **Chen, S., Shao, H. & Ji, X. (2021).** "Insights into Factors Affecting
   Traffic Accident Severity of Novice and Experienced Drivers: A Machine
   Learning Approach." *International Journal of Environmental Research and
   Public Health*, 18(23), 12725.
   PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC8656871/
   — Bekreftet: tittel, forfattere, tidsskrift/volum/artikkelnummer, og at
   metoden er CatBoost + SHAP feature importance, ikke en justert
   regresjonskoeffisient. Utvalgets land er ikke fastslått i denne
   verifiseringen.

5. **Oyugi, M. A. (2010).** "Actuarial Modelling for Insurance Claim Severity
   in Motor Comprehensive Policy using Industrial Statistical Distributions."
   International Congress of Actuaries (ASTIN-seksjonen), Cape Town.
   https://actuaries.org/app/uploads/2025/07/ICA2010_ASTIN_22_final_-paper_Oyugi.pdf
   (ressursside: https://actuaries.org/resources-post/actuarial-modelling-for-insurance-claim-severity-in-motor-comprehensive-policy-using-industrial-statistical-distributions/)
   — Bekreftet via IAAs ressursside: tittel, forfatter, år, kongress. Selve
   PDF-en kunne ikke tekstleses automatisk i to forsøk (komprimert PDF-strøm).
   Konklusjonen (lognormal passer bedre enn eksponential/Gamma/Weibull for
   kaskodata fra Nairobi, testet med Kolmogorov-Smirnov/Anderson-Darling) er
   korroborert av flere uavhengige sekundærkilder, men **primærteksten bør
   leses i fulltekst** før den siteres tyngre enn som en sidemerknad om
   distribusjonsvalg.

**Ikke brukt / bevisst utelatt fra referanselisten** (se begrunnelse i
teksten over): IIHS/HLDI-bulletinen om hestekrefter/vekt (kunne ikke
identifiseres som en spesifikk, navngitt bulletin), det usporede "fem
viktigste variabler"-sitatet, og de kommersielle sammenligningssidene om
drivstofftype/bilmerke (ikke fagfellevurdert, brukt kun som eksplisitt lavt
vektet anekdote i tabellen — ikke som verifisert funn).
