```markdown
# Overføring til ny sesjon: planlegging av severity-modellering

## Oppdrag og arbeidsform

Start i plan mode. Utarbeid en grundig og gjennomførbar plan for fase 2:
severity-modellering og storskadediagnostikk.

Planen skal skrives som en **egen Markdown-fil i `plans/`**, eksempelvis
`plans/severity_modeling_plan.md`. Ikke overskriv den eksisterende
`plans/glm_pricing_models_plan.md`.

Før planen ferdigstilles:
1. Kryssjekk de nå oppdaterte tallene i den opprinnelige planen mot notebookens
   faktiske resultater.
2. Gjennomfør litteraturarbeidet beskrevet nedenfor.
3. Skill eksplisitt mellom allerede vedtatte prinsipper og konkrete valg som
   fortsatt trenger brukerens godkjenning.

Ikke implementer eller fit nye modeller i denne planleggingssesjonen.
Ikke bruk 2024-data.

Samtalen skal være på norsk. Forklar intuitivt til en junior, med senior
aktuarfaglig presisjon. Bruk engelske variabel- og funksjonsnavn.
Unngå LaTeX i samtalen; notebookens senere dokumentasjon følger prosjektreglene.

## Prosjekt og relevante filer

Prosjektet er en porteføljebenchmark for motorforsikringsprising på spanske data.

- `glm_pricing_models.py`: modellspesifikasjoner, CV, frekvensresultater og
  beslutningsregister.
- `glm_pricing_models.ipynb`: lagrede resultattabeller og figurer.
- `analysis.py` / `analysis.ipynb`: deskriptiv analyse.
- `current.md`: brukerens foreløpige vurdering av severity-prediktorer.
- `plans/glm_pricing_models_plan.md`: eksisterende plan og historikk.
- `src/model_data.py`: felles datagrunnlag og avgrensninger.
- `src/glm_core.py`, `src/glm_diagnostics.py`, `src/phase_2/`:
  eksisterende støttefunksjoner. Mappenavnet `phase_2` betyr ikke at
  severity-modelleringen allerede er implementert.
- `src/abess_diagnostics.py`: separat frekvensdiagnostikk.

Les gjeldende AGENTS.md/CLAUDE.md. Bevar eksisterende brukerendringer.
Parede notebooks redigeres senere gjennom `.py`, ikke direkte i `.ipynb`.

## Status og tallkontroll før planlegging

Ferdig modelleringsfase er **fase 1: frekvens**.
Neste fase er **fase 2: severity og storskader**.

Brukeren opplyser at tallene i den opprinnelige planen nå er oppdatert.
Dette skal verifiseres, ikke antas.

Ved forrige gjennomgang ble følgende lest fra notebook-output:
- 55 246 poliseår.
- 37 126,014 eksponeringsår.
- 9 989 registrerte skader.
- 5 707 poliseår med skade.
- 5 698 positive skadeår i severity-rammen.
- Ni null-/nærnullkostnadsår utelatt fra Gamma-severity.
- Skadevektet severity inkludert disse: omtrent 871,771 EUR.
- Skadevektet severity i positivt utvalg: omtrent 872,645 EUR.

Disse verdiene er kontrollpunkter, ikke instruks om å overskrive nyere resultater.
Skill mellom:
- poliseår og registrerte skader;
- uvektet og skadevektet snitt;
- positivt severity-utvalg og alle registrerte skader;
- gjennomsnittsskade per poliseår og individuell skade;
- någjeldende populasjon og eldre resultater som inkluderte CC.

### ABESS-feilen som særlig må kontrolleres

Den gamle planen hadde feil retning på sammenligningen:
- F5 OOF-deviance: 1,121675.
- ABESS OOF-deviance: 1,121144.
- ABESS hadde dermed omtrent 0,000531 **lavere/b edre** deviance.
- Parvis cluster-SE: omtrent 0,000840.
- Gevinsten tilsvarer omtrent 0,63 standardfeil.

Kontroller at oppdatert tekst sier **bedre**, og at fortegnskonvensjonen er korrekt.
Forskjellen dokumenterer fortsatt ikke en robust generaliseringsgevinst.
Den låste frekvensmodellen F5 skal ikke endres gjennom denne planrevisjonen.

Kontroller også at lagrede notebook-resultater svarer til gjeldende kode.
Hvis dette ikke kan fastslås uten rekjøring, dokumenter avviket og nødvendig
verifikasjon. Ikke presenter ukontrollerte tall som bekreftede.

## Vedtatt hovedretning

Vi skal sammenligne:
1. En faglig begrunnet, litteraturforankret baseline som låses på forhånd.
2. En avgrenset datadrevet utvidelsesprosedyre.

Hovedspørsmålet er:
**Gir begrenset datadrevet tilpasning en stabil forbedring utover en modell
vi kunne spesifisere faglig på forhånd?**

Nestet gruppe-CV skal evaluere seleksjonsprosessen.
Litteraturbaselinen gir samtidig et selvstendig referansepunkt.

## Litteraturforankret baseline (ALLEREDE GJENNOMFØRT OG FINNES I MAPPEN SEVERITY LITTERATURGJENNOMGANG)

DETTE ER ALLEREDE GJENNOMFØRT OG FUNNENE FINNES I SEVERITY LITTERATURGJENNOMGANG MAPPEN.

Viktig: Vi er opptatt av prediksjon her, så sammenhengen i artikkelene behøver ikke være kausal.

Mål med litteraturgjennomgangen:
- Undersøk severity-drivere i motorforsikring.
- Prioriter spanske studier der relevante studier finnes.
- Sammenlignbar dekning og responsdefinisjon veier tyngre enn geografi alene.
- Din jobb er å validere at tolkningen fra disse artikkelene faktisk stemmer.
- Brukeren skal godkjenne baseline-variabler og funksjonsformer før nye
  kandidater testes mot CV-foldene.

Den spesifikke litteraturgjennomgangen av severity-drivere ble utsatt og
skal ikke omtales som allerede gjennomført. Et tidligere, bredere søk på
severity-metodikk finnes i samtalehistorikken, men erstatter ikke dette arbeidet.

Lag en evidenstabell i planen med:
- variabel;
- dokumentert sammenheng og eventuell retning;
- om effekten er justert for andre variabler;
- dekning og responsdefinisjon;
- kilde;
- overførbarhet til våre data;
- foreslått funksjonsform;
- hva som er litteraturstøttet, og hva som er en enkel arbeidsantakelse.

Skill mellom støtte for:
1. At variabelen er relevant.
2. Effektens retning.
3. Den konkrete funksjonsformen.

Ikke presenter valgt splinegrad eller logtransformasjon som et litteraturfunn
dersom kilden bare dokumenterer en generell sammenheng.
Ikke likestill manglende litteraturstøtte med manglende prediktiv verdi.

Litteraturbaselinen er forhåndsdefinert for kommende modellering, men prosjektet
har allerede brukt 2022–2023 i EDA og referansemodeller. Ikke kall den fullstendig
uavhengig av tidligere datainnsikt.

## Variabel seleksjon i grunn spesifikasjonen:

Grunnspesifikasjonen skal også inkludere variabler som ser særlig lovende ut basert på både den deskriptive analysen og litteraturgjennomgangen.

## Seleksjonsprosedyre: godkjent motforslag

Brukerens opprinnelige forslag var univariat GLM-screening mot nullmodellen.
Dette er erstattet av hovedagentens motforslag.

Begrunnelse:
- Univariat screening måler marginal sammenheng.
- En kandidat kan bare gjenspeile produktmiks eller bilverdi.
- En relevant justert effekt kan skjules i en marginal analyse.
- En felles terskel på treningsdevians favoriserer fleksible blokker.

Foretrukket løsning:
- Et lite, forhåndsdefinert sett med **utvidelser av den faste kjernen**.
- Sammenlign kjerne mot kjerne + kandidat.
- Vurder kandidatene direkte med indre gruppe-CV fremfor å legge inn
  univariat treningsscreening først.
- Behandle hele variabelblokker samlet: alle kategorinivåer eller hele splinen.
- Definer eventuelle faglig plausible par og erstatninger på forhånd.
- Ikke åpne nye kombinasjoner etter å ha sett valideringsresultatene.

Kjernen holdes fast gjennom sammenligningen av enkeltutvidelser.
Hvordan kvalifiserte tillegg eventuelt kombineres må spesifiseres.
To kandidater som hver bidrar mot kjernen kan forklare samme restsignal.

## Modellbudsjett og stoppregel

Det er vedtatt at kandidatrom, funksjonsformer og stoppregel låses på forhånd.
Nestet CV er ikke en tillatelse til ubegrenset modelljakt.

Planen må angi:
- fullstendig kandidatregister;
- obligatoriske og valgfrie blokker;
- tillatte funksjonsformer;
- eventuelle forhåndsdefinerte kombinasjoner;
- seleksjonsregel og regel ved nesten like resultater;
- maksimal mengde datadrevne valg;
- eksplisitt sluttpunkt uten nye runder etter resultatet.

Tidligere diskutert forslag til reduksjon:
1. Fast fullmodell.
2. Høyst fire forhåndsdefinerte blokker vurderes fjernet.
3. Én fjerning om gangen mot samme fullmodell.
4. Høyst én samlet redusert kandidat.
5. Hvis samlet reduksjon ikke består kontrollen, behold fullmodellen.
6. Ingen ny reduksjonsrunde.

**Fire blokker og fem reduksjonskandidater er ikke vedtatte tallgrenser.**
Vurder også om egen reduksjonsrunde er nødvendig når en liten litteraturkjerne
og avgrensede tillegg allerede er hoveddesignet.

## Nestet gruppe-CV: vedtatt

- Gruppér på `insured_id`, slik at samme ID ikke ligger på begge sider av en fold.
- Indre gruppe-CV velger modell innen den låste prosedyren.
- Ytre gruppe-CV evaluerer hele prosedyren mot litteraturbaselinen.
- Hele datadrevne seleksjonen gjentas innen hver ytre treningsfold.
- Imputasjon, splinebasis og øvrige lærte transformasjoner må læres på riktig
  treningsnivå, uten tilgang til valideringsutfall.
- Eventuell datadrevet screening må også ligge innenfor korrekt treningsnivå.
- Ytre resultater skal ikke brukes til gjentatt revidering av kandidatrommet.

Vi har valgt bort en separat sjette holdout-gruppe.
En slik gruppe ville ha omtrent 950 positive skadeår ved en sjettedel av dataene,
men få observasjoner for sjeldne hendelser og svak presisjon for små forbedringer.

Bootstrap kan beskrive usikkerhet, men skaper ikke nye uavhengige observasjoner
eller manglende storskader.

Antall indre og ytre folder er fortsatt ikke låst.
Planen må begrunne valget ut fra skadeår, unike ID-er og segmentstøtte.

Planen må også forklare:
- at ytterfoldene kan velge ulike modeller;
- hvordan endelig spesifikasjon velges på hele utviklingssettet;
- at ytre score gjelder seleksjonsprosedyren, ikke en direkte evaluering av
  sluttfitten på alle data;
- hvordan senere frekvens–severity-sammenkobling sikres med kompatible
  trenings- og valideringsindekser.

Cluster-SE håndterer noe av avhengigheten i scoreforskjeller, men korrigerer
ikke i seg selv seleksjonsoptimisme eller all usikkerhet fra modelltrening.

## Tidskontroll og skadeutvikling: vedtatt

Behold tidskontrollen **2022 → 2023**.

Det er nyttig å beholde en struktur som kan samordnes med frekvensmodellen,
men severity trenger egne diagnostiske vurderinger.

Implementeringen skal senere inkludere:
1. Standardisering av porteføljemiks:
   sammenlign år på en felles kovariatfordeling.
2. Årsvise forskjeller per produkt, skadeantallsgruppe og beløpsnivå,
   samt null-/nærnullkostnader.
3. Skille mellom globalt nivåskift og endrede segmentrelativiteter.

Kontrollene kan vise om miks forklarer deler av forskjellen.
De kan ikke sikkert identifisere kalenderutvikling versus reserveutvikling.

Dagens dokumenterte datagrunnlag mangler separate betalte beløp og reserver,
gjentatte verdsettinger og nødvendige oppgjørsdatoer.
Kort oppgjørstid for kasko er en arbeidsantakelse som reduserer bekymringen,
ikke bevis for at alle år er like modne.

Bruk betegnelsen **årskontroll**, ikke estimert skadeinflasjon.
Modellen gjelder registrert incurred, ikke dokumentert ultimate kostnad.
En god tidskontroll krever ikke at årsaken til nivåendringen er identifisert.

Planen må avgrense tidskontrollens rolle slik at den ikke blir en ekstra
iterativ optimaliseringsarena.

## Storskader: vedtatt

Dataene er aggregert per polise × år.
For N>1 kan én stor skade være skjult blant flere små.
Individuell skadefordeling kan ikke rekonstrueres fra aggregatet.

Før nye modellvalg:
- visualiser kostnadskonsentrasjon;
- oppsummer antall overskridelser og kostnadsandeler;
- skill samlet poliseårskostnad fra gjennomsnitt per registrert skade;
- dokumenter maskeringen som strukturell begrensning;
- vurder beløpsklumper som registreringsdiagnostikk, uten å anta årsaken.

N=1:
- brukes som sensitivitet med ett registrert krav;
- vurderes mot kovariatjustert OOF-forventning fra hovedmodellen;
- sammenlignes ikke bare med rått snitt for andre skadeantallsgrupper;
- er et selektert utvalg også etter justering for observerte kovariater;
- gir ikke nødvendigvis én fysisk hendelse, siden skadetellingen er uavklart.

Ikke bruk «mindre flåter» som forklaring: repoet dokumenterer ikke flåtestørrelse.
Ulik eksponering og skaderisiko kan allerede gi ulik sannsynlighet for N=1.

Den gamle regelen «over 5 % av kostnaden» erstattes.
Eventuell storskadebehandling krever:
- tilstrekkelig støtte for den konkrete behandlingen;
- evaluering mot **ukappet kostnad**;
- at overskytende kostnad ikke bare fjernes fra estimatet;
- foldvis læring av eventuelle tillegg og datadrevne terskler.

Kapping er ikke besluttet.
Terskler, støttekrav, kandidatbehandlinger og antall behandlinger må foreslås.
Skille tydelig mellom diagnostikk før fitting og modellavhengig diagnostikk
etter en ukappet baseline.

## Respons, modeller og evaluering: forslag som må konkretiseres

Følgende har vært anbefalt, men skal ikke behandles som fullt låste detaljer:

### Respons
Gjennomsnitt per registrert skade:
`property_incurred / property_claims`, med skadeantall som vekt.

Null-/nærnullkostnadsår inngår i frekvens, men ikke positiv Gamma-severity.
Forskjellen er liten på porteføljenivå, men skal dokumenteres og håndteres
ved sammenkobling med frekvens.

En alternativ oppdeling i fase 3 kan være:
sannsynlighet for positivt skadeår × forventet kostnad gitt positivt skadeår.
Denne unngår skadeantallet, men krever egen eksponeringshåndtering.
Den er ikke vedtatt som hovedmodell.

### Modellfamilier
Gamma med log-link er foreslått baseline.
Inverse Gaussian og lognormal har vært diskutert som utfordrere.
Velg og begrunn et begrenset sett.

Lognormal må predikere forventet beløp på originalskala.
Smearing og eventuell heterogen spredning må håndteres eksplisitt.
Bedre ubetinget AIC i EDA avgjør ikke hvilken regresjon som predikerer best.

### Evaluering
Anbefalt ramme:
- samme skadeantallsvektede Gamma-deviance for kandidatene;
- EUR-kalibrering: faktisk kostnad mot sum av skadeantall × predikert severity;
- kalibrering totalt, per produkt og relevante segmenter;
- kalibrering etter predikert severity;
- parvise scoreforskjeller og usikkerhet gruppert på `insured_id`;
- foldstabilitet, tidskontroll og innflytelsesdiagnostikk;
- senere evaluering av samlet ren premie.

Gamma-deviance vurderer relative feil og trenger derfor EUR-kontroller.
RMSE kan være sekundær beløpssensitiv diagnostikk.
MAE bør ikke velge en modell som skal estimere forventningen.

Frekvensens Poisson-baserte A/E-usikkerhetsgrenser skal ikke kopieres.
Cluster-bootstrap for finalistdiagnostikk er et forslag som må spesifiseres,
med tydelig omtale av hvilken usikkerhet den faktisk dekker.

Det gjenstår å låse primærscore, beslutningsterskler, forenklingsregel og
kriterier for diagnostisk overstyring.

## Viktige tolkningsforbehold

- Dette er en metodebenchmark med timingforbehold:
  registrerte risikofelt er ikke dokumenterte tegningsverdier.
- Eksisterende lekkasjeeksklusjoner videreføres.
- Bonus åpnes ikke som prediktor uten tidligere påkrevd dokumentasjon.
- Frekvensmodellens utelatte variabler er ikke automatisk irrelevante for severity.
- One-way-mønstre er ikke justerte effekter eller kausale forklaringer.
- Eksponeringsbasert merke-pooling garanterer ikke severity-støtte.
- Sammenheng mellom bilverdi og ytelse utelukker ikke at begge kan bidra.
- Observasjonsstøtte skal vurderes på severity-utvalget, ikke bare fullporteføljen.
- Tidligere EDA og referansemodeller har allerede brukt utviklingsdataene;
  nestet CV opphever ikke all tidligere påvirkning.

## Forventet leveranse fra neste sesjon

En egen severity-plan i `plans/` som inneholder:

1. Verifisert nåsituasjon og korrekte tall med sporbare referanser.
2. Evidenstabell fra Lunas litteratursøk, vurdert av hovedagenten.
3. Foreslått litteraturbaseline med eksplisitte funksjonsformer.
4. Begrenset kandidatregister og entydig seleksjons-/stoppregel.
5. Full beskrivelse av nestet gruppe-CV og endelig modellvalg.
6. Avgrenset tids- og modenhetsdiagnostikk.
7. Storskadediagnostikk og eventuelle betingelser for modellbehandling.
8. Respons-, fordelings- og evalueringsvalg.
9. Implementeringsrekkefølge, verifikasjonskrav og beslutningspunkter.
10. En kort liste over konkrete valg brukeren må godkjenne før modelltesting.

Bevar skillet mellom planlegging, godkjenning og implementering.
Ikke presenter forslag som allerede vedtatt, og ikke endre den frosne
frekvensmodellen som del av severity-planleggingen.
```