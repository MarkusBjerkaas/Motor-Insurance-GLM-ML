Fortsett arbeidet i /workspaces/MotorForsikring fra commit 7e86535.

  Les først prosjektets AGENTS.md-instruksjoner og plans/Severity_plan.md. Planen er autoritativ. handoff.md inneholder resultater fra det tidligere, smalere pilotløpet og er nå utdatert når det gjelder
  kandidatvalg og finalist. Pilotløpet finnes fortsatt i git-historikken ved commit dfad89a.

  Arbeidsform:
  - Du fungerer som senior aktuar og faglig kontrollør.
  - Deleger rett frem implementasjon, kodeopprydding og kjøring til Luna-agenter.
  - Gjennomgå og kritiser arbeidet selv.
  - Bruk en ny Luna-agent til å rette feil som avdekkes.
  - Dersom et stoppkriterium utløses eller en ny faglig beslutning kreves før neste modellsteg, stopp og spør meg.
  - Alt skriftlig skal være på norsk; variabel- og funksjonsnavn på engelsk.

  ABSOLUTT DATAGRENSE:
  - Aldri les, last, inspiser, transformer eller bruk 2024-data.
  - Bruk kun build_development_frames() og utviklingsårene 2022–2023.
  - Inverse Gaussian skal ikke implementeres eller fittes.

  Oppdrag:
  Revider det låste Gamma-kandidatløpet slik at disse tre kategoriske variablene inngår i den første seleksjonsrunden:

  1. fuel_type
  2. business_type
  3. payment_frequency
  4. vehicle_brand_pooled

  Bruk eksisterende S0 og de opprinnelige kandidatene, og legg til tre enkeltutfordrere:

  - FU1: S0 + fuel_type
  - BU1: S0 + business_type
  - PF1: S0 + payment_frequency

  Hver ny variabel skal være en egen kandidatblokk. S0 representerer uendret blokk. Dersom kandidaten kvalifiserer og vinner blokkens nesten-like-vurdering, skal blokken kunne inngå i den ene kombinerte
  kandidaten sammen med øvrige valgte blokker.

  Ikke legg til interaksjoner eller andre nye variabler.

  Før fitting:
  1. Revider plans/Severity_plan.md som en ny protokollversjon og logg endringen.
  2. Oppdater kandidatregister, blokkregister, parameterantall og modellbudsjett.
     - Det blir 13 faste spesifikasjoner inkludert S0.
     - 13 × 5 folder = 65 hovedtilpasninger.
     - Med høyst én kombinert kandidat blir maksimalt budsjett 70 hovedtilpasninger.
  3. Verifiser og lås kategorinivåer og referansenivåer fra utviklingsdata før fitting. Forventede referanser fra frekvensarbeidet er:
     - fuel_type: D
     - business_type: NB
     - payment_frequency: A
     Kontroller disse uten å berøre 2024.
  4. Oppdater støtte- og designkontrollene slik at de tre nye variablene inngår.
  5. Kontroller korrekte parametertall og full rang i alle fem folder før første fit.
  6. Oppdater syntetiske tester og budsjettassertions før modellkjøringen.

  Notebook og kode:
  - Rediger kun glm_pricing_severity.py, aldri .ipynb direkte.
  - Behold dagens notebookstruktur og korte celler.
  - Bruk eksisterende funksjoner i src/ og src_severity/ før du lager nye.
  - Generaliser resultatavhengig kode som nå antar at finalisten heter C1:
    - fjern assert FINALIST == "C1"

  Seleksjon:
  - Bruk de samme fem gruppefoldene, seed, Gamma/log-link, vekter, preprocessing og seleksjonsgrenser som i Severity_plan.md.
  - Kjør alle enkeltutfordrere mot samme S0.
  - Behold geografisærregelen for G2.
  - Velg ett alternativ per blokk med eksisterende nesten-like-regel.
  - Bygg høyst én kombinert kandidat dersom minst to blokker endres.
  - Ingen ny søkerunde eller manuell modelltilpasning etter at resultater er sett.

  Verifikasjon før kjøring:
  - Gjør en uavhengig kodegjennomgang av kandidatregister, blokklogikk, parameterantall, referansenivåer og budsjett.
  - Kjør Ruff og alle relevante syntetiske tester.
  - Rett eventuelle feil med en ny Luna-agent før hovedkjøringen.

  Deretter:
  1. Start notebooken fra ren kernel og kjør hele glm_pricing_severity.ipynb fra toppen.
  2. Ikke reduser de låste 2 000 bootstrap-trekkene.
  3. Kontroller alle kontrolltall og stoppregler.
  4. Dersom et faglig stopp utløses, stopp før sluttfit og spør meg.
  5. Hvis ingen stopp utløses, gjennomfør diagnostikk og sluttfit for den nye finalisten.
  7. Verifiser at notebooken kjører uten feil og at output er lagret.

  Markdown:
  Notebooken inneholder RESULTATPLASSHOLDER-celler i seksjon 4–7. Etter vellykket kjøring skal hver plassholder erstattes med en kort, konkret og faglig korrekt konklusjon basert på den nye outputen.

  Skriv blant annet:
  - gyldighet og scoreintervall
  - hvilke kandidater som kvalifiserte og hvorfor
  - finalist og nesten-like-vurdering
  - kalibrering og bootstrap
  - innflytelse
  - tidskontroll og segmentdrift
  - residualdiagnostikk
  - finalistens eksplisitte modellformel
  - koeffisienttolkning og begrensninger
  - nye resultatavhengige rader i beslutningsregisteret

  Rapporter til slutt:
  - endrede filer
  - kandidatbudsjett og faktisk antall fits
  - finalist og spesifikasjon
  - sentrale score- og diagnostikkresultater
  - alle stoppvurderinger
  - tester og kjørestatus
  - lokale checkpoint-commits

  Ikke push, merge, rebase eller lag en avsluttende commit uten godkjenning.