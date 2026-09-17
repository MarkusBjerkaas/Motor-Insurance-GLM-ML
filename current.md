## Lovende severity prediktorer: 

Generelt: Alle variabler som ble utelatt av argument om lekkasje eller usikkhet utelates av samme årsak som tidligere.


## Kategoriske:

Dekningstype ser lovende ut av samme grunn som frekvens, svært stort avvik i skadestørrelser.

Circulation AREA Ser lovende ut. Høyere severity i rurale områder gir mening ettersom det er høyere fart, samtidig som skader er sjeldne. 
    Forventet fortegn på koffisient dersom urban utelates - positivt.

Municipality ble droppet for frekvens, men bør testes for severity - defineres i variabel seleksjonsrunden: Virker litt lovende og bør nok tas med som kjerne.

Fuel type: Forventet fortegn om G brukes som dummy - positivt.

Payment frequency er usikker, ettersom bin S og Q har lav eksponering. 

NOEN SOM IKKE SER LOVENDE UT: Vanskelig å si. Muligens payment frequency ettersom det er ikke veldig stor variasjon, og jeg ville tenkt payment frequency er mer utsatt for frekvens forskjeller grunnet potensiell moral hazard, og at de har lav payment frequency fordi de muligens bytter forsikringsselskap ofte. På den andre siden kan en årsak til sjeldnere payment frequency være grunnet dårlig økonomi - som kan føre til at de er mer forsiktige i trafikken. Sansynligvis ikke veldig viktig prediktor for severity.


## Numeriske:
Driver age kan se ut til å ha en ikke linær sammenheng med severity. Ser relativt stabil ut i lavere aldere, men tydelig outlier i eldre alder. Kan muligens være pga måten vi binner på - så dette er muligens en lovende kandidat for ikke linær sammenheng.

Driving_experience_years ser ustabil ut. Må testes mot driver age på samme måte som tidligere. 

Log vehicle value ser ut til å være ikke linær. Bør muligens testes med polynom eller spline. God kandidat med ulik degrees of freedoms. 

Seats ser mer lovende ut her sammenlignet frekvensen. 

Bilmerke er fremdeles usikkert. Bør testes, men vil nok møte på samme problematikk som tidligere at det gir for mange degrees of freedom sammenlignet med gevinsten. Dette er nok en sterk kandidat for kredibilitetsextention.

## Korrelasjon:
Driving age og driving experience years har fremdeles samme multikollinaritet problematikk og kun en burde sansynligvis inkluderes. Performance og veichle value er og sterkt korrelert og funksjonsformen med utgangspunkt i plottet ser og ganske lik ut. Begge bør nok testes.





