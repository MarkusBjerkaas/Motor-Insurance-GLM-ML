$$
f(y;\mu,\phi,p)
=
a(y,\phi,p)\,
\exp\!\left\{
\frac{1}{\phi}
\left(
\frac{y\mu^{1-p}}{1-p}
-
\frac{\mu^{2-p}}{2-p}
\right)
\right\},
$$

$$
a(y,\phi,p)
=
\frac{1}{y}
\sum_{j\geq 1}
\frac{
y^{j\alpha}
}{
(p-1)^{j\alpha}
\phi^{j(1+\alpha)}
(2-p)^j
j!\,
\Gamma(j\alpha)
}.
$$

Her er

$$
\alpha=\frac{2-p}{p-1}.
$$

Leddene blir enorme og bråsmå på samme tid, så flyttall kan gå over- eller underflyt. Det gjelder særlig ved store $y/\phi$ (storskadene våre) og når $p$ ligger nær 1 eller 2. Summen må derfor beregnes i log-rom og rundt det dominerende leddet. `wright_bessel` fra scipy avhjelper det, men jeg ville likevel skrevet og validert denne koden selv, uten en etablert pakke å sammenligne mot.

Risikoen er altså at koden lett kan gi feil eller ustabile verdier uten at man ser det. Den er ikke uoverkommelig. Jeg overdrev nok litt ved å kalle metoden «sårbar». Riktigere er at den koster mest kode og mest validering. Det er grunnen til at jeg fortsatt anbefaler B.
