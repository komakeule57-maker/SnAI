# Inferenz-Test: Prinzessin `princess_SFT_6a8050b1332a5`, finale Fassung (Epoche 3/3) — Vergleich gegen historische Werte

Abschlusstest, angekündigt in `docs/inferenz/INFERENZ_PRINZESSIN_6A8050B1_EPOCHE1.md`
("der eigentliche Vergleich gegen die alte, voll austrainierte
Laptop-Prinzessin `trained_256_e40.snai` folgt nach Abschluss von Epoche 3")
vorgemerkt.
Diese Prinzessin ist der erste komplette Lauf mit dem gesamten seit dem
2026-08-15-Reset aktiven Fix-Stack: Loss-Masking, Weight-Decay/Dropout,
Label-Smoothing, GEN-`</s>`-Dichte-Retrofit, Wikitabellen-/Vorlagen-
Corpus-Cleanup und live Neuron-Revival.

## Testaufbau

- **Modell:** `factory/royal/princess_SFT_6a8050b1332a5.cortex`
- **Architektur:** 256 Dim, 2 Layers, 4 Heads (Net2Net-Wachstum aus einem
  128-Dim/2-Heads-Worker, nur Dim+Heads gewachsen, Layer-Zahl unverändert).
- **Training:** automatischer Schwarm-Pfad, gestartet 2026-08-15 17:08:21
  CEST (Worker 163748), fertig 18:53:15 CEST.
- **Held-out-Loss-Verlauf über 3 Epochen:** 5.980589 (PPL 395.67) →
  5.869183 → **5.862187** (PPL 351.49, finaler Best-Checkpoint). Deutlich
  abnehmender Grenzertrag zwischen Epoche 2 und 3 (−0.111 dann nur noch
  −0.007) — fast Plateau.
- **Globaler Trainings-Avg-Loss** (`royal_synthesis.log`, END-LOSS,
  jetzt korrekt dank des Logging-Fixes: 5.71729.
- **Inferenz-Parameter** (`snai_inference.php`, Standard-Sampling,
  `beam_width` weggelassen): Temperature 0.25, Repetition Penalty 1.15
  (Fenster 64), min_p 0.05, max. 256 neue Tokens — identisch zu allen
  bisherigen Referenztests.
- **Hardware:** Hauptrechner, RTX 3050, OpenCL-Pfad (`snai_inference.php`
  hat kein Dim-Gate, aktiviert GPU unabhängig von der Modellgröße).
- **Datum:** 2026-08-15, ca. 20:20 CEST.

## Ergebnisse (roh, unbearbeitet)

### Prompt 1: „Erzähl mir etwas über die Natur.“ (219 Zeichen, kein sauberer Stopp)
```
 einige Beispiel können Sie es ein Informationen einige Beispiele, dass du um der Zeit zu einer anderen Anmacken und andere Schüche oder verlegen und deine, wie die in den anderen von der Mögen.

Die:

-Sie des Kekürzt]
```

### Prompt 2: „Was ist ein Computer?“ (10 Zeichen, sofortiger Stopp)
```
 [Gekürzt]
```

### Prompt 3: „Beschreibe einen sonnigen Tag.“ (10 Zeichen, sofortiger Stopp, identisch zu Prompt 2)
```
 [Gekürzt]
```

## Beobachtungen

- **Das `[Gekürzt]`-Trunkierungsmarker-Artefakt aus dem Epoche-1-Zwischenstand
  ist in der finalen Fassung unverändert vorhanden.** Prompt 2 und 3 liefern
  exakt dasselbe Ergebnis wie beim Epoche-1-Test (`docs/inferenz/
  INFERENZ_PRINZESSIN_6A8050B1_EPOCHE1.md`): 10 Zeichen, identischer Marker.
  Die bereits dort identifizierte Ursache (der SFT-Sammler hängt bei
  hart gekappten SFT-Antworten wörtlich `"... [Gekürzt]"` vor das
  `</s>`-Stop-Token, betrifft ≈60% der `[SYMBIO]`-Antwort-Turns in der
  Epoche-1-Trainingsdatei dieser Prinzessin) ist weiterhin ungefixt
  (zurückgestellt bis nach diesem Vergleichstest).
- **Prompt 1 driftet in Wortsalat statt in den memorierten `[SYMBIO]`-Marker
  aus dem Epoche-1-Test** — andere Stichprobe (stochastisches Sampling ohne
  festen Seed), aber auch hier landet die Ausgabe am Ende in einem
  abgebrochenen `[Gekürzt]`-Fragment ("-Sie des Kekürzt]"). Keine
  inhaltliche Kohärenz zum Prompt.
- Kein Prompt endet über ein sauber vorhergesagtes `</s>` — die beiden
  kurzen Stopps sind Artefakt-getrieben, nicht "das Modell hat geantwortet
  und aufgehört".

## Vergleich mit historischen Werten

**1. Gegen den eigenen Epoche-1-Zwischenstand** (`docs/inferenz/
INFERENZ_PRINZESSIN_6A8050B1_EPOCHE1.md`): Held-out-Loss verbessert sich
5.980589 → 5.862187 (−0.118 über 2 weitere Epochen), aber praktisch das
gesamte Fehlerbild bleibt identisch — dasselbe `[Gekürzt]`-Kollapsmuster bei
denselben zwei Prompts, dieselbe Zeichenzahl. Zwei weitere Trainingsepochen
haben das dominante Artefakt nicht behoben, wie erwartet (der zugrunde
liegende Korpus-Bug wurde nicht gefixt, nur mehr davon trainiert).

**2. Gegen die alte, unregularisierte Laptop-Prinzessin `trained_256_e40.snai`**
(`docs/inferenz/INFERENZ_PRINZESSIN_MAXTRAIN_256_FINAL.md`, bereits am selben Tag mit
identischen Prompts/Parametern getestet — hier direkt übernommen statt
erneut auf dem Laptop ausgeführt, da Modell/Test/Fix-Stand seit dem Testdatum
unverändert sind). Dies ist der in der Epoche-1-Doku angekündigte Hauptvergleich:

| | `6a8050b1332a5` (neu, regularisiert) | `trained_256_e40` (alt, unregularisiert) |
|---|---|---|
| Training | 3 Epochen, automatischer Schwarm-Pfad, Net2Net 128→256 | 40 Epochen, Genesis (keine Net2Net-Vorgeschichte) |
| Fix-Stack | Loss-Masking, Weight-Decay/Dropout, Label-Smoothing, `</s>`-Retrofit, Wikitable-Cleanup, Neuron-Revival | keiner, altes Codestand ohne Batch-Shuffling |
| Korpus | Multi-Source (SFT, Loss-Masking auf Assistenten-Turns) | einzelner 24-MB-Korpus (`corpus_full.txt`) |
| Loss | Held-out 5.862187 (PPL 351) | Trainings-Fit 3.38581, kein Held-out-Tracking (vermutlich Memorierung) |
| Prompt 1 | 219 Zeichen, Wortsalat, kein Stopp | 827 Zeichen, flüssiges Deutsch, Tokenlimit erreicht |
| Prompt 2 | 10 Zeichen, `[Gekürzt]`-Kollaps | 1000 Zeichen, flüssiges Deutsch, Tokenlimit erreicht |
| Prompt 3 | 10 Zeichen, `[Gekürzt]`-Kollaps | 746 Zeichen, flüssiges Deutsch, Tokenlimit erreicht |
| Prompt-Kohärenz | keine | keine (Wikipedia-artige Abdrift) |

Trotz des vollen Fix-Stacks zeigt die neue, regularisierte Prinzessin bei
diesem Test **keinen qualitativen Vorsprung** gegenüber dem alten,
unregularisierten 40-Epochen-Maximaltraining — im Gegenteil: die alte
Laptop-Prinzessin liefert bei allen drei Prompts deutlich längeren,
grammatikalisch flüssigeren Text (wenn auch ebenso ohne inhaltliche
Prompt-Kohärenz), während die neue Prinzessin bei zwei von drei Prompts
sofort in ein Korpus-Artefakt kollabiert. Kein sauberer A/B-Test
(unterschiedliche Korpusgröße/-quelle/-ziel [SFT-Dialog vs. reiner
Fließtext], Trainingsdauer, Architekturherkunft Net2Net vs. Genesis,
n=3 pro Modell, stochastisches Sampling ohne festen Seed) — aber der
erhoffte qualitative Sprung durch die Regularisierungsmaßnahmen bleibt in
diesem konkreten Vergleich aus. Der dominante Fehlermodus der neuen
Prinzessin ist allerdings ein bereits klar isoliertes, bekanntes
Korpus-Artefakt und kein grundsätzliches Trainingsproblem.

**3. Gegen `docs/inferenz/INFERENZ_BASELINE.md`** (Legacy-1-Head-Prinzessin, Loss
6.56528, PPL ≈710): neue Prinzessin hat klar niedrigeren Loss (5.862 vs.
6.565) und PPL (351 vs. 710). Der Fehlermodus hat sich verschoben statt
verbessert — Baseline stoppte nie und kollabierte in Artikel-/Funktionswort-
Wiederholung ("der/die/dem"-Spam), die neue Prinzessin stoppt jetzt
zumindest, aber zu früh und auf einem Korpus-Artefakt statt auf echtem
Content-Ende.

**4. Gegen `docs/inferenz/INFERENZ_PRINZESSIN_POST_FIXES.md`** (GEN-Prinzessin
`6a7b734fa0a01`, Held-out/End-Loss 5.08337, bis dahin beste GEN-Prinzessin):
niedrigerer Loss als die neue SFT-Prinzessin (5.08 vs. 5.86) — aber anderer
Korpustyp (GEN vs. SFT, unterschiedliche Held-out-Sets, nicht direkt
vergleichbar) und andere Fix-Generation (post_fixes lief vor
Label-Smoothing/Neuron-Revival/Wikitable-Cleanup). Dort dominierte
memorierte System-Prompt-Boilerplate ("hilfreicher Assistent"-Text) als
Fehlermodus, hier das `[Gekürzt]`-Artefakt — unterschiedliche Korpora,
unterschiedliche dominante Artefakte, kein gemeinsamer Trend ablesbar.

## Einordnung

Held-out-Loss-Trend bleibt über alle 3 Epochen gesund, aber mit klar
abnehmendem Grenzertrag. Der eigentliche Befund dieses Vergleichstests: der
volle 2026-08-15-Fix-Stack (Loss-Masking, Weight-Decay/Dropout,
Label-Smoothing, `</s>`-Retrofit, Wikitable-Cleanup, Neuron-Revival) liefert
gegenüber der alten, unregularisierten Laptop-Prinzessin **keinen
sichtbaren qualitativen Sprung** — im direkten Vergleich wirkt die neue
Prinzessin sogar fragmentierter, dominiert von einem einzelnen,
identifizierten Korpus-Artefakt (`[Gekürzt]`-Trunkierungsmarker aus
dem SFT-Sammler). Da die Bedingung aus der Epoche-1-Doku ("Fix erst
nach dem finalen Epoche-3-Vergleichstest angehen") mit diesem Test erfüllt
ist, ist der Fix im SFT-Sammler jetzt der naheliegendste nächste
Schritt, um den am klarsten isolierbaren Fehlermodus aus dem Weg zu räumen
— hier nicht umgesetzt, nur als Empfehlung notiert.
