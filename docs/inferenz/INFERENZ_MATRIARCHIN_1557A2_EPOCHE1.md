# Inferenz-Test + Logit-Lens-Analyse: neue Matriarchin `1557a2`, Zwischenstand nach Epoche 1 von 6

Referenz-Inferenztest auf dem gerade fertigen
Epoche-1-Checkpoint, während Epoche 2 bereits lief — gleiches Muster wie
`docs/inferenz/INFERENZ_MATRIARCHIN_77A16F_EPOCHE1.md`. Zusätzlich:
**erster produktiver Einsatz des Logit-Lens-Werkzeugs**
(`diagnose/snai_logit_lens.php`, 2026-08-18) gegen die beiden klarsten
Kollaps-Übergänge aus den Standard-Prompts.

## Testaufbau

- **Modell:** `secured_models/matriarchin_GEN_1557a2_epoch1.apex` — Kopie
  von `factory/apex/matriarchin_GEN_1557a2.apex` unmittelbar nach Epoche 1,
  **vor** dem Start von Epoche 2 gesichert (byte-identisch per `sha256sum`
  verifiziert), da dieselbe Datei als Basis für Epoche 2 dient und danach
  überschrieben worden wäre. Liegt außerhalb von `factory/`.
- **Architektur:** 512 Dim, **6 Layers** (erster kompletter Lauf mit dem am
  16.08. korrigierten Matriarchin-Ladder-Verhältnis, vorher 4 Layers), 8
  Heads.
- **Herkunft:** APEX-SYNTHESE. Session `matriarchin_GEN_1557a2`,
  GEN-Kalibrierung über 6 geplante Epochen (10 Artefakte/Epoche, neue
  Epochenzahl seit dem Chinchilla-getriebenen Umbau, siehe
  `royal_distillery.php:131-146`).
- **Epoche 1** (aus `train_matriarchin_GEN_1557a2.jsonl`): Train-Loss
  5.056743, Held-out-Loss **5.347133** (PPL 210.0054), Dauer **12h 37m 23s**
  (`epoch_duration_sec: 45442.93`), GPU-Backend **CUDA** (bestätigt via
  `start_training.sh`-Workaround). Direkt danach automatisch als
  `best_matriarchin_GEN_1557a2.snai` gesichert.
- **Neuron-Revival-Hook aktiv:** direkt nach Epoche 1 automatisch **9 tote
  Neuronen revivt** (2× `L3_w_down`, 7× `L5_w_down`, von 27.648 geprüft).
  Der hier getestete Checkpoint enthält diese Änderung bereits.
- **Einordnung ggü. den beiden Vorgänger-Matriarchinnen nach Epoche 1**
  (selber Tag): Held-out-Loss hier (5.347133) liegt
  zwischen `77a16f` (5.340738, 4 Layer) und ist schlechter als `9d26ec`
  (5.313186, 4 Layer) — der zusätzliche Layer zeigt nach nur einer Epoche
  keinen Loss-Vorteil, kein sauberer A/B-Test (andere Korpusversion/
  Hyperparameter).
- **Inferenz-Parameter** (`snai_inference.php`, Standard-Sampling,
  `beam_width` weggelassen): Temperature 0.25, Repetition Penalty 1.15
  (Sliding Window 64 Tokens), min_p 0.05, max. 256 neue Tokens.
- **Hardware:** Hauptrechner, RTX 3050, OpenCL (Inferenz-Prozess
  unabhängig vom CUDA-Trainingsprozess). Lief **parallel** zum aktiven
  Epoche-2-Training auf derselben GPU — keine Fehler, kein beobachteter
  Konflikt.

## Ergebnisse (drei Standard-Prompts)

Alle drei Prompts kollabieren, keiner erreicht ein Stop-Token — Generierung
läuft jeweils bis zum 256-Token-Limit.

### Prompt 1: „Erzähl mir etwas über die Natur.“ (613 Zeichen)

Beginnt mit halbwegs satzförmigem, aber inhaltsleerem Text ("Natürlich. Die
Schaubeung, um den Saturn, und der Meiß, die mit einer..."), durchsetzt mit
vereinzelten `<unk>`-Fragmenten in Wortmitte (BPE-Artefakt). Kippt ab
"Die Gang<unk>chaft." in eine lange Kette reiner `<unk>. `-Wiederholungen
bis zum Limit.

### Prompt 2: „Was ist ein Computer?“ (349 Zeichen)

Ähnlich: kurzer, grammatisch lose zusammenhängender Anfang ("Natürlich!
Hier sind einige Beispiele für die Blinen- und Migung..."), bricht nach
"1. "Bernse":" in einen langen reinen Leerzeichen-Lauf ab (kein Text mehr,
nur Whitespace bis zum Limit).

### Prompt 3: „Beschreibe einen sonnigen Tag.“ (1073 Zeichen)

Schnellster Kollaps der drei: bereits nach "Ja.: Die" kippt die Ausgabe in
einen Leerzeichen-Lauf, danach in eine sehr lange Kette aus `<unk>`
(teils mit vereinzeltem `α`-Ausreißer) bis zum Limit.

## Logit-Lens-Analyse

Erster produktiver Einsatz von `diagnose/snai_logit_lens.php`
(2026-08-18). Zwei Kollaps-Übergänge inspiziert,
jeweils an der Position **direkt vor** dem ersten `<unk>` im echten Output
(bewusst gewählt, um Retokenisierungs-Verzerrung durch `<unk>`-Text im
rekonstruierten Prompt zu vermeiden). `top_k=6`, Sanity-Check (letzter Layer
== reguläre `forward()`-Ausgabe) in beiden Fällen bestanden.

### Fall A: Prompt 3, Position nach "Ja.: Die" (direkt vor dem Kollaps)

| Layer | Entropie (Bit) | Top-6 |
|---|---|---|
| L0 | 9.64 | ß(7%) `<unk>`(6%) Son(5%) ell(1%) ⌷(1%) andere(1%) |
| L1 | 10.04 | `<unk>`(6%) Son(4%) Zeit(3%) M(2%) Er(1%) ß(1%) |
| L2 | 10.45 | M(2%) `<unk>`(2%) S(2%) ⌷(1%) A(1%) L(1%) |
| L3 | 10.51 | S(2%) M(2%) ⌷(1%) `<unk>`(1%) A(1%) H(1%) |
| L4 | 10.62 | S(2%) M(2%) ⌷(1%) `<unk>`(1%) A(1%) T(1%) |
| L5 (final) | 10.58 | ⌷(2%) M(1%) S(1%) A(1%) G(1%) T(1%) |
| regulärer `forward()` | 10.58 | identisch zu L5 ✅ |

### Fall B: Prompt 1, Position nach "...die mit einer" (direkt vor dem ersten `<unk>`)

| Layer | Entropie (Bit) | Top-6 |
|---|---|---|
| L0 | 9.88 | ley(**13%**) andere(1%) kleinen(1%) kleine(1%) Vielzahl(1%) großen(1%) |
| L1 | 10.27 | ley(2%) kleinen(2%) großen(1%) anderen(1%) neuen(1%) ⌷(1%) |
| L2 | 10.36 | anderen(2%) kleinen(2%) großen(1%) St(1%) M(1%) Sch(1%) |
| L3 | 10.31 | anderen(2%) kleinen(1%) St(1%) M(1%) Sch(1%) großen(1%) |
| L4 | 10.29 | anderen(2%) Sch(1%) St(1%) M(1%) kleinen(1%) T(1%) |
| L5 (final) | 10.17 | Sch(1%) M(1%) anderen(1%) d(1%) T(1%) der(1%) |
| regulärer `forward()` | 10.17 | identisch zu L5 ✅ |

### Befund

Zwei auffällige Muster, beide **nicht** das erwartete Lehrbuch-Bild eines
gesunden Logit-Lens-Verlaufs (fortschreitende Schärfung/sinkende Entropie
von früh nach spät):

1. **Die Entropie bleibt über alle 6 Layer hinweg durchgehend hoch (9.6-10.6
   von max. 13 Bit) und sinkt zum Schluss kaum bis gar nicht** — selbst der
   letzte Layer ist an diesen Positionen praktisch so unentschlossen wie
   der erste. Zum Vergleich: der Corpus-Mittelwert liegt bei PPL 210 (≈7.7
   Bit) — an genau diesen beiden Kollaps-Übergangspunkten liegt die
   Unsicherheit klar darüber (2^10 ≈ 1024 statt ≈210 "gleichwertige"
   Kandidaten). Das Modell "weiß" an diesen Stellen in KEINEM Layer, was
   als Nächstes kommt — kein Hinweis auf einen einzelnen fehlerhaften
   Layer, eher auf generelle Unterdeckung an genau diesen Kontexten.
2. **Fall B zeigt sogar eine Umkehrung:** der früheste Layer (L0) hat mit
   "ley" (13%) das mit Abstand konkreteste Signal der ganzen Tabelle
   (vermutlich ein Fragment Richtung "Vielzahl/Käfer/Schmetterlinge" o.ä.,
   passend zum Naturthema) — dieses Signal verwässert sich aber durch die
   folgenden Layer wieder, statt sich zu verstärken, und endet in einer
   noch flacheren, generischeren Verteilung. Nur ein Einzelfall (n=1), aber
   bemerkenswert genug, um im Auge zu behalten, falls sich das Muster bei
   künftigen Lens-Läufen wiederholt.

Beide `<unk>`-Kandidaten selbst tauchen nur in den frühen/mittleren Layern
(L0-L4) unter den Top-6 auf, nicht mehr im letzten Layer — dort gewinnen
stattdessen einzelne Großbuchstaben-Fragmente (M, S, A, G, T) mit
ähnlich flachen ~1-2%. Auch das passt eher zu "Modell hat an dieser Stelle
schlicht nichts gelernt" als zu "ein bestimmter Layer erzeugt aktiv den
`<unk>`-Kollaps".

## Einordnung

Deckt sich im Kollaps-Muster mit `INFERENZ_MATRIARCHIN_77A16F_EPOCHE1.md`
(dort 2-3 von 3 Prompts kollabiert) — hier sogar 3 von 3, aber nicht direkt
vergleichbar (anderer Korpus-Stand, andere Architektur: 6 statt 4 Layer).
Erwartbar roh nach nur 1 von 6 geplanten Epochen, kein Regressionsbefund.

**Wichtigster Ertrag dieses Tests ist methodisch, nicht inhaltlich:** die
Logit-Lens-Befunde sprechen dafür, dass der Kollaps bei diesem
Zwischenstand eher ein **Trainingsstand-/Datenmengen-Symptom** ist (diffuse
Unsicherheit über die gesamte Tiefe) als ein **lokalisierter
Architektur-/Layer-Bug** (der sich als scharfer Sprung in einem einzelnen
Layer zeigen würde). Das stützt tendenziell die bereits vereinbarte
Priorität — erst die Chinchilla-Materialanhebung abwarten/vorantreiben,
Layer-Analyse-Interventionen zurückstellen — auch wenn n=2 Positionen
statt eines systematischen Sweeps über viele Kollaps-Stellen keine
belastbare Schlussfolgerung erlaubt. Kein Grund, das Werkzeug jetzt zu
vertiefen; es steht startklar für spätere, systematischere Läufe (z.B.
über mehrere Kollaps-Stellen gemittelt, oder im Vergleich mit einem
späteren, weiter trainierten Checkpoint desselben Laufs).
