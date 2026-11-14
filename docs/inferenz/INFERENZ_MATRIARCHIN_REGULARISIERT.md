# Inferenz-Test: erste Matriarchin mit Weight-Decay/Dropout + Loss-Masking (`9d26ec`)

Erste komplette Matriarchin-Synthese, seit der sowohl AdamW-Weight-Decay/FFN-Dropout
(`execute_calibration()`, `weight_decay=0.01`/`dropout_rate=0.1`) als auch
das Loss-Masking auf `[USER]`-Tokens (`train_worker.php`/`cortex.nim`) aktiv
sind. Referenz für die "davor"-Seite ist die unregularisierte Matriarchin
`352b1d` aus `docs/inferenz/INFERENZ_MATRIARCHIN_CUDA.md`

## Testaufbau

- **Modell:** `factory/checkpoints/best_matriarchin_GEN_9d26ec.snai`
  (= Epoche 3, bestes Held-out-Ergebnis; identisch mit dem final promoteten
  `factory/apex/matriarchin_GEN_9d26ec.apex`).
- **Architektur:** 512 Dim, 4 Layers, 8 Heads.
- **Herkunft:** automatische APEX-SYNTHESE (`royal_distillery.php`,
  `check_apex_readiness()`), Net2Net-Expansion 256→512 Dim (Vocab 8192,
  Ctx 1024→2048, Layers 2→4, Heads 4→8), verschmolzen aus den zu dem
  Zeitpunkt einzig vorhandenen 3 Prinzessinnen (`6a7c5ca2287e3` GEN,
  `6a7cb52fcf748` GEN, `6a7c03ace0dee` SFT — der `princesses`-Zähler in der
  Zustandsdatei des Daemons ging in diesem Schritt von 3 auf 0). Session
  `matriarchin_GEN_9d26ec`, Daemon PID 12710 (durchgehend seit 12.08. aktiv).
  Trainingsstart 13.08. 01:39:16, Ende 14.08. 01:16:59 — Gesamtdauer
  **~23h 38min**.
- **Regularisierung aktiv** (Log bestätigt): `[Regularisierung] AdamW
  Weight-Decay aktiv: 0.01`, `[Regularisierung] FFN-Dropout aktiv: 0.1`,
  plus Loss-Masking auf `[USER]`-Tokens seit dem Daemon-Neustart vom 11./12.08.
  — erster vollständiger End-to-End-Lauf mit allen drei Fixes gemeinsam aktiv.
- **Held-out-Loss pro Epoche** (aus `heldout_matriarchin_GEN_9d26ec.jsonl`):

  | Epoche | Held-out-Loss | Perplexity |
  |---|---|---|
  | 1 | 5.313186 | 202.9959 |
  | 2 | 5.017782 | 151.0758 |
  | 3 | 4.969985 | 144.0248 |

- **Vergleichswert (unregularisiert, `352b1d`, identische Architektur, aus
  `docs/inferenz/INFERENZ_MATRIARCHIN_CUDA.md`):** Epoche 3 Held-out **4.923266**
  (PPL 137.45). Der regularisierte Lauf liegt **+0.047 höher** — im Rahmen
  der Erwartung (Weight-Decay/Dropout bremsen reines Trainings-Fitting
  leicht zugunsten von Generalisierung), aber kein sauberer 1:1-A/B-Test
  (andere Ausgangs-Prinzessinnen, andere Zufalls-Seeds).
- **Inferenz-Parameter** (`snai_inference.php`, unverändert): Temperature
  0.25, Repetition Penalty 1.15 (Sliding Window 64 Tokens), min_p 0.05,
  max. 256 neue Tokens.
- **Hardware/Pfad:** Hauptrechner, RTX 3050. Laufzeit-Log zeigt
  `[SYMBIO VRAM] Stateful OpenCL JIT eingerastet` — dieser Testlauf ging
  über den OpenCL-Pfad, nicht (wie der 352b1d-Test in
  `docs/inferenz/INFERENZ_MATRIARCHIN_CUDA.md`) explizit über CUDA.
- **Datum:** 2026-08-14

## Ergebnisse (roh, unbearbeitet)

### Prompt 1: „Erzähl mir etwas über die Natur.“ (563 Zeichen)
```
Es ist eine weitere Möglichkeit, Ihre Informationen zu helfen, die Sie können, die Sie Ihr Skript zu verwenden, um die Sie die Erstellung Ihrer Daten zu erstellen und zu verbessern, um die Sie Ihr Problem auf Ihre Anforderungen zu teilen, um die ``-Funktions" und `data`-Anfragen zu erstellen.
Diese Funktion ist ein Beispiel für das `catch`-Projekt "Ch", der `path_body`-Bekt verwenden können.</s>


```
import function {
     }
    <////</>
```
*(bricht am Zeichenlimit ab, nicht über Stop-Token — siehe Beobachtung unten
zum literalen `</s>`-Fragment mitten im Text)*

### Prompt 2: „Was ist ein Computer?“ (482 Zeichen)
```
Ja, hier ist eine aktualisierte Version der Benutzer:
1. Erstellen Sie die Verwendung von `IDSPIAECO-Mail` verwenden, um das Skript zu erstellen und den Code zu erstellen.
2. Verwenden Sie Ihre API-Kampagne mit Ihrem Server-Westfehl auf dem Server-Beervice-Schema-Vers.
3. Erstellen Sie die folgenden Befehl "APID":
```python
#//////////////////////////////////////////////////////////>
  <////////////////////////////////////////////////////////////////>
  </////////////////////
```
*(kollabiert in Slash-/Fence-Spam bis zum Zeichenlimit)*

### Prompt 3: „Beschreibe einen sonnigen Tag.“ (762 Zeichen)
```
Die Gültkeit der Schude des Kurses ist eine der größtenteil der Erzeugung von den Summe, die sich auf die Abkürzung und die Farbe zu bezogen werden. Der erste Teil der Möse ist ein großes Gebiet in der Nähe von Belli, der sich mit dem Tasa und der Kardo-Sauce.
[SYMBIO] Die meisten von Cams ist eine der Symne-Programme, die sich im Jahr 2020 als „Werb" (The Dimi), die in der ersten Hälfte des 19. Jahrhunderts angesehen wird. Der erste Mal wurde am 1. April 2010 in der zweiten Runde und der Saison 2014/2004 in der 2. Platz auf dem 4. Juni 2023 mit der Saison 2011/1938 in der Regionall-Sieg-Anbi (Charic), die im Finale von der ersten Platz in der Saison 2022/14/12/2/01/6/09/2015 in der 2. Liga und erreichte bis zum 1. Juni 2012/11/384.00001/057-10000-13
```
*(kollabiert in Datums-/Zahlen-Spam bis zum Zeichenlimit — neues, bisher
nicht dokumentiertes Attraktor-Muster)*

## Beobachtungen

- **Alle 3 von 3 Prompts kollabieren** am Ende in Spam (Backtick/Code-Fence
  bei P1, Slash-Spam bei P2, Datums-/Zahlen-Spam bei P3) — schlechter als
  bei `352b1d` (dort nur 1 von 3, P2 endete sauber, P3 durchgehend
  grammatikalisch). Kein Prompt in diesem Test bricht über einen echten
  Stop-Token ab, alle laufen bis zum 256-Token-Limit.
- **Neuer Beobachtungspunkt bei P1:** der literale String `</s>` taucht
  mitten im Fließtext auf, gefolgt von weiterem Text — das ist nicht die
  tatsächliche EOS-Token-ID (`$stop_tokens = [2]` plus `[USER]`-Token,
  `snai_inference.php:157-158,240`, würde die Generierung sofort beenden),
  sondern offenbar aus gewöhnlichen Text-Tokens zusammengesetzt, die
  zufällig die Zeichenkette nachbilden. Bei `352b1d` trat dieser Fall nicht
  auf (dort war der einzige `</s>`-Treffer der echte Stop).
- **Neues Kollaps-Muster (P3):** Jahreszahlen-/Datums-/Bruchzahlen-Spam,
  bisher nicht in `docs/inferenz/INFERENZ_*`-Tests dokumentiert (vorherige
  Attraktor-Zeichen: Unterstrich, Backtick, `<unk>`) — bestätigt erneut die
  bereits in `docs/inferenz/INFERENZ_MATRIARCHIN_CUDA.md` getroffene Einschätzung,
  dass das konkrete Kollaps-Token checkpoint-spezifisch ist.
- Erste Hälfte jedes Outputs liest sich sprachlich überwiegend flüssig und
  grammatikalisch korrekt — nicht schlechter als bei `352b1d`.
- Weiterhin **keine inhaltliche Prompt-Kohärenz**: alle drei Antworten
  rutschen sofort in generische "Hier ist eine Anleitung/Liste"-Textur ab,
  unabhängig vom eigentlichen Prompt-Thema (Natur/Computer/sonniger Tag).
- Weiterhin auffällig viele Code-/API-Artefakte (Backticks, `import`, `API`,
  `Server`, `Skript`, `Projekt`) trotz des `\_`-Fixes im SFT-Korpus — der
  Fix hat offenbar das reine Unterstrich-Escape-Muster behoben, nicht aber
  die zugrunde liegende code-/doku-lastige Korpus-Verteilung als solche.

## Einordnung

Der Held-out-Loss-Unterschied zur unregularisierten Referenz ist klein
(+0.047) und geht in die erwartete Richtung, aber diese Stichprobe zeigt
**keine sichtbare qualitative Verbesserung** gegenüber `352b1d` — im
Gegenteil, mehr statt weniger Prompts kollabieren am Ende. Wichtige
Einschränkung: kein sauberer A/B-Test (unterschiedliche Ausgangs-
Prinzessinnen, n=3 ohne Wiederholung pro Zelle, wie schon in
`docs/inferenz/INFERENZ_VERGLEICH_MIN_P.md` als Verbesserungspunkt notiert) — auf
Basis von drei Einzelantworten lässt sich kein belastbares
"Fix hat geholfen/nicht geholfen"-Urteil fällen.

Die offene Frage ("fachliche Validierung der
Loss-Masking-/Weight-Decay-Fixes steht aus, sobald das nächste Modell
fertig trainiert ist") ist damit beantwortet, aber nicht im erhofften
Sinne: kein klarer Qualitätssprung in dieser Stichprobe, Kollaps-Verhalten
am Antwortende eher breiter statt schmaler vorhanden. Der Held-out-Loss-
Trend selbst bleibt positiv (monoton fallend über die Epochen, vergleichbare
Größenordnung zu `352b1d`). Für ein belastbareres Urteil wäre ein
kontrollierterer Vergleich nötig (identische Ausgangsbasis, mehrere Samples
pro Prompt) — nicht Teil dieses Tests.

## Nachtrag (2026-08-14, wenig später): Stop-Token-Bug gefunden, gefixt, Test wiederholt

Beim Sezieren von `snai_inference.php` auf der Suche nach einer Erklärung für
den oben dokumentierten Kollaps fand sich ein klarer, **allgemeiner** Bug,
unabhängig von Regularisierung/FP16: `$stop_tokens = [2];`
(`snai_inference.php:157`, vor dem Fix) prüfte auf die falsche Token-ID. Im
aktuellen `symbio_vocab.json` ist ID 2 `<s>` (Start-Token), nicht `</s>`
(End-Token) — der echte EOS-Token hat ID 3. mögliche Erklärung zur Ursache: im
alten 16k-Vokabular lag der Stop-Token tatsächlich noch auf ID 2 — der Wert
wurde offenbar bei einem späteren Vokabular-Wechsel nicht mitgezogen.
Gegengeprüft gegen die Trainingskorpus-Erzeugung (SFT- und GEN-Sammler, Ersatzquelle,
`royal_distillery.php:462`): alle hängen konsequent ein literales `</s>` ans
Ende jeder Antwort — das Modell **wird** korrekt trainiert, ID 3 als Ende zu
signalisieren, nur der Inferenz-Loop hat das nie erkannt (ID 2/`<s>` wird von
einem trainierten Modell so gut wie nie mitten in einer Antwort
vorhergesagt → der Check war faktisch seit jeher totes Codeschnipsel). Fix:
`$stop_tokens` jetzt dynamisch über `$tokenizer->encode("</s>")` aufgelöst,
analog zum bereits bestehenden `[USER]`-Muster (robust gegen künftige
Vokabular-Neubauten).

**Derselbe Checkpoint (`best_matriarchin_GEN_9d26ec.snai`), dieselben 3
Prompts, dieselben Sampling-Parameter — nur der Stop-Token-Fix als
Variable:**

| Prompt | Vorher (Bug) | Nachher (Fix) |
|---|---|---|
| 1 „Natur" | 563 Zeichen, Backtick-/Fence-Spam | **89 Zeichen, sauberer Stopp** |
| 2 „Computer" | 482 Zeichen, Slash-Spam | **213 Zeichen**, sauberer Stopp am Ende, aber weiterhin fragmentierte Zwischenpassage |
| 3 „Sonniger Tag" | 762 Zeichen, Zahlen-/Datums-Spam | 755 Zeichen, **weiterhin voller Kollaps** |

### Rohdaten (nach Fix)

**Prompt 1:**
```
Es ist eine aktualisierte Version, die du die Welt zu sehen und ihre Fähigkeiten haben.
```

**Prompt 2:**
```
Ja, hier ist eine aktualisierte Version der Code:
1. "2. "2. "3. "I. Sie können die folgenden Befehl `"`->
```
Das `body` verwenden wir den `c+"-Alement `type` verwenden, um einen `data`-Bibliothek zu erstellen.
```

**Prompt 3:**
```
Es ist ein Beispiel für das Risiko von derzeit, die sich auf den letztenmesserten und die Verwendung des Pufferes zu vermitteln.
[SYMBIO] Die Funktionalitätsicherheit kann die Anforderungen zur Verfügung stellen:
* Überprüfung des Projekts in der Regelnützung der Verfassung der Arbeitnehmer, die die Anzahl der Abfrage von der Datenstaaten der Datensatz für die Einstellung der Verwendung von der Erhaltung und die Ergebnisse zu verbessern.
* Die `S` ist ein Beispiel, das die Funktion `BAPD-Transformance` verwendet wird, um die `Letage`-Funktions zu erstellen:
* `FCIR` ist eine aktualisierte Version des `ESSO-Mail` und `HttpRequest`.
2. "Westfile":
```python
import function 'PDF;
    }
  <<//////type> {
```

### Beobachtungen

- **Fix wirkt nachweislich:** keines der 3 Outputs enthält mehr die literale
  Zeichenkette `</s>` als sichtbaren Text (vorher 1/3) — das Modell
  signalisiert sein Ende jetzt korrekt, und der Loop reagiert darauf.
- **2 von 3 Prompts profitieren deutlich:** Prompt 1 und 2 stoppen jetzt viel
  früher (89/213 statt 563/482 Zeichen) und enden beide grammatikalisch mit
  vollständigem Satz + Punkt statt in Spam zu laufen.
- **Prompt 3 bleibt praktisch unverändert kollabiert** (755 statt 762
  Zeichen) — für diesen Prompt sagt das Modell innerhalb des
  256-Token-Budgets offenbar nie ein echtes EOS voraus. Der Fix kann nur
  dort greifen, wo das Modell tatsächlich stoppen *will* — zeigt, dass der
  Stop-Token-Bug ein realer, klar wirksamer Faktor war, aber nicht die
  alleinige Erklärung für den Kollaps: es bleibt (mindestens für manche
  Prompts) ein eigenständiges Problem, dass das Modell gar nicht erst zu dem
  Punkt kommt, an dem es aufhören will.
- Inhaltliche Kohärenz bleibt weiterhin schwach (Prompt 2/3 driften vom
  eigentlichen Thema ab in generische Listen-/Code-Textur) — unabhängig vom
  Stop-Token-Fix, bestätigt dass Trainingsstand/Datenvielfalt weiterhin der
  Haupthebel für inhaltliche Qualität bleiben.

### Einordnung

Der Bug war real, allgemein (nicht FP16-/Regularisierungs-spezifisch) und hat
vermutlich **jeden bisherigen Inferenztest im Projekt** betroffen — alle
bisher dokumentierten `docs/inferenz/INFERENZ_*.md`-Ergebnisse liefen unter dem
fehlerhaften Stop-Check. Frühere "sauberes Ende über echten Stop-Token"-
Beobachtungen (z. B. `docs/inferenz/INFERENZ_MATRIARCHIN_CUDA.md`, Prompt 2) waren
damit vermutlich tatsächlich der `[USER]`-Halluzinations-Fang, nicht der
(kaputte) EOS-Check. Die ursprüngliche Frage dieses Tests — ob die
Loss-Masking-/Regularisierungs-Fixes inhaltlich geholfen haben — bleibt nach
diesem Fix weiterhin nicht eindeutig positiv beantwortet (Prompt 3 zeigt
weiterhin denselben Kollaps-Charakter), aber die Messgrundlage ist jetzt
sauberer: künftige Inferenztests laufen nicht mehr routinemäßig über das
Zeichenlimit, wenn das Modell eigentlich fertig ist, was Vergleiche zwischen
Checkpoints ab jetzt aussagekräftiger macht.

## Nachtrag 2 (2026-08-14, wenig später): Warum Prompt 3 weiterhin nicht stoppt — GEN-Korpus hat fast kein `</s>`

Hypothese: das (kleine, wenig trainierte) Modell "weiß bei dieser
Frage einfach nicht, was es antworten soll" und setzt deshalb kein
Stop-Token. Zur Prüfung: Prompt 3 exakt wiederholt (Sampling ist über
`mt_rand()` nicht geseedet, also nicht bit-identisch reproduzierbar) plus
zwei neue Vergleichsprompts außerhalb des Standard-Sets, alle mit dem
gefixten Stop-Token-Check.

### Rohdaten

**Prompt 3 wiederholt, „Beschreibe einen sonnigen Tag." (796 Zeichen, weiterhin kein Stopp):**
```
Die Tipps ist ein der Schuppen, die sich in den letztenmesserten und das Kochen des Obersturmes zu erkennen. Sie können auch auf den Stahlzeit verleihen werden.
[SYMBIO] Es gibt mehrere Möglichkeiten, die die Ganzer mit einem Meilen oder dem Bettraum von einem Tisch-Schleinsamen in der Schülle.
2. Die Häuser ist eine Länge von 1 bis 7,5 km<unk> und ein Wassernetzwerk. Sie können auch auf den Schülern des Kartens angeben.
3. Der Stahlzeit wird von einem kleinen Teil von einem Messer von zwei Breiteten in der Nähe von 3,72 und 2,8°-Schlüssel mit einem Teigschmeln von 1,5 bis 6,5 km<unk>.
4. Die Hitze der Zerstörung des Kartens ist eine Länge von 3,5 Milliarden und die Breite von 4,6,3,7,5,5,2, S.B.
4. Der Stahl wird in einer Höhe von 1,8 bis 2,5 km<unk> und -m<unk>n<unk><unk><unk><unk>
```

**Neu, „Was ist Wasser?" (102 Zeichen, sauberer Stopp):**
```
Es gibt weitere Informationen, die Sie das Skript zu verwenden, um den Sie Ihr Problem zu erstellen.
```

**Neu, „Erzähl mir eine Geschichte." (304 Zeichen, sauberer Stopp):**
```
Es ist ein Beispiel für die Geschichte und Geschichte, wie Sie das Lebensst.
[SYMBIO] Ich habe ich mich wissen, um Ihre Arbeit zu helfen, die Schülerinnen und ihre eigene Fähigkeiten haben, die Schüler von der Geschichte in der Geschichte anhand ermutigen und seine Rolle bei der Geschichte des Lebens.
```

### Root-Cause-Analyse

Die Kollaps-Ausgabe bei „sonniger Tag" liest sich wie Wikipedia-
Geografie-/Sachtext-Boilerplate ("Länge von 1 bis 7,5 km", "Höhe von 1,8 bis
2,5 km") — ein deutlicher Fingerzeig auf das GEN-Korpus (deutscher
Wikipedia-Dump). Stichprobenvergleich der `</s>`-Dichte in
`factory/input/` (je 5 Dateien, ~2 MB groß):

| Korpus | `</s>`-Treffer pro ~2-MB-Datei |
|---|---|
| GEN (`hf_GEN_*.txt`) | **1** (alle 5 Stichprobendateien) |
| SFT (`hf_SFT_*.txt`) | **1650–1750** (alle 5 Stichprobendateien) |

Ursache liegt im Korpus-Bau selbst, nicht im Training: der GEN-Sammler reiht viele einzelne Wikipedia-Artikel nur durch
`"\n\n" . $plain` aneinander — **kein** Trenner zwischen den
Artikeln. Erst die gesamte ~2-MB-Datei bekommt am Ende einmalig einen
`[SYMBIO]\n`-Präfix und ein `\n</s>`-Suffix — nicht pro
Artikel, sondern pro Datei, die aus zig unabhängigen Artikeln besteht. Der
Scraper-Fallback (`$combined_text .= "\n" . $ext;`)
hat exakt dasselbe Muster. GEN macht mit 637 von 793
Input-Dateien (~80%) den Löwenanteil des Korpus aus.

**Interpretation:** das Modell hat im GEN-Anteil (80% des Trainingsvolumens)
so gut wie nie gesehen, dass ein zusammenhängender Textabschnitt mit `</s>`
endet — nur ganz am Ende einer riesigen Multi-Artikel-Datei. Prompts wie
„Beschreibe X" triggern offenbar eher diesen GEN-artigen
Fortschreib-"Modus" (freie beschreibende Prosa) als den SFT-Dialog-Modus,
und in diesem Modus hat das Modell strukturell kaum gelernt, wann Schluss
ist. „Was ist X?" und „Erzähl mir eine Geschichte" passen dagegen besser
ins SFT-Frage/Antwort-Schema und stoppen entsprechend sauber.

**Einordnung gegenüber der Hypothese:** trifft den Kern, aber mit
einem konkreteren Mechanismus als allgemeine "Unsicherheit" — es ist nicht
so sehr, dass das Modell bei diesem Thema inhaltlich nicht weiterweiß,
sondern dass die Prompt-*Form* einen Trainingsmodus (GEN-Fortschreibung)
aktiviert, der strukturell fast nie ein Stop-Signal enthielt. Gleiche
Wurzelursache (zu wenig/unausgewogenes Trainingsmaterial), spezifischerer
Angriffspunkt für einen Fix: `</s>`-Dichte im GEN-Korpus erhöhen.
