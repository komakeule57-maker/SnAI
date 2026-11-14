# Inferenz-Test: Prinzessin, 256-Dim-Maximaltraining (Epoche 40/40, final)

Abschlusstest des Laptop-Maximaltrainings (`longtrain_256`)
— Fortsetzung von `docs/inferenz/INFERENZ_PRINZESSIN_MAXTRAIN_256.md`
(Zwischenstand Epoche 25). Der Lauf ist am 2026-08-15 um 09:55 CEST mit allen
40 Epochen fertig geworden (Worker-PID 1567856, `train_cluster.php`-Prozess
danach beendet, kein Kill nötig — regulärer Abschluss).

## Vorbereitung: Stop-Token-Fix auf den Laptop nachgezogen

Der Laptop läuft bewusst auf einem älteren Codestand (kein Batch-Shuffling,
für einen späteren A/B-Vergleich). Dabei war `snai_inference.php` noch auf
dem Stand *vor* dem am 2026-08-14 auf dem Hauptrechner gefundenen und
gefixten Stop-Token-Bug:
`$stop_tokens = [2]` prüfte auf `<s>` (Start-Token) statt `</s>`
(tatsächliche Ende-Token-ID im aktuellen `symbio_vocab.json`; auf dem Laptop
verifiziert: `</s>` = ID 3, `[USER]` = ID 5). Dieser Bug betraf laut
Haupt-Notiz vermutlich jeden bisherigen Inferenztest im Projekt, auch den
Epoche-25-Zwischenstand oben.

Vor diesem Test 1:1 denselben Fix wie auf dem Hauptrechner angewendet
(`snai_inference.php`, Zeile ~154): `$stop_tokens` jetzt dynamisch über
`$tokenizer->encode("</s>")` aufgelöst statt hartkodiert `[2]`. Nur diese
eine Stelle geändert, **kein** Beam-Search-Sync (Hauptrechner-Feature vom
14.08., auf dem Laptop bislang nicht vorhanden) — außerhalb des Umfangs
dieses Tests. Alte Version gesichert unter
`snai_inference.php.bak_prestop_fix` (Laptop). `php -l` sauber, keine
sonstigen Codeänderungen.

## Testaufbau

- **Modell:** `factory/testruns/longtrain_256/trained_256_e40.snai` (Laptop,
  finales Schwarm-Modell laut `train.log`: „Schwarm-Modell bereit").
- **Trainings-Loss:** Epoche 40 (final) Avg Loss **3.38581** — gegenüber
  3.663391 bei Epoche 25 weiter gefallen. Weiterhin reiner
  Trainings-Fit-Wert, kein Held-out-Tracking bei diesem Lauf (bei ~1,3 Tokens/Parameter
  vermutlich überwiegend Memorierung).
- **Herkunft:** wie beim Epoche-25-Test — Genesis-Modell (kein
  Net2Net-Rauschen), voller 24-MB-Korpus (`corpus_full.txt`, 198.373 Zeilen,
  8.219× `</s>` enthalten — anders als der damals fürs Haupttraining
  diagnostizierte, fast `</s>`-freie GEN-Korpus nicht auffällig dünn),
  Warmup+Cosine-Decay, altem Codestand ohne Batch-Shuffling.
- **Architektur:** 256 Dim, 2 Layer, 4 Heads.
- **Inferenz-Parameter** (`snai_inference.php`, sonst unverändert):
  Temperature 0.25, Repetition Penalty 1.15, max. 256 neue Tokens,
  Stop-Token-Fix wie oben angewendet.
- **Hardware/Pfad:** Laptop, AMD Radeon Graphics (renoir, OpenCL/RustiCL-
  Fallback, kein echtes `CL_DEVICE_TYPE_GPU` gemeldet).
- **Datum:** 2026-08-15.

## Ergebnisse (roh, unbearbeitet)

### Prompt 1: „Erzähl mir etwas über die Natur." (827 Zeichen, kein Stopp — Tokenlimit erreicht)
```
Die Natur und eine Natur zu Natur, die sich auf der Natur und die Naturrechtszeit zu beenden. Es wird auch durch den „Die" genannt. Eine „Bef" bezeichnet die „Schweil".

Der Spiegel ist ein Beispiel, das in der Natur und die Natur und die Naturschutz, die hätten sich im Laufe der Zeitalter zu verstehen.

In der Welt hat die „Groß".

Die Lindensteins (* „PD-Mitgliede") und ist eine deutsche Verständnis von Natur und die Natur des Kirchenrechts."

Das Grundstück ist der Natur- und Kunstgeschichte.

Die DDocke ist seit dem Ende des Zweiten Weltkriegs, in der Kirche, in Deutschland und der Natur.

Kurt Langmann:

Neues Bild (* 19. Januar 1887 in Scheidelberg; <unk> 17. Februar 1851 in Wien) war ein deutscher Diplom, d.&nbsp;a.)

Der erste jüdische Geschichte in der Stadt.

Hans-Ursprünglich des Landes Baden-Württemberg
```

### Prompt 2: „Was ist ein Computer?" (1000 Zeichen, kein Stopp — Tokenlimit erreicht)
```
Es ist wichtig, dass das Modell von einem Principal Engineer und es ist möglich, das Modell als auch eine Vielzahl von Anwendungen mit einer Vielzahl von Netzwerk, die zur Erlebualität zu bewerten. Die Auftragung von Datensätzen werden in der Regel mit einer Vielzahl von Gegenständen verwendet, die zur Netze zu verwalten. Die Aufteilung wird durch eine Aufrechterhaltung eines Modelles, das zur Verteilung von Hackanerkennung von Fülern, die zur Übertragung von Hängen und mit dem Bits, dem Verfahren zur Herstellung von Hitlers, dem die Behandlung der Gesundheitsprozesses, dem heutigen Symptomen, der zur Behandlung der Gesundheit des Mensums, der durch den technischen Prozess der Baureihe von Hinduismus und der Gesundheit von Helsinklerin und der Vergangenen Kenntnisse der Biotechnik, der zur Verarbeitung von Hutor und der Biosphäre, die zur Verpackung der Verantwortung der Verpackung und dem Bioxidamentoktor zu verursachen.

Die Bundesrepublik China wurde von der Verfassungslosen Biover
```

### Prompt 3: „Beschreibe einen sonnigen Tag." (746 Zeichen, kein Stopp — Tokenlimit erreicht)
```
Um eine Dungea-Sammlung des Kunstparks der Tierbranche zwischen den beiden Platzierungen und dem Zweiten Weltkrieg/Schluba-Spieler bei der Qualifikation zur Saison 2023/19. Am Ende der Spielzeit 2011/082.09 gewann die Mannschaft in der Saison 2006/1282/29/384/42.html

Nachruf/www/mkg/su.jpg|Gigsliste

Die

Im Mai 2017

Also

Das

Vorlage

Erfahrung

Diese Liste

Chorus (* 12. März 2009 in der Folge) ist ein deutscher Fußballspieler.

Im Jahr 2013 wurde er als Meister der Vereinigung der Ligasträger der FC.

Europameisterschaft Kurt ist der Name folgender Personen:

Theodor von der Stadt Nelsberg-Fürttemberg

Broschi

Karl Kelsen (* 22. Juni 1945 in Lastowballspieler, einer Gruppe von 1557 bis zum 1. September 2007 in der Region VfR-115
```

## Beobachtungen

- **Kein `<unk>`-Kollaps mehr** bei keinem der drei Prompts — anders als beim
  Epoche-25-Zwischenstand, wo Prompt 3 in eine `<unk>`-Wiederholungsschleife
  kollabierte. Einzige Verbesserung gegenüber dem Zwischenstand, die sich
  klar isolieren lässt.
- **Weiterhin keine inhaltliche Kohärenz zum Prompt** — driftet bei allen
  drei Versuchen sofort in Wikipedia-artige Biografie-/Sport-/Orts-
  Aufzählungen ab, identisches Muster wie bei Epoche 25.
- **Stop-Token-Fix greift technisch, aber keiner der drei Läufe nutzt ihn** —
  alle drei erreichen das 256-Token-Limit statt vorzeitig auf `</s>` zu
  stoppen. Anders als beim Matriarchin-Retest nach demselben Fix
  (`docs/inferenz/INFERENZ_MATRIARCHIN_REGULARISIERT.md`, Nachtrag: dort 2 von 3
  Prompts mit sauberem Stopp) sagt dieses Modell bei keinem der drei
  Standard-Prompts ein `</s>` mit ausreichender Konfidenz voraus. Nicht
  durch dünne `</s>`-Dichte im Korpus erklärbar wie beim damaligen
  Matriarchin-Root-Cause (dieser Korpus hat 8.219 Treffer auf 24 MB, im
  Schnitt alle ~3 KB, deutlich dichter als der damals diagnostizierte
  Fast-Null-Fall) — offene Beobachtung, in diesem Test nicht weiter
  untersucht.
- **Möglicher Wikitext-Markup-Rest bei Prompt 3** („Vorlage", „Vorlagenname"-
  artige Fragmente, `Nachruf/www/mkg/su.jpg|Gigsliste`) — erinnert an die am
  gleichen Tag auf dem Hauptrechner gefundene und gefixte Wikitabellen-/
  Vorlagen-Leck-Ursache ("Wikitabellen-/Vorlagen-Leck..."). Dieser Laptop-Korpus
  (`corpus_full.txt`) ist aber ein separater, statischer Snapshot von vor
  diesem Fix und wurde nicht retrofittet — plausibel, dass dieselbe
  Leck-Klasse hier ungefixt vorliegt. Nicht untersucht/behoben, da
  außerhalb des angefragten Testumfangs (separater Branch/Korpus) — nur als
  Hinweis notiert.
- Nebenbefund, kein Blocker: `[cortex] WARN: adam_t fehlt im Checkpoint, m/v
  sind aber warm -> adam_t = 1000` — bekannte Inferenz-Randnotiz, betrifft
  nicht die Generation-Qualität (Ursache/Fix bereits für den Hauptrechner-
  Pfad dokumentiert ; auf dem Laptop nicht nachgezogen, da außerhalb des
  Testumfangs).

## Einordnung

Trainings-Loss ist über die letzten 15 Epochen weiter gefallen
(3.663391 → 3.38581), und der `<unk>`-Kollaps-Fehlermodus aus dem
Zwischenstand ist verschwunden — aber die grundsätzliche Diagnose vom
Epoche-25-Test bleibt unverändert: grammatikalisch überwiegend flüssiges
Deutsch, aber keine inhaltliche Prompt-Kohärenz, stattdessen durchgängiges
Abdriften in Wikipedia-artige Aufzählungsprosa. Da kein Held-out-Loss-
Tracking existiert, bleibt offen, wie viel des gesunkenen Trainings-Loss
Generalisierung vs. Memorierung ist (~1,3 Tokens/Parameter/Durchlauf bei 40
Epochen auf demselben 24-MB-Korpus spricht eher für Memorierung). Damit ist dieser Maximaltraining-
Lauf abgeschlossen; kein weiterer automatischer Fortsetzungslauf auf dem
Laptop vorgesehen (letzter bekannter Stand, ggf. beim nächsten Laptop-Check
gegenprüfen).
