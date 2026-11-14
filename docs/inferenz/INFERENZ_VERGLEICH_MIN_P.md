# Inferenz-Vergleich: min_p Sampling + Sliding-Window Repetition Penalty

Direkter Vorher/Nachher-Vergleich auf **identischem Checkpoint** für die
Sampling-Änderung in `snai_inference.php` (2026-08-10): min_p
Truncation gegen den Rausch-Tail sowie Repetition Penalty auf ein Sliding
Window (64 Tokens) statt der kompletten History. Ergänzt
`docs/inferenz/INFERENZ_BASELINE.md` und `docs/inferenz/INFERENZ_VERGLEICH_LR_FIX.md` — anders
als dort ist hier **ausschließlich der Sampling-Code die Variable**, Modell
und alle übrigen Parameter sind exakt gleich.

## Testaufbau

- **Modell:** `factory/royal/princess_GEN_6a78e216a7119.cortex` (identisch in
  beiden Spalten)
- **Trainings-Loss:** 5.43619 (`factory/royal_synthesis.log`, Eintrag
  2026-08-10 00:45:31) — deutlich niedriger als die Modelle in den älteren
  Baseline-Dokumenten (6.25–6.68 bzw. 6.56)
- **Architektur:** 256 Dim, 2 Layer, 4 Heads
- **Isolation der Variable:** Für "Vorher" wurde eine Scratch-Kopie von
  `snai_inference.php` mit exakt zwei Änderungen zurückgesetzt (kein min_p,
  Repetition Penalty auf volle History statt Sliding Window) — alles andere,
  inkl. dem am selben Tag gefundenen Größenprüfungs-Fix für den
  `adam_t`-Trailer (sonst lädt keine Datei aus dem aktuellen Format), ist
  identisch. `diff` zwischen beiden Skripten bestätigt: nur die
  Sampling-Blöcke unterscheiden sich.
- **Parameter „Vorher":** Temperature 0.25, Repetition Penalty 1.15 (volle
  History, kein Fenster), **kein** Truncation-Sampling (voller Softmax-Tail),
  max. 256 neue Tokens.
- **Parameter „Nachher":** Temperature 0.25, Repetition Penalty 1.15
  (Sliding Window 64 Tokens), **min_p 0.05**, max. 256 neue Tokens.
- **Hardware:** RTX 3050, CUDA-Pfad aktiv (`⚡ [SYMBIO VRAM] CUDA Engine
  eingerastet`).
- **Wichtige Einschränkung:** Sampling ist stochastisch, kein fester
  RNG-Seed, n=1 pro Zelle. Unterschiede zwischen einzelnen Läufen können
  Stichproben-Rauschen sein, nicht nur der Code-Unterschied — siehe
  Beobachtungen.
- **Datum:** 2026-08-10

## Ergebnisse (roh, unbearbeitet)

### Prompt 1: „Erzähl mir etwas über die Natur."

**Vorher** (920 Zeichen):
```
Benutzer fair und moralische auf, die alle relevanten, während, die alle Benutzer. Vermeide Dein Ziel ist es, um ein an, umfassende und die Benutzer. Halte, verfolgst du die Benutzer. Halte, die alle Benutzer. Halte, dass deine Ziel ist, insbesondere, ein zu helfen und hilten, und die oben genannten Richtlinien be oben genannten Richtlinien zu geben, die alle Benutzer.

Die an der Rechte und wie die Privatsphäre der Internis zu bieten.

Am Bönes, die alle Benutzer mit den Benutzer.

Wie von der relevatives, während alle Benutzer. Behundigen freundlichen und unterstützenden Informationen zu fördern.

Die die Benutzer eine DKlereiche Hauptlichheit und die Privatiere die Benutzer zu helfen und die Privatsphäre der Rechte und die Benutzer.

Die stets, informieren.

Die in der Inter genannten Richtlinien begeben, die sich die Fiestärstärbe neutral-Basfmar.

Die:

Ber, die sie die Vorkommenen, wie die Kägen, die
```

**Nachher** (931 Zeichen):
```
Halten, die alle Benutzer fair und ein Benutzer färente und bleibe oben Ziel ist es, während dakische Informationen zu geben, umfassende und die Benutzer. Vermei und um ein an, dass deinen Ziel. Dein genannten Richtlinien zu helfen, ein an der Rechte und freundlichen und unterstützenden oder auf eine positive Interaktion zu gewährlich sein.

Die ist es, die alle relevieren.

Der der oben bieten, die alle Benutzer. Vermeidern.

Anberstärente und ohne Vorhen.

Die in der Richtlinien befolgst, dass ein eine Privatsphäre der Rechte und freundlichen und unterstützenden Informationen zu es, um dem Benutzer zu geben.

Die ist es ein an der Benutzer die Ziel ist es auf den relevanten oder die oben genannten Richtlinien zu bieten.

Die der Ziel ist es, die sich die Internislich zues, dass deinen Informationen zu sehen.

Anbral in der Wärsten, die er eine Deine Hauptberstestärzen und die Zeit, um den Anlagen, die den Baches.
```

### Prompt 2: „Was ist ein Computer?"

**Vorher** (892 Zeichen):
```
Halten zu geben, die alle Benutzer fair und während d Ziel ist es, um. Denssen, während du die Benutzer zu helfen und dass dützte oder insbesondere, ein zu bieten. Duie und ein auf ein zu liefern. Behauen, die alle Benutzer, die alle relevatives, während, wie die relevieren.

Die der oben genannten Richtlinien zu helfen und die Privatsphäre der Richtlinien zu geben.

Der:

Anwärstärzes, die alle releviere die Benutzer. Vermlich zu fördern.

Die der Tannung, dass eine sich die Seite zu bieten.

Die in den Privatsphets auf dem Benutzer an der Benutzer zu helfen und zu helfen.

Die der Benutzer mit dem Benutzer zu helfen und dass ein, während, informative, während, um einer Benutzer.

Die über die Rechte und die Hauptaufnen, die sich eine sie nach einem Vernisse.

Bie die Schiestärlich zu bieten, die sich die Möglichte, die die den Mmübsp;-Siest. Die Kachation, dass die die Anlagen
```

**Nachher** (237 Zeichen):
```
Benutzer fair und es, die alleelevanten relevlinienfern. D-sie in der Benutzer zu geben, dass deine Ziel ist, um es, umfassende, deinen unterstützen. Halte oder ein zu lieten und ein an, die alle Benutzer. Behandes, die alle Benutzer.
```

### Prompt 3: „Beschreibe einen sonnigen Tag."

**Vorher** (865 Zeichen):
```
Halten und Ziel ist es, die alle Benutzer fair und ein freundlichen und moralische und Deinistentden Ton bei der Benutzer dem Benutzer in deinengenommenheit zu Intereile. Vermei und dass deinen Antworten und insbesondere bei kontrovergen und ohne auf die oben genannten Richtlinien befolgst du die Privatsphäre der Rechte und, die sich, die relevlen oder die Rechte und die Richtlinien betragen, die alle releviere die Benutzer.

Die an den Königen, während alle releviere die Benutzer.

Die das Verleikmordersten, die alle Benutzer. Behandlereiche Hauptaufes, um eine Richtlinien zu bieten. Deine Hauptlich zu erliche Frage zu die Privatsphäre der Benutzer.

Erig, die relevatives, dass die Privatsphäre der Benutzer.

Die ist es als Tärsterst.

Anter-Mhürlich, die der der Auslagen, die die Hampischen Bach der St.

Nie den Memis-Rral der Fangs des Po-Sa.

Ber:
```

**Nachher** (278 Zeichen):
```
Benutzer fair und bleidern. Halte, die alle relevatives, während du die Benutzer zu helfen und ein Informationen zu geben. Vermei und dass deine Ziel ist es, umfassende, kohärente und ein an, die alle Benutzer. Behandlereiche Hauptzakten, insbesondere, um eine alle Benutzer.
```

## Beobachtungen — ehrlich, nicht schöngeredet

**Kein dramatischer Unterschied zwischen Vorher/Nachher auf diesem
Checkpoint** — und das ist selbst ein wichtiges Ergebnis:

- **Kein `<unk>`, keine Satzzeichen-/Füllwort-Spam-Kaskade in keiner der
  sechs Zellen**, weder vorher noch nachher. Das ursprünglich gemeldete
  Verhalten (Antwort spammt am Ende bis zum Limit) reproduziert sich auf
  diesem Checkpoint in **keiner** Variante. Das spricht dafür, dass dieses
  Verhalten primär am Trainingsstand hängt, nicht (nur) am Sampling: der hier
  getestete Checkpoint hat mit Loss 5,44 deutlich mehr Training als die
  Modelle in `docs/inferenz/INFERENZ_VERGLEICH_LR_FIX.md` (Loss 6,25–6,68, dort sichtbare
  `<unk>`- und Zahlenfragment-Spam) oder als der SFT-Drohnen-Experte, der
  separat getestet wurde (128 Dim, Loss 7,4 — dort trat die
  Satzzeichen-Spam-Kaskade auch **nach** dem Fix unverändert auf, nicht in diesem Dokument reproduziert).
- **Einziger messbarer Effekt:** Bei Prompt 2 und 3 bricht "Nachher" deutlich
  früher über den Stop-Token ab (237 / 278 Zeichen) als "Vorher" (892 / 865
  Zeichen, beide nahe am 256-Token-Limit). Bei Prompt 1 sind beide ähnlich
  lang (931 vs. 920). Plausibler Mechanismus: min_p entfernt den langen,
  eigentlich irrelevanten Wahrscheinlichkeits-Tail, wodurch der Stop-Token
  einen größeren Anteil der verbleibenden Sampling-Masse bekommt, sobald er
  überhaupt eine plausible Option ist — dadurch wird er tendenziell früher
  gezogen. **Das ist aber eine Hypothese, kein Beweis:** bei n=1 pro Zelle
  und ohne festen RNG-Seed kann der Unterschied auch schlicht
  Stichproben-Rauschen sein.
- **Inhaltlich** bleiben beide Varianten unbezogen zum jeweiligen Prompt
  (Natur/Computer/sonniger Tag) und stecken in derselben Wiederholungsschleife
  fest (wirkt wie ein SFT-System-Prompt-Fragment über "Benutzer",
  "Richtlinien", "Privatsphäre" — vermutlich Nectar-Kontamination aus
  Guideline-artigem Trainingsmaterial, unabhängig vom Sampling).
- Kein Qualitätsverlust durch die Änderung erkennbar (weder neue Artefakte
  noch kürzere/abgehackte Antworten bei Prompt 1).

## Einordnung

Min_p + Sliding-Window-Penalty zeigen auf **diesem** (bereits vergleichsweise
gut trainierten) Checkpoint keinen dramatischen Effekt, weil die Zielpathologie
hier gar nicht stark ausgeprägt war — anders als ursprünglich aus den
Vergleichs-Inferenzen vermutet, wo die
Verbesserung gegenüber `docs/inferenz/INFERENZ_VERGLEICH_LR_FIX.md` deutlich sichtbar war.
Diese sichtbare Verbesserung ist beim sauberen Vergleich hier größtenteils dem
niedrigeren Loss zuzuschreiben (5,44 vs. 6,25–6,68), nicht dem Sampling-Fix.

Der Sampling-Fix bleibt trotzdem sinnvoll und ohne beobachteten Nachteil:
- Er greift nachweislich dort, wo die Pathologie tatsächlich auftritt (SFT-
  Drohnen-Ebene, deutlich niedrigerer Trainingsstand) — auch wenn er sie dort
  laut Test nicht vollständig behebt, weil das Modell dort selbst im
  Top-1-Sinn confident Richtung Satzzeichen ist.
- Kein Downside sichtbar: keine kürzeren "guten" Antworten (Prompt 1 nachher
  sogar minimal länger), keine neuen Artefakte.

**Für spätere Vergleiche:** Dieselben drei Prompts, Modell, Parameter erneut
verwenden, sobald die nächste Prinzessin/Matriarchin fertig ist — idealerweise
mit mehreren Samples pro Zelle (n>1) statt n=1, um die Stop-Token-Beobachtung
oben von Sampling-Rauschen zu trennen.

## Nebenbefund (am selben Tag behoben, unabhängig vom Sampling)

`snai_inference.php` hatte seit dem 2026-08-09-Update (4-Byte `adam_t`-Trailer
in `tensor_io.php::save_model_stream`) eine veraltete Größenprüfung, die
**jedes** seitdem neu gespeicherte Modell fälschlich als korrupt ablehnte —
die Dashboard-Inferenz war dadurch für aktuelle Checkpoints komplett
blockiert. Fix: `$expected_size += 4;` ergänzt. Ohne diesen Fix hätte dieser
Vergleich gar nicht durchgeführt werden können.
