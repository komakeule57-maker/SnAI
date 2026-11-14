# Inferenz-Test: erste Matriarchin (vor dem LR-Fix)

Erster Inferenztest mit einer echten Matriarchin (512 Dim, 4 Layer, 8 Heads) —
Referenzpunkt vor dem LR-Boost-Fix, für den späteren Vergleich mit einer
Matriarchin, die komplett unter dem Fix trainiert wurde.

## Testaufbau

- **Modell:** `factory/apex/matriarchin_GEN_b760a8.apex` (Hauptrechner, RTX 3050)
- **Trainings-Loss:** Achtung, `royal_synthesis.log` loggt fälschlich Epoche 1s
  Loss (6.05171) statt des tatsächlichen Endwerts — siehe Bug in
  `royal_distillery.php:488` (`preg_match()` statt `preg_match_all()` auf den
  kompletten 3-Epochen-Output, matcht den ersten statt den letzten Treffer).
  **Echter Endwert nach Epoche 3: 5.56625** (aus dem Daemon-Log nachgezogen:
  Epoche 1 = 6.05171, Epoche 2 = 5.71142, Epoche 3 = 5.56625).
  **Nachtrag 2026-08-11:** Bug im Code behoben, greift aber erst ab dem nächsten Neustart des Steuer-Daemons —
  bis dahin bleibt der oben beschriebene manuelle Nachzieh-Workaround nötig.
- **Herkunft:** reguläre Schwarm-Pipeline, 3 Prinzessinnen verschmolzen,
  Net2Net-Expansion 256→512 Dim, LR-Boost-Fix zu diesem Zeitpunkt **noch nicht**
  auf dem Hauptrechner aktiv.
- **Parameter:** Temperature 0.25, Repetition Penalty 1.15, max. 256 neue
  Tokens (identisch zu den vorherigen Vergleichsdokumenten).
- **Datum:** 2026-08-04

## Ergebnisse (roh)

### Prompt 1: „Erzähl mir etwas über die Natur.“ (648 Zeichen)
```
Sicher, die „bspm Jahrministerische Schätigen und in der Sie-Vertar)

Dereinigte den ersten Filme in der Gände des B<unk><unk>cyi (Ler)

Angt ein Beispiel der Kanga-Schant:

Neseinigungsminister
Bach der Universität Harten ist eine Beidung von der Stadt-Prof, um das Dözier und der Dés-Gerglisch, wie die Nendos und die Goller und der Tin-Bandel, dass die sich für die Milligulierte und in der Wägot-Martig in der Gemeinde-C-Kapani.

Weutsche, die die Ruadierung des Prickemlichen Alter des Lundel.

Derutscher Fäze, der Bauket, die Ableben, dass das in der J.-Aniversität am 3.

Das Gebäude, die Tessorgen des Bugo mit dem Dagal, der S.&nbsp;F.)
```

### Prompt 2: „Was ist ein Computer?“ (958 Zeichen)
```
Ein setzung der Aktor des Wissenschaften von B<unk>-Verteinigte die Stadtberter (Leglisch), um eine an der Tundenen und in der S.-Profärten, dass die „Kürnen“), um den ersten Stanischen Kapel-Wand), dass das Kartung der Pgotischen Kinns.

Die Guelle – In:

Der Jahr 2016: Die Mahinische Frau (Scha)
Amach:

Baro/R.jani<unk><unk><unk>...<unk>  [kollabiert in eine lange <unk>-Wiederholungsschleife bis zum Zeichenlimit]
```

### Prompt 3: „Beschreibe einen sonnigen Tag.“ (575 Zeichen)
```
Die erste Fußballspieler und war dem Jahr 1995, dass die Vie in der Stadt-Derglisch, die „Schanti), die sich der Universität des Zweiten Weltkrieg wurde.

Am Jahr 2018 ist ein deutscher Politiker
199862.

Weutscher Film:

Haras Géd (Ba)

Nach der Gemeinde „Lürzt])
In der US-Verto (Prsg.)

Jrsg.

Zubert Pg|Derutsche (Aniversität“)

Dereinigungsministers
Die Sie eine Schätter des Bahkang, um die Nichen von der Kulatischen Timation
Miline
Die Dielan (*)

Die Fonti-Fa-Gerglisch)

Die B<unk>coni (Ler)

Die Bos-Se-Cret

Alne –

Die Rappe
Die Kitin:

Die Bordner

The Lecto
```

## Beobachtungen

- **Deutlich weniger Artikel-Wiederholungsschleife** als bei den Prinzessinnen
  (256 Dim) — kein endloses "der/die/den"-Looping mehr.
- **Spürbar mehr echte Wikipedia-Struktur:** Universität, Politiker, Zweiter
  Weltkrieg, Jahreszahlen (1995, 2016, 2018), Klammer-Referenzen — sieht auf
  den ersten Blick "seriöser"/enzyklopädischer aus als die Prinzessin-Outputs.
- **Weiterhin frei erfunden:** kein einziges der generierten "Wörter" wie
  "Kanga-Schant", "Derglisch", "Wägot-Martig", "Neseinigungsminister" ist
  echtes Deutsch — Wortfetzen-Collage, keine echte Lexik.
- **Keine inhaltliche Kohärenz zum Prompt** bei allen drei Versuchen — weder
  Natur noch Computer noch "sonniger Tag" werden tatsächlich behandelt.
- **Neuer Fehlermodus bei Prompt 2:** Kollaps in eine lange `<unk>`-Token-
  Wiederholungsschleife bis zum Zeichenlimit — bei den Prinzessinnen eher
  Leerzeilen/Sonderzeichen, hier stattdessen ein sich wiederholendes
  Unbekannt-Token.

## Einordnung

Subjektiv deutlich überzeugender als die Prinzessin-Ergebnisse trotz nur 3
Trainings-Epochen — bestätigt, dass Modellgröße/Architektur bei diesem Tier
bereits mehr hergeben, als die kurze Kalibrierung ausschöpft. Dieses Dokument
ist der Vorher-Referenzpunkt: die nächste Matriarchin-Synthese läuft komplett
unter dem LR-Boost-Fix — direkter Vergleich
folgt, sobald die fertig ist.
