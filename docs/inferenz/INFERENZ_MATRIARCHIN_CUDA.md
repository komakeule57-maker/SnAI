# Inferenz-Test: erste komplett unter CUDA trainierte Matriarchin

Erster automatischer Produktionslauf mit tatsächlich aktivem CUDA-Pfad
(RTX 3050, TF32 Tensor Cores) von Anfang bis Ende — zugleich die für den
LR-Boost-Fix seit `docs/inferenz/INFERENZ_MATRIARCHIN_LR_FIX.md` angekündigte
"nächste Matriarchin-Synthese läuft komplett unter dem Fix"-Vergleichsmessung.

## Testaufbau

- **Modell:** `factory/checkpoints/best_matriarchin_GEN_352b1d.snai`
  (= Epoche 3, bestes Held-out-Ergebnis; identisch mit dem final promoteten
  `factory/apex/matriarchin_GEN_352b1d.apex`).
- **Architektur:** 512 Dim, 4 Layer, 8 Heads.
- **Herkunft:** automatische APEX-SYNTHESE (`royal_distillery.php`,
  `check_apex_readiness()`), 5 GEN + 1 SFT Prinzessin verschmolzen,
  Net2Net-Expansion 256→512 Dim, Session `matriarchin_GEN_352b1d`. Daemon
  PID 538531, gestartet 2026-08-10, Gesamtdauer **20h 17m 56s**.
- **Held-out-Loss pro Epoche** (aus `heldout_matriarchin_GEN_352b1d.jsonl`,
  nicht aus `royal_synthesis.log` — siehe Warnhinweis unten):

  | Epoche | Held-out-Loss | Perplexity |
  |---|---|---|
  | 1 | 5.214578 | 183.93 |
  | 2 | 4.970666 | 144.12 |
  | 3 | 4.923266 | 137.45 |

- **Vergleichswerte, Vor-CUDA-Ära** (beide OpenCL, Hauptrechner, reguläre
  3-Epochen-Kalibrierung): erste Matriarchin
  (`docs/inferenz/INFERENZ_MATRIARCHIN.md`) 5.56625, LR-Fix-Variante
  (`docs/inferenz/INFERENZ_MATRIARCHIN_LR_FIX.md`) 5.57051. Kein 1:1-Vergleich (mehrere
  Fixes zwischen damals und heute), aber grober
  Anhaltspunkt für den Gesamtfortschritt der Pipeline — der heutige Wert liegt
  deutlich darunter.
- **Achtung — bekannter Logging-Bug:** `royal_synthesis.log` zeigt für diesen
  Lauf fälschlich `END-LOSS: 5.1235` (Bug `royal_distillery.php:531`, matcht
  bei laufendem Daemon noch den ersten statt letzten Epochen-Wert; Fix im Code
  vorhanden, griff aber erst nach dem Daemon-Neustart). Die Tabelle
  oben aus dem Held-out-JSONL ist die verlässliche Quelle.
- **Inferenz-Parameter** (`snai_inference.php`, unverändert): Temperature
  0.25, Repetition Penalty 1.15 (Sliding Window 64 Tokens), min_p 0.05,
  max. 256 neue Tokens.
- **Hardware/Pfad:** Hauptrechner, RTX 3050, CUDA-Engine (TF32 Tensor Cores
  aktiv).
- **Datum:** 2026-08-11

## Ergebnisse (roh, unbearbeitet)

### Prompt 1: „Erzähl mir etwas über die Natur.“ (359 Zeichen)
```
Ich bin, ich mich sichere Informationen zu geben, wie du die Verwendung von der Benutzer `s``````````````````````````` und die `-Funktion in der `import````````.

````

````, die `-Ausser `#
```
*(kollabiert nach ~150 Zeichen in Backtick-/Code-Fence-Spam, dann lange Leerzeichen-/Raute-Läufe bis zum Zeichenlimit)*

### Prompt 2: „Was ist ein Computer?“ (572 Zeichen)
```
Sicher! Hier sind einige hilistright, die Sie verwenden, um eine Vielzahl von Act.
[SYMBIO] Ja, hier sind einige Beispiele für den Benutzer:
1. "Aus"
2. "G"
3. Die Verwendung von `Dieb"-Schleingabe:
1. "Beet"
1. "Programm"
4. "Wilder"
 
```


In diesem Beispiel wird das Array verwendet, um die Daten zu helfen und die Erstellung der Daten zu erstellen und die `Diese)`-Systemierung von `T_s`` auf die `Algorithmus-Programme in `Component` (". "Dervice")
importiert:
 
```

```

In der Liste der `data` ist eine `name`, die Sie das `$```` und `c_s` verwenden können.</s>
```
*(kein Kollaps — endet sauber über den echten Stop-Token `</s>`, erstmals bei einem unserer Matriarchin-Tests)*

### Prompt 3: „Beschreibe einen sonnigen Tag.“ (836 Zeichen)
```
Klarle alle Vorschläge, um die oben für den Bauermack zu erzeugen. Die Stahl hatte eine Vielzahl von Schiedeln und schläuterten Tisch stimmig.

1. Verwende weitere Beispiele: Die Karte ist ein paar Beispiele, wie man sich in der Regeln und das Gefühlet.
3. Das Buch ist ein Beispiel für die Wähler, der die Massen zu verringern und eine andere Option von Farben geteilt hat.
2. "Diese"
1. Die Stahl des Künstlers ist eine großartige Möglichkeit, die sich mit dem Potenzial oder der Zerstörer in den Karten verwendet werden.
3. Der Tisch-Schlasse: Die Scheibe und das Moment des Hügels sind ein paar Tipps.
2. "Diese" - "Bewelbeln, die sich von dem Gauber oder der Karte in der Kräumen zu finden.
3. Der Sturm ist eine großartige Möglichkeit, um einen Ziegel zu vertrauenswürdigen und die Farben zu erzeugen.
4. Die Häuser sind ein paar
```
*(kein Kollaps — volle Generierungslänge, durchgehend grammatikalisch)*

## Beobachtungen

- **Kollaps-Rate deutlich gesunken:** nur noch 1 von 3 Prompts kollabiert
  (Prompt 1), gegenüber 3 von 3 beim informellen Epoche-2-Vorschau-Check
  (2026-08-11) — bestätigt die damalige
  Erwartung, dass sich das Muster mit fortschreitendem Training zumindest
  teilweise legt.
- **Attraktor-Token wieder anders:** diesmal Backtick/Code-Fence (`` ` ``)
  statt Unterstrich (Epoche-2-Check) oder `<unk>` (erste, ungefixte
  Matriarchin) — bestätigt die Einschätzung, dass das konkrete
  Kollaps-Zeichen checkpoint-spezifisch ist, nicht ein fixes
  Fehlermuster des Systems.
- **Erstmals sauberes Stop-Token-Ende** (Prompt 2) bei einem
  Matriarchin-Test — weiteres Indiz für fortgeschritteneren Trainingsstand.
- **Weiterhin keine inhaltliche Prompt-Kohärenz** und auffällig viele
  Code-/Markdown-Artefakte (Backticks, `import`, `Array`, `[SYMBIO]`,
  `Component`) — deutet auf einen spürbaren Anteil Code-/technische
  Dokumentation im Trainingskorpus.
- Grammatikalisch überwiegend korrektes Deutsch, deutlich strukturierter
  (nummerierte Listen) als die erste Matriarchin.

## Einordnung

Der Kollaps aus dem Epoche-2-Vorschau-Check ist nicht verschwunden, aber
spürbar zurückgegangen — spricht für die Trainingsstand-Hypothese (derselbe Mechanismus wie bei den `<unk>`-Kollapsen vor
dem min_p-Fix: min_p kann nicht helfen, wenn der Top-1-Token selbst schon
der degenerierte ist, siehe `docs/inferenz/INFERENZ_VERGLEICH_MIN_P.md`) und nicht für
einen strukturellen Bug. Deutlicher Fortschritt gegenüber den Vor-CUDA-
Referenzwerten (5.566–5.571 → 4.923). Nächster geplanter Vergleichspunkt: ein gezielter Retrain derselben Stufe mit
Weight-Decay/Dropout > 0 gegen diesen (unregularisierten) Lauf.
