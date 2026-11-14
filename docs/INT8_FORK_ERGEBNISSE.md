# INT8-Fork: Ergebnisse der Chinchilla-Prinzessin (Auswertung 2026-08-27)

**Quelle:** `snai_edge_int8_full` auf dem Laptop (<lan-ip>), Lauf
`run_princess_chinchilla.sh`, Logs unter `~/.cache/prin/`.

> **KORREKTUREN (2026-08-27)**
> Drei Punkte dieser Auswertung waren falsch und sind unten im Text
> richtiggestellt, nicht überschrieben:
>
> 1. **Die Zahlentabelle stammt aus dem billigen Trainer-Indikator**, nicht aus
>    einer 60-Fenster-Auswertung. Nachgeprüft: die Werte stehen wörtlich in
>    `factory/checkpoints/best_prin_int8_s{0,13}.json`, und die zugehörigen
>    `heldout_i8_*.jsonl` weisen `heldout_tokens` im Bereich 500–1000 aus
>    ( 999; zwei von mir gezogene Dateien: 511). Das ist
>    rund 1/200 der belastbaren Messung. Die richtigen Zahlen stehen jetzt in
>    „Die Messwerte".
> 2. **Der Abbruch war kein Rätsel** — Segment  14 wurde um
>    21:15:32 beendet.
> 3. **Die Ursache des Segment-6-Sprungs war falsch zugeordnet.** Nicht der
>    SFT-Nachschub-Fix, sondern eine bewusste blockweise Durchmischung von
>    `pseg06–33` am 26.08. um 20:32.
>
> Ein vierter Punkt war eine unzulässige Verallgemeinerung von mir und ist
> unten als eigener Abschnitt ergänzt: „der Hauptzweig" hat den Fix **nicht**
> überall — nur die Kopie auf <lan-ip>.

## Status: der Lauf ist bei Segment 13 von 34 stehengeblieben

Segment 13 endete sauber um 20:55:52. Segment 14 lief an und bricht im Log um
**21:15:23 mitten im Training ab** (Step 110, keine Fehlermeldung, kein
Traceback). Das Treiberskript läuft nicht mehr, kein Worker ist aktiv. Das
sieht nach einem Kill von außen aus, nicht nach einem Absturz — ein SIGSEGV
oder ein PHP-Fatal hätte eine Spur hinterlassen.

Erreicht sind damit 13 von 34 Segmenten, also **38 % des geplanten Laufs**.
Die Checkpoints `prin_int8_s{4,8,11,12,13}.snai` liegen noch (das Skript räumt
ältere planmäßig weg und behält jedes vierte).

## Die Messwerte

**Die belastbaren Zahlen** (60 Fenster):

| | Segment 0 | Segment 13 |
|---|---|---|
| In-Domain | 7,0531 (PPL 1156) | **3,8633 (PPL 48)** |
| Transfer | 7,8252 (PPL 2503) | **4,6471 (PPL 104)** |

**Die Tabelle unten ist der 999-Token-Indikator des Trainers und als Kennzahl
unbrauchbar.** Sie steht hier nur noch, weil sie zeigt, *wie* unbrauchbar: sie
hat einen echten Einbruch bei Segment 4 (+0,79 Loss, 4,0 σ) vollständig
verschluckt und stattdessen die glatte Folge 6,858 → 6,856 → 6,846 gemeldet.
Wer nur auf diese Spalte schaut, sieht eine ruhige Kurve, wo eine Störung war.

| Segment | Held-out (Indikator, ~1/200 der Daten) | PPL | End-Loss | GradNorm (Median) |
|---|---|---|---|---|
| 0 | 7,786208 | 2407,17 | 5,36041 | 795 |
| 1 | 7,434404 | 1693,25 | 4,16939 | 895 |
| 2 | 6,857573 | 951,06 | 3,78606 | 798 |
| 3 | 6,855981 | 949,54 | 3,97352 | 792 |
| 4 | 6,846287 | 940,38 | 3,21848 | 693 |
| 5 | 6,619363 | 749,47 | 3,75474 | 782 |
| **6** | **5,213466** | **183,73** | 4,34770 | 908 |
| 7 | 5,008233 | 149,64 | 4,00846 | 889 |
| 8 | 4,888303 | 132,73 | 4,01102 | 909 |
| 9 | 4,784464 | 119,64 | 3,96895 | 940 |
| 10 | 4,751934 | 115,81 | 3,93427 | 1010 |
| 11 | 4,716437 | 111,77 | 3,94266 | 1114 |
| 12 | 4,613207 | 100,81 | 3,88332 | 1140 |
| 13 | 4,597702 | 99,26 | 3,81215 | 1177 |

Held-out fällt monoton von 7,786 auf 4,598, die Perplexität von 2407 auf 99,3.
Rund 2,3 Stunden je Segment, durchgehend zwei Worker.

## Befund 1: der Sprung bei Segment 6 ist ein Korpuswechsel, kein Lernsprung

Zwischen Segment 5 (6,619) und 6 (5,213) fällt der Held-out um **1,41** — mehr
als in den fünf Segmenten davor zusammen. Die Dateizeitstempel erklären es:

```
pseg04.txt, pseg05.txt   2026-08-26 04:31
pseg06.txt, pseg07.txt   2026-08-26 20:30
```

Der Nektar wurde **mitten im Lauf neu gebaut**.

**Korrektur zur Ursache:** ich hatte das dem SFT-Nachschub-Fix und der
Lang-Modus-Umstellung desselben Tages zugeschrieben. Falsch. Es wurde
`pseg06–33` am 26.08. um 20:32 **blockweise durchmischt**, weil
buchweise Segmente nachweislich Fortschritt zerstörten. Die Zeitstempel waren
richtig gelesen, der Schluss daraus nicht — ich habe eine mir bekannte
Änderung desselben Tages als Erklärung genommen, statt nach der tatsächlichen
zu fragen.

**Und der Sprung ist real, nicht messtechnisch:** er erscheint auch auf dem
unveränderten Wikipedia-Held-out (6,8869 → 5,1458, 26,8 σ). Meine Formulierung
„kein Lernsprung" war insofern zu scharf.

Was bleibt: es ist ein **Regimewechsel der Daten**, kein organischer
Lernfortschritt. Die Kurve über Segment 5/6 hinweg darf nicht als eine Kurve
gelesen werden.

## Befund 2 (der schwerwiegende): der Fork trainiert mit dem Clipping-Bug vom 18.08.

Gegenüberstellung der Quelltexte, beide heute gelesen:

| | Hauptzweig | `snai_edge_int8_full` |
|---|---|---|
| Clip-Aufruf | `clipGlobalGradientNorm(state, clip_norm)` | `clipGlobalGradientNorm(state, 1.0)` — **beide** Aufrufstellen (Z. 3847, 5285) |
| Schwelle | `let clip_norm = max(1.0'f32, trainable_count.cfloat)` | fest verdrahtet `1.0` |
| geloggter Wert | `last_grad_norm = last_grad_norm / clip_norm` → Mittelwert-Raum | `last_grad_norm = global_norm` → **Summen-Raum, keine Umrechnung** |

Das ist Zeile für Zeile der Zustand, der am 18.08. im Hauptzweig gefunden und
behoben wurde: die Schwelle steht um den Faktor `trainable_count` zu eng und
feuert in **100 % aller Schritte**. Der Audit-Eintrag von damals nennt als
Messwerte vor dem Fix „GradNorm-Minimum 254,8, Median 1303,7 bei Schwelle 1,0".
Der Fork liegt heute bei Medianen von **693 bis 1177** — dieselbe
Größenordnung, dasselbe Bild.

Praktisch heißt das: Adam bekommt in jedem Schritt ein auf Norm 1 gepresstes
Gradientensignal. Die Richtung stimmt, der Betrag ist weg.

**Verschärfend: der Effekt wächst über den Lauf.** Der GradNorm-Median steigt
ab Segment 4 monoton (693 → 782 → 908 → 889 → 909 → 940 → 1010 → 1114 → 1140 →
1177). Da die Schwelle konstant 1,0 bleibt, wird der Skalierungsfaktor immer
kleiner — die effektive Lernrate sinkt über den Lauf, ohne dass das jemand so
entschieden hätte. Der Lauf hat also nebenbei einen ungeplanten, mit der Zeit
schärfer werdenden LR-Zerfall.

Der Fork stammt laut eigener Notiz vom 2026-08-18, der Fix wurde am selben Tag
um 19:13 gebaut. Die Trennung liegt also genau auf der Kante; der Fork hat den
Fix nicht mitbekommen und ihn in den INT8-Pfaden auch später nicht nachgezogen
(`cortex.nim` dort zuletzt am 25.08. angefasst).

## Was daraus folgt — und was ausdrücklich NICHT

**Weiterhin gültig:** alle *fork-internen* Vergleiche. Die INT8-gegen-FP32-Arme
liefen beide durch dasselbe Clipping, der dokumentierte Abstand von +0,057
Held-out ist davon unberührt. Auch die Aussagen über Laufzeit
(INT8 46 % schneller auf Prinzessinnen-Geometrie), Speicher (−18 % durch
quantisierte Adam-Momente) und den 5,4×-Inferenzvorteil hängen nicht am
Optimierer.

**Nicht mehr haltbar:** jede absolute Qualitätsaussage aus dem Fork, und jeder
Vergleich seiner Zahlen mit denen des Hauptzweigs. Der Held-out von 4,598 nach
13 Segmenten steht neben den 3,458 der FP32-Chinchilla-Prinzessin — die beiden
messen aber weder dieselbe Datei noch denselben Optimierer:

- **Anderer Holdout.** Der Fork nutzt `factory/holdout_princess/ho_princess.txt`
  (325.533 B), der Hauptzweig `holdout/holdout.txt` (524.288 B). Verschiedene
  Texte, verschiedene Tokenzahl.
- **Anderer Optimierer-Zustand.** Gefixtes gegen ungefixtes Clipping.
- **Anderer Datenmix** und, siehe Befund 1, ein Korpuswechsel mitten im Lauf.

Vier unabhängige Unterschiede — ein Zahlenvergleich zwischen den beiden Läufen
trägt nichts.

## Korrektur: „der Hauptzweig hat den Fix" ist zu grob — er existiert genau einmal

Der Fix liegt
**nicht im Hauptzweig**, sondern in *einer Arbeitskopie* davon:

| Ort | `cortex.nim` vom | Clipping |
|---|---|---|
| `snai_edge` auf **<lan-ip>** (diese Maschine) | 18.08. 08:47 | `clip_norm = max(1.0'f32, trainable_count)` ✅ |
| `snai_edge` auf **<lan-ip>** (Laptop) | **15.08. 12:57** | `clipGlobalGradientNorm(state, 1.0)` ❌ |
| `snai_edge_int8_full` (Laptop) | 25.08. | `1.0` an beiden Stellen ❌ |
| `snai_edge_fp16` (<lan-ip>) | 18.08. 12:24 | `1.0`, kein `clip_norm` ❌ |

**Drei von vier Kopien tragen den Vor-Fix-Stand**, darunter der Hauptzweig auf
dem Laptop — der dort einen produktiven Scout-Daemon fährt. Meine Formulierung
„zwei Forks hängen hinterher" war damit falsch; das Problem ist nicht die
Forks, sondern dass es **keinen Mechanismus gibt, der irgendeine Kopie über
irgendeine Änderung informiert**: niemand sieht einer Datei an, dass ihr
etwas fehlt.

Ob der abgebrochene Lauf fortgesetzt oder neu gestartet wird, ist damit
ohnehin eine eigene Frage — eine Fortsetzung ab Segment 14 mit
gefixtem Clipping würde einen zweiten Bruch in dieselbe Kurve setzen, nach dem
Korpuswechsel bei Segment 6 den zweiten.
