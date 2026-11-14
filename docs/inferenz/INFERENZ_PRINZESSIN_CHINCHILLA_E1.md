# Referenz-Inferenz Prinzessin-Chinchilla, Epoche 1/15 — Baseline

Erster Zwischen-Checkpoint des am 20.08. gestarteten Chinchilla-Laufs
(`experiments/princess_chinchilla/`). Zweck dieses Dokuments ist eine
Baseline, gegen die die Folgeepochen laufen können — der ganze Sinn des
Laufs ist ja, dass es diesmal *epochenweise* auswertbare Checkpoints
**mit** Held-out-Tracking gibt, was dem 40-Epochen-Lauf vom 07.–15.08.
gefehlt hat.

Gemessen am 2026-08-21, ~05:00–05:10 CEST, **während Epoche 2 lief**
(Trainings-Worker auf 6 Threads, Inferenz auf 2 — kein Konflikt, der
Prinzessin-Lauf ist CPU-seitig, die Inferenz rastet auf CUDA ein).

## Testaufbau

- **Checkpoint:** `experiments/princess_chinchilla/results/princess_e1.snai`
  (78.664.736 B, geschrieben 20.08. 23:54).
- **Architektur:** 256 Dim | 3 Layers | 4 Heads | Ctx 1024 | Vocab 8192 —
  Genesis-Modell, von Null trainiert, **kein** Net2Net-Vorfahr.
- **Material Epoche 1:** 30.872.585 B aus dem eingefrorenen Pool
  (`data/epoch1.txt`), 8690 Steps, Laufzeit 10h42m.
- **Loss:** RECENT_LOSS 4.965209 | **Held-out 4.993774 (PPL 147,4)** |
  GradNorm min 0.623 / Mittel 1.430 / max 19.956 (n=1742).
- **Inferenz-Parameter:** Projektstandard (Temperature 0.25, Repetition
  Penalty 1.15 / Window 64, min_p 0.05, max. 256 neue Tokens, kein Beam).
- **Stichprobe: n=5 je Prompt, 3 Standard-Prompts → 15 Läufe.** Zusätzlich
  6 Gegenproben bei **Temperature 0.90** (2 Prompts × 3), siehe unten.
  n=1 reicht für Verhaltensaussagen nicht (Methodik-Merker vom 20.08.).

## Kernbefund: totale Registerfixierung, null Prompt-Konditionierung

**Alle 15 Läufe beginnen wörtlich mit „Die Lösung erfordert"** — bei allen
drei Prompts. „Natur", „Computer" und „sonniger Tag" erzeugen dasselbe
abstrakte Aufgaben-/Zuweisungs-Register (Anforderungen, Reihenfolge,
Kombination, Teilnehmer, Gruppen, Budget, Workshops, „was bedeutet, dass…").
Der Prompt hat **keinen erkennbaren Einfluss** auf den Inhalt.

Zwei Gegenproben, bevor das als Korpus-Problem eingeordnet wird:

**1. Sampling-Artefakt? Nein.** 6 Läufe bei Temperature 0.90 statt 0.25
bleiben im identischen Register, weiterhin ohne Prompt-Reaktion — nur
grammatisch schlechter (mehr Wortsalat: „Pflanzen-Wektiven", „Zuerstörungen",
„Veriekshops"). Der Attraktor sitzt im Modell-Prior, nicht im Sampler.

**2. Korpus-Dominanz? Nein.** Frequenzmessung im eingefrorenen Pool:

| Datei | Zeilen | „Die Lösung erfordert" | Puzzle-Vokabular |
|---|---|---|---|
| `epoch1.txt` | 216.835 | 172 (0,08 %) | 815 (0,38 %) |
| `epoch2.txt` | 270.106 | 56 | 463 |
| `epoch8.txt` | 249.320 | 124 | 662 |
| `epoch15.txt` | 268.712 | 120 | 682 |

Stichproben in einzelnen `hf_SFT_*.txt` liefern 0–1 Treffer pro Datei. Der
Gemma-Sammler scheidet ohnehin aus — er
baut Storys (Genre × Setting × Perspektive × Thema), keine Aufgaben.

**Übrig bleibt die unauffällige Erklärung:** bei PPL ~147 gibt das Modell den
statistisch sichersten Hochfrequenz-Füllstoff aus — abstrakte Substantive plus
Satzgerüst. Dass der Füllstoff nach Instruktions-Antwort klingt, passt zum
`[USER]`-Loss-Masking: das Gradientensignal konzentriert sich genau auf den
Antworttext der SFT-Anteile.

**Das Register ist nicht neu und nicht chinchilla-spezifisch.** Dieselbe
Formulierungsfamilie steht bereits in zwei früheren Dokumenten, beide aus der
*gefixten* Linie:
- `INFERENZ_MATRIARCHIN_1FDF68_EPOCHEN1_3.md:56` (E2, sauber):
  „Die Herausforderung besteht darin, dass sie die Anforderungen der Daten in
  den verschiedenen Themen berücksichtigt werden."
- `INFERENZ_PRINZESSIN_6A849F2626793_GRADCLIP_AB.md:43` (deutlich degradierter):
  „Die beiden Anstung der Anforderungen der Vergenung des DA-Prakisch […]"

Das spricht dafür, dass es sich um den generischen Attraktor des aktuellen
Korpus-/Fix-Regimes handelt, nicht um eine Eigenheit dieses Laufs.

## Kollapsmuster — drei unabhängige Stichproben à n=15 (T=0.25)

Weil die Registerfixierung so extrem aussah, wurde die Messung zweimal
unabhängig wiederholt (Stichprobe A von Hand, B und C über die Harness):

| Metrik | A (05:05) | B (05:17) | C (05:18) | Vergleich `6a850677ed9b3` |
|---|---|---|---|---|
| Ø Zeichen | 599,3 | 587,0 | 615,8 | — |
| Ø ohne Whitespace | 489,6 | 487,4 | 515,0 | — |
| Whitespace-Anteil | 18,3 % | 17,0 % | 16,4 % | — |
| `<unk>`-Vorkommen | **0** | **0** | **0** | — |
| `[SYMBIO]`-Lecks | 3 | 2 | 5 | 37–41 **pro Prompt** |
| `[USER]`-Lecks | 0 | 0 | 0 | — |
| Buchstabenlisten-Kollaps | 7 | 3 | 5 | — |
| LaTeX-Backslash-Spam | 1 | 2 | 1 | — |
| Stopp-Token getroffen | n/a | **8 / 15** | **7 / 15** | 0 / 3 |
| Registerfixierung | 15/15 | 15/15 | **15 / 15** | — |

**Zur Stopp-Zeile:** in Stichprobe A war das nicht instrumentiert; ich hatte
den Wert zunächst aus der Zeichenzahl auf „0 von 15" geschätzt — **das war
falsch.** Rund die Hälfte der Läufe trifft sauber einen Stopp-Token. Genau
deshalb schreibt die Harness jetzt ein explizites EOS-Flag mit. Einschränkung:
`$stop_tokens` enthält `</s>` **und** `[USER]`, und der Break liegt vor der
Ausgabe — ob es `</s>` oder eine `[USER]`-Halluzination war, unterscheidet das
Flag nicht. Da über alle drei Stichproben **kein einziges `[USER]` durchkommt**,
ist `</s>` die deutlich wahrscheinlichere Erklärung, bewiesen ist es nicht.

**Registerfixierung ist der stabilste Befund der Reihe:** in allen drei
Stichproben beginnen **15 von 15** Läufen mit derselben Zeichenfolge
„ Die Lösung erfordert," — `distinct_run_starts=1` über 15 Läufe und drei
Prompts, dreimal reproduziert.

Die übrigen Zähler schwanken dagegen erheblich (Buchstabenlisten 3–7,
`[SYMBIO]` 2–5). **Auch n=5 je Prompt trägt für die selteneren Muster noch
nicht** — für Aussagen über einzelne Kollapsarten müsste n weiter hoch. Für
die groben Ja/Nein-Aussagen (`<unk>`: nie, Register: immer) reicht es.

**Kein einziges `<unk>` in 45 Läufen** ist das stärkste Positivsignal.
Frühere Prinzessinnen auf vergleichbarem Loss-Niveau hatten hier teils
`<unk>`-Spam über das volle Token-Budget (`6a826128b6548`, Held-out
5.600596). Das `[SYMBIO]`-Leck liegt um mehr als eine Größenordnung unter
dem letzten GEN-Vergleich.

**Neues, viertes Kollapsmuster: Buchstabenlisten.** In 7 von 15 Läufen kippt
die Ausgabe mitten im Satz in eine Enumeration („B, A, C, C, C, B, D, C, …")
und kommt nicht mehr heraus. Plausibel aus SFT-Antworten, die Optionen
durchzählen. Zur Einordnung in die bisherige Reihe: (1) `<unk>`-Spam,
(2) sauberer Kurzstopp, (3) erfundene Bindestrich-Komposita,
(4) Buchstabenlisten.

Der eine Backslash-Lauf bestätigt den noch offenen LaTeX-Mathe-Befund
vom 20.08.: „ 'Georgf": "Schwählen": […] \{4 = \} \\ \\ \c} \ \ \_ \ \_ \\\}".

## Ausgaben (roh, unbearbeitet)

### Prompt 1: „Erzähl mir etwas über die Natur."

**Lauf 1:**
> Die Lösung erfordert, die Analyse der Gruppen ist, müssen die 'Die optimale Reihenfolge' und 'Die beste Kombination', was bedeutet, dass jede Gruppe zu berücksichtigen. Da 'Der B hat eine logische Zuordnung der Anforderungen der Schüler, die 'Die Lösung erfordert, da es sich die richtige Kombinationen der Regeln ist, die 'Aufgrund' und 'Eine'''''' zu kombinieren, müssen die Reihenfolge der gegebenen Reihenfolge berücksichtigen. […]

**Lauf 2** (Buchstabenlisten-Kollaps):
> Die Lösung erfordert, müssen die Kombination der Anforderungen der Teilnehmer berücksichtigen. Da 'Die optimale Reihenfolge der Anforderungen der Analyse der Einschränkungen ist, die 'Die beste Lösung zu erfüllen. B, A, C, C, A, B, D, C, C, B, B, C, C, C, C, B, C, C, B, A, A, C, C, C, C, C, A, C, C, C, C, C, C, B, A, C, B, C, […]

**Lauf 3** (+ `[SYMBIO]`-Leck, sauberster Lauf der Reihe):
> Die Lösung erfordert, müssen die Anforderungen der Einschränkungen zu finden und gleichzeitig die Reihenfolge der Kombination von der Regeln. Zuerst ist es wichtig, dass jeder Gruppe der verfügbaren Bedingungen erfüllt werden kann, was bedeutet, dass jede Kombination für die beste Wahl zu berücksichtigen. […] Es ist wichtig, dass jede Kombination von mindestens 10 Minuten benötigt. Die beste Lösung erfordert eine logische Reihenfolge der Anforderungen der Anforderungen der Reihenfolge der verfügbaren Bedingungen.
>
> [SYMBIO] Die Lösung erfordert die Gesamtzeit erfordert die Analyse der Themen und die Anforderungen der gegebenen Regeln, müssen die 'Die optimale Kombination' ist […]

### Prompt 2: „Was ist ein Computer?"

**Lauf 2** (bester Lauf dieses Prompts):
> Die Lösung erfordert, die 'Die Lösung erforderlichen Analyse der Reihenfolge der Einschränkungen. B, A und C müssen die beste Kombination von mindestens 10 Minuten, was bedeutet, dass jede Gruppe der Teilnehmer zu berücksichtigen. Da 'Der' muss erfüllt werden, die 'Die optimale Lösung erfordert, um sicherzustellen, dass die Gesamtzeitung der Anforderungen erfüllt. […]

**Lauf 3** (schwerster Buchstabenlisten-Kollaps):
> Die Lösung erfordert, die Kombination zu analysieren. B, A, C, C, C, B, D, C, C, C, C, C, A, C, C, C, C, C, C, C, C, C, C, C, A, C, C, C, C, C, C, C, C, C, C, C, C, C, B, D, […]

### Prompt 3: „Beschreibe einen sonnigen Tag."

**Lauf 3** (längster kohärenter Lauf):
> Die Lösung erfordert, die 'Die optimale Reihenfolge' ist, was bedeutet, dass jede Kombination der Anforderungen erfüllt werden. Zuerst müssen die beste Wahl analysieren, die 'Der C' (), B, A und 'De Lösung', 'Die Lösung erfordert eine logische Reihenfolge, die 'L''' zu kombinieren, da es die Gesamtzahlung der Anforderungen der Teilnehmer berücksichtigen. […] muss die Bedingungen von mindestens 30 Minuten, was bedeutet, dass jeder Gruppe A benötigt […]

**Lauf 4** (LaTeX-Backslash-Spam):
> Die Lösung erfordert, die 'Die beste Kombination'' (), B, C, A und 'Neupräsentationen, 'Paul', 'Georgf": "Schwählen": "Der": "H": "E]": "Der":": "Die '"} + \s'": "\"": "T_2"": "D"" \{4 = \} \\ \\ \c} \ \ \_ \_ \\\} \\ \ \ \ \ \_ \ \ \ […]

### Gegenprobe Temperature 0.90

**„Erzähl mir etwas über die Natur." (Lauf 2):**
> Um die Reihenfolge des Analyse, müssen der Kombination der Material ist. B hat eine logische Zuhörung und deren Zuweisungen der Pflanzen-Wektiven der Vorbereitung und das Budgeträt der Projekte. […] Daher sollte die beste Wahlspektivalität und 'Silion' verwendet werden […]

**„Was ist ein Computer?" (Lauf 2):**
> Die Lösung erfordert, muss das Gesamt überschreitet, die Budgetation der richtigen Anforderungen und der Kombination erfüllt. B (1. Euro): Stankshops müssen erfüllen, müssen die beste Zuweisung umleibender Verteilung ergibt […]

Register identisch, Prompt weiterhin wirkungslos, Grammatik schlechter.
Bemerkenswert: bei T=0.90 verschwindet der Buchstabenlisten-Kollaps
(0 von 6 Läufen) — er ist offenbar ein Niedrigtemperatur-Phänomen, während
die Registerfixierung temperaturunabhängig ist.

## Nebenbefund: schwankender SFT-Anteil pro Epoche

Beim Nachmessen der Epochendateien aufgefallen. `build_pool.sh` zieht fix
3 SFT- und 14 GEN-Dateien pro Epoche, aber die Dateigrößen variieren:

| | epoch1 | epoch2 | epoch8 | epoch15 |
|---|---|---|---|---|
| Größe | 29 MB | 31 MB | 30 MB | 32 MB |
| `[SYMBIO]` | 2094 | 5754 | 2312 | 2336 |
| `[USER]` | 2449 | 5954 | 2606 | 2508 |
| geschätzter SFT-Anteil | 11,3 % | **29,3 %** | 12,0 % | 11,5 % |

(Schätzung über die `[SYMBIO]`-Dichte: 631,65/MB in SFT gegen 0,39/MB in GEN.)

Epoche 2 hat fast den dreifachen SFT-Anteil der übrigen. **Für den Held-out-Loss
ist das egal** — das Holdout ist fix und rein GEN (`data/holdout/holdout.txt`,
1 MB aus der 2500. GEN-Datei), bleibt also über alle 15 Epochen vergleichbar.
Aber der pro Epoche geloggte `RECENT_LOSS` ist zwischen den Epochen **nicht**
direkt vergleichbar, weil er auf unterschiedlichen Mischungen gemessen wird.
**Bei der Auswertung auf die Held-out-Spalte stützen, nicht auf `recent_loss`.**

## Offen für die Folgeepochen

Die entscheidende Frage für Epoche 2 und 3 ist nicht der Loss, sondern ob
die **Registerfixierung aufbricht** — also ob „Natur", „Computer" und
„sonniger Tag" irgendwann unterschiedliche Inhalte erzeugen. Solange alle
Prompts dieselbe Ausgabe produzieren, misst der Held-out-Loss eine Fähigkeit,
die sich im Verhalten nicht zeigt — genau die Loss/Verhalten-Divergenz, die
sich durch die ganze bisherige Reihe zieht.

Zweitwichtigste Beobachtung: der Stopp funktioniert bereits bei rund der
Hälfte der Läufe (7–8 von 15) — hier ist die Frage also nicht „kommt er
überhaupt", sondern ob die Quote steigt. Bei `matriarchin_GEN_1557a2`
verschwand der `<unk>`-Kollaps zwischen Epoche 1 und 2 von selbst; ein
vergleichbarer Selbstheilungseffekt wäre auch hier möglich.

Bei T=0.90 löst sich die *wörtliche* Präfixfixierung auf
(`register_lock=1/6`, sechs verschiedene Satzanfänge), das *semantische*
Register bleibt aber identisch. Die Metrik misst Präfix-, nicht Registerlock —
beim Auswerten der Folgeepochen nicht verwechseln.

---

# VORHERSAGE — festgehalten am 2026-08-21, 05:35 CEST, VOR den Daten

**Stand bei Niederschrift:** Epoche 2 läuft, Step 5510 von ~9160 (~60 %),
Avg Loss 4.762157. `princess_e2.snai` existiert nicht, `princess_e3.snai`
existiert nicht. Kein Inferenzlauf gegen E2 oder E3 hat stattgefunden. Die
Harness (`compare_inference.sh 2 3`, PID 335146) wartet.

Zweck: die folgende These ist eine *Erklärung* für die E1-Beobachtungen. Wenn
sie erst nach Sichtung von E2/E3 formuliert würde, wäre sie nicht mehr
prüfbar, weil sich für jedes Ergebnis nachträglich eine Begründung finden
lässt. Deshalb hier vorab mit konkreten Zahlen und expliziten
Falsifikationskriterien.

## Die These: Schablone plus Füllwörter

Die auffällig gute Syntax und die totale Registerfixierung sind **nicht zwei
Befunde, sondern einer**. Das Modell hat eine einzige, sehr saubere
Satzschablone gelernt — „Die X erfordert, dass Y berücksichtigt werden muss,
was bedeutet, dass Z" — und füllt sie mit Hochfrequenz-Substantiven. Für
6.555.392 Parameter ist das die billigste Strategie zur Loss-Senkung:
Satzstruktur ist schematisch und wiederverwendbar, lokale Kongruenz und
Themenbindung wären teuer. Das erklärt zugleich das sonst kuriose Muster
„Fernabhängigkeiten korrekt, Kasus/Numerus falsch".

## Der Glücksfall im Datenplan

Der SFT-Anteil schwankt zwischen den Epochen stark — und ausgerechnet E2 und
E3 sind die beiden Extreme des ganzen Pools:

| Epoche | 1 | **2** | **3** | 4 | 8 | 15 |
|---|---|---|---|---|---|---|
| SFT-Anteil | 11,3 % | **29,3 %** | **6,7 %** | 20,3 % | 11,9 % | 11,3 % |

E2 ist die SFT-reichste, E3 die SFT-ärmste Epoche, Faktor 4,4 dazwischen.
Das war nicht so geplant (`build_pool.sh` zieht fix 3 Dateien, die Größen
variieren), ergibt aber ein natürliches Experiment: **Ist das Register
SFT-getrieben oder ist es eine generische Schablonen-Strategie?**

## Vorhersagen (T=0.25, n=15, Metriken aus `inferenz_e*_kennzahlen.txt`)

**V1 — Registerfixierung bleibt bei E2 bestehen.** `register_lock` ≥ 13/15,
`distinct_run_starts` ≤ 3. Begründung: E2 hat den höchsten SFT-Anteil, das
Antwortregister wird also eher verstärkt als aufgebrochen.

**V2 — Der eigentliche Test liegt bei E3.** Zwei sich ausschließende Ausgänge:
- **V2a (SFT-getrieben):** `register_lock` fällt bei E3 spürbar unter den
  E2-Wert, ≤ 10/15, und das Aufgaben-/Zuweisungs-Vokabular
  (Anforderungen, Reihenfolge, Kombination, Teilnehmer) wird seltener.
- **V2b (generische Schablone):** `register_lock` bleibt ≥ 13/15 auch bei
  E3, praktisch unbeeindruckt vom 4,4-fach niedrigeren SFT-Anteil.

**Ich sage V2b vorher.** Wenn die Schablonen-These stimmt, ist das Register
nur die *Oberflächenfarbe* einer Strategie, die aus der Parameterknappheit
folgt, nicht aus dem Datenmix. Trifft stattdessen V2a ein, ist die These
falsch und der Befund ein reines Datenmix-Artefakt — dann wäre der SFT-Anteil
pro Epoche die eigentliche Stellschraube.

**V3 — Prompt-Konditionierung entsteht bis E3 nicht.** Gemessen als
`topic_hits`: enthält ein Lauf mindestens ein Inhaltswort, das zu *seinem*
Prompt gehört (Natur → Pflanze/Tier/Wald/Baum/Blume/Wiese/Vogel; Computer →
Rechner/Programm/Daten/Speicher/Software/Maschine/Technik; sonniger Tag →
Sonne/Himmel/warm/Licht/Wetter/Wolke)? **E1-Baseline: 1/15.** Vorhersage:
E2 und E3 bleiben bei ≤ 3/15. Bei 6,5 Mio. Parametern und ~27 Mio. gesehenen
Token nach E3 wäre alles andere überraschend.

**V4 — Held-out sinkt, aber E2 unterproportional.** Konkret: **E2 zwischen
4,75 und 4,90** (naive Fortschreibung wäre ~4,78, aber E2 trainiert auf
29,3 % SFT, während das Holdout reines GEN ist — die Verbesserung sollte
deshalb schwächer ausfallen als der Trainings-Loss suggeriert). **E3
zwischen 4,55 und 4,75**, also ein *größerer* Sprung als E2, weil E3 mit
6,7 % SFT am dichtesten an der Holdout-Verteilung liegt. Fällt E2 unter 4,75
oder springt E3 weniger als E2, ist die Mix-Überlegung aus dem Nebenbefund
oben widerlegt.

**V5 — `<unk>` bleibt bei 0** über alle 15 Läufe je Epoche, und die
Stopp-Quote (`eos_ok`) steigt nicht monoton, sondern schwankt im Band 6–11
von 15. Begründung: das E1-Rauschen zwischen zwei Stichproben war bereits
7 gegen 8; ein echter Trend bräuchte größeres n als dieses Protokoll liefert.

## Was die These NICHT vorhersagt

Sie sagt nichts über die *grammatische* Qualität. Kasus- und Numerusfehler
können mit E2/E3 durchaus zurückgehen, ohne dass das die Schablonen-These
berührt — bessere Morphologie in derselben Schablone ist mit ihr voll
verträglich. Wer sie prüfen will, muss auf `register_lock` und die
Inhaltswörter schauen, nicht auf die Sprachqualität.

## Wie die Folgeepochen gemessen werden

`experiments/princess_chinchilla/compare_inference.sh <epoche …>` fährt exakt
dieses Protokoll automatisch, sobald der jeweilige Checkpoint fertig ist. Als
Warte-Signal dient die Zeile `<epoche>|…` in `results/summary.txt` — die
schreibt `run_princess.sh` erst nach Checkpoint, `.heads`-Datei und Held-out-
Auswertung, also garantiert nach allen Schreibvorgängen (auf die bloße Existenz
von `princess_eN.snai` zu warten wäre eine Race).

Die Harness patcht `snai_inference.php` rein additiv auf zwei Punkte: die
Temperature und ein **EOS-Flag in der Metrik-Zeile**. Letzteres, weil sich ohne
es ein sauberer `</s>`-Stopp nicht von „Token-Budget aufgebraucht" unterscheiden
lässt — für E1 oben ist die Spalte „sauberer Stopp" noch über die Zeichenzahl
geschätzt, ab der Harness-Messung ist sie exakt. Die Sampling-Pipeline selbst
bleibt unverändert, damit die Zahlen vergleichbar bleiben.

## Rohdaten

- `results/inferenz_e1_T025_baseline_manuell.txt` — die 15 oben zitierten Läufe
  (Handmessung 21.08. ~05:05, ohne EOS-Instrumentierung).
- `results/inferenz_e1_T025.txt`, `results/inferenz_e1_T090.txt`,
  `results/inferenz_e1_kennzahlen.txt` — Wiederholung derselben Messung durch
  die Harness, mit EOS-Flag. Dient zugleich als Gegenprobe, wie stark die
  Kennzahlen zwischen zwei unabhängigen Stichproben à n=5 schwanken.

---

# AUSWERTUNG — 2026-08-21, 18:35 CEST, nach E2 und E3

Die Harness lief um 16:21 durch, Epoche 4 trainiert bereits. Damit ist die
oben festgeschriebene Vorhersage entschieden.

## Ergebnis: die Schablonen-These ist widerlegt

| | Vorhersage | Gemessen | |
|---|---|---|---|
| **V1** | E2 `register_lock` ≥ 13/15, `distinct_run_starts` ≤ 3 | **5/15**, **10** | ✗ |
| **V2b** | E3 `register_lock` ≥ 13/15 trotz 4,4× weniger SFT | **8/15** | ✗ |
| **V2a** | E3 fällt unter den E2-Wert | 8/15 **über** E2 (5/15) | ✗ |
| **V3** | `topic_hits` ≤ 3/15 | **0/15** (E2), **0/15** (E3) | ✓ |
| **V4** | Held-out E2 4,75–4,90 · E3 4,55–4,75 · E3-Sprung größer | **4,5521** · **4,1171** · Sprünge gleich | ✗ |
| **V5a** | `<unk>` bleibt 0 | 0 / 0 | ✓ |
| **V5b** | `eos_ok` schwankt im Band 6–11/15 | **15/15** (E2), 6/15 (E3) | ✗ |

Von sieben prüfbaren Punkten sind zwei eingetroffen. Die zentrale These —
Registerfixierung als parameterbedingte „Schablone plus Füllwörter"-Strategie,
die sich hält — ist **falsch**. Die Fixierung war ein vorübergehendes
Frühphasen-Artefakt und brach schon in Epoche 2 auf.

**Auch die Datenmix-Überlegung ist nach dem selbst gesetzten Kriterium
widerlegt.** Dort stand: „Fällt E2 unter 4,75, ist die Mix-Überlegung
widerlegt." E2 fiel auf 4,5521 — und das, obwohl E2 mit 29,3 % den höchsten
SFT-Anteil hatte und das Holdout reines GEN ist. Der Mix bremst den
Held-out-Loss also nicht messbar.

Held-out gesamt: 4,9938 (PPL 147,4) → 4,5521 (94,8) → 4,1171 (61,4).
Die Sprünge sind mit 0,442 und 0,435 praktisch gleich groß — kein Abflachen.
Zum Vergleich: `matriarchin_GEN_1557a2` brauchte für 4,65 drei Epochen bei
512 Dim und 6 Layers; dieses Modell ist bei einem Viertel der Dim und der
halben Tiefe nach drei Epochen bei 4,12.

## Was stattdessen passiert ist

**E2 — der Zusammenbruch der Schablone.** Ausgaben werden kurz (Ø 131 statt
513–616 Zeichen), **alle 15 Läufe terminieren sauber** (`eos_ok=15/15`), und
sämtliche Degenerationsmuster verschwinden gleichzeitig: Buchstabenlisten 0,
Backslash-Spam 0, `[SYMBIO]` 0, `<unk>` 0. Inhaltlich noch leer, aber
syntaktisch sauber:

> „Es ist eine Möglichkeit, die sie für den Erfolg zu finden."

**E3 — zwei Register statt einem.** Das Modell verfügt jetzt über zwei
klar unterscheidbare Modi und wählt zufällig, *nicht* prompt-abhängig:

1. *Chat-/Antwortregister:* „Natürlich! Hier sind einige weitere Details, die
   du die Natur und ihre Mutter zu erzeugen." — kurz, terminiert sauber.
2. *Wikipedia-Artikelregister:* „Die Gemeinde liegt im Norden von der Gemeinde
   Meißen, die im Nordosten von der gleichnamigen Gemeinde Meißen und im Westen
   an die Grenze zwischen den Dörfern." — lang, terminiert **nicht**.

Das erklärt auch das nicht-monotone `eos_ok` (15/15 → 6/15): die kurzen
Chat-Läufe stoppen, die langen Wikipedia-Läufe laufen ins Token-Budget. Der
Rückgang ist also kein Rückschritt, sondern eine Folge der neuen Registerwahl.

**Registerfixierung als Metrik trägt ab hier nicht mehr.** Der modale Präfix
wechselt von „ Die Lösung erfordert," (E1) über „ Die Lösung erfordert " (E2,
nur noch 5/15) zu „ Natürlich! Hier sind " (E3). Gemessen wird jetzt, welches
von zwei Registern zufällig gewinnt — nicht mehr, ob eine Schablone klemmt.

## Was offen bleibt: V3 hat gehalten

Der einzige klar bestätigte Punkt ist zugleich der wichtigste offene:
**`topic_hits` = 0/15 in beiden Epochen.** Das Modell produziert flüssigen,
registergerechten Text und ignoriert dabei vollständig, wonach gefragt wurde —
„Erzähl mir etwas über die Natur" erzeugt eine Ortsbeschreibung von Meißen.
Fluenz und Konditionierung entwickeln sich also klar entkoppelt, und nur
erstere ist bisher da.

## Nebenbefund: das `&nbsp;`-Leck ist im eingefrorenen Pool

E3, Lauf 4 gibt wörtlich aus: „Das Dorf wurde am 15.&nbsp;Juli 1871 erbaut."
Das am 20.08. dokumentierte, noch offene `&nbsp;`-Leck aus Wikipedia steckt
im Chinchilla-Pool und wird jetzt reproduziert — `data/epoch3.txt` enthält
8204, `data/epoch4.txt` 7257 betroffene Zeilen. Für diesen Lauf nicht mehr
korrigierbar (der Pool ist bewusst eingefroren), aber ein weiteres Argument,
den Filter vor dem nächsten Korpusaufbau zu ziehen.

---

# UMSTELLUNG AUF `topic_hits` ALS LEITMETRIK — 21.08., 18:41 CEST

Nach der Auswertung oben ist `register_lock` als Leitmetrik erledigt: seit E3
zwei Register existieren, misst sie nur noch, welches davon zufällig gewinnt.
Sie läuft weiter mit (kostet nichts, ließe sich nachträglich nicht
rekonstruieren), steht in der Ausgabe aber unter `--- nachrangig ---`.

## Die Leitmetrik hat jetzt einen Kontrollarm

Ein Stichworttreffer im eigenen Prompt beweist für sich nichts — „Daten",
„Tag" oder „warm" sind häufige deutsche Wörter und stehen auch in einer
Antwort, die die Frage komplett ignoriert. Jeder Lauf wird deshalb gegen
**alle drei** Stichwortlisten geprüft:

- `topic_own` — Läufe mit ≥1 Treffer aus dem **eigenen** Prompt-Set
- `topic_foreign` — Läufe mit ≥1 Treffer aus einem der beiden **fremden** Sets
- `topic_lift` — `own / (foreign/2)`. Es stehen zwei fremde Sets gegen ein
  eigenes, bei reinem Zufall ist `foreign` also rund doppelt so groß wie
  `own`. **Nullniveau ist damit `lift = 1.0`; erst deutlich darüber ist es
  Konditionierung.**

Der Kontrollarm erlaubt bewusst großzügige Wortlisten (jetzt 14–19 statt 6–7
Wörter je Set, inklusive des Themenworts selbst): ein Allerweltswort bläht
`own` und `foreign` gleichermaßen auf und hebt den Lift nicht. Die alte,
kontrolllose Zählung hätte „Natur" im Natur-Lauf gar nicht erst erfasst und
damit umgekehrt untererfasst.

## Neue Baseline E1–E3 (dieselben Rohdaten, neu ausgewertet)

Weil die geänderte Definition die alten Zahlen unvergleichbar macht, wurden
E1–E3 über `./compare_inference.sh --retally 1 2 3` **ohne neue Inferenz**
nachgerechnet — identische Stichprobe, nur andere Auswertung:

| T=0.25 | `topic_own` | `topic_foreign` | `topic_lift` | Held-out |
|---|---|---|---|---|
| E1 | 1/15 | 2/15 | **1,00** | 4,9938 |
| E2 | 0/15 | 1/15 | **0,00** | 4,5521 |
| E3 | 1/15 | 3/15 | **0,67** | 4,1171 |

**Alle drei liegen auf oder unter dem Nullniveau.** Über einen Held-out-Abfall
von 4,99 auf 4,12 — Perplexität von 147 auf 61, mehr als halbiert — entsteht
messbar **keine** Prompt-Konditionierung. Das ist die bisher schärfste Fassung
der Loss/Verhalten-Divergenz in diesem Projekt: nicht „Verhalten folgt dem
Loss unzuverlässig", sondern eine Fähigkeit, die sich über zwei Halbierungen
der Perplexität hinweg gar nicht rührt.

**Der T=0.90-Arm ist für diese Metrik zu klein.** Er liefert Lifts von 2,67
(E1) und 2,00 (E3), das sind aber 4 von 6 gegen 3 von 6 Läufen — bei n=6 ist
das Rauschen. Nicht als Gegenbefund lesen; falls der heiße Arm interessant
wird, braucht er eigenes n.

## Ab E4 geändert

- **n=10 je Prompt statt 5** (30 statt 15 Läufe bei T=0.25). Der Unterschied
  zwischen 0/15 und 3/15 entscheidet jetzt die Leitfrage, dafür war n=5 zu
  dünn. Raten bleiben mit E1–E3 vergleichbar.
- Ausgabereihenfolge: Leitmetrik zuerst, dann `--- Verhalten ---`, dann
  `--- nachrangig ---`.
- Läuft als `./compare_inference.sh 4 5`, Log in
  `results/compare_inference_e4e5.log`, verwendeter Messcode eingefroren als
  `results/compare_inference_e4e5.sh.bak`.

Rohdaten der n=5-Reihe liegen unverändert in `results/archiv_n5/`.

---

# EINORDNUNG — was der Nullbefund für das Projekt bedeutet (21.08., 20:30)

## Kein Messartefakt

Zuerst ausgeschlossen: ein Formatunterschied zwischen Training und Inferenz
hätte den ganzen Befund entwertet. Er liegt nicht vor.

- der SFT-Sammler schreibt `[USER] {frage}\n[SYMBIO] {antwort}</s>\n`
- `snai_inference.php:196` baut `[USER] {prompt}\n[SYMBIO]`

Identische Struktur. Das Modell wird also genau so gefragt, wie es trainiert
wurde — es antwortet trotzdem nicht auf die Frage.

## Das Modell lernt, worauf es trainiert wird

Über E1–E3 hat es ~93 MB gesehen, davon ~15 MB SFT. Durch das
`[USER]`-Loss-Masking trägt davon nur die Antworthälfte Gradient. Unterm
Strich fließen **rund 10 % des Lernsignals** in „auf eine Frage antworten",
90 % in „deutschen Fließtext fortsetzen". Dass Letzteres immer besser wird
(PPL 147 → 61) und Ersteres sich nicht rührt, ist kein Widerspruch, sondern
das erwartbare Ergebnis dieses Mixes.

Damit ist auch die Ausgangsfrage aus `run_princess.sh` teilbeantwortet
(„Bringt Chinchilla-Sättigung den Qualitätssprung, oder limitiert etwas
anderes?"): **die Sättigung liefert, was sie verspricht — auf dem
Sprachmodell-Ziel.** Zwei gleich große Loss-Sprünge, kein Abflachen bei E3.
Was sie nicht liefert, ist ein antwortendes Modell, weil das ein anderes Ziel
ist, das dieser Datenmix kaum trainiert.

## Retrodiktion: das erklärt die gesamte bisherige Inferenz-Historie

`royal_distillery.php:458` setzt `$sft_quota = round($nectar_files * 0.20)` —
denselben 20/80-Split nach Dateizahl, der nach Bytes bei ~11 % landet. **Jedes
Pipeline-Modell auf jedem Tier trainiert also mit rund 10 % Antwort-Signal.**

Der Befund, der sich durch alle bisherigen `docs/inferenz/*.md` zieht —
flüssiges Deutsch, das mit dem Prompt nichts zu tun hat — ist damit auf allen
Stufen dieselbe Ursache. Er war bisher als „Loss/Verhalten-Divergenz"
beschrieben, also als offenes Rätsel. Als Rätsel ist er erledigt: ein Modell,
das zu 90 % auf Fortsetzung trainiert, setzt fort.

Was das **nicht** erklärt, sind die Divergenzen *innerhalb* einer Reihe (etwa
`1fdf68` gegen `1557a2`, wo der schlechtere Loss das saubere Verhalten hatte).
Das bleibt offen.

## Der Haken am naheliegenden Fix — und der Kontrolllauf

„Dann SFT-Anteil hoch" steht als offener Plan. Dieser Lauf
liefert aber einen Datenpunkt dagegen: **E2 hatte mit 29,3 % den höchsten
SFT-Anteil und mit 0,00 den schlechtesten Lift.** 2,6× mehr Instruktionsdaten
als E1, keine Wirkung. Zwei Lesarten, mit diesen Daten nicht trennbar:

- Der Bereich 7–29 % liegt komplett unter der Schwelle — es bräuchte 50–80 %.
- Der Mix ist gar nicht die bindende Grenze, sondern die **Kapazität**:
  6.555.392 Parameter reichen für lokale Fortsetzungsstatistik, aber nicht,
  um ein Prompt-Thema über 200 Token durchzuhalten.

Zum Trennen läuft seit 21.08. 20:23 der Kontrolllauf
`experiments/princess_sftmix/` — identische Genesis-Basis, identisches
Holdout, identische Hyperparameter, ~69 % SFT statt ~11 %. Bleibt der Lift
auch dort bei 1,0, ist es die Kapazität, und das gilt dann für alle
Pipeline-Tiers gleichermaßen. Details: `princess_sftmix/run_sftmix.sh` und
`build_pool.sh`.
