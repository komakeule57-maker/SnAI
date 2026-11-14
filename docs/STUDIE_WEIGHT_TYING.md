# Machbarkeitsstudie: Weight Tying (Embedding ↔ `w_out`)

**Stand 2026-08-27. Reine Code-Analyse.**
Auftrag: klären, ob das Binden der Ausgabematrix an die Embedding-Matrix in
dieser Engine machbar ist, was es kostet und wo die Risiken liegen.

## Warum überhaupt: das Parameterbudget

Errechnet aus der Größenformel in `royal_distillery.php::validate_tensor_file`,
Vokabular 8192:

| Stufe | Embed | `w_out` | **Rumpf** | Tabellen-Anteil |
|---|---|---|---|---|
| Drohne (128/512/2) | 1.048.576 | 1.048.576 | **393.728** | **84,2 %** |
| Prinzessin (256/1024/3) | 2.097.152 | 2.097.152 | **2.360.832** | **64,0 %** |
| Matriarchin (512/2048/6) | 4.194.304 | 4.194.304 | 18.880.512 | 30,8 % |
| Empress (1024/4096/12) | 8.388.608 | 8.388.608 | 151.019.520 | 10,0 % |

**Die Drohne ist keine kleine Transformer-Architektur, sondern eine große
Nachschlagetabelle mit einem winzigen Transformer daran.** 393.728 Parameter
rechnen tatsächlich. Bei der Prinzessin sind es 36 %.

Bei gleicher Gesamtgröße, nur durch Teilen der beiden Tabellen:

| | heute | gebunden | Rumpf |
|---|---|---|---|
| Drohne | 2 Layer | **7 Layer** | 393.728 → **1.378.048** (3,5×) |
| Prinzessin | 3 Layer | **6 Layer** | 2.360.832 → **4.721.664** (2,0×) |
| Matriarchin | 6 Layer | 7 Layer | +17 % |

Der Hebel sitzt unten und verschwindet nach oben. Zusätzlich schrumpft der
Adam-Zustand um eine ganze Matrix: der Prinzessinnen-Checkpoint fiele von
78,6 MB auf ~53,5 MB.

## Befund 1: die Formen passen exakt

```
cortex.nim:1018   initM(s.embeddings, vocab, dim)
cortex.nim:1023   initM(s.w_out,      vocab, dim)
```

Identische Orientierung. Der Embedding-Lookup liest Zeile `token`, die
Ausgabeprojektion rechnet `logits[v] = dot(w_out.row[v], norm_buf)` — beide
zeilenweise über das Vokabular. **Tying wäre eine direkte Substitution, keine
Transposition.** Das war die Hauptsorge und sie ist ausgeräumt.

## Befund 2: die GPU-Backends sind gar nicht betroffen

| Datei | Treffer auf `w_out` |
|---|---|
| `cortex.nim` | 57 |
| `cortex_opencl.nim` | **0** |
| `cortex_cuda.cu` | **0** |
| `cortex_cuda_bridge.nim` | **0** |

Die Backends rechnen über generische Matrix-Ops (`matMulBatchedD`,
`matMulTBatchedD`, `gradAccumBatchedD`), die `w_out` nur als Argument
bekommen. **Kein einziger GPU-Kernel muss angefasst werden** — der Punkt, an
dem der INT8-Fork die meiste Zeit verloren hat, entfällt hier.

## Befund 3: kein Sparse-Update geht verloren

`upEmb` (cortex.nim:871) und `upM` sind beide dichte Schleifen über
`w.rows * w.cols`. Ich hatte vermutet, der Embedding-Pfad nutze ein
Scatter-Update über nur berührte Zeilen, das beim Binden verloren ginge — tut
er nicht. Gebunden gäbe es **ein** dichtes Update statt zwei, der Adam-Schritt
wird also schneller, nicht langsamer.

## Risiko 1 (das größte): die Gradienten liegen in verschiedenen Speichern

```
cortex.nim:1402  # g_embeddings und die rms-Gradienten sind bewusst NICHT dabei -
                 # die entstehen weiterhin auf der CPU (Scatter bzw. Norm-Backward).
cortex.nim:1408  zeroMat(state.g_embeddings)     <- nur CPU
cortex.nim:1409  zeroGrad(state.g_w_out)          <- CPU UND GPU-Puffer
```

`g_embeddings` entsteht auf der **CPU** (Scatter über die Eingabetoken),
`g_w_out` auf der **GPU** (`gradAccumBatchedD`, cortex.nim:1672). Gebunden
müssten beide Beiträge in **dieselbe** Matrix summiert werden.

Das ist keine Kleinigkeit. Genau diese Fehlerklasse — „der Gradient lag im
falschen Speicher" — hat den INT8-Fork mehrfach aufgehalten, und sie ist
**still**: das Training läuft weiter, die Zahlen sehen plausibel aus, nur ein
Teil des Gradienten fehlt.

Sauberer Weg: die GPU-Akkumulation nach dem Batch zurückholen (`pullGrad`
existiert bereits, cortex.nim:1905) und den CPU-Scatter danach daraufaddieren.
Kostet eine zusätzliche Host↔Device-Übertragung je Schritt über eine
`vocab × dim`-Matrix. Bei der Prinzessin sind das 8 MB je Schritt — messbar,
aber verkraftbar; auf Drohnen-/Prinzessinnenebene läuft ohnehin alles auf der
CPU (GPU erst ab dim ≥ 512).

## Risiko 2: das Gradient-Clipping ändert seine Bedeutung

`clipGlobalGradientNorm` summiert heute zwei getrennte Beiträge:

```
cortex.nim:921   total_sq += calcL2Sq(s.g_embeddings)
cortex.nim:922   total_sq += calcL2Sq(s.g_w_out)
```

Gebunden ist es **ein** Term über eine Matrix, die die Summe beider Gradienten
trägt. Die globale Norm ändert sich dadurch, und mit ihr das Verhalten der
Clip-Schwelle. Angesichts der Projekthistorie — der Clipping-Bug vom 18.08. hat
wochenlang jeden Schritt verfälscht — sollte nach dem Umbau die GradNorm-
Verteilung explizit nachgemessen werden, nicht angenommen.

## Risiko 3: `w_out` hat heute eine eigene Initialisierung

```
init_base.php:100  write_tensor_with_adam($fp, $vocab*$dim, false, 0.0001); // w_out (Softmax-Schutz)
```

Die Ausgabematrix startet bewusst sehr klein (0,0001), damit die Logits am
Anfang flach sind. Das Embedding startet anders. Gebunden gibt es nur **eine**
Initialisierung, und die Entscheidung, welche, ist keine Formalie: startet die
gemeinsame Matrix auf Embedding-Niveau, sind die Anfangslogits deutlich größer
als bisher. Standardlösung wäre, die Logits beim Auslesen zu skalieren statt
die Matrix klein zu halten — das ist aber eine zusätzliche Änderung im Forward.

## Änderungsfläche

**`cortex.nim` — rund 20 echte Stellen** (57 Treffer, viele davon Kommentare):

| Bereich | Zeilen | Aufwand |
|---|---|---|
| Felddeklaration | 62 | trivial |
| Allokation | 1023–1026 | trivial (auf Embedding zeigen) |
| Forward | 1381, 1508, 1615, 1637 | trivial (Substitution) |
| Backward | 1671, 1672, 1699, 1792 | **hier sitzt Risiko 1** |
| Adam-Update | 852, 877 | eine Zeile entfällt |
| Clipping | 921–922, 939–940, 953–954 | **Risiko 2** |
| FFI-Zeiger (Typ 1) | 1190 | trivial |
| Freigabe | 2020–2023 | Doppel-Free vermeiden |
| NaN-Scan / `blast` | 2174, 2292, 2325, 2343 | trivial |

**PHP — 6 Stellen:**

| Datei | Zeile | Was |
|---|---|---|
| `tensor_io.php` | 142, 226 | Lesen/Schreiben des Tensors |
| `royal_distillery.php` | 97 | Größenprüfung (`* 3` für w+m+v) |
| `royal_distillery.php` | 913 | Net2Net-Upscale — entfällt, wird **einfacher** |
| `init_base.php` | 100 | Genesis-Init, siehe Risiko 3 |
| `tensor_ffi_adapter.php` | 27 | Tensor-Typ-Tabelle |

**Dateiformat:** `w_out` verschwindet aus der Datei. Braucht ein Flag oder eine
neue Magic, damit Altmodelle weiter laden. `SNAI_V6` ist als Bezeichner
**verbrannt** — der FP16- und der INT8-Fork haben ihn unabhängig voneinander
mit gegenläufiger Bedeutung belegt. Hier wäre
`SNAI_V7` oder ein Bit im bestehenden Header zu nehmen.

## Einschätzung

**Machbar, mittlerer Aufwand, ein ernstzunehmendes Risiko.** Die Struktur ist
günstiger als erwartet: gleiche Matrixform, keine GPU-Kernel, kein verlorenes
Sparse-Update. Der Aufwand steckt fast vollständig in der Gradienten-
Zusammenführung (Risiko 1) und in zwei Folgeentscheidungen (Clipping-Verhalten,
Initialisierung).

Was dagegen spricht: es ist eine Engine-Änderung mit Formatwechsel, also genau
die Kategorie, bei der dieses Projekt in der Vergangenheit die teuersten Fehler
gemacht hat — und die stillen Varianten davon (falscher Speicher, verändertes
Clipping) sind erst nach Tagen sichtbar.

Was dafür spricht: Faktor 3,5 auf den rechnenden Teil der Drohne, Faktor 2 bei
der Prinzessin, ohne einen Euro Hardware, plus ein Drittel kleinere Checkpoints.

## Vorschlag für den Nachweis

Billig und aussagekräftig, bevor irgendetwas Produktives umgestellt wird:

1. **Drohnen-A/B**, gleiche Gesamtgröße, gleicher eingefrorener Korpus, gleiche
   Prompts: heute (128/512/**2** Layer, ungebunden) gegen gebunden
   (128/512/**7** Layer). Auf Drohnenebene sind das Stunden, nicht Tage.
2. **GradNorm-Verteilung** beider Arme protokollieren — Risiko 2 wird damit
   gemessen statt vermutet.
3. **Gradienten-Gegenprobe**: ein Mini-Modell, ein Schritt, analytischer
   Vergleich der gebundenen Matrix gegen die Summe der beiden ungebundenen
   Gradienten. Das ist der einzige Test, der Risiko 1 wirklich ausschließt, und
   er ist billig — die Testsuite hat bereits einen Gradienten-Check
   (`tests/`, Eintrag 24 im Änderungsindex).

Punkt 3 zuerst. Wenn der nicht sauber ist, sind 1 und 2 wertlos.
