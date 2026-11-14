# DemiGoddess-Testlauf — Planung (Kompromiss-Modell, CPU-Pfad)

## Ziel

Klären, ob die volle DemiGoddess-Spezifikation (2048 Dim, 16 Layer, siehe
`EVOLUTION_LADDER` in `royal_distillery.php`) auf der vorhandenen Hardware
(RTX 3050, 6 GB VRAM, 16 GB System-RAM) unter reinem FP32 überhaupt zeitlich
praktikabel trainierbar ist — **kein Produktionslauf**, reiner
Machbarkeits-/Geschwindigkeitstest. Eine Epoche mit viel Trainingsmaterial
reicht als Testkriterium  — kein vollständiger
Mehr-Epochen-Kalibrierungslauf, kein Held-out-Vergleichsanspruch wie bei den
`docs/inferenz/INFERENZ_*.md`-Tests.

## Warum nicht die normale Schwarm-Pipeline

Kein Weg über `royal_distillery.php`s Evolution-Ladder (Net2Net-Wachstum durch
alle Zwischenstufen Drohne→...→Empress→DemiGoddess). Stattdessen direktes
Genesis-Init auf die volle DemiGoddess-Dimension (analog `init_base.php`), als
eigenständiges Einzeltraining außerhalb der normalen Pipeline.

**"Kompromiss" bezieht sich bewusst nicht auf die Layer-Zahl:** die volle
16-Layer-Spec bleibt erhalten — eine Reduktion der Layer-Zahl wurde in der
vorausgegangenen Diskussion explizit verworfen ("wäre am Ziel vorbei"). Der Kompromiss besteht
stattdessen aus zwei anderen Stellschrauben: reduzierter Kontext (Ctx) und
erzwungenes CPU-Backend statt GPU.

## Kernparameter (vorläufig)

| Parameter | Wert | Begründung |
|---|---|---|
| Vocab | 8192 | Projekt-Standard |
| Dim | 2048 | volle DemiGoddess-Spec |
| FFN | 8192 (4× Dim) | Projekt-Standard-Ratio (`init_base.php`-Default) |
| Layer | 16 | volle DemiGoddess-Spec, bewusst nicht reduziert |
| Ctx | 2048 (statt Leiter-Standard 8192) | RAM-Budget, siehe Rechnung unten |
| Backend | erzwungen CPU | VRAM (6 GB) reicht bei Dim=2048 nur für ~5-6 Layer, System-RAM (16 GB) für die vollen 16 |

## RAM-Rechnung (Basis für die Ctx-Wahl)

Gewicht+Adam-Zustand (w, m, v, g) pro Layer skaliert nur mit Dim/FFN, nicht
mit Ctx: bei Dim=2048 sind das ~768 MiB/Layer, bei 16 Layern **12,0 GiB** fix
— unabhängig vom gewählten Ctx. KV-Cache+Gradient skaliert dagegen mit
`16·Dim·Ctx` Byte/Layer und ist der eigentliche Hebel:

| Ctx | KV-Cache+Grad (16 Layer) | Gesamt (+ fix ~0,9-1,1 GiB für Embeddings/`w_out`/Logits-Cache/Rest) |
|---|---|---|
| 8192 (Leiter-Standard) | 4,0 GiB | **~17,5 GiB** — passt nicht in 16 GB |
| 4096 | 2,0 GiB | **~14,9 GiB** — passt knapp, ~1 GB Puffer |
| 2048 | 1,0 GiB | **~13,9 GiB** — ~2 GB Puffer |

**Empfehlung:** mit Ctx=2048 starten (mehr Sicherheitsabstand für OS/andere
Prozesse), Ctx=4096 nur als Stretch-Option falls 2048 spürbar Luft lässt.
Diese Rechnung deckt nur Gewichte+Adam+KV-Cache/Gradient ab — ephemere
Scratch-Buffer (Attention-Scores etc.) und OS-Overhead sind nicht exakt
beziffert, siehe "Offene Fragen".

## Geschwindigkeits-Unsicherheit

Rein aus der Architektur grob abgeschätzter Mehraufwand pro Trainingsschritt
gegenüber der (GPU-trainierten) Matriarchin (512 Dim, 4 Layer): Faktor
(2048/512)² × (16/4) = **~64×** allein aus Dim²/Layer-Skalierung. Der
zusätzliche Faktor durch den Verlust der GPU-Parallelität ist **nicht**
seriös aus dem Code hochrechenbar (hängt von OMP-Thread-Zahl, Cache-Verhalten
bei 2048×2048-Matrizen etc. ab). Matriarchin-Epochen brauchten bereits ~7-8h
auf GPU — eine DemiGoddess-Epoche auf reiner CPU könnte sich von "einige
Tage" bis "mehrere Wochen" strecken, je nach diesem unbekannten Zusatzfaktor.

**Deshalb vor dem eigentlichen Epochen-Lauf: ein kurzer Zeit-Benchmark**
(10-20 echte Trainingsschritte, Zeit/Schritt messen, auf eine volle Epoche
hochrechnen) — erst danach entscheiden, ob der volle Lauf überhaupt gestartet
wird.

## Offene technische Fragen / nächste Schritte

1. **CPU-Erzwingung braucht ein kleines eigenständiges Skript.**
   `TensorFFI::set_gpu_threshold()` (`tensor_ffi_adapter.php:260`, wrapt
   `cortex_set_gpu_threshold()`) existiert bereits FFI-seitig, wird aber von
   keinem bestehenden CLI-Pfad aufgerufen — kein `train_worker.php`/
   `train_cluster.php`-Argument dafür. Braucht ein eigenes Test-Skript
   (analog zum `diagnose/*.php`-Muster: Genesis-Init + Threshold weit über
   die größte Matrixgröße setzen + Trainingsschleife + Zeit/Schritt-Logging),
   nicht die Produktionsskripte anfassen. Noch nicht geschrieben.
2. **Trainingsmaterial noch nicht konkretisiert.** "Sehr viel
   Trainingsmaterial" — welche Korpus-Quelle/-Menge für
   den Test? GEN/SFT-Mix aus `factory/input/` direkt, oder ein eigener
   Ausschnitt?
3. Ephemere GPU-Scratch-Buffer/Treiber-Overhead sind für den CPU-Pfad
   irrelevant (kein VRAM-Aspekt mehr), aber die genaue Host-RAM-Nutzung
   durch Aktivierungs-/Gradienten-Scratch-Buffer (`act_ffn_out`, `d_ffn`,
   Dropout-Masken etc.) ist in der obigen Rechnung nur grob mit "fix ~0,9-1,1
   GiB" abgeschätzt, nicht Byte-genau nachgerechnet.
4. Ergebnis (Zeit/Schritt, Hochrechnung, ob und wie der volle Lauf
   durchgeführt wurde) hier nachtragen, sobald der Benchmark gelaufen ist.
