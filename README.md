# Symbiotic Nano-AI

Proof of Concept: Dieses Framework ist der lebende Beweis, dass man nicht auf Python angewiesen ist, um eine schnelle, hardwareagnostische KI-Architektur zu bauen. Es entkoppelt das neuronale Training komplett von gängigen pip-Abhängigkeiten und Virtual Environments.

Ein dezentrales, ressourceneffizientes Framework für das Training, die Synthese und die Inferenz von Large Language Models (LLMs) auf Consumer-Hardware.

Das System kombiniert die Flexibilität von PHP mit der nackten Performance einer C/Nim-Engine via Zero-Copy FFI (Foreign Function Interface). Es ermöglicht verteiltes Lernen (Federated Averaging) über P2P-Netzwerke und dynamisches Modell-Wachstum durch Network Morphism (Net2Net).

## 🚀 Kern-Features

**P2P Schwarm-Training (Federated Averaging):** Verteiltes Lernen auf mehreren PCs. Das System synchronisiert Trainingsdaten und verschmilzt lokal trainierte Gewichte vollautomatisch über das lokale Netzwerk. Kommunikation läuft direkt über UDP/TCP (Gossip-Protokoll), ohne zentralen Server und ohne Cloud-Abhängigkeit.

> Hinweis: Dies ist echtes Full-Parameter-Training. Es handelt sich **nicht** um LoRA, DoRA oder andere PEFT-Adapter, sondern um das Verschmelzen vollständiger, dichter Matrizen.

**Network Morphism (Destille):** Verlustfreies Upscaling von Basismodellen (128 Dimensionen) auf höhere Architekturen (256, 512, 1024 und darüber), inklusive Mitskalierung des Adam-Zustands.

**Autonome Steuerung:** Ein Daemon überwacht Rohdaten, trainiert automatisiert Basis-Drohnen und verschmilzt diese bei kritischer Masse zu größeren Netzen.

**Zero-Copy Inferenz:** Direkte Speicheradressierung der Tensoren in RAM/VRAM unter Umgehung des PHP-Overheads. Die Gewichte werden nie durch PHP-Arrays gereicht.

**Drei Rechenpfade:** CUDA, OpenCL und CPU. Das Backend wird zur Laufzeit ermittelt (`cortex_init_gpu()`); die GPU greift ab 512 Dimensionen, kleinere Modelle rechnen bewusst auf der CPU.

## 🛠 Systemanforderungen

- **OS:** Linux (Ubuntu/Debian/Arch/Manjaro)
- **PHP:** 8.1 oder höher (CLI-Modus). **Wichtig:** In der `php.ini` muss `ffi.enable=true` gesetzt sein.
- **Nim Compiler:** 1.6+ (baut die Engine)
- **C-Compiler mit OpenMP:** Die Engine wird mit `--threads:on` und `-fopenmp` übersetzt. Ohne OpenMP läuft sie, aber jede parallele Schleife seriell.
- **OpenCL (optional):** für GPU-beschleunigte Inferenz und Training
- **CUDA (optional):** wird eingebunden, wenn `nvcc` beim Bauen gefunden wird; sonst überspringt `./symbio.sh init` das CUDA-Backend und OpenCL/CPU tragen allein.

## 📦 Installation

```bash
git clone https://github.com/komakeule57-maker/SnAI.git
cd SnAI
chmod +x symbio.sh
./symbio.sh init
```

`./symbio.sh init` ist **nicht optional** — die kompilierten Bibliotheken liegen bewusst nicht im Repository. Der Befehl legt die Ordnerstruktur an (`lib/`, `factory/…`) und baut:

| Artefakt | Quelle | Bedingung |
|---|---|---|
| `lib/libcortex.so` | `core/cortex.nim` | immer |
| `lib/libtensorutils.so` | `core/snai_tensor_utils.nim` | immer |
| `lib/libcortex_cuda.so` | `core/cortex_cuda.cu` | nur wenn `nvcc` vorhanden |

Nach dem Linken prüft das Skript zwei Dinge selbst, statt sie zu glauben: dass der RUNPATH exakt `$ORIGIN` lautet (sonst findet `libcortex.so` die CUDA-Bibliothek nicht und fällt **still** auf OpenCL zurück) und dass alle Pflichtsymbole exportiert sind. Schlägt eine der Prüfungen fehl, bricht der Build ab.

## ⚙️ Nutzung

Das Framework kann auf zwei Arten betrieben werden: vollautonom im Schwarm-Modus oder manuell über das CLI.

### Modus 1: Vollautonomer Schwarm (empfohlen)

Startet den P2P-Gossip-Node und den Autonomie-Daemon nebeneinander; Strg-C fährt beide ab.

```bash
./symbio.sh swarm
```

Ablauf:

1. Rohe Textdateien nach `factory/input/` legen (siehe [Korpusformat](#-korpusformat)).
2. Der Daemon erkennt die Daten, trainiert eine Basis-Drohne (128 Dim) und legt sie in `factory/experts/` ab. Der verbrauchte Text wandert nach `factory/archive/`.
3. Sobald 3 Drohnen existieren, stellt der Daemon sie nach `factory/tributes/`, verschmilzt sie, skaliert auf 256 Dim und kalibriert nach — das Ergebnis landet als Prinzessin in `factory/royal/`.

Der Node belegt dabei drei Ports: **UDP 43720** (Pheromone), **TCP 43721** (Tensor-Transfer) und **TCP 4372** (Hive-Fingerprint).

### Modus 2: Manuelle Kontrolle (CLI)

**Modell trainieren (BPTT)**

```bash
./symbio.sh cli train <input_text.txt> <output_modell.snai> [epochen]
./symbio.sh cli train factory/input/data.txt factory/experts/drone.snai 20
```

Die Genesis-Dimension liegt fest bei 128 (Drohne); ein Dimensions-Argument gibt es nicht. Höhere Stufen entstehen durch Destillation, nicht durch direktes Training.

**Modell befragen (Zero-Copy Inferenz)**

```bash
./symbio.sh cli infer <modell.snai> "<prompt>" [beam_width]
./symbio.sh cli infer factory/experts/drone.snai "Die Bahnstrecke"
```

Ohne `beam_width` läuft Temperature/min-p-Sampling; ab `2` schaltet Beam-Search ein. Der Prompt wird intern als `[USER] <text>\n[SYMBIO]` formatiert.

**Modell destillieren (Merge + Net2Net Upscale + Kalibrierung)**

```bash
./symbio.sh cli distill [Prinzessin|Matriarchin|Empress|Demi Goddess|Goddess]
./symbio.sh cli distill Prinzessin
```

Verschmelzung, Expansion und Kalibrierung laufen als **eine** Einheit — die früheren Einzelschritte `cli merge` und `cli upscale` gibt es nicht mehr. Für `Prinzessin` müssen 3 Drohnen in `factory/tributes/` liegen; im autonomen Betrieb stellt der Daemon sie selbst dorthin.

## 🧬 Evolutionsleiter

Jede Stufe entsteht aus der vorigen. Die Werte stehen in `EVOLUTION_LADDER` in `src/royal_distillery.php`; die Layer-Zahlen bis Empress folgen dem Verhältnis von Pythia-70M (512 Dim / 6 Layer), nicht einem konstanten Dim/Layer-Faktor.

| Stufe | Dim | Kontext | Layer | Quelle | Ablage |
|---|---|---|---|---|---|
| Drohne | 128 | 512 | 2 | Genesis | `factory/experts/` |
| Prinzessin | 256 | 1024 | 3 | 3 Drohnen | `factory/royal/` |
| Matriarchin | 512 | 2048 | 6 | Prinzessin | `factory/apex/` |
| Empress | 1024 | 4096 | 12 | Matriarchin | `factory/apex/` |
| Demi Goddess | 2048 | 8192 | 16 | Empress | `factory/apex/` |
| Goddess | 4096 | 8192 | 32 | Demi Goddess | `factory/apex/` |

## 📝 Korpusformat

Reiner Fliesstext funktioniert unverändert — solche Dokumente trainieren vollständig.

Für Dialogdaten sind `[USER]`, `[SYMBIO]` und `[SYSTEM]` registrierte Einzel-Tokens:

```
[USER] Definiere eine Python Funktion:
[SYMBIO] def hallo():
print('Test')</s>
```

Enthält ein Dokument diese Marker, greift **Loss-Masking**: nur die `[SYMBIO]`-Antwort liefert Gradienten, der `[USER]`-Text nicht. Das Vokabular umfasst 8192 Token (`lib/symbio_vocab.json`, BPE mit Metaspace `U+2581`).

## 💾 Modellformat (SNAI_V5)

Ein Checkpoint ist ein roher Strom, kein Container:

- **28-Byte-Header:** 8 Byte gepaddetes Magic `SNAI_V5`, dann 5× `uint32` Little-Endian — `vocab`, `hidden`, `ffn`, `ctx`, `layers`
- **Nutzlast:** `float32` Little-Endian. Pro Tensor werden Gewicht und die beiden Adam-Momente **verschränkt** abgelegt (`w`, `m`, `v`) — daher die dreifache Dateigrösse gegenüber reinen Gewichten.
- **Trailer:** optionaler `int32` `adam_t` (Schrittzähler)
- **Sidecars:** `<modell>.heads` (Attention-Köpfe) und `<modell>.growth` (Net2Net-Expansionsgrenze). Beide gehören zum Modell und müssen mitkopiert werden.

> ⚠️ **Modelle im alten Format `SNAI_V4` können nicht gelesen werden.** V4 enthielt nur Gewichte ohne Adam-Zustand, und das Vokabular hat sich von 16384 auf 8192 Token geändert — die Embedding-Matrizen wären selbst nach einer Formatkonvertierung bedeutungslos. Einen Konverter gibt es nicht. Wer den Juni-Stand braucht, findet ihn im Commit `3bae226`.

## 📂 Architektur / Verzeichnisstruktur

| Pfad | Inhalt |
|---|---|
| `core/` | Quelltext der Nim/C/CUDA-Tensor-Engine |
| `src/` | PHP-Steuerlogik (Daemon, Node, Cluster-Master, Destille, CLI) |
| `lib/` | Tokenizer-Vokabular; hierhin baut `symbio.sh init` die `.so`-Dateien |
| `tools/` | Tokenizer-Schmiede (`bpe_forge`) zur einmaligen Vokabular-Erzeugung |
| `factory/` | Laufzeit-Territorium, wird beim `init` angelegt |

Innerhalb von `factory/`:

| Pfad | Inhalt |
|---|---|
| `input/` | rohe Textdaten (`.txt`) für das Training |
| `archive/` | verbrauchte Textdaten, werden für Kalibrierungen wiederverwendet |
| `experts/` | Basis-Drohnen (128 Dim) |
| `tributes/` | Vorraum der Destille: die 3 Drohnen der nächsten Synthese |
| `royal/` | Prinzessinnen (256 Dim) |
| `apex/` | Matriarchin und höher |
| `royal_fragments/` | verbrauchte Drohnen nach einer erfolgreichen Synthese |
| `checkpoints/` | Zwischenstände je Epoche und Worker |

## ⚖️ Lizenz & Kommerzielle Nutzung

Dieses Projekt ist Open Source und unter der GNU AGPLv3 lizenziert. Das bedeutet, dass du den Code für private, akademische oder Open-Source-Projekte frei nutzen, verändern und weitergeben darfst, solange du dich an die Bedingungen der AGPLv3 hältst (welche vorschreibt, dass abgeleitete Werke ebenfalls unter dieser Lizenz veröffentlicht werden müssen).

**Für Unternehmen:**
Wenn ihr diesen Code in einem kommerziellen oder proprietären Umfeld (inklusive SaaS) nutzen, integrieren oder verbreiten möchtet, ohne den strengen Open-Source-Zwang der AGPLv3 zu übernehmen, ist eine separate kommerzielle Lizenz erforderlich. Bitte kontaktiert mich für kommerzielle Anfragen.
