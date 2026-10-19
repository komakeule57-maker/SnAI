*[Deutsch](README.md) · **English***

# Symbiotic Nano-AI

Proof of concept: this framework is living evidence that you do not need Python to build a fast, hardware-agnostic AI architecture. It decouples neural training entirely from the usual pip dependencies and virtual environments.

A decentralised, resource-efficient framework for training, synthesising and running inference on Large Language Models (LLMs) on consumer hardware.

The system combines the flexibility of PHP with the raw performance of a C/Nim engine via zero-copy FFI (Foreign Function Interface). It supports distributed learning (federated averaging) over P2P networks and dynamic model growth through network morphism (Net2Net).

## 🚀 Core features

**P2P swarm training (federated averaging):** Distributed learning across several machines. The system synchronises training data and merges locally trained weights automatically over the local network. Communication runs directly over UDP/TCP (gossip protocol), with no central server and no cloud dependency.

> Note: this is genuine full-parameter training. It is **not** LoRA, DoRA or any other PEFT adapter, but the merging of complete, dense matrices.

**Network morphism (the distillery):** Lossless upscaling of base models (128 dimensions) to larger architectures (256, 512, 1024 and beyond), including the Adam state, which is scaled along with the weights.

**Autonomous operation:** A daemon watches for raw data, trains base drones automatically and merges them into larger networks once critical mass is reached.

**Zero-copy inference:** Tensors are addressed directly in RAM/VRAM, bypassing PHP overhead. Weights never pass through PHP arrays.

**Three compute paths:** CUDA, OpenCL and CPU. The backend is determined at runtime (`cortex_init_gpu()`); the GPU kicks in from 512 dimensions upwards, while smaller models run on the CPU by design.

## 🛠 Requirements

- **OS:** Linux (Ubuntu/Debian/Arch/Manjaro)
- **PHP:** 8.1 or newer (CLI mode). **Important:** `ffi.enable=true` must be set in `php.ini`.
- **Nim compiler:** 1.6+ (builds the engine)
- **A C compiler with OpenMP:** the engine is built with `--threads:on` and `-fopenmp`. Without OpenMP it still runs, but every parallel loop runs serially.
- **OpenCL (optional):** for GPU-accelerated inference and training
- **CUDA (optional):** included if `nvcc` is found at build time; otherwise `./symbio.sh init` skips the CUDA backend and OpenCL/CPU carry the load alone.

## 📦 Installation

```bash
git clone https://github.com/komakeule57-maker/SnAI.git
cd SnAI
chmod +x symbio.sh
./symbio.sh init
```

`./symbio.sh init` is **not optional** — the compiled libraries are deliberately kept out of the repository. The command creates the directory structure (`lib/`, `factory/…`) and builds:

| Artifact | Source | Condition |
|---|---|---|
| `lib/libcortex.so` | `core/cortex.nim` | always |
| `lib/libtensorutils.so` | `core/snai_tensor_utils.nim` | always |
| `lib/libcortex_cuda.so` | `core/cortex_cuda.cu` | only if `nvcc` is present |

After linking, the script verifies two things rather than assuming them: that the RUNPATH is exactly `$ORIGIN` (otherwise `libcortex.so` cannot find the CUDA library and falls back to OpenCL **silently**), and that every required symbol is exported. If either check fails, the build aborts.

## ⚙️ Usage

The framework can be run in two ways: fully autonomously in swarm mode, or manually through the CLI.

### Mode 1: Autonomous swarm (recommended)

Starts the P2P gossip node and the autonomy daemon side by side; Ctrl-C shuts both down.

```bash
./symbio.sh swarm
```

How it proceeds:

1. Drop raw text files into `factory/input/` (see [corpus format](#-corpus-format)).
2. The daemon picks up the data, trains a base drone (128 dim) and places it in `factory/experts/`. The consumed text moves on to `factory/archive/`.
3. Once three drones exist, the daemon stages them in `factory/tributes/`, merges them, upscales to 256 dim and calibrates — the result lands in `factory/royal/` as a Prinzessin.

The node occupies three ports: **UDP 43720** (pheromones), **TCP 43721** (tensor transfer) and **TCP 4372** (hive fingerprint).

### Mode 2: Manual control (CLI)

**Train a model (BPTT)**

```bash
./symbio.sh cli train <input_text.txt> <output_model.snai> [epochs]
./symbio.sh cli train factory/input/data.txt factory/experts/drone.snai 20
```

The genesis dimension is fixed at 128 (drone); there is no dimension argument. Larger tiers come from distillation, not from direct training.

**Long runs: warmup and decay over steps**

Warmup and cosine decay in `train_cluster.php` count in epochs. If you train a large corpus in segments (each segment one call with one epoch), you get neither, just the full learning rate throughout. Three environment variables fill that gap. They count over the Adam step counter `adam_t` stored in the checkpoint, so they carry across segment boundaries:

| Variable | Effect |
|---|---|
| `SNAI_WARMUP_STEPS=N` | learning rate rises linearly over the first N steps |
| `SNAI_LR_DECAY_STEPS=M` | cosine from the end of warmup to step M, floor afterwards |
| `SNAI_LR_MIN_RATIO=r` | floor = r × learning rate (default 0.1) |

Without these variables nothing changes. A recipe tried on a model with width 512, 6 layers and context 1024 (not on the 128 drone): learning rate `0.0001`, `SNAI_WARMUP_STEPS=500`, `SNAI_LR_DECAY_STEPS` = steps of the whole run, weight decay `0.1` (10th argument of `train_cluster.php`), one worker, no dropout, no label smoothing. With learning rate 0.001 and no warmup, attention saturated there within a few hundred steps; a long run with that setting never learned to copy from its context in 20 segments.

**Query a model (zero-copy inference)**

```bash
./symbio.sh cli infer <model.snai> "<prompt>" [beam_width]
./symbio.sh cli infer factory/experts/drone.snai "Die Bahnstrecke"
```

Without `beam_width` the sampler uses temperature/min-p; from `2` upwards beam search kicks in. Internally the prompt is formatted as `[USER] <text>\n[SYMBIO]`.

**Distil a model (merge + Net2Net upscale + calibration)**

```bash
./symbio.sh cli distill [Prinzessin|Matriarchin|Empress|Demi Goddess|Goddess]
./symbio.sh cli distill Prinzessin
```

Merging, expansion and calibration run as **one** unit — the former separate steps `cli merge` and `cli upscale` no longer exist. `Prinzessin` requires three drones in `factory/tributes/`; in autonomous mode the daemon stages them itself.

> The tier names are literal arguments, taken from `EVOLUTION_LADDER` in the source. They stay in German even in this document — `distill Princess` will not work.

## 🧬 Evolution ladder

Each tier grows out of the previous one. The values live in `EVOLUTION_LADDER` in `src/royal_distillery.php`; the layer counts up to Empress follow the ratio of Pythia-70M (512 dim / 6 layers) rather than a constant dim-per-layer factor.

| Tier | Dim | Context | Layers | Grown from | Stored in |
|---|---|---|---|---|---|
| `Drohne` (drone) | 128 | 512 | 2 | genesis | `factory/experts/` |
| `Prinzessin` (princess) | 256 | 1024 | 3 | 3 drones | `factory/royal/` |
| `Matriarchin` (matriarch) | 512 | 2048 | 6 | princess | `factory/apex/` |
| `Empress` | 1024 | 4096 | 12 | matriarch | `factory/apex/` |
| `Demi Goddess` | 2048 | 8192 | 16 | empress | `factory/apex/` |
| `Goddess` | 4096 | 8192 | 32 | demi goddess | `factory/apex/` |

## 📝 Corpus format

Plain running text works as-is — such documents train in full.

For dialogue data, `[USER]`, `[SYMBIO]` and `[SYSTEM]` are registered single tokens:

```
[USER] Definiere eine Python Funktion:
[SYMBIO] def hallo():
print('Test')</s>
```

When a document contains these markers, **loss masking** applies: only the `[SYMBIO]` response contributes gradients, the `[USER]` text does not. If a custom vocabulary lacks the markers as single tokens, the worker turns masking off with a warning instead of guessing.

**Vocabulary.** `lib/symbio_vocab.json` holds 8192 tokens (BPE with the `U+2581` metaspace), built from a cleaned German corpus. Characters outside the base alphabet pass through as UTF-8 bytes (256 byte tokens, `byte_fallback`), so `<unk>` no longer occurs, not even for emoji or foreign scripts. To build your own:

```bash
tools/bpe_init/bpe_forge <corpus_dir> lib/symbio_vocab.json
```

It reads every `*.txt`/`*.php` in the directory. Before training, `src/korpus_sanitize.php` (`korpus_bereinigen()`) removes runs of scripts that cannot be transliterated, together with their language label, e.g. "(chinesisch 新镇)".

> ⚠️ **A new vocabulary applies since October 2026.** Models trained with the old one (up to commit `b5a3a66`) do not match it: the same token number now stands for a different piece of text. Retrain old models, or run them with the old `lib/symbio_vocab.json` from that commit.

## 💾 Model format (SNAI_V5)

A checkpoint is a raw stream, not a container:

- **28-byte header:** 8 bytes of padded magic `SNAI_V5`, then 5× `uint32` little-endian — `vocab`, `hidden`, `ffn`, `ctx`, `layers`
- **Payload:** `float32` little-endian. For each tensor the weight and both Adam moments are stored **interleaved** (`w`, `m`, `v`) — hence the threefold file size compared to weights alone.
- **Trailer:** optional `int32` `adam_t` (step counter)
- **Sidecars:** `<model>.heads` (attention heads) and `<model>.growth` (Net2Net expansion boundary). Both belong to the model and must be copied along with it.

> ⚠️ **Models in the old `SNAI_V4` format cannot be read.** V4 held weights only, without the Adam state, and the vocabulary changed from 16384 to 8192 tokens — the embedding matrices would be meaningless even after a format conversion. There is no converter. If you need the June state, it is at commit `3bae226`.

## 📂 Architecture / directory layout

| Path | Contents |
|---|---|
| `core/` | source of the Nim/C/CUDA tensor engine |
| `src/` | PHP control logic (daemon, node, cluster master, distillery, CLI) |
| `lib/` | tokenizer vocabulary; `symbio.sh init` builds the `.so` files here |
| `tools/` | tokenizer forge (`bpe_forge.nim`, Nim standard library only) for one-off vocabulary generation |
| `factory/` | runtime territory, created during `init` |

Inside `factory/`:

| Path | Contents |
|---|---|
| `input/` | raw text data (`.txt`) for training |
| `archive/` | consumed text data, reused for calibration |
| `experts/` | base drones (128 dim) |
| `tributes/` | the distillery's antechamber: the three drones for the next synthesis |
| `royal/` | princesses (256 dim) |
| `apex/` | matriarch and above |
| `royal_fragments/` | drones consumed by a successful synthesis |
| `checkpoints/` | intermediate state per epoch and worker |

## ⚖️ Licence & commercial use

This project is open source and licensed under the GNU AGPLv3. That means you may use, modify and redistribute the code freely for private, academic or open-source projects, as long as you abide by the terms of the AGPLv3 (which requires derivative works to be published under the same licence).

**For companies:**
If you wish to use, integrate or distribute this code in a commercial or proprietary setting (including SaaS) without taking on the strict open-source obligations of the AGPLv3, a separate commercial licence is required. Please get in touch for commercial enquiries.

> This section is a summary for convenience. The binding text is the [LICENSE](LICENSE) file.
