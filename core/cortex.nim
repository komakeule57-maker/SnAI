# Dateiname: cortex.nim
# Funktion: Symbio Nano-AI Framework - Cortex Core Backend (FFI, Inference, BPTT)
# ==============================================================================
# VERSION: V11.11  (Etappe 1, Schritt 1 - gebuendeltes w_out ist Standard)
# Gemessen gegen V11.8 auf expert_SFT (dim=128, vocab=16384, 256 Positionen):
#   CPU  456,5 ms -> 335,0 ms (1,36x)   GPU  2672,9 ms -> 1671,7 ms (1,60x)
#   Loss in allen Laeufen bitidentisch -> Default jetzt AN.
#   Abschaltbar bleibt er: cortex_set_batched_wout(0) stellt V11.8-Verhalten her.
# V11.9 hatte in den CPU-Fallbacks der beiden Backward-Wrapper eine
# cache-feindliche Schleifenordnung (Schrittweite 512 B bzw. 64 KB in der
# innersten Schleife) -> Bucket 3 stieg von 255 ms auf 4012 ms. Behoben:
# beide nutzen jetzt exakt die zeilenweise Ordnung des bewaehrten alten
# Pfades, nur um die Positionsachse erweitert. GPU-Pfad unveraendert.
# Vorgaenger: V11.8 (OpenMP-Rueckbau) | Backend: cortex_opencl_v86.nim (V8.6)
# cortex_set_batched_wout(0) schaltet zurueck auf den alten Pfad (A/B-Vergleich).
# ==============================================================================
import std/math
import std/random
# V11.7 BAUFLAGS (zwingend, sonst laufen die ||-Schleifen seriell):
#   nim c -d:danger --threads:on --passC:-fopenmp --passL:-fopenmp \
#         --app:lib --out:libcortex.so cortex.nim
import std/times
import cortex_opencl
import cortex_cuda_bridge

# ==============================================================================
# SYMBIO NANO-AI FRAMEWORK (Code-4372)
# Modul: Cortex Core (Nim FFI Backend) - PRODUCTION V11.1 (ADAM & FFN FIX)
# ------------------------------------------------------------------------------
# MAXIME: E > H. Perfekte Mathematik trifft auf ausbruchsicheres I/O.
# UPDATE V11: Unterstützung für das SNAI_V5 Format (Adam Persistence).
# AUDIT-FIX V11.1: Heap-Buffer-Overflow bei dynamischer ffnDim behoben.
# AUDIT-FIX V11.1: Sanity-Checks für Header-Werte (OOM-Prevention) integriert.
# FIX V11.4b: Fehlerausgaben ohne String-Konkatenation (varargs writeLine).
#             Beseitigt die -Wstringop-overflow Warnungen unter -d:danger + LTO.
# PERF V11.4: w_out-Logits und der komplette Matrix-Backward laufen auf der GPU.
# AUDIT-FIX V11.3: Net2Net-Gain-Guard an cortex_load_layer + cortex_validate.
# AUDIT-FIX V11.2: Loader ist jetzt selbstverifizierend (Byte-Bilanz + v>=0-Test),
#                  raeumt bei Fehlern auf und stellt adam_t wieder her.
# ==============================================================================

{.pragma: ffi, exportc, dynlib, cdecl.}

type
  FloatArray = UncheckedArray[cfloat]
  IntArray   = UncheckedArray[cint]
  FloatPtrArray = UncheckedArray[ptr FloatArray]
  MatrixArray   = UncheckedArray[Matrix]

  Matrix* = object
    rows*: cint
    cols*: cint
    data*: ptr FloatArray

  CortexState* = object
    vocabSize*, hiddenDim*, ffnDim*, ctxSize*, numLayers*: cint
    adam_t*: cint 
    numHeads*, headSize*: cint
    cachedTokens*: cint
    
    embeddings*, m_embeddings*, v_embeddings*, g_embeddings*: Matrix
    w_out*, m_w_out*, v_w_out*, g_w_out*: Matrix
    
    rms_final_weight*, m_rms_final_weight*, v_rms_final_weight*, g_rms_final_weight*: ptr FloatArray

    rms_att_weights*, m_rms_att_weights*, v_rms_att_weights*, g_rms_att_weights*: ptr FloatPtrArray
    rms_ffn_weights*, m_rms_ffn_weights*, v_rms_ffn_weights*, g_rms_ffn_weights*: ptr FloatPtrArray
    w_q*, m_w_q*, v_w_q*, g_w_q*: ptr MatrixArray
    w_k*, m_w_k*, v_w_k*, g_w_k*: ptr MatrixArray
    w_v*, m_w_v*, v_w_v*, g_w_v*: ptr MatrixArray
    w_o*, m_w_o*, v_w_o*, g_w_o*: ptr MatrixArray
    w_up*, m_w_up*, v_w_up*, g_w_up*: ptr MatrixArray
    w_down*, m_w_down*, v_w_down*, g_w_down*: ptr MatrixArray

    k_cache*, v_cache*, act_q*: ptr MatrixArray
    d_k_cache*, d_v_cache*: ptr MatrixArray
    logits_cache*: Matrix
    
    act_hidden*, act_norm_att*, act_att_out*: ptr FloatPtrArray
    act_hidden_ffn*, act_norm_ffn*, act_ffn_in*, act_ffn_out*: ptr FloatPtrArray
    # Update 2026-08-09 (docs/ROADMAP.md, Nachruest-Paket #6 "Dropout"):
    # gleiche Form/Groesse wie act_ffn_out (ctx*ffnDim je Layer) - haelt pro
    # Element entweder 0.0 (verworfen) oder 1.0/(1.0-rate) (behalten, bereits
    # skaliert - "Inverted Dropout", damit im Eval/Inferenz-Pfad KEINE
    # separate Kompensation noetig ist, siehe applyDropoutForward()). Muss von
    # Forward bis zum entsprechenden Backward-Aufruf desselben Layers
    # ueberleben (beide sind getrennte Durchlaeufe ueber alle Positionen,
    # nicht Position-fuer-Position verschraenkt) - deshalb ein persistenter
    # State-Puffer, keine lokale Variable.
    dropout_mask*: ptr FloatPtrArray
    act_hidden_final*: ptr FloatArray
    # V11.9: fuer den gebuendelten w_out-Pfad. act_norm_final haelt die Ausgabe
    # der finalen RMSNorm je Position, d_norm_all das Ergebnis des gebuendelten
    # w_out-Backwards. Beide ctx*dim Floats -> bei ctx=1024, dim=256 je 1 MB.
    act_norm_final*, d_norm_all*: ptr FloatArray

    # AUDIT: fuer den gebuendelten (layer-major) Attention-Pfad. d_hidden_all
    # ersetzt d_hidden als Positions-persistenter Gradienten-Akkumulator ueber
    # die gesamte Layer-Countdown-Schleife (statt pro Position neu). d_att_out_all
    # und d_q_all sind die gebuendelten Ein-/Ausgaben fuer attentionBackwardBatchedD.
    # Alle drei ctx*dim Floats, wie act_norm_final/d_norm_all.
    d_hidden_all*, d_att_out_all*, d_q_all*: ptr FloatArray

    # AUDIT: fuer die gebuendelte Q/K/V/O/Up/Down-Mathematik (matMulBatchedD &
    # Co., dieselbe Infrastruktur wie beim gebuendelten w_out). mm_hd_scratch
    # (ctx*dim) und mm_ffn_scratch (ctx*ffn) sind reiner Ein-Layer-Zwischenspeicher,
    # sequenziell innerhalb eines Sub-Pass wiederverwendet - kein Positions- oder
    # Layer-persistenter Zustand noetig.
    mm_hd_scratch*, mm_ffn_scratch*: ptr FloatArray

    j_space_cache*: ptr FloatPtrArray

    hidden_state_buf*, norm_buf*, att_out_buf*, ffn_buf*, q_buf*: ptr FloatArray
    d_logits_buf*, scores_buf*, d_w_buf*: ptr FloatArray
    d_hidden*, d_norm*, d_att_out*, d_q*, d_ffn*: ptr FloatArray

var gpu_initialized: bool = false

# Update 2026-08-08: intelligenter Backend-Switch (siehe cortex_init_gpu()
# unten und docs/ROADMAP.md). gpu_initialized bleibt der reine "irgendein
# GPU-Backend aktiv?"-Schalter (von useGpuMat() und den meisten Gate-Checks
# unveraendert weiterverwendet) - gpu_backend haelt zusaetzlich WELCHES.
type GpuBackend = enum gbNone, gbOpenCL, gbCuda
var gpu_backend: GpuBackend = gbNone

proc cortex_get_gpu_backend*(): cint {.ffi.} =
  ## Fuer Diagnose/Logging von aussen (PHP-Seite): 0=keins, 1=OpenCL, 2=CUDA.
  case gpu_backend
  of gbNone: 0.cint
  of gbOpenCL: 1.cint
  of gbCuda: 2.cint

# V11.4: EINE Stelle entscheidet ueber CPU/GPU. Forward, Backward, Zeroing und
# Sync muessen zwingend dieselbe Antwort bekommen - sonst holt ein Sync
# Device-Nullen ueber ein auf der CPU berechnetes Gradientenfeld.
# V11.5: Die Schwelle ist jetzt eine ELEMENTZAHL, keine Kantenlaenge.
# Grund: ein GPU-Roundtrip kostet eine feste Latenz (gemessen ~1,8 ms). Eine
# 256x256-Matrix sind 65k MACs - die rechnet die CPU in 66 us. Die GPU zu
# fragen ist dort 27x langsamer. Erst ab ~1,8M Elementen lohnt der Weg.
var gpu_min_elems: int = 1_000_000

proc cortex_set_gpu_threshold*(elems: cint) {.ffi.} =
  gpu_min_elems = (if elems < 0: 0 else: elems.int)

proc cortex_get_gpu_threshold*(): cint {.ffi.} =
  gpu_min_elems.cint

# ==============================================================================
# WEIGHT-DECAY (AdamW-Style, dekorreliert vom Adam-Update selbst).
# Siehe docs/ROADMAP.md, "Trainings-Absicherung: Nachruest-Paket" Punkt 1.
# Default 0.0 = exaktes No-Op, identisch zum bisherigen reinen-Adam-Verhalten -
# bestehende Laeufe/Checkpoints sind unberuehrt, solange niemand explizit
# cortex_set_weight_decay() aufruft. Greift NUR an echten Gewichtsmatrizen
# (Embeddings, w_q/k/v/o, w_up/down, w_out) - RMSNorm-Gains werden bewusst
# ausgenommen (sollen um 1.0 pendeln, nicht gegen 0 gezogen werden), siehe
# upA()/oclUpA() unten, die global_weight_decay NICHT weiterreichen.
# ==============================================================================
var global_weight_decay: cfloat = 0.0

proc cortex_set_weight_decay*(wd: cfloat) {.ffi.} =
  global_weight_decay = (if wd < 0.0: 0.0 else: wd)

proc cortex_get_weight_decay*(): cfloat {.ffi.} =
  global_weight_decay

# Update 2026-08-09 (docs/ROADMAP.md, Nachruest-Paket #9 "Gradient-Norm-
# Logging"): clipGlobalGradientNorm() berechnete die globale L2-Norm schon
# immer fuer die Clip-Entscheidung, hat sie aber bisher nirgends nach aussen
# gereicht. Gleiches Muster wie global_weight_decay oben - ein Prozess haelt
# genau einen CortexState, ein globaler Wert reicht.
var last_grad_norm: cfloat = 0.0

proc cortex_get_last_grad_norm*(): cfloat {.ffi.} =
  last_grad_norm

# ==============================================================================
# DROPOUT (siehe docs/ROADMAP.md, "Trainings-Absicherung: Nachruest-Paket"
# Punkt 6, umgesetzt im Rahmen "Engine-/Framework-Verbesserungen" 2026-08-09).
# Default 0.0 = exaktes No-Op, gleiches Prinzip wie global_weight_decay oben -
# bestehende Laeufe sind unberuehrt, solange niemand explizit
# cortex_set_dropout_rate() aufruft. Greift NUR in cortex_train_step() (auf
# der FFN-Aktivierung, nach SiLU, vor der Down-Projektion) - cortex_forward()
# (Inferenz/Held-out-Eval) kennt kein Dropout, das ist die komplette
# Train/Eval-Unterscheidung: zwei bereits strukturell getrennte Code-Pfade,
# keine zusaetzliche Modus-Variable noetig.
# ==============================================================================
var global_dropout_rate: cfloat = 0.0
var dropout_rng_seeded = false

proc cortex_set_dropout_rate*(rate: cfloat) {.ffi.} =
  # Obergrenze 0.9 statt 1.0 - bei rate=1.0 waere der Skalierungsfaktor
  # 1.0/(1.0-rate) eine Division durch Null.
  global_dropout_rate = (if rate < 0.0: 0.0 elif rate > 0.9: 0.9 else: rate)
  if global_dropout_rate > 0.0 and not dropout_rng_seeded:
    randomize()  # einmalig, aus der System-Uhr - siehe ROADMAP-Kommentar
    dropout_rng_seeded = true

proc cortex_get_dropout_rate*(): cfloat {.ffi.} =
  global_dropout_rate

# Inverted Dropout: skaliert die ueberlebenden Elemente bereits im Training um
# 1.0/(1.0-rate) hoch, damit cortex_forward() (Inferenz) UNVERAENDERT bleiben
# kann - keine separate "* (1-rate)"-Kompensation dort noetig, die leicht
# vergessen werden koennte. buf wird in-place skaliert, mask haelt denselben
# Wert (0.0 oder 1.0/(1.0-rate)) pro Element fest, fuer den spaeteren
# Backward-Aufruf auf GENAU diesem Layer.
proc applyDropoutForward(buf: ptr FloatArray, mask: ptr FloatArray, n: int) {.inline.} =
  if global_dropout_rate <= 0.0: return
  let keepScale = 1.0'f32 / (1.0'f32 - global_dropout_rate)
  for i in 0 ..< n:
    if rand(1.0) < global_dropout_rate.float:
      mask[i] = 0.0
      buf[i] = 0.0
    else:
      mask[i] = keepScale
      buf[i] *= keepScale

# Kettenregel fuer y = mask * x: dL/dx = dL/dy * mask - exakt dieselbe Maske
# wie im zugehoerigen Forward-Aufruf, deshalb kein erneutes Wuerfeln hier.
proc applyDropoutBackward(grad: ptr FloatArray, mask: ptr FloatArray, n: int) {.inline.} =
  if global_dropout_rate <= 0.0: return
  for i in 0 ..< n:
    grad[i] *= mask[i]

# ==============================================================================
# LABEL SMOOTHING (siehe docs/ROADMAP.md, "Naechste Schritte", Punkt 7).
# Default 0.0 = exaktes No-Op, gleiches Prinzip wie global_weight_decay/
# global_dropout_rate oben - bestehende Laeufe sind unberuehrt, solange
# niemand explizit cortex_set_label_smoothing() aufruft. Wirkt NUR auf die
# drei One-Hot-Gradient-Stellen (logits[v] - target), die auch der
# Loss-Masking-Fix vom 2026-08-12 angefasst hat - ersetzt dort das harte Ziel
# 1.0/0.0 durch das geglaettete Ziel (1-eps)/eps/(vocabSize-1). Die
# geloggte/getrackte total_loss (reines -ln(p_korrekt)) bleibt bewusst
# unveraendert, damit Held-out-Loss-Werte mit bisherigen Laeufen vergleichbar
# bleiben - Standardpraxis (z.B. fairseq trennt "loss" und "nll_loss"
# genauso), nur das Gradientensignal wird geglaettet.
# ==============================================================================
var global_label_smoothing: cfloat = 0.0

proc cortex_set_label_smoothing*(eps: cfloat) {.ffi.} =
  # Obergrenze 0.9 analog Dropout - bei eps=1.0 waere das Zielsignal fuer die
  # korrekte Klasse identisch zur Gleichverteilung, kein sinnvolles Training mehr.
  global_label_smoothing = (if eps < 0.0: 0.0 elif eps > 0.9: 0.9 else: eps)

proc cortex_get_label_smoothing*(): cfloat {.ffi.} =
  global_label_smoothing

# Geglaettetes Ziel fuer Klasse v gegen das wahre Label tgid. Bei eps=0.0
# identisch zum bisherigen harten One-Hot-Ziel (exaktes No-Op).
proc smoothedTarget(v, tgid: int, vocabSize: cint): cfloat {.inline.} =
  if global_label_smoothing <= 0.0:
    return (if v == tgid: 1.0.cfloat else: 0.0.cfloat)
  let eps = global_label_smoothing
  if v == tgid:
    (1.0'f32 - eps)
  else:
    eps / (vocabSize - 1).cfloat

# ==============================================================================
# LR-BOOST fuer frisch Net2Net-expandierte Parameter (siehe docs/ROADMAP.md,
# Abschnitt "Differenzierte Learning-Rate fuer frisch Net2Net-expandierte
# Parameter"). upscale_matrix_stream() in royal_distillery.php uebernimmt aus
# der Quelle nur den [0:old_dim)x[0:old_dim)-Block unveraendert, jede Zeile/
# Spalte darueber hinaus ist frisches Rauschen um Null - bei einer
# Dim-Verdopplung sind das bis zu 75% der Zellen einer quadratischen Matrix.
# Ohne Differenzierung lernen diese mit derselben, fuer die Gesamtheit moderat
# gewaehlten Rate wie die bereits konvergierten Altgewichte.
# growth_old_dim < 0 = kein Boundary gesetzt = kein Boost, identisch zum
# bisherigen Verhalten (Default, z.B. fuer Genesis-Modelle ohne
# Expansionshistorie).
# ==============================================================================
var growth_old_dim: cint = -1
var growth_old_ffn: cint = -1
var growth_old_layers: cint = -1
var growth_boost: cfloat = 1.0

proc cortex_set_growth_boundary*(old_dim, old_ffn, old_layers: cint, boost: cfloat) {.ffi.} =
  growth_old_dim = old_dim
  growth_old_ffn = old_ffn
  growth_old_layers = old_layers
  growth_boost = boost

proc elemBoost(idx, cols, boundR, boundC: cint): cfloat {.inline.} =
  if cols <= 0: return 1.0
  let row = idx div cols
  let col = idx mod cols
  if row >= boundR or col >= boundC: return growth_boost
  return 1.0

# ==============================================================================
# V11.6 PROFILER - keine Theorie mehr, nur Messwerte.
# Buckets: 0 matVecMul GPU | 1 matVecMul CPU | 2 backwardAcc GPU
#          3 backwardAcc CPU | 4 Attention GPU | 5 Clip+Adam | 6 Step gesamt
# "Rest" = Bucket 6 minus 0..5 = Normen, Softmax, SwiGLU, Embedding-Scatter.
# ==============================================================================
var prof: array[8, float64]

proc cortex_profile_reset*() {.ffi.} =
  for i in 0 ..< 8: prof[i] = 0.0

proc cortex_profile_get*(idx: cint): cfloat {.ffi.} =
  if idx < 0 or idx > 7: return -1.0
  return (prof[idx] * 1000.0).cfloat

{.emit: """
#ifdef _OPENMP
#include <omp.h>
int cortex_omp_probe(void) { return omp_get_max_threads(); }
#else
int cortex_omp_probe(void) { return 0; }
#endif
""".}

proc cortex_omp_probe(): cint {.importc, nodecl.}

# Gibt die Zahl der OpenMP-Threads zurueck, 0 = ohne -fopenmp gebaut.
# 0 heisst: alle ||-Schleifen laufen seriell, der Umbau ist wirkungslos.
proc cortex_get_threads*(): cint {.ffi.} =
  cortex_omp_probe()

# ==============================================================================
# Update 2026-08-08: BACKEND-DISPATCH-SCHICHT
# Duenne 1:1-Wrapper um ocl_*/cuda_* - jede bestehende Aufrufstelle im Rest
# dieser Datei ruft ab jetzt gpuX() statt ocl_X() direkt auf, verzweigt aber
# NICHT selbst auf den Backend-Typ (das bleibt hier zentral gebuendelt, damit
# nicht an 15 Stellen im File dieselbe case-Unterscheidung wiederholt wird).
# Die umgebende gpu_initialized/useGpuMat-Gate-Logik an den Aufrufstellen
# bleibt unveraendert - nur der innere Funktionsname aendert sich.
# ==============================================================================

proc gpuReleaseBuffer(p: pointer) {.inline.} =
  case gpu_backend
  of gbCuda: cuda_release_buffer(p)
  of gbOpenCL: ocl_release_buffer(p)
  of gbNone: discard

proc gpuSyncToHost(p: pointer) {.inline.} =
  case gpu_backend
  of gbCuda: cuda_sync_to_host(p)
  of gbOpenCL: ocl_sync_to_host(p)
  of gbNone: discard

# AUDIT-FIX 2026-08-18 (docs/ROADMAP.md, Engine-Audit Befund 6): das Gegenstueck
# zu gpuSyncToHost fehlte in dieser Dispatch-Schicht komplett - obwohl BEIDE
# Backends es bereitstellen (ocl_sync_to_device/cuda_sync_to_device) und
# cortex_cuda.cu in cuda_get_buffer() ausdruecklich darauf hinweist: ein
# Host-Puffer wird NUR beim allerersten Zugriff hochgeladen, wer die Host-Seite
# danach von aussen aendert (cortex_load_layer, Net2Net-Upscale, frisches Laden
# von Platte) MUSS explizit nachschieben. Aufgerufen wird das jetzt in
# cortex_load_layer/cortex_load_adam_layer, jeweils nur wenn bereits ein
# GPU-Backend aktiv ist - im normalen Ablauf (train_worker.php laedt das Modell
# VOR init_gpu()) bleibt das ein exaktes No-Op, es legt also keine VRAM-Puffer
# an, die es vorher nicht gab.
proc gpuSyncToDevice(p: pointer, bytes: csize_t) {.inline.} =
  case gpu_backend
  of gbCuda: cuda_sync_to_device(p, bytes)
  of gbOpenCL: ocl_sync_to_device(p, bytes)
  of gbNone: discard

proc gpuZeroBuffer(p: ptr cfloat, bytes: csize_t) {.inline.} =
  case gpu_backend
  of gbCuda: cuda_zero_buffer(p, bytes)
  of gbOpenCL: ocl_zero_buffer(p, bytes)
  of gbNone: discard

proc gpuMatVecMul(m, v, o: ptr cfloat, rows, cols: cint) {.inline.} =
  case gpu_backend
  of gbCuda: cuda_matVecMul(m, v, o, rows, cols)
  of gbOpenCL: ocl_matVecMul(m, v, o, rows, cols)
  of gbNone: discard

proc gpuMatVecMulT(m, d_out, d_in: ptr cfloat, rows, cols, accumulate: cint) {.inline.} =
  case gpu_backend
  of gbCuda: cuda_matVecMulT(m, d_out, d_in, rows, cols, accumulate)
  of gbOpenCL: ocl_matVecMulT(m, d_out, d_in, rows, cols, accumulate)
  of gbNone: discard

proc gpuGradAccum(gm, d_out, i_v: ptr cfloat, rows, cols: cint) {.inline.} =
  case gpu_backend
  of gbCuda: cuda_gradAccum(gm, d_out, i_v, rows, cols)
  of gbOpenCL: ocl_gradAccum(gm, d_out, i_v, rows, cols)
  of gbNone: discard

proc gpuMatMulBatched(m, i_v, o_v: ptr cfloat, rows, cols, batch, maxBatch: cint) {.inline.} =
  # Update 2026-08-09: cuda_matMulBatched() cached d_in/d_out in cuda_get_buffer()
  # unter dem Host-Pointer MIT fester Groesse (siehe cortex_cuda.cu) - anders als
  # matMulTBatched/gradAccumBatched/attentionForwardBatched, die fuer variable
  # Batchgroessen den Ephemeral-Grow-Pattern nutzen. Vorher wurde hier `batch`
  # (die tatsaechliche, PRO AUFRUF variierende Fenstergroesse - siehe `act` in
  # cortex_train_step) fuer BEIDE Parameter (batch_size UND max_batch) durch-
  # gereicht. Sobald ein kuerzeres Restfenster (letztes, nicht volles ctx-
  # Fenster einer Chunk-Datei) denselben Host-Pointer mit einer ANDEREN Groesse
  # registrierte als zuvor, schlug cuda_get_buffer() mit "Buffer Size Mismatch"
  # fehl und lieferte NULL zurueck - die Gewichts-Aktualisierung fuer genau
  # diesen Schritt wurde daraufhin still uebersprungen (kein Crash, siehe Guard
  # in cuda_matMulBatched: "if (!d_M || !d_in || !d_out) return;"). Bis zu
  # diesem Fund NIE in einem echten Lauf sichtbar geworden, weil CUDA in der
  # Produktion mangels $ORIGIN-Rpath nie tatsaechlich griff (siehe Roadmap-
  # Update) - erst der heutige echte End-to-End-Test hat es aufgedeckt. Fix:
  # `maxBatch` ist die FESTE Allokationsgroesse der beteiligten Host-Puffer
  # (immer state.ctxSize, siehe Aufrufer unten) - `batch` bleibt die echte,
  # variierende Elementzahl fuer die eigentliche Berechnung.
  case gpu_backend
  of gbCuda: cuda_matMulBatched(m, i_v, o_v, rows, cols, batch, maxBatch)
  of gbOpenCL: ocl_matMulBatched(m, i_v, o_v, rows, cols, batch)
  of gbNone: discard

proc gpuMatMulTBatched(m, d_out, d_in: ptr cfloat, rows, cols, batch, accumulate: cint) {.inline.} =
  case gpu_backend
  of gbCuda: cuda_matMulTBatched(m, d_out, d_in, rows, cols, batch, accumulate)
  of gbOpenCL: ocl_matMulTBatched(m, d_out, d_in, rows, cols, batch, accumulate)
  of gbNone: discard

proc gpuGradAccumBatched(gm, d_out, i_v: ptr cfloat, rows, cols, batch: cint) {.inline.} =
  case gpu_backend
  of gbCuda: cuda_gradAccumBatched(gm, d_out, i_v, rows, cols, batch)
  of gbOpenCL: ocl_gradAccumBatched(gm, d_out, i_v, rows, cols, batch)
  of gbNone: discard

proc gpuAttentionForward(q, k, v, o: ptr cfloat, pos, numHeads, headSize, dim, ctx: cint) {.inline.} =
  case gpu_backend
  of gbCuda: cuda_attentionForward(q, k, v, o, pos, numHeads, headSize, dim, ctx)
  of gbOpenCL: ocl_attentionForward(q, k, v, o, pos, numHeads, headSize, dim, ctx)
  of gbNone: discard

proc gpuAttentionBackward(q, k, v, d_att_out, d_q, d_k, d_v: ptr cfloat, pos, numHeads, headSize, dim, ctx: cint) {.inline.} =
  case gpu_backend
  of gbCuda: cuda_attentionBackward(q, k, v, d_att_out, d_q, d_k, d_v, pos, numHeads, headSize, dim, ctx)
  of gbOpenCL: ocl_attentionBackward(q, k, v, d_att_out, d_q, d_k, d_v, pos, numHeads, headSize, dim, ctx)
  of gbNone: discard

proc gpuAttentionForwardBatched(q_all, k, v, out_all: ptr cfloat, act, numHeads, headSize, dim, ctx: cint) {.inline.} =
  case gpu_backend
  of gbCuda: cuda_attentionForwardBatched(q_all, k, v, out_all, act, numHeads, headSize, dim, ctx)
  of gbOpenCL: ocl_attentionForwardBatched(q_all, k, v, out_all, act, numHeads, headSize, dim, ctx)
  of gbNone: discard

proc gpuAttentionBackwardBatched(q_all, k, v, d_att_out_all, d_q_all, d_k, d_v: ptr cfloat, act, numHeads, headSize, dim, ctx: cint) {.inline.} =
  case gpu_backend
  of gbCuda: cuda_attentionBackwardBatched(q_all, k, v, d_att_out_all, d_q_all, d_k, d_v, act, numHeads, headSize, dim, ctx)
  of gbOpenCL: ocl_attentionBackwardBatched(q_all, k, v, d_att_out_all, d_q_all, d_k, d_v, act, numHeads, headSize, dim, ctx)
  of gbNone: discard

proc gpuAdamStep(w, m, v, g: ptr cfloat, lr, b1, b2, b1t, b2t, scale: cfloat, size, is_sparse, cols, bound_r, bound_c: cint, boost, wd: cfloat) {.inline.} =
  case gpu_backend
  of gbCuda: cuda_adamStep(w, m, v, g, lr, b1, b2, b1t, b2t, scale, size, is_sparse, cols, bound_r, bound_c, boost, wd)
  of gbOpenCL: ocl_adamStep(w, m, v, g, lr, b1, b2, b1t, b2t, scale, size, is_sparse, cols, bound_r, bound_c, boost, wd)
  of gbNone: discard

proc useGpuMat(m: Matrix): bool {.inline.} =
  gpu_initialized and (m.rows.int * m.cols.int) >= gpu_min_elems

proc gradBytes(m: Matrix): csize_t {.inline.} =
  cast[csize_t](m.rows.uint64 * m.cols.uint64 * sizeof(cfloat).uint64)

proc allocMatrix(r, c: cint): Matrix =
  result.rows = r
  result.cols = c
  result.data = cast[ptr FloatArray](allocShared0(cast[int](r.uint64 * c.uint64 * sizeof(cfloat).uint64)))

proc freeMatrix(m: var Matrix) =
  if not m.data.isNil:
    if gpu_initialized: gpuReleaseBuffer(cast[pointer](m.data))
    deallocShared(m.data)
  m.data = nil

# --- MATH CORE ---
# Update 2026-08-08 (docs/ROADMAP.md, "RoPE-Basis an wachsende Kontextlaenge
# anpassen"): NTK-aware Basis-Skalierung statt fest verdrahteter 10000.
# ROPE_REF_CTX = Ctx der kleinsten Stufe der Evolutionsleiter (Drohne, 512) -
# dort ist ratio=1 und die Basis bleibt exakt beim klassischen Default (keine
# Aenderung fuer bereits trainierte Drohnen-Checkpoints). Fuer groessere Ctx
# (bis 8192 bei DemiGoddess/Goddess, das 16-Fache) skaliert die Basis
# proportional mit, damit die Rotationsfrequenzen nicht auf einen viel
# laengeren Kontext extrapolieren muessen, als sie kalibriert wurden -
# dieselbe Technik wie NTK-aware-Scaling/YaRN bei Long-Context-Modellen:
#   base_neu = base_alt * (ctx / ROPE_REF_CTX) ^ (head_size / (head_size - 2))
const ROPE_BASE = 10000.0
const ROPE_REF_CTX = 512.0

proc ropeBase(head_size, ctx: cint): float32 {.inline.} =
  if head_size <= 2: return ROPE_BASE.float32
  let ratio = ctx.float64 / ROPE_REF_CTX
  let exponent = head_size.float64 / (head_size.float64 - 2.0)
  (ROPE_BASE * pow(ratio, exponent)).float32

proc applyRoPE(vec: ptr FloatArray, dim, head_size, ctx, pos: cint, backward: bool = false) =
  let num_heads = dim div head_size
  let sign: cfloat = if backward: -1.0 else: 1.0
  let base = ropeBase(head_size, ctx)
  for h in 0 ..< num_heads:
    let h_off = h * head_size
    for i in countup(0, head_size - 1, 2):
      let freq = 1.0 / pow(base, i.float32 / head_size.float32)
      let theta = pos.float32 * freq
      let cos_t = cos(theta).cfloat
      let sin_t = sign * sin(theta).cfloat
      let v0 = vec[h_off + i]
      let v1 = vec[h_off + i + 1]
      vec[h_off + i] = v0 * cos_t - v1 * sin_t
      vec[h_off + i + 1] = v0 * sin_t + v1 * cos_t

proc matVecMul*(m: Matrix, i_v, o_v: ptr FloatArray) =
  let t0 = epochTime()
  if useGpuMat(m):
    gpuMatVecMul(cast[ptr cfloat](m.data), cast[ptr cfloat](i_v), cast[ptr cfloat](o_v), m.rows, m.cols)
    prof[0] += epochTime() - t0
  else:
    let rows = m.rows.int
    let cols = m.cols.int
    # V11.7: jede Zeile schreibt ausschliesslich o_v[r] -> race-frei parallel.
    for r in 0 || (rows - 1):
      var s: cfloat = 0.0
      let off = r * cols
      for c in 0 ..< cols:
        s += m.data[off + c] * i_v[c]
      o_v[r] = s
    prof[1] += epochTime() - t0

proc matVecMulBackwardAcc*(m, gm: Matrix, i_v, d_out, d_in: ptr FloatArray, accumulate: bool = false) =
  # V11.4: Backward-Offload. gm bleibt persistent im VRAM und akkumuliert ueber
  # alle Token - pro Token wandern nur die beiden kleinen Vektoren.
  let t0 = epochTime()
  if useGpuMat(m):
    if not d_in.isNil:
      gpuMatVecMulT(cast[ptr cfloat](m.data), cast[ptr cfloat](d_out),
                     cast[ptr cfloat](d_in), m.rows, m.cols,
                     (if accumulate: 1.cint else: 0.cint))
    gpuGradAccum(cast[ptr cfloat](gm.data), cast[ptr cfloat](d_out),
                  cast[ptr cfloat](i_v), m.rows, m.cols)
    prof[2] += epochTime() - t0
    return

  let rowsB = m.rows.int
  let colsB = m.cols.int

  if not d_in.isNil:
    # V11.8: zurueck auf die originale zeilenweise Reihenfolge. Die getauschte
    # Variante aus V11.7 war parallelisierbar, las m aber spaltenweise - bei
    # 16384x256 (w_out) zerlegt das die Cache-Lines. Bleibt vorerst seriell.
    if not accumulate:
      for c in 0 ..< colsB:
        d_in[c] = 0.0
    for r in 0 ..< rowsB:
      let do_v = d_out[r]
      let off = r * colsB
      for c in 0 ..< colsB:
        d_in[c] += m.data[off + c] * do_v
        
  # V11.7: Zeile r schreibt ausschliesslich gm-Zeile r -> race-frei parallel.
  for r in 0 || (rowsB - 1):
    let do_v = d_out[r]
    let off = r * colsB
    for c in 0 ..< colsB:
      gm.data[off + c] += do_v * i_v[c]
  prof[3] += epochTime() - t0

# ==============================================================================
# V11.9: GEBUENDELTE MATRIX-OPERATIONEN UEBER DIE POSITIONSACHSE
# Layout durchgehend row-major: IN[p*cols + c], OUT[p*rows + r].
# Bewusst hinter einem Flag: der alte Pfad bleibt Zeile fuer Zeile erhalten,
# damit sich beide im selben Binary gegeneinander messen lassen.
# ==============================================================================
var batched_wout: cint = 1   # V11.11: Standard AN (verifiziert schneller auf CPU und GPU)

proc cortex_set_batched_wout*(enabled: cint) {.ffi.} =
  batched_wout = enabled

proc cortex_get_batched_wout*(): cint {.ffi.} =
  batched_wout

# ==============================================================================
# WEIGHT TYING (2026-08-27) - Embedding und w_out teilen sich eine Matrix.
# Herleitung und Risikoanalyse: docs/STUDIE_WEIGHT_TYING.md
#
# WARUM: bei den kleinen Stufen sind die beiden vocab*dim-Tabellen der groesste
# Posten ueberhaupt - Drohne 84,2 %, Prinzessin 64,0 % aller Parameter. Geteilt
# passt bei gleicher Gesamtgroesse eine Drohne mit 7 statt 2 Layern und eine
# Prinzessin mit 6 statt 3.
#
# WIE - und warum NICHT einfach alles aliasen:
#   Gewichte (w_out/m/v) werden auf die Embedding-Matrizen gezeigt, teilen sich
#   also wirklich den Speicher. Der GRADIENT g_w_out bleibt dagegen ein eigener
#   Puffer und wird erst nach pullGrad() explizit auf g_embeddings addiert.
#   Grund: g_embeddings entsteht auf der CPU (Scatter, Zeile ~1764), g_w_out auf
#   der GPU (gradAccumBatchedD). Waeren auch die Gradienten aliast, wuerde
#   pullGrad() - das NACH dem Scatter laeuft - die CPU-Beitraege mit dem
#   GPU-Stand ueberschreiben. Das waere ein stiller Fehler: Training laeuft
#   weiter, Zahlen sehen plausibel aus, ein Teil des Gradienten fehlt.
#   Dieselbe Fehlerklasse hat den INT8-Fork mehrfach aufgehalten.
#
# Der Flag muss VOR cortex_init/cortex_load_stream gesetzt sein - er entscheidet
# ueber die Allokation. Default 0 = bisheriges Verhalten, Bit fuer Bit.
#
# Dateiformat bleibt unveraendert: die FFI-Extraktion liest s.w_out.data, was im
# gebundenen Fall auf die Embeddings zeigt. Ein gebundenes Modell schreibt also
# einen w_out-Block, der eine Kopie der Embeddings ist - redundant auf Platte,
# dafuer laedt es auch in einer ungebundenen Engine korrekt. Der Platzgewinn auf
# Platte waere ein zweiter, separater Schritt (Formatwechsel, siehe Studie).
# ==============================================================================
var weight_tying: cint = 0

proc cortex_set_weight_tying*(enabled: cint) {.ffi.} =
  weight_tying = enabled

proc cortex_get_weight_tying*(): cint {.ffi.} =
  weight_tying

# Nur fuer Diagnose: laesst applyGlobalAdamUpdate() aus, damit der Gradient nach
# dem Schritt lesbar bleibt (adamStep konsumiert ihn sonst, siehe Kommentar bei
# cortex_extract_grad). Default 0 - im Produktivpfad nie gesetzt.
var skip_adam: cint = 0

proc cortex_set_skip_adam*(enabled: cint) {.ffi.} =
  skip_adam = enabled

# OUT[p*rows + r] = sum_c M[r*cols + c] * IN[p*cols + c]
# maxBatch: feste Allokationsgroesse von i_all/o_all (state.ctxSize beim
# Aufrufer) - siehe Kommentar in gpuMatMulBatched oben. Unabhaengig von batch,
# das pro Aufruf mit der tatsaechlichen Fenstergroesse variieren darf.
proc matMulBatchedD(m: Matrix, i_all, o_all: ptr FloatArray, batch: int, maxBatch: int) =
  let t0 = epochTime()
  if useGpuMat(m):
    gpuMatMulBatched(cast[ptr cfloat](m.data), cast[ptr cfloat](i_all),
                      cast[ptr cfloat](o_all), m.rows, m.cols, batch.cint, maxBatch.cint)
    prof[0] += epochTime() - t0
    return
  let rows = m.rows.int
  let cols = m.cols.int
  for p in 0 ..< batch:
    let ib = p * cols
    let ob = p * rows
    for r in 0 || (rows - 1):
      var s: cfloat = 0.0
      let off = r * cols
      for c in 0 ..< cols:
        s += m.data[off + c] * i_all[ib + c]
      o_all[ob + r] = s
  prof[1] += epochTime() - t0

# D_IN[p*cols + c] = sum_r M[r*cols + c] * D_OUT[p*rows + r]
# AUDIT: accumulate-Parameter ergaenzt (Standard false, bestehende w_out-Aufrufe
# unveraendert) - noetig, weil w_q/w_k/w_v alle in denselben d_norm-Gradienten
# summieren (exakt wie matVecMulBackwardAcc es pro Position schon immer tat).
proc matMulTBatchedD(m: Matrix, d_out_all, d_in_all: ptr FloatArray, batch: int, accumulate: bool = false) =
  let t0 = epochTime()
  if useGpuMat(m):
    gpuMatMulTBatched(cast[ptr cfloat](m.data), cast[ptr cfloat](d_out_all),
                       cast[ptr cfloat](d_in_all), m.rows, m.cols, batch.cint, (if accumulate: 1.cint else: 0.cint))
    prof[2] += epochTime() - t0
    return
  let rows = m.rows.int
  let cols = m.cols.int
  # V11.10: Positionen sind unabhaengig -> parallel ueber p. Innen exakt die
  # zeilenweise Ordnung des alten Pfades (m wird sequenziell gestreamt).
  for p in 0 || (batch - 1):
    let ob = p * rows
    let ib = p * cols
    if not accumulate:
      for c in 0 ..< cols:
        d_in_all[ib + c] = 0.0
    for r in 0 ..< rows:
      let do_v = d_out_all[ob + r]
      let off = r * cols
      for c in 0 ..< cols:
        d_in_all[ib + c] += m.data[off + c] * do_v
  prof[3] += epochTime() - t0

# gm[r*cols + c] += sum_p D_OUT[p*rows + r] * I_V[p*cols + c]
# Ein Thread pro Zeile r -> jeder schreibt nur seine eigene gm-Zeile, race-frei.
proc gradAccumBatchedD(m, gm: Matrix, d_out_all, i_all: ptr FloatArray, batch: int) =
  let t0 = epochTime()
  if useGpuMat(m):
    gpuGradAccumBatched(cast[ptr cfloat](gm.data), cast[ptr cfloat](d_out_all),
                         cast[ptr cfloat](i_all), m.rows, m.cols, batch.cint)
    prof[2] += epochTime() - t0
    return
  let rows = m.rows.int
  let cols = m.cols.int
  # V11.10: r aussen -> jeder Thread besitzt genau eine gm-Zeile (race-frei,
  # bleibt im L1). p in der Mitte, c innen streamend ueber i_all. Der einzige
  # strided Zugriff (d_out_all) passiert nur rows*batch mal statt
  # rows*cols*batch mal - Faktor 128 weniger bei cols=128.
  for r in 0 || (rows - 1):
    let off = r * cols
    for p in 0 ..< batch:
      let do_v = d_out_all[p * rows + r]
      let ib = p * cols
      for c in 0 ..< cols:
        gm.data[off + c] += do_v * i_all[ib + c]
  prof[3] += epochTime() - t0

# ==============================================================================
# AUDIT: GEBUENDELTE ATTENTION - analog zu V11.9 (batched_wout), aber fuer den
# eigentlich teuersten GPU-Pfad. Ursache der urspruenglichen 20x-Verlangsamung
# auf schwachen/treiber-limitierten GPUs: 'act' einzelne Dispatches pro Layer,
# jeder mit vollem Kernel-Launch + Transfer-Overhead inkl. eines BLOCKIERENDEN
# Reads. Erfordert layer-major statt position-major Reihenfolge im aufrufenden
# Trainings-Loop (siehe cortex_train_step, batchedAttn-Zweig): Q/K/V muessen
# fuer ALLE Positionen eines Layers feststehen, bevor Attention fuer eine davon
# berechnet wird.
# Standard AN: verifiziert per diag_batched.php-Methodik (Loss-Vergleich CPU/GPU,
# alt/neu, 1 und 8 Heads, 20 Schritte) - bitidentisch bis auf Float32-Rauschen,
# 11x Speedup auf dim=512 (49,6s -> 4,6s Step-Zeit), Attention-Backward-Anteil
# allein 43x (32,8s -> 0,76s). Analog zu batched_wout's Standard AN.
# ==============================================================================
var batched_attention: cint = 1

proc cortex_set_batched_attention*(enabled: cint) {.ffi.} =
  batched_attention = enabled

proc cortex_get_batched_attention*(): cint {.ffi.} =
  batched_attention

# Liest state.act_q[l] (fuer alle 'act' Positionen bereits gefuellt von Sub-Pass 1
# im Aufrufer), schreibt state.act_att_out[l]. CPU-Zweig ist Zeile-fuer-Zeile
# identisch zur alten Inline-Mathematik im Positions-Loop.
proc attentionForwardBatchedD(state: ptr CortexState, l, act: cint) =
  let t0 = epochTime()
  let dim = state.hiddenDim
  let numHeads = state.numHeads
  let headSize = state.headSize
  let q_all = cast[ptr FloatArray](state.act_q[l].data)
  let out_all = state.act_att_out[l]

  if gpu_initialized and batched_attention != 0:
    gpuAttentionForwardBatched(cast[ptr cfloat](q_all), cast[ptr cfloat](state.k_cache[l].data),
                                 cast[ptr cfloat](state.v_cache[l].data), cast[ptr cfloat](out_all),
                                 act, numHeads, headSize, dim, state.ctxSize)
    prof[0] += epochTime() - t0
    return

  for p in 0 ..< act:
    let off_dim = p * dim
    let q_p = cast[ptr FloatArray](addr q_all[off_dim])
    let out_p = cast[ptr FloatArray](addr out_all[off_dim])
    for d in 0 ..< dim: out_p[d] = 0.0
    for h in 0 ..< numHeads:
      let h_off = h * headSize
      var max_s: cfloat = -9999.0
      for t in 0 .. p:
        let tk = cast[ptr FloatArray](addr state.k_cache[l].data[t * dim])
        var sc: cfloat = 0.0
        for d in 0 ..< headSize: sc += q_p[h_off + d] * tk[h_off + d]
        sc /= sqrt(headSize.float32)
        state.scores_buf[t] = sc
        if sc > max_s: max_s = sc
      var sum_e: cfloat = 0.0
      for t in 0 .. p:
        state.scores_buf[t] = exp(state.scores_buf[t] - max_s)
        sum_e += state.scores_buf[t]
      for t in 0 .. p:
        let w = state.scores_buf[t] / sum_e
        let tv = cast[ptr FloatArray](addr state.v_cache[l].data[t * dim])
        for d in 0 ..< headSize: out_p[h_off + d] += tv[h_off + d] * w
  prof[1] += epochTime() - t0

# Liest state.act_q[l], state.d_att_out_all (fuer alle Positionen von Sub-Pass 1
# im Aufrufer gefuellt). Schreibt state.d_q_all sowie state.d_k_cache[l]/
# state.d_v_cache[l] (bereits am Funktionsanfang von cortex_train_step genullt).
# CPU-Zweig laeuft bewusst weiterhin als Countdown ueber die Positionen - das
# ist keine GPU-Notloesung, sondern noetig, damit die += Akkumulation in
# d_k_cache/d_v_cache exakt wie im alten Pfad funktioniert (siehe Kernel-
# Kommentar zu attentionBackwardBatchedKey fuer die GPU-seitige Race-Analyse).
proc attentionBackwardBatchedD(state: ptr CortexState, l, act: cint) =
  let t0 = epochTime()
  let dim = state.hiddenDim
  let numHeads = state.numHeads
  let headSize = state.headSize
  let q_all = cast[ptr FloatArray](state.act_q[l].data)
  let datt_all = state.d_att_out_all
  let dq_all = state.d_q_all

  if gpu_initialized and batched_attention != 0:
    gpuAttentionBackwardBatched(cast[ptr cfloat](q_all), cast[ptr cfloat](state.k_cache[l].data),
                                  cast[ptr cfloat](state.v_cache[l].data), cast[ptr cfloat](datt_all),
                                  cast[ptr cfloat](dq_all), cast[ptr cfloat](state.d_k_cache[l].data),
                                  cast[ptr cfloat](state.d_v_cache[l].data), act, numHeads, headSize,
                                  dim, state.ctxSize)
    prof[2] += epochTime() - t0
    return

  let scale = 1.0 / sqrt(headSize.float32)
  for p in countdown(act - 1, 0):
    let off_dim = p * dim
    let q_p = cast[ptr FloatArray](addr q_all[off_dim])
    let d_att_out_p = cast[ptr FloatArray](addr datt_all[off_dim])
    let d_q_p = cast[ptr FloatArray](addr dq_all[off_dim])
    for d in 0 ..< dim: d_q_p[d] = 0.0
    for h in 0 ..< numHeads:
      let h_off = h * headSize
      var max_s: cfloat = -9999.0
      for t in 0 .. p:
        let k_t = cast[ptr FloatArray](addr state.k_cache[l].data[t * dim])
        var sc: cfloat = 0.0
        for d in 0 ..< headSize: sc += q_p[h_off + d] * k_t[h_off + d]
        sc *= scale
        state.scores_buf[t] = sc
        if sc > max_s: max_s = sc
      var sum_e: cfloat = 0.0
      for t in 0 .. p:
        state.scores_buf[t] = exp(state.scores_buf[t] - max_s)
        sum_e += state.scores_buf[t]
      var sum_w_dw: cfloat = 0.0
      for t in 0 .. p:
        let w = state.scores_buf[t] / sum_e
        state.scores_buf[t] = w
        let v_t = cast[ptr FloatArray](addr state.v_cache[l].data[t * dim])
        var dw: cfloat = 0.0
        for d in 0 ..< headSize: dw += d_att_out_p[h_off + d] * v_t[h_off + d]
        let d_v_t = cast[ptr FloatArray](addr state.d_v_cache[l].data[t * dim])
        for d in 0 ..< headSize: d_v_t[h_off + d] += d_att_out_p[h_off + d] * w
        state.d_w_buf[t] = dw
        sum_w_dw += w * dw
      for t in 0 .. p:
        let w = state.scores_buf[t]
        let ds = w * (state.d_w_buf[t] - sum_w_dw) * scale
        let k_t = cast[ptr FloatArray](addr state.k_cache[l].data[t * dim])
        let d_k_t = cast[ptr FloatArray](addr state.d_k_cache[l].data[t * dim])
        for d in 0 ..< headSize:
          d_q_p[h_off + d] += ds * k_t[h_off + d]
          d_k_t[h_off + d] += ds * q_p[h_off + d]
  prof[3] += epochTime() - t0

proc applyRMSNorm(i_v, o_v: ptr FloatArray, dim: cint, w: ptr FloatArray) =
  var ss: cfloat = 0.0
  for d in 0 ..< dim: 
    ss += i_v[d] * i_v[d]
  let inv = 1.0 / sqrt((ss / dim.float32) + 1e-5)
  for d in 0 ..< dim: 
    o_v[d] = i_v[d] * inv * (if w.isNil: 1.0.cfloat else: w[d])

proc applyRMSNormBackwardAcc*(d_out, a_i: ptr FloatArray, dim: cint, w, gw: ptr FloatArray) =
  var ss: cfloat = 0.0
  for d in 0 ..< dim: 
    ss += a_i[d] * a_i[d]
  let inv = 1.0 / sqrt((ss / dim.float32) + 1e-5)
  
  var sum_dy_nx: cfloat = 0.0
  for d in 0 ..< dim: 
    let w_val = if w.isNil: 1.0.cfloat else: w[d]
    sum_dy_nx += d_out[d] * w_val * (a_i[d] * inv)
  let mean_dy_nx = sum_dy_nx / dim.float32

  for d in 0 ..< dim:
    let w_val = if w.isNil: 1.0.cfloat else: w[d]
    let nx = a_i[d] * inv
    if not gw.isNil: 
      gw[d] += d_out[d] * nx
    d_out[d] = (d_out[d] * w_val - nx * mean_dy_nx) * inv

# --- SPARSE ADAM CORE ---
proc adamStep(w, m, v, g: ptr cfloat, lr, b1, b2, b1t, b2t, scale: cfloat, is_sparse: bool = false, boost: cfloat = 1.0, wd: cfloat = 0.0) {.inline.} =
  if scale == 0.0: return
  var grad = g[] / scale
  if is_sparse and abs(grad) < 1e-8: return
  if grad.isNaN or grad > 9999.0 or grad < -9999.0:
    g[] = 0.0
    return

  m[] = b1 * m[] + (1.0 - b1) * grad
  v[] = b2 * v[] + (1.0 - b2) * grad * grad
  let m_hat = m[] / b1t
  let v_hat = v[] / b2t
  w[] -= (lr * boost) * m_hat / (sqrt(v_hat) + 1e-8)
  if wd > 0.0: w[] -= lr * wd * w[]
  g[] = 0.0

proc applyGlobalAdamUpdate(state: ptr CortexState, lr: cfloat, scale: cfloat) =
  # AUDIT-FIX 2026-08-18 (docs/ROADMAP.md, Engine-Audit Befund 7): adam_t wurde
  # bisher als allererstes erhoeht - auch bei scale == 0.0 (Fenster ohne eine
  # einzige trainierbare Position, moeglich bei reinen [USER]-Fenstern in
  # SFT-Chunks). adamStep() kehrt dort sofort zurueck, es findet also gar kein
  # Update statt; der Bias-Correction-Zaehler zaehlte trotzdem mit und lief
  # gegenueber der tatsaechlichen Zahl der Updates davon. Jetzt: kein Schritt,
  # kein Zaehler.
  if scale == 0.0: return
  state.adam_t += 1
  let b1: cfloat = 0.9
  let b2: cfloat = 0.999
  let b1t = 1.0 - pow(b1, state.adam_t.float32)
  let b2t = 1.0 - pow(b2, state.adam_t.float32)

  # Boundary-Werte fuer den LR-Boost: bei inaktivem Growth-Tracking (kein
  # cortex_set_growth_boundary()-Aufruf, growth_old_dim < 0) werden die
  # Grenzen auf die AKTUELLEN Dimensionen gesetzt - row/col koennen die nie
  # erreichen, der Boost triggert also nie. Identisch zum alten Verhalten.
  let boostActive = growth_old_dim >= 0
  let boundDimGlobal = if boostActive: growth_old_dim else: state.hiddenDim

  if gpu_initialized:
    template oclUpM(w, m, v, g, boundR, boundC) = gpuAdamStep(cast[ptr cfloat](w.data), cast[ptr cfloat](m.data), cast[ptr cfloat](v.data), cast[ptr cfloat](g.data), lr, b1, b2, b1t, b2t, scale, w.rows * w.cols, 0.cint, w.cols, boundR, boundC, growth_boost, global_weight_decay)
    template oclUpEmb(w, m, v, g, boundC) = gpuAdamStep(cast[ptr cfloat](w.data), cast[ptr cfloat](m.data), cast[ptr cfloat](v.data), cast[ptr cfloat](g.data), lr, b1, b2, b1t, b2t, scale, w.rows * w.cols, 1.cint, w.cols, w.rows.cint, boundC, growth_boost, global_weight_decay)
    template oclUpA(w, m, v, g, size, boundC) = gpuAdamStep(cast[ptr cfloat](w), cast[ptr cfloat](m), cast[ptr cfloat](v), cast[ptr cfloat](g), lr, b1, b2, b1t, b2t, scale, size, 0.cint, size, 1.cint, boundC, growth_boost, 0.0.cfloat)

    oclUpEmb(state.embeddings, state.m_embeddings, state.v_embeddings, state.g_embeddings, boundDimGlobal)
    # WEIGHT TYING: w_out IST hier die Embedding-Matrix. Ein zweiter Adam-Schritt
    # darauf waere ein doppeltes Update derselben Gewichte mit demselben (bereits
    # zusammengefuehrten) Gradienten.
    if weight_tying == 0:
      oclUpM(state.w_out, state.m_w_out, state.v_w_out, state.g_w_out, state.w_out.rows.cint, boundDimGlobal)
    oclUpA(state.rms_final_weight, state.m_rms_final_weight, state.v_rms_final_weight, state.g_rms_final_weight, state.hiddenDim, boundDimGlobal)

    for l in 0 ..< state.numLayers:
      let isNewLayer = boostActive and (l.cint >= growth_old_layers)
      let boundDim = if not boostActive: state.hiddenDim elif isNewLayer: 0.cint else: growth_old_dim
      let boundFfn = if not boostActive: state.ffnDim elif isNewLayer: 0.cint else: growth_old_ffn

      oclUpA(state.rms_att_weights[l], state.m_rms_att_weights[l], state.v_rms_att_weights[l], state.g_rms_att_weights[l], state.hiddenDim, boundDim)
      oclUpA(state.rms_ffn_weights[l], state.m_rms_ffn_weights[l], state.v_rms_ffn_weights[l], state.g_rms_ffn_weights[l], state.hiddenDim, boundDim)
      oclUpM(state.w_q[l], state.m_w_q[l], state.v_w_q[l], state.g_w_q[l], boundDim, boundDim)
      oclUpM(state.w_k[l], state.m_w_k[l], state.v_w_k[l], state.g_w_k[l], boundDim, boundDim)
      oclUpM(state.w_v[l], state.m_w_v[l], state.v_w_v[l], state.g_w_v[l], boundDim, boundDim)
      oclUpM(state.w_o[l], state.m_w_o[l], state.v_w_o[l], state.g_w_o[l], boundDim, boundDim)
      oclUpM(state.w_up[l], state.m_w_up[l], state.v_w_up[l], state.g_w_up[l], boundFfn, boundDim)
      oclUpM(state.w_down[l], state.m_w_down[l], state.v_w_down[l], state.g_w_down[l], boundDim, boundFfn)
  else:
    template upM(w, m, v, g, boundR, boundC) =
      for i in 0 ..< (w.rows * w.cols): adamStep(addr w.data[i], addr m.data[i], addr v.data[i], addr g.data[i], lr, b1, b2, b1t, b2t, scale, false, elemBoost(i.cint, w.cols, boundR, boundC), global_weight_decay)
    template upEmb(w, m, v, g, boundC) =
      for i in 0 ..< (w.rows * w.cols): adamStep(addr w.data[i], addr m.data[i], addr v.data[i], addr g.data[i], lr, b1, b2, b1t, b2t, scale, true, elemBoost(i.cint, w.cols, w.rows.cint, boundC), global_weight_decay)
    template upA(w, m, v, g, size, boundC) =
      for i in 0 ..< size: adamStep(addr w[i], addr m[i], addr v[i], addr g[i], lr, b1, b2, b1t, b2t, scale, false, elemBoost(i.cint, size, 1.cint, boundC))

    upEmb(state.embeddings, state.m_embeddings, state.v_embeddings, state.g_embeddings, boundDimGlobal)
    # WEIGHT TYING: siehe GPU-Zweig oben - kein zweites Update derselben Matrix.
    if weight_tying == 0:
      upM(state.w_out, state.m_w_out, state.v_w_out, state.g_w_out, state.w_out.rows.cint, boundDimGlobal)
    upA(state.rms_final_weight, state.m_rms_final_weight, state.v_rms_final_weight, state.g_rms_final_weight, state.hiddenDim, boundDimGlobal)

    for l in 0 ..< state.numLayers:
      let isNewLayer = boostActive and (l.cint >= growth_old_layers)
      let boundDim = if not boostActive: state.hiddenDim elif isNewLayer: 0.cint else: growth_old_dim
      let boundFfn = if not boostActive: state.ffnDim elif isNewLayer: 0.cint else: growth_old_ffn

      upA(state.rms_att_weights[l], state.m_rms_att_weights[l], state.v_rms_att_weights[l], state.g_rms_att_weights[l], state.hiddenDim, boundDim)
      upA(state.rms_ffn_weights[l], state.m_rms_ffn_weights[l], state.v_rms_ffn_weights[l], state.g_rms_ffn_weights[l], state.hiddenDim, boundDim)
      upM(state.w_q[l], state.m_w_q[l], state.v_w_q[l], state.g_w_q[l], boundDim, boundDim)
      upM(state.w_k[l], state.m_w_k[l], state.v_w_k[l], state.g_w_k[l], boundDim, boundDim)
      upM(state.w_v[l], state.m_w_v[l], state.v_w_v[l], state.g_w_v[l], boundDim, boundDim)
      upM(state.w_o[l], state.m_w_o[l], state.v_w_o[l], state.g_w_o[l], boundDim, boundDim)
      upM(state.w_up[l], state.m_w_up[l], state.v_w_up[l], state.g_w_up[l], boundFfn, boundDim)
      upM(state.w_down[l], state.m_w_down[l], state.v_w_down[l], state.g_w_down[l], boundDim, boundFfn)

proc calcL2Sq(m: Matrix): cfloat {.inline.} =
  var sum: cfloat = 0.0
  let total = m.rows * m.cols
  for i in 0 ..< total: sum += m.data[i] * m.data[i]
  return sum

proc calcL2Sq(arr: ptr FloatArray, size: cint): cfloat {.inline.} =
  var sum: cfloat = 0.0
  for i in 0 ..< size: sum += arr[i] * arr[i]
  return sum

proc scaleMat(m: Matrix, scale: cfloat) {.inline.} =
  let total = m.rows * m.cols
  for i in 0 ..< total: m.data[i] *= scale

proc scaleArr(arr: ptr FloatArray, size: cint, scale: cfloat) {.inline.} =
  for i in 0 ..< size: arr[i] *= scale

proc zeroMat(m: Matrix) {.inline.} =
  let total = m.rows * m.cols
  for i in 0 ..< total: m.data[i] = 0.0

proc zeroArr(arr: ptr FloatArray, size: cint) {.inline.} =
  for i in 0 ..< size: arr[i] = 0.0

proc clipGlobalGradientNorm(s: ptr CortexState, max_norm: cfloat) =
  var total_sq: cfloat = 0.0
  total_sq += calcL2Sq(s.g_embeddings)
  # WEIGHT TYING: g_w_out wurde oben bereits auf g_embeddings addiert. Hier
  # nochmal mitzuzaehlen waere eine Doppelzaehlung - die globale Norm fiele zu
  # gross aus und das Clipping wuerde zu frueh und zu stark greifen. Nach der
  # Historie dieses Projekts (Clipping-Bug 18.08.) ist das die Stelle, an der so
  # ein Fehler am teuersten waere.
  if weight_tying == 0:
    total_sq += calcL2Sq(s.g_w_out)
  total_sq += calcL2Sq(s.g_rms_final_weight, s.hiddenDim)
  for l in 0 ..< s.numLayers:
    total_sq += calcL2Sq(s.g_w_q[l])
    total_sq += calcL2Sq(s.g_w_k[l])
    total_sq += calcL2Sq(s.g_w_v[l])
    total_sq += calcL2Sq(s.g_w_o[l])
    total_sq += calcL2Sq(s.g_w_up[l])
    total_sq += calcL2Sq(s.g_w_down[l])
    total_sq += calcL2Sq(s.g_rms_att_weights[l], s.hiddenDim)
    total_sq += calcL2Sq(s.g_rms_ffn_weights[l], s.hiddenDim)
  
  let global_norm = sqrt(total_sq)
  last_grad_norm = global_norm

  if global_norm.isNaN or global_norm > max_norm:
    if global_norm.isNaN or global_norm == 0.0:
      zeroMat(s.g_embeddings)
      zeroMat(s.g_w_out)
      zeroArr(s.g_rms_final_weight, s.hiddenDim)
      for l in 0 ..< s.numLayers:
        zeroMat(s.g_w_q[l])
        zeroMat(s.g_w_k[l])
        zeroMat(s.g_w_v[l])
        zeroMat(s.g_w_o[l])
        zeroMat(s.g_w_up[l])
        zeroMat(s.g_w_down[l])
        zeroArr(s.g_rms_att_weights[l], s.hiddenDim)
        zeroArr(s.g_rms_ffn_weights[l], s.hiddenDim)
    else:
      let scale = max_norm / global_norm
      scaleMat(s.g_embeddings, scale)
      # WEIGHT TYING: g_w_out ist nach der Zusammenfuehrung tot - es geht in
      # keinen Adam-Schritt mehr ein und wird zu Beginn des naechsten Schritts
      # genullt. Skalieren waere ein wirkungsloser Durchlauf ueber vocab*dim.
      if weight_tying == 0:
        scaleMat(s.g_w_out, scale)
      scaleArr(s.g_rms_final_weight, s.hiddenDim, scale)
      for l in 0 ..< s.numLayers:
        scaleMat(s.g_w_q[l], scale)
        scaleMat(s.g_w_k[l], scale)
        scaleMat(s.g_w_v[l], scale)
        scaleMat(s.g_w_o[l], scale)
        scaleMat(s.g_w_up[l], scale)
        scaleMat(s.g_w_down[l], scale)
        scaleArr(s.g_rms_att_weights[l], s.hiddenDim, scale)
        scaleArr(s.g_rms_ffn_weights[l], s.hiddenDim, scale)

# ================================================================
# FFI EXPORTS
# ===============================================================

# AUDIT-FIX: ffn Parameter explizit aufnehmen (verhindert Heap-Buffer-Overflow)
proc cortex_init*(vocab, dim, ffn, ctx, layers, heads: cint): ptr CortexState {.ffi.} =
  var s = cast[ptr CortexState](allocShared0(sizeof(CortexState)))
  s.vocabSize = vocab
  s.hiddenDim = dim
  s.ffnDim = ffn
  s.ctxSize = ctx
  s.numLayers = layers
  
  s.numHeads = max(1, min(heads, dim))
  s.headSize = dim div s.numHeads
  if (s.headSize mod 2) != 0: s.headSize -= 1
  if s.headSize < 2: s.headSize = 2
  s.numHeads = dim div s.headSize

  # AUDIT-FIX 2026-08-18 (docs/ROADMAP.md, Engine-Audit Befund 5): headSize wird
  # oben auf eine gerade Zahl abgerundet (RoPE braucht Paare). Ist dim danach
  # kein Vielfaches von headSize, decken die Heads nur numHeads*headSize < dim
  # Kanaele ab - die restlichen werden in der Attention auf 0 gesetzt und nie
  # wieder beschrieben (out_p[d] = 0.0 fuer alle d, danach schreiben nur die
  # Head-Segmente h_off..h_off+headSize). Ergebnis waeren dauerhaft tote Kanaele
  # OHNE jede Meldung. Mit den aktuellen Zweierpotenz-Dims der EVOLUTION_LADDER
  # (128/256/512/1024/2048/4096) plus suggest_num_heads() nicht ausloesbar -
  # deshalb hier bewusst NUR eine Warnung und KEINE stille Korrektur der
  # Geometrie: eine nachtraeglich veraenderte Head-Zahl wuerde die Bedeutung
  # bereits trainierter Attention-Gewichte aendern, genau die Fehlerklasse,
  # gegen die head_config.php mit der .heads-Sidecar schuetzt.
  if s.numHeads * s.headSize != dim:
    stderr.writeLine("[cortex] WARN: dim=", $dim, " ist kein Vielfaches von headSize=",
                     $s.headSize, " (numHeads=", $s.numHeads, ") - die obersten ",
                     $(dim - s.numHeads * s.headSize),
                     " Kanaele sind keinem Head zugeordnet und bleiben dauerhaft 0.")


  s.cachedTokens = 0
  s.adam_t = 0
  
  template initM(m, r, c) = 
    m = allocMatrix(r, c)
  template initA(a, size) = 
    a = cast[ptr FloatArray](allocShared0(cast[int](size.uint64 * sizeof(cfloat).uint64)))
  template initP(p, l, T) = 
    p = cast[T](allocShared0(cast[int](l.uint64 * sizeof(p[0]).uint64)))
  template initHist(p, l, size) = 
    p = cast[ptr FloatPtrArray](allocShared0(cast[int](l.uint64 * sizeof(pointer).uint64)))
    for i in 0 ..< l: 
      p[i] = cast[ptr FloatArray](allocShared0(cast[int](size.uint64 * sizeof(cfloat).uint64)))

  initM(s.embeddings, vocab, dim)
  initM(s.m_embeddings, vocab, dim)
  initM(s.v_embeddings, vocab, dim)
  initM(s.g_embeddings, vocab, dim)
  
  # WEIGHT TYING: Gewichte und Adam-Momente zeigen auf die Embedding-Matrizen,
  # der Gradient bleibt eigenstaendig (Begruendung oben beim Flag).
  if weight_tying != 0:
    s.w_out   = s.embeddings
    s.m_w_out = s.m_embeddings
    s.v_w_out = s.v_embeddings
  else:
    initM(s.w_out, vocab, dim)
    initM(s.m_w_out, vocab, dim)
    initM(s.v_w_out, vocab, dim)
  initM(s.g_w_out, vocab, dim)
  
  initA(s.rms_final_weight, dim)
  initA(s.m_rms_final_weight, dim)
  initA(s.v_rms_final_weight, dim)
  initA(s.g_rms_final_weight, dim)

  initP(s.rms_att_weights, layers, ptr FloatPtrArray)
  initP(s.m_rms_att_weights, layers, ptr FloatPtrArray)
  initP(s.v_rms_att_weights, layers, ptr FloatPtrArray)
  initP(s.g_rms_att_weights, layers, ptr FloatPtrArray)
  
  initP(s.rms_ffn_weights, layers, ptr FloatPtrArray)
  initP(s.m_rms_ffn_weights, layers, ptr FloatPtrArray)
  initP(s.v_rms_ffn_weights, layers, ptr FloatPtrArray)
  initP(s.g_rms_ffn_weights, layers, ptr FloatPtrArray)
  
  initP(s.w_q, layers, ptr MatrixArray)
  initP(s.m_w_q, layers, ptr MatrixArray)
  initP(s.v_w_q, layers, ptr MatrixArray)
  initP(s.g_w_q, layers, ptr MatrixArray)
  
  initP(s.w_k, layers, ptr MatrixArray)
  initP(s.m_w_k, layers, ptr MatrixArray)
  initP(s.v_w_k, layers, ptr MatrixArray)
  initP(s.g_w_k, layers, ptr MatrixArray)
  
  initP(s.w_v, layers, ptr MatrixArray)
  initP(s.m_w_v, layers, ptr MatrixArray)
  initP(s.v_w_v, layers, ptr MatrixArray)
  initP(s.g_w_v, layers, ptr MatrixArray)
  
  initP(s.w_o, layers, ptr MatrixArray)
  initP(s.m_w_o, layers, ptr MatrixArray)
  initP(s.v_w_o, layers, ptr MatrixArray)
  initP(s.g_w_o, layers, ptr MatrixArray)
  
  initP(s.w_up, layers, ptr MatrixArray)
  initP(s.m_w_up, layers, ptr MatrixArray)
  initP(s.v_w_up, layers, ptr MatrixArray)
  initP(s.g_w_up, layers, ptr MatrixArray)
  
  initP(s.w_down, layers, ptr MatrixArray)
  initP(s.m_w_down, layers, ptr MatrixArray)
  initP(s.v_w_down, layers, ptr MatrixArray)
  initP(s.g_w_down, layers, ptr MatrixArray)
  
  initP(s.k_cache, layers, ptr MatrixArray)
  initP(s.v_cache, layers, ptr MatrixArray)
  initP(s.act_q, layers, ptr MatrixArray)
  
  initP(s.d_k_cache, layers, ptr MatrixArray)
  initP(s.d_v_cache, layers, ptr MatrixArray)

  for l in 0 ..< layers:
    initA(s.rms_att_weights[l], dim)
    initA(s.m_rms_att_weights[l], dim)
    initA(s.v_rms_att_weights[l], dim)
    initA(s.g_rms_att_weights[l], dim)
    
    initA(s.rms_ffn_weights[l], dim)
    initA(s.m_rms_ffn_weights[l], dim)
    initA(s.v_rms_ffn_weights[l], dim)
    initA(s.g_rms_ffn_weights[l], dim)
    
    initM(s.w_q[l], dim, dim)
    initM(s.m_w_q[l], dim, dim)
    initM(s.v_w_q[l], dim, dim)
    initM(s.g_w_q[l], dim, dim)
    
    initM(s.w_k[l], dim, dim)
    initM(s.m_w_k[l], dim, dim)
    initM(s.v_w_k[l], dim, dim)
    initM(s.g_w_k[l], dim, dim)
    
    initM(s.w_v[l], dim, dim)
    initM(s.m_w_v[l], dim, dim)
    initM(s.v_w_v[l], dim, dim)
    initM(s.g_w_v[l], dim, dim)
    
    initM(s.w_o[l], dim, dim)
    initM(s.m_w_o[l], dim, dim)
    initM(s.v_w_o[l], dim, dim)
    initM(s.g_w_o[l], dim, dim)
    
    initM(s.w_up[l], s.ffnDim, dim)
    initM(s.m_w_up[l], s.ffnDim, dim)
    initM(s.v_w_up[l], s.ffnDim, dim)
    initM(s.g_w_up[l], s.ffnDim, dim)
    
    initM(s.w_down[l], dim, s.ffnDim)
    initM(s.m_w_down[l], dim, s.ffnDim)
    initM(s.v_w_down[l], dim, s.ffnDim)
    initM(s.g_w_down[l], dim, s.ffnDim)
    
    initM(s.k_cache[l], ctx, dim)
    initM(s.v_cache[l], ctx, dim)
    initM(s.act_q[l], ctx, dim)
    
    initM(s.d_k_cache[l], ctx, dim)
    initM(s.d_v_cache[l], ctx, dim)
    
    for i in 0 ..< dim: 
      s.rms_att_weights[l][i] = 1.0
      s.rms_ffn_weights[l][i] = 1.0

  for i in 0 ..< dim: 
    s.rms_final_weight[i] = 1.0
    
  initM(s.logits_cache, ctx, vocab)

  let ctxDim = ctx.int * dim.int
  let ctxFfn = ctx.int * s.ffnDim.int

  initHist(s.act_hidden, layers, ctxDim)
  initHist(s.act_norm_att, layers, ctxDim)
  initHist(s.act_att_out, layers, ctxDim)
  initHist(s.act_hidden_ffn, layers, ctxDim)
  initHist(s.act_norm_ffn, layers, ctxDim)
  initHist(s.act_ffn_in, layers, ctxFfn)
  initHist(s.act_ffn_out, layers, ctxFfn)
  initHist(s.dropout_mask, layers, ctxFfn)

  initHist(s.j_space_cache, layers, dim)
  
  initA(s.act_hidden_final, ctxDim)
  initA(s.act_norm_final, ctxDim)
  initA(s.d_norm_all, ctxDim)
  initA(s.d_hidden_all, ctxDim)
  initA(s.d_att_out_all, ctxDim)
  initA(s.d_q_all, ctxDim)
  initA(s.mm_hd_scratch, ctxDim)
  initA(s.mm_ffn_scratch, ctxFfn)

  initA(s.hidden_state_buf, dim)
  initA(s.norm_buf, dim)
  initA(s.att_out_buf, dim)
  initA(s.ffn_buf, s.ffnDim)
  initA(s.q_buf, dim)
  
  initA(s.d_logits_buf, vocab)
  initA(s.scores_buf, ctx)
  initA(s.d_w_buf, ctx)
  
  initA(s.d_hidden, dim)
  initA(s.d_norm, dim)
  initA(s.d_att_out, dim)
  initA(s.d_q, dim)
  initA(s.d_ffn, s.ffnDim)
  
  return s

proc getExpectedSize(s: ptr CortexState, t: cint): cint {.inline.} =
  case t:
  of 0, 1: return s.vocabSize * s.hiddenDim
  of 2, 10, 11: return s.hiddenDim
  of 12, 13, 14, 15: return s.hiddenDim * s.hiddenDim
  of 16, 17: return s.ffnDim * s.hiddenDim
  else: return 0

proc getTensorPtr(s: ptr CortexState, l: cint, t: cint, m: bool = false, v: bool = false): ptr cfloat =
  if l >= s.numLayers or l < -1: return nil 
  if l == -1:
    if t == 0: return if m: cast[ptr cfloat](s.m_embeddings.data) elif v: cast[ptr cfloat](s.v_embeddings.data) else: cast[ptr cfloat](s.embeddings.data)
    if t == 1: return if m: cast[ptr cfloat](s.m_w_out.data) elif v: cast[ptr cfloat](s.v_w_out.data) else: cast[ptr cfloat](s.w_out.data)
    if t == 2: return if m: cast[ptr cfloat](s.m_rms_final_weight) elif v: cast[ptr cfloat](s.v_rms_final_weight) else: cast[ptr cfloat](s.rms_final_weight)
  else:
    if t == 10: return if m: cast[ptr cfloat](s.m_rms_att_weights[l]) elif v: cast[ptr cfloat](s.v_rms_att_weights[l]) else: cast[ptr cfloat](s.rms_att_weights[l])
    if t == 11: return if m: cast[ptr cfloat](s.m_rms_ffn_weights[l]) elif v: cast[ptr cfloat](s.v_rms_ffn_weights[l]) else: cast[ptr cfloat](s.rms_ffn_weights[l])
    if t == 12: return if m: cast[ptr cfloat](s.m_w_q[l].data) elif v: cast[ptr cfloat](s.v_w_q[l].data) else: cast[ptr cfloat](s.w_q[l].data)
    if t == 13: return if m: cast[ptr cfloat](s.m_w_k[l].data) elif v: cast[ptr cfloat](s.v_w_k[l].data) else: cast[ptr cfloat](s.w_k[l].data)
    if t == 14: return if m: cast[ptr cfloat](s.m_w_v[l].data) elif v: cast[ptr cfloat](s.v_w_v[l].data) else: cast[ptr cfloat](s.w_v[l].data)
    if t == 15: return if m: cast[ptr cfloat](s.m_w_o[l].data) elif v: cast[ptr cfloat](s.v_w_o[l].data) else: cast[ptr cfloat](s.w_o[l].data)
    if t == 16: return if m: cast[ptr cfloat](s.m_w_up[l].data) elif v: cast[ptr cfloat](s.v_w_up[l].data) else: cast[ptr cfloat](s.w_up[l].data)
    if t == 17: return if m: cast[ptr cfloat](s.m_w_down[l].data) elif v: cast[ptr cfloat](s.v_w_down[l].data) else: cast[ptr cfloat](s.w_down[l].data)
  return nil

# ==============================================================================
# NET2NET GAIN GUARD (V11.3)
# ------------------------------------------------------------------------------
# Ein RMSNorm-Gain von 0 ist kein "aus", sondern ein Sattelpunkt:
#   forward : o_v[d] = i_v[d] * inv * 0 = 0
#   backward: der Kanal liefert exakt 0 Gradient in alle nachgelagerten Spalten
# Der Kanal bleibt damit dauerhaft stumm und ist durch Training nicht mehr
# erreichbar. Genau das passiert, wenn eine Breiten-Expansion die neuen
# Gain-Werte nicht auf 1.0 setzt.
# ==============================================================================

var gain_guard_enabled: cint = 1

proc cortex_set_gain_guard*(enabled: cint) {.ffi.} =
  gain_guard_enabled = enabled

# Sucht einen zusammenhaengenden toten Block AM ENDE eines Gain-Arrays.
# Nur dieses Muster ist die Expansions-Signatur - gestreute kleine Werte
# koennen legitim antrainiert sein und werden bewusst ignoriert.
proc deadTailStart(p: ptr FloatArray, n: cint): cint =
  if p.isNil or n < 8: return -1
  let eps: cfloat = 0.05
  if abs(p[n-1]) >= eps: return -1
  var i = n - 1
  while i >= 0 and abs(p[i]) < eps:
    dec i
  let start = i + 1
  if (n - start) * 8 < n: return -1   # kleiner als 1/8 der Breite -> kein Muster
  return start

proc checkGains(p: ptr cfloat, size: cint, what: string): bool {.discardable.} =
  let arr = cast[ptr FloatArray](p)
  let ds = deadTailStart(arr, size)
  if ds < 0: return false
  stderr.writeLine("[cortex] NET2NET-GUARD: ", what, " hat tote Kanaele [",
                   $ds, "..", $(size-1), "] von ", $size, ".")
  if gain_guard_enabled != 0:
    for i in ds ..< size: arr[i] = 1.0
    stderr.writeLine("[cortex] NET2NET-GUARD: auf 1.0 korrigiert. ",
                     "Die Ursache gehoert trotzdem in den Morph-Code.")
  return true

proc cortex_load_layer*(s: ptr CortexState, layer_idx: cint, tensor_type: cint, raw_weights: ptr cfloat, size: cint) {.ffi.} =
  if size != getExpectedSize(s, tensor_type): return 
  let p = getTensorPtr(s, layer_idx, tensor_type)
  if p.isNil: return
  copyMem(p, raw_weights, size * sizeof(cfloat))

  # AEGIS: greift genau in dem Moment, in dem der Morph die Gewichte reinreicht.
  if tensor_type == 2 or tensor_type == 10 or tensor_type == 11:
    checkGains(p, size, "Gain-Tensor (type " & $tensor_type & ", layer " & $layer_idx & ")")

  # AUDIT-FIX 2026-08-18 (Befund 6, siehe gpuSyncToDevice oben): frisch
  # eingespielte Gewichte muessen die GPU erreichen. Ohne das rechnete ein
  # bereits initialisiertes Backend stumm mit dem Stand weiter, der beim
  # allerersten cuda_get_buffer()/ocl_get_buffer() hochgeladen wurde.
  if gpu_initialized:
    gpuSyncToDevice(cast[pointer](p), cast[csize_t](size.int * sizeof(cfloat)))

proc cortex_extract_layer*(s: ptr CortexState, layer_idx: cint, tensor_type: cint, raw_weights: ptr cfloat, size: cint) {.ffi.} =
  if size != getExpectedSize(s, tensor_type): return
  let p = getTensorPtr(s, layer_idx, tensor_type)
  if not p.isNil: copyMem(raw_weights, p, size * sizeof(cfloat))

# ==============================================================================
# DIAGNOSE-ZUGRIFF AUF DIE GRADIENTEN (2026-08-27)
# ------------------------------------------------------------------------------
# Read-only, ausserhalb des Trainingspfads, kein Aufruf aus der Pipeline.
#
# Gebaut fuer die Weight-Tying-Gegenprobe: nur so laesst sich pruefen, ob der
# gebundene Gradient wirklich die Summe der beiden ungebundenen ist. Vorher war
# der Gradient von PHP aus ueberhaupt nicht einsehbar - jeder Numerik-Audit
# musste ihn indirekt ueber Gewichtsdifferenzen erschliessen, was bei Adam
# (erster Schritt ~ lr*sign(g)) die Betragsinformation verliert.
#
# ZEITPUNKT DES LESENS - WICHTIG: nach cortex_train_step() sind die g_* LEER.
# adamStep() setzt `g[] = 0.0`, sobald es den Wert verbraucht hat (Zeile 15 dort),
# der Gradient wird also konsumiert und nicht nur gelesen. Ein Leseversuch nach
# dem Schritt liefert ueberall Nullen, waehrend cortex_get_last_grad_norm()
# einen plausiblen Wert meldet - eine Kombination, die leicht als kaputte
# Extraktion fehlgedeutet wird (genau so ist es beim Bau dieser Sonde passiert).
#
# Wer den Gradienten sehen will, muss Adam ueberspringen:
# cortex_set_skip_adam(1). Dann bleibt nach dem Schritt exakt der GECLIPPTE
# Gradient stehen. Richtung und Verhaeltnis sind davon unberuehrt, ein Clip ist
# ein uniformer Skalar.
# ==============================================================================
proc getGradPtr(s: ptr CortexState, l: cint, t: cint): ptr cfloat =
  if l >= s.numLayers or l < -1: return nil
  if l == -1:
    if t == 0: return cast[ptr cfloat](s.g_embeddings.data)
    if t == 1: return cast[ptr cfloat](s.g_w_out.data)
    if t == 2: return cast[ptr cfloat](s.g_rms_final_weight)
  else:
    if t == 10: return cast[ptr cfloat](s.g_rms_att_weights[l])
    if t == 11: return cast[ptr cfloat](s.g_rms_ffn_weights[l])
    if t == 12: return cast[ptr cfloat](s.g_w_q[l].data)
    if t == 13: return cast[ptr cfloat](s.g_w_k[l].data)
    if t == 14: return cast[ptr cfloat](s.g_w_v[l].data)
    if t == 15: return cast[ptr cfloat](s.g_w_o[l].data)
    if t == 16: return cast[ptr cfloat](s.g_w_up[l].data)
    if t == 17: return cast[ptr cfloat](s.g_w_down[l].data)
  return nil

proc cortex_extract_grad*(s: ptr CortexState, layer_idx: cint, tensor_type: cint, raw: ptr cfloat, size: cint) {.ffi.} =
  if size != getExpectedSize(s, tensor_type): return
  let p = getGradPtr(s, layer_idx, tensor_type)
  if not p.isNil: copyMem(raw, p, size * sizeof(cfloat))

proc cortex_load_adam_layer*(s: ptr CortexState, layer_idx: cint, tensor_type: cint, is_v: bool, raw: ptr cfloat, size: cint) {.ffi.} =
  if size != getExpectedSize(s, tensor_type): return
  let p = getTensorPtr(s, layer_idx, tensor_type, not is_v, is_v)
  if p.isNil: return
  copyMem(p, raw, size * sizeof(cfloat))
  # AUDIT-FIX 2026-08-18 (Befund 6): identisch zu cortex_load_layer - m/v landen
  # im GPU-Adam-Kernel und muessen bei bereits aktivem Backend nachgezogen werden.
  if gpu_initialized:
    gpuSyncToDevice(cast[pointer](p), cast[csize_t](size.int * sizeof(cfloat)))

proc cortex_extract_adam_layer*(s: ptr CortexState, layer_idx: cint, tensor_type: cint, is_v: bool, raw: ptr cfloat, size: cint) {.ffi.} =
  if size != getExpectedSize(s, tensor_type): return
  let p = getTensorPtr(s, layer_idx, tensor_type, not is_v, is_v)
  if not p.isNil: copyMem(raw, p, size * sizeof(cfloat))

proc cortex_reset_cache*(s: ptr CortexState) {.ffi.} =
  s.cachedTokens = 0

proc cortex_forward*(state: ptr CortexState, raw_in: ptr cint, num_tok: cint, raw_logits: ptr cfloat) {.ffi.} =
  let tokens = cast[ptr IntArray](raw_in)
  let logits = cast[ptr FloatArray](raw_logits)

  # AUDIT-FIX 2026-08-18 (docs/ROADMAP.md, Engine-Audit Befund 4): frueher lief
  # die Schleife bis num_tok und brach bei vollem KV-Cache per "break" ab. Die
  # Logits werden aber NUR im Zweig "i == num_tok - 1" geschrieben - nach einem
  # vorzeitigen break blieb raw_logits also voellig unberuehrt, und der Aufrufer
  # bekam ohne jede Fehlermeldung die Verteilung des VORHERIGEN Aufrufs zurueck
  # (in snai_inference.php: eine scheinbar haengende Generierung statt eines
  # sauberen Endes). Jetzt wird die verarbeitbare Menge vorab bestimmt, und die
  # Logits gehoeren immer zum letzten TATSAECHLICH verarbeiteten Token.
  let free_slots = state.ctxSize - state.cachedTokens
  let processable = min(num_tok, free_slots)
  if processable <= 0:
    # Kein einziges Token verarbeitbar (Cache randvoll). Es gibt keinen
    # Fehlerkanal aus dieser Signatur heraus - deshalb die Logits deterministisch
    # nullen (statt stumm den alten Stand stehen zu lassen) und laut auf STDERR
    # melden, damit der Zustand ueberhaupt diagnostizierbar ist.
    for v in 0 ..< state.vocabSize: logits[v] = 0.0
    stderr.writeLine("[cortex] WARN: cortex_forward() ohne freien KV-Cache-Slot (cachedTokens=",
                     $state.cachedTokens, ", ctxSize=", $state.ctxSize,
                     ") - Logits genullt. cortex_reset_cache() noetig.")
    return

  for i in 0 ..< processable:
    var tid = tokens[i]
    if tid < 0 or tid >= state.vocabSize: tid = 0
    let pos = state.cachedTokens

    for d in 0 ..< state.hiddenDim: 
      state.hidden_state_buf[d] = state.embeddings.data[tid * state.hiddenDim + d]
    
    for l in 0 ..< state.numLayers:
      applyRMSNorm(state.hidden_state_buf, state.norm_buf, state.hiddenDim, state.rms_att_weights[l])
      
      matVecMul(state.w_q[l], state.norm_buf, state.q_buf)
      applyRoPE(state.q_buf, state.hiddenDim, state.headSize, state.ctxSize, pos)

      let kp = cast[ptr FloatArray](addr state.k_cache[l].data[pos * state.hiddenDim])
      let vp = cast[ptr FloatArray](addr state.v_cache[l].data[pos * state.hiddenDim])
      matVecMul(state.w_k[l], state.norm_buf, kp)
      applyRoPE(kp, state.hiddenDim, state.headSize, state.ctxSize, pos)
      matVecMul(state.w_v[l], state.norm_buf, vp)
      
      if gpu_initialized:
          gpuAttentionForward(cast[ptr cfloat](state.q_buf), cast[ptr cfloat](state.k_cache[l].data), cast[ptr cfloat](state.v_cache[l].data),
                               cast[ptr cfloat](state.att_out_buf), pos.cint, state.numHeads.cint, state.headSize.cint, state.hiddenDim.cint, state.ctxSize.cint)
      else:
          for d in 0 ..< state.hiddenDim:
            state.att_out_buf[d] = 0.0
            
          for h in 0 ..< state.numHeads:
            let h_off = h * state.headSize
            var max_s: cfloat = -9999.0
            for t in 0 .. pos:
              let tk = cast[ptr FloatArray](addr state.k_cache[l].data[t * state.hiddenDim])
              var sc: cfloat = 0.0
              for d in 0 ..< state.headSize: 
                sc += state.q_buf[h_off + d] * tk[h_off + d]
              sc /= sqrt(state.headSize.float32)
              state.scores_buf[t] = sc
              if sc > max_s: max_s = sc
              
            var sum_e: cfloat = 0.0
            for t in 0 .. pos: 
              state.scores_buf[t] = exp(state.scores_buf[t] - max_s)
              sum_e += state.scores_buf[t]
              
            for t in 0 .. pos:
              let w = state.scores_buf[t] / sum_e
              let tv = cast[ptr FloatArray](addr state.v_cache[l].data[t * state.hiddenDim])
              for d in 0 ..< state.headSize: 
                state.att_out_buf[h_off + d] += tv[h_off + d] * w
      
      matVecMul(state.w_o[l], state.att_out_buf, state.d_norm)
      for d in 0 ..< state.hiddenDim: 
        state.hidden_state_buf[d] += state.d_norm[d]
      
      applyRMSNorm(state.hidden_state_buf, state.norm_buf, state.hiddenDim, state.rms_ffn_weights[l])
      matVecMul(state.w_up[l], state.norm_buf, state.ffn_buf)
      for d in 0 ..< state.ffnDim: 
        let v = state.ffn_buf[d]
        state.ffn_buf[d] = v * (1.0 / (1.0 + exp(-v)))
      matVecMul(state.w_down[l], state.ffn_buf, state.att_out_buf)
      for d in 0 ..< state.hiddenDim: 
        state.hidden_state_buf[d] += state.att_out_buf[d]
        
      if i == processable - 1:
        for d in 0 ..< state.hiddenDim:
          state.j_space_cache[l][d] = state.hidden_state_buf[d]

    applyRMSNorm(state.hidden_state_buf, state.norm_buf, state.hiddenDim, state.rms_final_weight)

    if i == processable - 1:
      # V11.4: identische Mathematik, jetzt ueber den GPU-Pfad.
      matVecMul(state.w_out, state.norm_buf, logits)
    
    state.cachedTokens += 1

proc cortex_train_step*(state: ptr CortexState, raw_in: ptr cint, raw_tg: ptr cint, raw_mask: ptr cint, num_tok: cint, lr: cfloat): cfloat {.ffi.} =
  let act = min(num_tok, state.ctxSize)
  if act <= 0: return 0.0
  let t_step = epochTime()
  let useBatchedWout = (batched_wout != 0)
  let batchedAttn = (batched_attention != 0)
  let t_in = cast[ptr IntArray](raw_in)
  let t_tg = cast[ptr IntArray](raw_tg)
  let t_mask = cast[ptr IntArray](raw_mask)
  var total_loss: cfloat = 0.0
  # Loss-Masking (docs/ROADMAP.md, "Loss-Masking auf [USER]-Tokens"): zaehlt
  # nur Positionen, deren Ziel-Token trainierbar ist (t_mask[pos] != 0), damit
  # die zurueckgegebene Durchschnitts-Loss nicht durch nicht-trainierte
  # [USER]-Positionen verwaessert wird.
  var trainable_count: cint = 0
  
  # V11.4: Gradienten, die auf der GPU akkumuliert werden, muessen AUCH dort
  # genullt werden. g_embeddings und die rms-Gradienten sind bewusst NICHT dabei -
  # die entstehen weiterhin auf der CPU (Scatter bzw. Norm-Backward).
  template zeroGrad(g: Matrix) =
    zeroMat(g)
    if useGpuMat(g): gpuZeroBuffer(cast[ptr cfloat](g.data), gradBytes(g))

  zeroMat(state.g_embeddings)
  zeroGrad(state.g_w_out)
  zeroArr(state.g_rms_final_weight, state.hiddenDim)
  
  for l in 0 ..< state.numLayers:
    zeroGrad(state.g_w_q[l])
    zeroGrad(state.g_w_k[l])
    zeroGrad(state.g_w_v[l])
    zeroGrad(state.g_w_o[l])
    zeroGrad(state.g_w_up[l])
    zeroGrad(state.g_w_down[l])
    zeroArr(state.g_rms_att_weights[l], state.hiddenDim)
    zeroArr(state.g_rms_ffn_weights[l], state.hiddenDim)
    
    if gpu_initialized:
      let full_cache_bytes = cast[csize_t](state.ctxSize * state.hiddenDim * sizeof(cfloat))
      gpuZeroBuffer(cast[ptr cfloat](state.d_k_cache[l].data), full_cache_bytes)
      gpuZeroBuffer(cast[ptr cfloat](state.d_v_cache[l].data), full_cache_bytes)
    else:
      for i in 0 ..< (act * state.hiddenDim): 
        state.d_k_cache[l].data[i] = 0.0
        state.d_v_cache[l].data[i] = 0.0

  if batchedAttn:
    # ==========================================================================
    # AUDIT: LAYER-MAJOR FORWARD (gebuendelte Attention)
    # Mathematisch identisch zum Positions-Major-Pfad unten, nur umsortiert:
    # fuer jeden Layer werden erst ALLE Positionen durch RMSNorm+QKV geschickt,
    # dann EIN gebuendelter Attention-Aufruf fuer den ganzen Layer, dann ALLE
    # Positionen durch O-Projektion+FFN. state.act_hidden_final dient dabei als
    # Positions-persistenter Roll-Puffer ueber die Layer hinweg (ersetzt
    # hidden_state_buf, das im alten Pfad pro Position frisch aus den Embeddings
    # startete) - nach dem letzten Layer haelt es exakt das, was sein Name schon
    # immer versprach: den finalen Hidden-State je Position.
    # ==========================================================================
    for p in 0 ..< act:
      var tid = t_in[p]
      if tid < 0 or tid >= state.vocabSize: tid = 0
      let off_dim = p * state.hiddenDim
      for d in 0 ..< state.hiddenDim:
        state.act_hidden_final[off_dim + d] = state.embeddings.data[tid * state.hiddenDim + d]

    for l in 0 ..< state.numLayers:
      # Sub-Pass 1: RMSNorm (echt Positions-abhaengig, bleibt Schleife) direkt
      # in act_norm_att[l] geschrieben, dann Q/K/V in EINEM Rutsch fuer alle
      # Positionen - dieselbe Batched-Matmul-Infrastruktur wie beim w_out-Pfad.
      for i in 0 ..< (act * state.hiddenDim):
        state.act_hidden[l][i] = state.act_hidden_final[i]
      for p in 0 ..< act:
        let off_dim = p * state.hiddenDim
        applyRMSNorm(cast[ptr FloatArray](addr state.act_hidden_final[off_dim]),
                     cast[ptr FloatArray](addr state.act_norm_att[l][off_dim]),
                     state.hiddenDim, state.rms_att_weights[l])

      matMulBatchedD(state.w_q[l], state.act_norm_att[l], cast[ptr FloatArray](state.act_q[l].data), act, state.ctxSize.int)
      matMulBatchedD(state.w_k[l], state.act_norm_att[l], cast[ptr FloatArray](state.k_cache[l].data), act, state.ctxSize.int)
      matMulBatchedD(state.w_v[l], state.act_norm_att[l], cast[ptr FloatArray](state.v_cache[l].data), act, state.ctxSize.int)

      for p in 0 ..< act:
        let off_dim = p * state.hiddenDim
        applyRoPE(cast[ptr FloatArray](addr state.act_q[l].data[off_dim]), state.hiddenDim, state.headSize, state.ctxSize, p.cint)
        applyRoPE(cast[ptr FloatArray](addr state.k_cache[l].data[off_dim]), state.hiddenDim, state.headSize, state.ctxSize, p.cint)

      attentionForwardBatchedD(state, l.cint, act.cint)

      # Sub-Pass 3: O-Projektion gebuendelt -> mm_hd_scratch, dann Skip-Add +
      # FFN-RMSNorm (Positions-abhaengig), dann Up/Down gebuendelt.
      matMulBatchedD(state.w_o[l], state.act_att_out[l], state.mm_hd_scratch, act, state.ctxSize.int)
      for i in 0 ..< (act * state.hiddenDim):
        state.act_hidden_final[i] += state.mm_hd_scratch[i]
      for i in 0 ..< (act * state.hiddenDim):
        state.act_hidden_ffn[l][i] = state.act_hidden_final[i]
      for p in 0 ..< act:
        let off_dim = p * state.hiddenDim
        applyRMSNorm(cast[ptr FloatArray](addr state.act_hidden_final[off_dim]),
                     cast[ptr FloatArray](addr state.act_norm_ffn[l][off_dim]),
                     state.hiddenDim, state.rms_ffn_weights[l])

      matMulBatchedD(state.w_up[l], state.act_norm_ffn[l], state.act_ffn_in[l], act, state.ctxSize.int)
      for i in 0 ..< (act * state.ffnDim):
        let v = state.act_ffn_in[l][i]
        state.act_ffn_out[l][i] = v * (1.0 / (1.0 + exp(-v)))
      applyDropoutForward(state.act_ffn_out[l], state.dropout_mask[l], act * state.ffnDim)
      matMulBatchedD(state.w_down[l], state.act_ffn_out[l], state.mm_hd_scratch, act, state.ctxSize.int)
      for i in 0 ..< (act * state.hiddenDim):
        state.act_hidden_final[i] += state.mm_hd_scratch[i]

    for p in 0 ..< act:
      var tgid = t_tg[p]
      if tgid < 0 or tgid >= state.vocabSize: tgid = 0
      let off_dim = p * state.hiddenDim
      let final_hidden = cast[ptr FloatArray](addr state.act_hidden_final[off_dim])
      applyRMSNorm(final_hidden, state.norm_buf, state.hiddenDim, state.rms_final_weight)

      if useBatchedWout:
        for d in 0 ..< state.hiddenDim:
          state.act_norm_final[off_dim + d] = state.norm_buf[d]
      else:
        var max_l: cfloat = -9999.0
        let logits = cast[ptr FloatArray](addr state.logits_cache.data[p * state.vocabSize])
        matVecMul(state.w_out, state.norm_buf, logits)
        for v in 0 ..< state.vocabSize:
          if logits[v] > max_l: max_l = logits[v]
        var sum_p: cfloat = 0.0
        for v in 0 ..< state.vocabSize:
          logits[v] = exp(logits[v] - max_l)
          sum_p += logits[v]
        for v in 0 ..< state.vocabSize:
          logits[v] /= sum_p
        if t_mask[p] != 0:
          total_loss += -ln(logits[tgid] + 1e-7)
          inc trainable_count
  else:
   for pos in 0 ..< act:
    var tid = t_in[pos]
    var tgid = t_tg[pos]
    if tid < 0 or tid >= state.vocabSize: tid = 0
    if tgid < 0 or tgid >= state.vocabSize: tgid = 0
    let off_dim = pos * state.hiddenDim
    let off_ffn = pos * state.ffnDim
    for d in 0 ..< state.hiddenDim:
      state.hidden_state_buf[d] = state.embeddings.data[tid * state.hiddenDim + d]

    for l in 0 ..< state.numLayers:
      for d in 0 ..< state.hiddenDim:
        state.act_hidden[l][off_dim + d] = state.hidden_state_buf[d]
      applyRMSNorm(state.hidden_state_buf, state.norm_buf, state.hiddenDim, state.rms_att_weights[l])
      for d in 0 ..< state.hiddenDim:
        state.act_norm_att[l][off_dim + d] = state.norm_buf[d]

      let q_p = cast[ptr FloatArray](addr state.act_q[l].data[off_dim])
      matVecMul(state.w_q[l], state.norm_buf, q_p)
      applyRoPE(q_p, state.hiddenDim, state.headSize, state.ctxSize, pos)

      let kp = cast[ptr FloatArray](addr state.k_cache[l].data[off_dim])
      let vp = cast[ptr FloatArray](addr state.v_cache[l].data[off_dim])
      matVecMul(state.w_k[l], state.norm_buf, kp)
      applyRoPE(kp, state.hiddenDim, state.headSize, state.ctxSize, pos)
      matVecMul(state.w_v[l], state.norm_buf, vp)

      if gpu_initialized:
          gpuAttentionForward(cast[ptr cfloat](q_p), cast[ptr cfloat](state.k_cache[l].data), cast[ptr cfloat](state.v_cache[l].data),
                               cast[ptr cfloat](state.att_out_buf), pos.cint, state.numHeads.cint, state.headSize.cint, state.hiddenDim.cint, state.ctxSize.cint)
      else:
          for d in 0 ..< state.hiddenDim:
            state.att_out_buf[d] = 0.0

          for h in 0 ..< state.numHeads:
            let h_off = h * state.headSize
            var max_s: cfloat = -9999.0
            for t in 0 .. pos:
              let tk = cast[ptr FloatArray](addr state.k_cache[l].data[t * state.hiddenDim])
              var sc: cfloat = 0.0
              for d in 0 ..< state.headSize:
                sc += q_p[h_off + d] * tk[h_off + d]
              sc /= sqrt(state.headSize.float32)
              state.scores_buf[t] = sc
              if sc > max_s: max_s = sc

            var sum_e: cfloat = 0.0
            for t in 0 .. pos:
              state.scores_buf[t] = exp(state.scores_buf[t] - max_s)
              sum_e += state.scores_buf[t]

            for t in 0 .. pos:
              let w = state.scores_buf[t] / sum_e
              let tv = cast[ptr FloatArray](addr state.v_cache[l].data[t * state.hiddenDim])
              for d in 0 ..< state.headSize:
                state.att_out_buf[h_off + d] += tv[h_off + d] * w

      for d in 0 ..< state.hiddenDim:
        state.act_att_out[l][off_dim + d] = state.att_out_buf[d]
      matVecMul(state.w_o[l], state.att_out_buf, state.d_norm)
      for d in 0 ..< state.hiddenDim:
        state.hidden_state_buf[d] += state.d_norm[d]

      for d in 0 ..< state.hiddenDim:
        state.act_hidden_ffn[l][off_dim + d] = state.hidden_state_buf[d]
      applyRMSNorm(state.hidden_state_buf, state.norm_buf, state.hiddenDim, state.rms_ffn_weights[l])
      for d in 0 ..< state.hiddenDim:
        state.act_norm_ffn[l][off_dim + d] = state.norm_buf[d]

      matVecMul(state.w_up[l], state.norm_buf, state.ffn_buf)
      for d in 0 ..< state.ffnDim:
        state.act_ffn_in[l][off_ffn + d] = state.ffn_buf[d]
      for d in 0 ..< state.ffnDim:
        let v = state.ffn_buf[d]
        state.ffn_buf[d] = v * (1.0 / (1.0 + exp(-v)))
      applyDropoutForward(state.ffn_buf, cast[ptr FloatArray](addr state.dropout_mask[l][off_ffn]), state.ffnDim)
      for d in 0 ..< state.ffnDim:
        state.act_ffn_out[l][off_ffn + d] = state.ffn_buf[d]
      matVecMul(state.w_down[l], state.ffn_buf, state.att_out_buf)
      for d in 0 ..< state.hiddenDim:
        state.hidden_state_buf[d] += state.att_out_buf[d]

    for d in 0 ..< state.hiddenDim:
      state.act_hidden_final[off_dim + d] = state.hidden_state_buf[d]
    applyRMSNorm(state.hidden_state_buf, state.norm_buf, state.hiddenDim, state.rms_final_weight)

    if useBatchedWout:
      # V11.9: nur die Norm-Ausgabe sichern. w_out laeuft nach der Schleife
      # gebuendelt ueber ALLE Positionen - ein Kernel statt act Kernels.
      for d in 0 ..< state.hiddenDim:
        state.act_norm_final[off_dim + d] = state.norm_buf[d]
    else:
      var max_l: cfloat = -9999.0
      let logits = cast[ptr FloatArray](addr state.logits_cache.data[pos * state.vocabSize])
      matVecMul(state.w_out, state.norm_buf, logits)
      for v in 0 ..< state.vocabSize:
        if logits[v] > max_l: max_l = logits[v]
      var sum_p: cfloat = 0.0
      for v in 0 ..< state.vocabSize:
        logits[v] = exp(logits[v] - max_l)
        sum_p += logits[v]
      for v in 0 ..< state.vocabSize:
        logits[v] /= sum_p
      if t_mask[pos] != 0:
        total_loss += -ln(logits[tgid] + 1e-7)
        inc trainable_count

  # ============================================================================
  # V11.9: GEBUENDELTER w_out-BLOCK
  # Ein matMulBatched fuer alle Logits, dann Softmax je Position, dann
  # d_logits in-place in logits_cache - und ein einziges Paar Backward-Aufrufe
  # statt act Stueck. Reihenfolge-unabhaengig, deshalb aus der countdown-
  # Schleife herausziehbar (Attention-Gradienten bleiben dort unberuehrt).
  # ============================================================================
  if useBatchedWout:
    let logits_all = cast[ptr FloatArray](state.logits_cache.data)
    matMulBatchedD(state.w_out, state.act_norm_final, logits_all, act, state.ctxSize.int)

    for pos in 0 ..< act:
      var tgid = t_tg[pos]
      if tgid < 0 or tgid >= state.vocabSize: tgid = 0
      let base = pos * state.vocabSize
      var max_l: cfloat = -9999.0
      for v in 0 ..< state.vocabSize:
        if logits_all[base + v] > max_l: max_l = logits_all[base + v]
      var sum_p: cfloat = 0.0
      for v in 0 ..< state.vocabSize:
        logits_all[base + v] = exp(logits_all[base + v] - max_l)
        sum_p += logits_all[base + v]
      for v in 0 ..< state.vocabSize:
        logits_all[base + v] /= sum_p
      if t_mask[pos] != 0:
        total_loss += -ln(logits_all[base + tgid] + 1e-7)
        inc trainable_count

    # d_logits IN-PLACE: ab hier haelt logits_cache die Gradienten, nicht mehr
    # die Wahrscheinlichkeiten. Die Logits werden danach nicht mehr gebraucht.
    # Maskierte Positionen (t_mask[pos] == 0, [USER]-Ziel-Token) bekommen einen
    # Null-Gradienten - kein Backprop-Signal fuer vom Menschen vorgegebenen Text.
    for pos in 0 ..< act:
      var tgid = t_tg[pos]
      if tgid < 0 or tgid >= state.vocabSize: tgid = 0
      let base = pos * state.vocabSize
      if t_mask[pos] != 0:
        for v in 0 ..< state.vocabSize:
          logits_all[base + v] = logits_all[base + v] - smoothedTarget(v, tgid, state.vocabSize)
      else:
        for v in 0 ..< state.vocabSize:
          logits_all[base + v] = 0.0

    matMulTBatchedD(state.w_out, logits_all, state.d_norm_all, act)
    gradAccumBatchedD(state.w_out, state.g_w_out, logits_all, state.act_norm_final, act)

  if batchedAttn:
    # ==========================================================================
    # AUDIT: LAYER-MAJOR BACKWARD (gebuendelte Attention)
    # Spiegel des Forward-Umbaus. state.d_hidden_all uebernimmt die Rolle von
    # d_hidden, jetzt aber Positions-persistent statt pro Position frisch auf
    # 0 gesetzt - es sammelt den Gradienten fuer jede Position ueber die
    # gesamte Countdown-Schleife der Layer.
    # ==========================================================================
    for p in 0 ..< act:
      let off_dim = p * state.hiddenDim
      let final_hidden = cast[ptr FloatArray](addr state.act_hidden_final[off_dim])
      if useBatchedWout:
        for d in 0 ..< state.hiddenDim:
          state.d_norm[d] = state.d_norm_all[off_dim + d]
      else:
        var tgid = t_tg[p]
        if tgid < 0 or tgid >= state.vocabSize: tgid = 0
        let logits = cast[ptr FloatArray](addr state.logits_cache.data[p * state.vocabSize])
        if t_mask[p] != 0:
          for v in 0 ..< state.vocabSize:
            state.d_logits_buf[v] = logits[v] - smoothedTarget(v, tgid, state.vocabSize)
        else:
          for v in 0 ..< state.vocabSize:
            state.d_logits_buf[v] = 0.0
        applyRMSNorm(final_hidden, state.norm_buf, state.hiddenDim, state.rms_final_weight)
        matVecMulBackwardAcc(state.w_out, state.g_w_out, state.norm_buf, state.d_logits_buf, state.d_norm, false)
      applyRMSNormBackwardAcc(state.d_norm, final_hidden, state.hiddenDim, state.rms_final_weight, state.g_rms_final_weight)
      for d in 0 ..< state.hiddenDim:
        state.d_hidden_all[off_dim + d] = state.d_norm[d]

    for l in countdown(state.numLayers - 1, 0):
      # Sub-Pass A: FFN-Backward gebuendelt. d_hidden_all ist bereits exakt das,
      # was vorher pro Position in d_norm kopiert wurde - direkt als d_out_all
      # verwendbar, keine Kopie noetig.
      matMulTBatchedD(state.w_down[l], state.d_hidden_all, state.mm_ffn_scratch, act)
      gradAccumBatchedD(state.w_down[l], state.g_w_down[l], state.d_hidden_all, state.act_ffn_out[l], act)

      applyDropoutBackward(state.mm_ffn_scratch, state.dropout_mask[l], act * state.ffnDim)
      for i in 0 ..< (act * state.ffnDim):
        let x = state.act_ffn_in[l][i]
        let sig = 1.0 / (1.0 + exp(-x))
        state.mm_ffn_scratch[i] *= sig * (1.0 + x * (1.0 - sig))

      matMulTBatchedD(state.w_up[l], state.mm_ffn_scratch, state.mm_hd_scratch, act)
      gradAccumBatchedD(state.w_up[l], state.g_w_up[l], state.mm_ffn_scratch, state.act_norm_ffn[l], act)

      for p in 0 ..< act:
        let off_dim = p * state.hiddenDim
        applyRMSNormBackwardAcc(cast[ptr FloatArray](addr state.mm_hd_scratch[off_dim]),
                                 cast[ptr FloatArray](addr state.act_hidden_ffn[l][off_dim]),
                                 state.hiddenDim, state.rms_ffn_weights[l], state.g_rms_ffn_weights[l])
      for i in 0 ..< (act * state.hiddenDim):
        state.d_hidden_all[i] += state.mm_hd_scratch[i]

      # Sub-Pass B: O-Projektion-Backward gebuendelt -> d_att_out_all.
      matMulTBatchedD(state.w_o[l], state.d_hidden_all, state.d_att_out_all, act)
      gradAccumBatchedD(state.w_o[l], state.g_w_o[l], state.d_hidden_all, state.act_att_out[l], act)

      let t_att = epochTime()
      attentionBackwardBatchedD(state, l.cint, act.cint)
      prof[4] += epochTime() - t_att

      for p in 0 ..< act:
        let off_dim = p * state.hiddenDim
        applyRoPE(cast[ptr FloatArray](addr state.d_q_all[off_dim]), state.hiddenDim, state.headSize, state.ctxSize, p.cint, true)
        applyRoPE(cast[ptr FloatArray](addr state.d_k_cache[l].data[off_dim]), state.hiddenDim, state.headSize, state.ctxSize, p.cint, true)

      # Sub-Pass C: Q/K/V-Backward gebuendelt. Alle drei speisen denselben
      # act_norm_att[l] - ihre d_in-Beitraege muessen deshalb aufsummiert
      # werden (matMulTBatchedD's accumulate-Parameter, siehe dessen Kommentar).
      matMulTBatchedD(state.w_q[l], state.d_q_all, state.mm_hd_scratch, act, false)
      gradAccumBatchedD(state.w_q[l], state.g_w_q[l], state.d_q_all, state.act_norm_att[l], act)
      matMulTBatchedD(state.w_k[l], state.d_k_cache[l].data, state.mm_hd_scratch, act, true)
      gradAccumBatchedD(state.w_k[l], state.g_w_k[l], state.d_k_cache[l].data, state.act_norm_att[l], act)
      matMulTBatchedD(state.w_v[l], state.d_v_cache[l].data, state.mm_hd_scratch, act, true)
      gradAccumBatchedD(state.w_v[l], state.g_w_v[l], state.d_v_cache[l].data, state.act_norm_att[l], act)

      for p in 0 ..< act:
        let off_dim = p * state.hiddenDim
        applyRMSNormBackwardAcc(cast[ptr FloatArray](addr state.mm_hd_scratch[off_dim]),
                                 cast[ptr FloatArray](addr state.act_hidden[l][off_dim]),
                                 state.hiddenDim, state.rms_att_weights[l], state.g_rms_att_weights[l])
      for i in 0 ..< (act * state.hiddenDim):
        state.d_hidden_all[i] += state.mm_hd_scratch[i]

    for p in 0 ..< act:
      var tid = t_in[p]
      if tid < 0 or tid >= state.vocabSize: tid = 0
      let off_dim = p * state.hiddenDim
      for d in 0 ..< state.hiddenDim:
        state.g_embeddings.data[tid * state.hiddenDim + d] += state.d_hidden_all[off_dim + d]
  else:
   for pos in countdown(act - 1, 0):
    var tid = t_in[pos]
    var tgid = t_tg[pos]
    if tid < 0 or tid >= state.vocabSize: tid = 0
    if tgid < 0 or tgid >= state.vocabSize: tgid = 0
    let off_dim = pos * state.hiddenDim
    let off_ffn = pos * state.ffnDim
    let logits = cast[ptr FloatArray](addr state.logits_cache.data[pos * state.vocabSize])

    if not useBatchedWout:
      if t_mask[pos] != 0:
        for v in 0 ..< state.vocabSize:
          state.d_logits_buf[v] = logits[v] - smoothedTarget(v, tgid, state.vocabSize)
      else:
        for v in 0 ..< state.vocabSize:
          state.d_logits_buf[v] = 0.0
    for d in 0 ..< state.hiddenDim:
      state.d_hidden[d] = 0.0

    let final_hidden = cast[ptr FloatArray](addr state.act_hidden_final[off_dim])
    if useBatchedWout:
      # d_norm wurde bereits gebuendelt berechnet - nur uebernehmen.
      for d in 0 ..< state.hiddenDim:
        state.d_norm[d] = state.d_norm_all[off_dim + d]
    else:
      applyRMSNorm(final_hidden, state.norm_buf, state.hiddenDim, state.rms_final_weight)
      matVecMulBackwardAcc(state.w_out, state.g_w_out, state.norm_buf, state.d_logits_buf, state.d_norm, false)
    applyRMSNormBackwardAcc(state.d_norm, final_hidden, state.hiddenDim, state.rms_final_weight, state.g_rms_final_weight)

    for d in 0 ..< state.hiddenDim:
      state.d_hidden[d] += state.d_norm[d]

    for l in countdown(state.numLayers - 1, 0):
      for d in 0 ..< state.hiddenDim:
        state.d_norm[d] = state.d_hidden[d]

      let act_ffn_out_p = cast[ptr FloatArray](addr state.act_ffn_out[l][off_ffn])
      matVecMulBackwardAcc(state.w_down[l], state.g_w_down[l], act_ffn_out_p, state.d_norm, state.d_ffn, false)

      applyDropoutBackward(state.d_ffn, cast[ptr FloatArray](addr state.dropout_mask[l][off_ffn]), state.ffnDim)
      let act_ffn_in_p = cast[ptr FloatArray](addr state.act_ffn_in[l][off_ffn])
      for d in 0 ..< state.ffnDim:
        let x = act_ffn_in_p[d]
        let sig = 1.0 / (1.0 + exp(-x))
        state.d_ffn[d] *= sig * (1.0 + x * (1.0 - sig))

      let act_norm_ffn_p = cast[ptr FloatArray](addr state.act_norm_ffn[l][off_dim])
      matVecMulBackwardAcc(state.w_up[l], state.g_w_up[l], act_norm_ffn_p, state.d_ffn, state.d_norm, false)

      let act_hidden_ffn_p = cast[ptr FloatArray](addr state.act_hidden_ffn[l][off_dim])
      applyRMSNormBackwardAcc(state.d_norm, act_hidden_ffn_p, state.hiddenDim, state.rms_ffn_weights[l], state.g_rms_ffn_weights[l])

      for d in 0 ..< state.hiddenDim:
        state.d_hidden[d] += state.d_norm[d]
      for d in 0 ..< state.hiddenDim:
        state.d_norm[d] = state.d_hidden[d]

      let act_att_out_p = cast[ptr FloatArray](addr state.act_att_out[l][off_dim])
      matVecMulBackwardAcc(state.w_o[l], state.g_w_o[l], act_att_out_p, state.d_norm, state.d_att_out, false)

      let q_p = cast[ptr FloatArray](addr state.act_q[l].data[off_dim])

      let t_att = epochTime()
      if gpu_initialized:
          gpuAttentionBackward(cast[ptr cfloat](q_p), cast[ptr cfloat](state.k_cache[l].data), cast[ptr cfloat](state.v_cache[l].data), cast[ptr cfloat](state.d_att_out),
                                cast[ptr cfloat](state.d_q), cast[ptr cfloat](state.d_k_cache[l].data), cast[ptr cfloat](state.d_v_cache[l].data),
                                pos.cint, state.numHeads.cint, state.headSize.cint, state.hiddenDim.cint, state.ctxSize.cint)
          prof[4] += epochTime() - t_att
      else:
          for d in 0 ..< state.hiddenDim:
            state.d_q[d] = 0.0
          for h in 0 ..< state.numHeads:
            let h_off = h * state.headSize
            let scale = 1.0 / sqrt(state.headSize.float32)
            var max_s: cfloat = -9999.0
            for t in 0 .. pos:
              let k_t = cast[ptr FloatArray](addr state.k_cache[l].data[t * state.hiddenDim])
              var sc: cfloat = 0.0
              for d in 0 ..< state.headSize:
                sc += q_p[h_off + d] * k_t[h_off + d]
              sc *= scale
              state.scores_buf[t] = sc
              if sc > max_s: max_s = sc

            var sum_e: cfloat = 0.0
            for t in 0 .. pos:
              state.scores_buf[t] = exp(state.scores_buf[t] - max_s)
              sum_e += state.scores_buf[t]

            var sum_w_dw: cfloat = 0.0
            for t in 0 .. pos:
              let w = state.scores_buf[t] / sum_e
              state.scores_buf[t] = w
              let v_t = cast[ptr FloatArray](addr state.v_cache[l].data[t * state.hiddenDim])
              var dw: cfloat = 0.0
              for d in 0 ..< state.headSize:
                dw += state.d_att_out[h_off + d] * v_t[h_off + d]

              let d_v_t = cast[ptr FloatArray](addr state.d_v_cache[l].data[t * state.hiddenDim])
              for d in 0 ..< state.headSize:
                d_v_t[h_off + d] += state.d_att_out[h_off + d] * w

              state.d_w_buf[t] = dw
              sum_w_dw += w * dw

            for t in 0 .. pos:
              let w = state.scores_buf[t]
              let ds = w * (state.d_w_buf[t] - sum_w_dw) * scale
              let k_t = cast[ptr FloatArray](addr state.k_cache[l].data[t * state.hiddenDim])
              let d_k_t = cast[ptr FloatArray](addr state.d_k_cache[l].data[t * state.hiddenDim])
              for d in 0 ..< state.headSize:
                state.d_q[h_off + d] += ds * k_t[h_off + d]
                d_k_t[h_off + d] += ds * q_p[h_off + d]

      let current_d_k = cast[ptr FloatArray](addr state.d_k_cache[l].data[off_dim])
      let current_d_v = cast[ptr FloatArray](addr state.d_v_cache[l].data[off_dim])

      applyRoPE(state.d_q, state.hiddenDim, state.headSize, state.ctxSize, pos, true)
      applyRoPE(current_d_k, state.hiddenDim, state.headSize, state.ctxSize, pos, true)

      let act_norm_att_p = cast[ptr FloatArray](addr state.act_norm_att[l][off_dim])

      matVecMulBackwardAcc(state.w_q[l], state.g_w_q[l], act_norm_att_p, state.d_q, state.d_norm, false)
      matVecMulBackwardAcc(state.w_k[l], state.g_w_k[l], act_norm_att_p, current_d_k, state.d_norm, true)
      matVecMulBackwardAcc(state.w_v[l], state.g_w_v[l], act_norm_att_p, current_d_v, state.d_norm, true)

      let act_hidden_p = cast[ptr FloatArray](addr state.act_hidden[l][off_dim])
      applyRMSNormBackwardAcc(state.d_norm, act_hidden_p, state.hiddenDim, state.rms_att_weights[l], state.g_rms_att_weights[l])
      for d in 0 ..< state.hiddenDim:
        state.d_hidden[d] += state.d_norm[d]

    for d in 0 ..< state.hiddenDim:
      state.g_embeddings.data[tid * state.hiddenDim + d] += state.d_hidden[d]

  # V11.4: gm hat ueber alle Token im VRAM akkumuliert - einmal pro Step abholen,
  # damit Clipping und Adam auf dem Host denselben Stand sehen.
  template pullGrad(g: Matrix) =
    if useGpuMat(g): gpuSyncToHost(cast[pointer](g.data))

  pullGrad(state.g_w_out)
  for l in 0 ..< state.numLayers:
    pullGrad(state.g_w_q[l])
    pullGrad(state.g_w_k[l])
    pullGrad(state.g_w_v[l])
    pullGrad(state.g_w_o[l])
    pullGrad(state.g_w_up[l])
    pullGrad(state.g_w_down[l])

  # ============================================================================
  # WEIGHT TYING: Gradienten zusammenfuehren.
  #
  # Genau HIER und nirgends sonst - der Punkt ist bewusst gewaehlt:
  #   davor  liegt pullGrad(), der GPU-Anteil steht also im Host-Speicher;
  #   dahinter liegt clipGlobalGradientNorm(), das die Norm ueber den FERTIGEN
  #            Gradienten bilden muss, und applyGlobalAdamUpdate().
  # Frueher zusammenfuehren hiesse, den GPU-Anteil zu verpassen; spaeter hiesse,
  # dass Clipping und Adam auf einem halben Gradienten arbeiten.
  #
  # Nach dieser Schleife traegt g_embeddings die Summe beider Beitraege (Scatter
  # aus dem Embedding-Lookup + Ausgabeprojektion). g_w_out ist danach redundant
  # und wird im naechsten Schritt ohnehin genullt.
  if weight_tying != 0:
    let embElems = state.vocabSize.int * state.hiddenDim.int
    for i in 0 ..< embElems:
      state.g_embeddings.data[i] += state.g_w_out.data[i]

  let t_upd = epochTime()
  # AUDIT-FIX 2026-08-18 (docs/ROADMAP.md, Engine-Audit Befund 1):
  # clipGlobalGradientNorm() lief bisher mit einer fest verdrahteten 1.0 - und
  # zwar auf dem ueber ALLE Positionen des Fensters AUFSUMMIERTEN Gradienten,
  # weil die Division durch trainable_count erst danach in adamStep()
  # ("grad = g[] / scale") passiert. Die 1.0 ist aber die uebliche Konvention
  # fuer den MITTLEREN Gradienten - die Schwelle stand damit um genau den
  # Faktor trainable_count (1024 bei Prinzessin, 2048 bei Matriarchin) zu eng.
  # Folge, empirisch belegt an 6329 geloggten Schritten in
  # factory/beehive_stdout.log: kleinste je gemessene Norm 254.8, Median
  # 1303.7 - der Clip-Zweig wurde in JEDEM Schritt genommen, ohne Ausnahme.
  # Aus dem Sicherheitsventil gegen Ausreisser war eine feste
  # Gradienten-Normalisierung geworden, und das GradNorm-Logging aus
  # Nachruest-Paket #9 konnte strukturell nie Instabilitaet melden.
  # Fix: die Schwelle mitskalieren. ||g||/scale <= 1.0 <=> ||g|| <= scale,
  # das ist exakt "mittlerer Gradient auf Norm 1.0 geclippt". max(1.0, ...)
  # nur als Division-durch-Null-Schutz - der Adam-Skalierer unten bleibt
  # bewusst das ROHE trainable_count, damit scale==0.0 (Fenster komplett ohne
  # trainierbare Position) weiterhin in adamStep()s "if scale == 0.0: return"
  # laeuft und NICHT versehentlich einen reinen Momentum-/Weight-Decay-Schritt
  # ausloest.
  let clip_norm = max(1.0'f32, trainable_count.cfloat)
  clipGlobalGradientNorm(state, clip_norm)
  # last_grad_norm haelt nach clipGlobalGradientNorm() die Norm im SUMMEN-Raum.
  # Hier auf den Mittelwert-Raum umgerechnet, damit der geloggte GradNorm-Wert
  # direkt gegen die Clip-Schwelle 1.0 lesbar ist (genau das war Sinn von
  # Nachruest-Paket #9). ACHTUNG beim Vergleich mit Altlogs: Werte vor dem
  # 2026-08-18 sind Summen-Normen, also grob um trainable_count groesser.
  last_grad_norm = last_grad_norm / clip_norm
  # Loss-Masking: skaliert den akkumulierten Gradienten auf die trainierbaren
  # Positionen statt die volle Fenstergroesse - sonst wuerden Fenster mit viel
  # [USER]-Text (Gradient dort bereits oben auf 0 gesetzt) einen kuenstlich
  # abgeschwaechten Update-Schritt bekommen. scale==0.0 (Fenster komplett ohne
  # trainierbare Position) laesst adamStep() den Schritt bereits sauber
  # ueberspringen (siehe "if scale == 0.0: return" dort).
  if skip_adam == 0:
    applyGlobalAdamUpdate(state, lr, trainable_count.cfloat)
  prof[5] += epochTime() - t_upd
  prof[6] += epochTime() - t_step
  return total_loss / max(1, trainable_count).cfloat

proc cortex_free*(state: ptr CortexState) {.ffi.} =
  if state.isNil: return
  
  template freeM(m) = 
    freeMatrix(m)
    
  template freeA(a) = 
    if not a.isNil: 
      if gpu_initialized: gpuReleaseBuffer(cast[pointer](a))
      deallocShared(a)
      a = nil
      
  template freeHist(p, l) = 
    if not p.isNil:
      for i in 0 ..< l: 
        freeA(p[i])
      deallocShared(p)
      p = nil

  freeM(state.logits_cache)
  freeA(state.act_hidden_final)
  freeA(state.act_norm_final)
  freeA(state.d_norm_all)
  freeA(state.d_hidden_all)
  freeA(state.d_att_out_all)
  freeA(state.d_q_all)
  freeA(state.mm_hd_scratch)
  freeA(state.mm_ffn_scratch)
  freeA(state.hidden_state_buf)
  freeA(state.norm_buf)
  freeA(state.att_out_buf)
  freeA(state.ffn_buf)
  freeA(state.d_logits_buf)
  freeA(state.scores_buf)
  freeA(state.d_w_buf)
  freeA(state.q_buf)
  freeA(state.d_hidden)
  freeA(state.d_norm)
  freeA(state.d_att_out)
  freeA(state.d_q)
  freeA(state.d_ffn)
  
  freeHist(state.act_hidden, state.numLayers)
  freeHist(state.act_norm_att, state.numLayers)
  freeHist(state.act_att_out, state.numLayers)
  freeHist(state.act_hidden_ffn, state.numLayers)
  freeHist(state.act_norm_ffn, state.numLayers)
  freeHist(state.act_ffn_in, state.numLayers)
  freeHist(state.act_ffn_out, state.numLayers)
  freeHist(state.dropout_mask, state.numLayers)

  freeHist(state.j_space_cache, state.numLayers)
  
  if not state.act_q.isNil:
    for l in 0 ..< state.numLayers: 
      freeM(state.act_q[l])
      freeM(state.d_k_cache[l])
      freeM(state.d_v_cache[l])
    deallocShared(state.act_q)
    deallocShared(state.d_k_cache)
    deallocShared(state.d_v_cache)
  
  freeM(state.embeddings)
  freeM(state.m_embeddings)
  freeM(state.v_embeddings)
  freeM(state.g_embeddings)
  # WEIGHT TYING: w_out/m/v zeigen auf die Embedding-Matrizen, die direkt darueber
  # bereits freigegeben wurden - ein zweites freeMatrix() waere ein Double-Free.
  # freeMatrix() setzt m.data zwar auf nil, aber auf SEINER Kopie: Matrix ist ein
  # Werttyp, die Zuweisung oben hat den Zeiger kopiert, nicht verlinkt. Die
  # Aliase wissen also nichts vom Freigeben des Originals.
  # g_w_out ist IMMER ein eigener Puffer und wird immer freigegeben.
  if weight_tying == 0:
    freeM(state.w_out)
    freeM(state.m_w_out)
    freeM(state.v_w_out)
  freeM(state.g_w_out)
  
  freeA(state.rms_final_weight)
  freeA(state.m_rms_final_weight)
  freeA(state.v_rms_final_weight)
  freeA(state.g_rms_final_weight)
  
  if not state.w_q.isNil:
    for l in 0 ..< state.numLayers:
      freeA(state.rms_att_weights[l])
      freeA(state.m_rms_att_weights[l])
      freeA(state.v_rms_att_weights[l])
      freeA(state.g_rms_att_weights[l])
      freeA(state.rms_ffn_weights[l])
      freeA(state.m_rms_ffn_weights[l])
      freeA(state.v_rms_ffn_weights[l])
      freeA(state.g_rms_ffn_weights[l])
      freeM(state.w_q[l])
      freeM(state.m_w_q[l])
      freeM(state.v_w_q[l])
      freeM(state.g_w_q[l])
      freeM(state.w_k[l])
      freeM(state.m_w_k[l])
      freeM(state.v_w_k[l])
      freeM(state.g_w_k[l])
      freeM(state.w_v[l])
      freeM(state.m_w_v[l])
      freeM(state.v_w_v[l])
      freeM(state.g_w_v[l])
      freeM(state.w_o[l])
      freeM(state.m_w_o[l])
      freeM(state.v_w_o[l])
      freeM(state.g_w_o[l])
      freeM(state.w_up[l])
      freeM(state.m_w_up[l])
      freeM(state.v_w_up[l])
      freeM(state.g_w_up[l])
      freeM(state.w_down[l])
      freeM(state.m_w_down[l])
      freeM(state.v_w_down[l])
      freeM(state.g_w_down[l])
      freeM(state.k_cache[l])
      freeM(state.v_cache[l])
      
    deallocShared(state.rms_att_weights)
    deallocShared(state.m_rms_att_weights)
    deallocShared(state.v_rms_att_weights)
    deallocShared(state.g_rms_att_weights)
    deallocShared(state.rms_ffn_weights)
    deallocShared(state.m_rms_ffn_weights)
    deallocShared(state.v_rms_ffn_weights)
    deallocShared(state.g_rms_ffn_weights)
    deallocShared(state.w_q)
    deallocShared(state.m_w_q)
    deallocShared(state.v_w_q)
    deallocShared(state.g_w_q)
    deallocShared(state.w_k)
    deallocShared(state.m_w_k)
    deallocShared(state.v_w_k)
    deallocShared(state.g_w_k)
    deallocShared(state.w_v)
    deallocShared(state.m_w_v)
    deallocShared(state.v_w_v)
    deallocShared(state.g_w_v)
    deallocShared(state.w_o)
    deallocShared(state.m_w_o)
    deallocShared(state.v_w_o)
    deallocShared(state.g_w_o)
    deallocShared(state.w_up)
    deallocShared(state.m_w_up)
    deallocShared(state.v_w_up)
    deallocShared(state.g_w_up)
    deallocShared(state.w_down)
    deallocShared(state.m_w_down)
    deallocShared(state.v_w_down)
    deallocShared(state.g_w_down)
    deallocShared(state.k_cache)
    deallocShared(state.v_cache)

  deallocShared(state)

# ==============================================================================
# FFI EXPORT: STREAM LOADER (V11.2 - SELF-VERIFYING)
# ------------------------------------------------------------------------------
# Layout-Annahme: [Header][ pro Tensor: W | m | v ][ optional int32 adam_t ]
# Diese Annahme wird VOR dem Laden ueber die Byte-Bilanz der Datei geprueft
# und NACH dem Laden ueber die harte Invariante v >= 0 bewiesen.
# Kein stilles Laden von Muell mehr: jeder Fehler gibt nil + Fehlercode zurueck.
# ==============================================================================

type
  ModelHeader {.packed.} = object
    magic: array[8, char]
    vocab_size: uint32
    hidden_dim: uint32
    ffn_dim: uint32
    ctx_size: uint32
    num_layers: uint32

const
  LOAD_OK          = 0.cint   # alles sauber
  LOAD_ERR_OPEN    = 1.cint   # Datei nicht oeffenbar
  LOAD_ERR_HEADER  = 2.cint   # Header unvollstaendig
  LOAD_ERR_MAGIC   = 3.cint   # kein SNAI_V5
  LOAD_ERR_SANITY  = 4.cint   # Header-Werte unplausibel
  LOAD_ERR_INIT    = 5.cint   # cortex_init fehlgeschlagen
  LOAD_ERR_SIZE    = 6.cint   # Dateigroesse passt zu keinem bekannten Layout
  LOAD_ERR_TRUNC   = 7.cint   # Datei bricht mittendrin ab
  LOAD_ERR_LAYOUT  = 8.cint   # v-Tensor negativ -> Reihenfolge im File anders
  LOAD_ERR_NAN     = 9.cint   # NaN/Inf in den Gewichten

var last_load_error: cint = LOAD_OK

proc cortex_get_load_error*(): cint {.ffi.} =
  return last_load_error

# Stichprobenscan: NaN/Inf immer, Negativwerte optional (fuer v-Tensoren)
proc scanBad(p: ptr FloatArray, n: int, mustBePositive: bool): bool =
  if p.isNil or n <= 0: return false
  let step = max(1, n div 8192)
  var i = 0
  while i < n:
    let x = p[i]
    if x.isNaN or x > 1e30 or x < -1e30: return true
    if mustBePositive and x < 0.0: return true
    i += step
  return false

proc isAllZero(p: ptr FloatArray, n: int): bool =
  if p.isNil or n <= 0: return true
  let step = max(1, n div 8192)
  var i = 0
  while i < n:
    if p[i] != 0.0: return false
    i += step
  return true

# Gesundheitscheck fuer einen State im Speicher. Bitmaske:
#   1 = tote Gain-Kanaele   2 = NaN/Inf in Gewichten   4 = adam_t 0 trotz warmem m/v
# 0 = sauber, -1 = kein State.
proc cortex_validate*(s: ptr CortexState): cint {.ffi.} =
  if s.isNil: return -1
  var flags: cint = 0

  if deadTailStart(s.rms_final_weight, s.hiddenDim) >= 0: flags = flags or 1
  for l in 0 ..< s.numLayers:
    if deadTailStart(s.rms_att_weights[l], s.hiddenDim) >= 0: flags = flags or 1
    if deadTailStart(s.rms_ffn_weights[l], s.hiddenDim) >= 0: flags = flags or 1

  let embSize = s.vocabSize.int * s.hiddenDim.int
  if scanBad(s.embeddings.data, embSize, false) or
     scanBad(s.w_out.data, embSize, false) or
     scanBad(s.rms_final_weight, s.hiddenDim.int, false): flags = flags or 2

  if s.adam_t <= 0 and not isAllZero(s.v_embeddings.data, embSize): flags = flags or 4
  return flags

# Misst die reine Roundtrip-Latenz eines GPU-Calls in Millisekunden.
# Damit laesst sich die Schwelle oben empirisch statt geschaetzt setzen:
# lohnend ist die GPU ab etwa (Latenz_in_ms * 1e6) Elementen.
proc cortex_bench_gpu*(state: ptr CortexState, iters: cint): cfloat {.ffi.} =
  if state.isNil or not gpu_initialized or iters <= 0: return -1.0
  let d = state.hiddenDim
  let m = cast[ptr cfloat](state.w_q[0].data)
  let i_v = cast[ptr cfloat](state.norm_buf)
  let o_v = cast[ptr cfloat](state.d_norm)
  gpuMatVecMul(m, i_v, o_v, d, d)          # Aufwaermen: Buffer anlegen
  let t0 = epochTime()
  for _ in 0 ..< iters:
    gpuMatVecMul(m, i_v, o_v, d, d)
  return (((epochTime() - t0) * 1000.0) / iters.float).cfloat

proc cortex_load_stream*(filepath_c: cstring, num_heads: cint): ptr CortexState {.ffi.} =
  last_load_error = LOAD_OK
  let filepath = $filepath_c
  var f: File
  if not open(f, filepath, fmRead):
    last_load_error = LOAD_ERR_OPEN
    return nil

  var header: ModelHeader
  if f.readBuffer(addr header, sizeof(ModelHeader)) != sizeof(ModelHeader):
    last_load_error = LOAD_ERR_HEADER
    f.close()
    return nil

  var magicStr = ""
  for c in header.magic:
    if c != '\0': magicStr.add(c)

  if magicStr != "SNAI_V5":
    last_load_error = LOAD_ERR_MAGIC
    stderr.writeLine("[cortex] FATAL: Magic '", magicStr, "' != 'SNAI_V5'.")
    f.close()
    return nil

  let
    nVocab  = header.vocab_size.int
    nDim    = header.hidden_dim.int
    nFfn    = header.ffn_dim.int
    nCtx    = header.ctx_size.int
    nLayers = header.num_layers.int

  # AUDIT-FIX V11.2: ffn_dim und ctx_size wurden vorher NICHT geprueft.
  if nVocab  < 1 or nVocab  > 4_000_000 or
     nDim    < 2 or nDim    > 65_536 or
     nFfn    < 1 or nFfn    > 262_144 or
     nCtx    < 1 or nCtx    > 1_048_576 or
     nLayers < 1 or nLayers > 512:
    last_load_error = LOAD_ERR_SANITY
    stderr.writeLine("[cortex] FATAL: Header unplausibel (V=", $nVocab, " D=", $nDim,
                     " F=", $nFfn, " C=", $nCtx, " L=", $nLayers, ").")
    f.close()
    return nil

  # --- BYTE-BILANZ: erwartete Nutzlast exakt ausrechnen ---
  let embSize = nVocab * nDim
  let dim2    = nDim * nDim
  let ffnSize = nFfn * nDim

  let elemsPerLayer = 2*nDim + 4*dim2 + 2*ffnSize
  let elemsTotal    = 2*embSize + nDim + nLayers*elemsPerLayer
  let wBytes        = elemsTotal.int64 * sizeof(cfloat).int64
  let payload       = f.getFileSize() - sizeof(ModelHeader).int64

  var withAdam = false   # Datei enthaelt m und v
  var withT    = false   # Datei enthaelt adam_t als int32-Trailer

  if payload == wBytes * 3:
    withAdam = true
  elif payload == wBytes * 3 + 4:
    withAdam = true
    withT = true
  elif payload == wBytes:
    discard
  elif payload == wBytes + 4:
    withT = true
  else:
    last_load_error = LOAD_ERR_SIZE
    stderr.writeLine("[cortex] FATAL: Groessen-Mismatch. Payload=", $payload,
                     " B. Erwartet: ", $wBytes, " (nur W) oder ", $(wBytes*3),
                     " (W+m+v). Delta zu W+m+v: ", $(payload - wBytes*3), " B.")
    f.close()
    return nil

  let s = cortex_init(nVocab.cint, nDim.cint, nFfn.cint, nCtx.cint, nLayers.cint, num_heads)
  if s.isNil:
    last_load_error = LOAD_ERR_INIT
    f.close()
    return nil

  # AUDIT-FIX V11.2: jeder Abbruch gibt jetzt auch den State frei (vorher: Leak).
  template rd(dest: pointer, count: int) =
    let need = count * sizeof(cfloat)
    if f.readBuffer(dest, need) != need:
      last_load_error = LOAD_ERR_TRUNC
      stderr.writeLine("[cortex] FATAL: Datei bricht ab (", $count, " Floats erwartet).")
      f.close()
      cortex_free(s)
      return nil

  template blast(pw, pm, pv: pointer, count: int) =
    rd(pw, count)
    if withAdam:
      rd(pm, count)
      rd(pv, count)

  # --- GLOBALE TENSOREN ---
  blast(s.embeddings.data, s.m_embeddings.data, s.v_embeddings.data, embSize)
  # WEIGHT TYING: s.w_out zeigt hier auf s.embeddings. Wuerde der w_out-Block
  # der Datei dorthin gelesen, ueberschriebe er die gerade geladenen Embeddings.
  # Bei einem gebunden gespeicherten Modell waere das folgenlos (beide Bloecke
  # sind identisch), beim Laden eines UNGEBUNDENEN Checkpoints in eine gebundene
  # Engine dagegen still falsch - die Embeddings waeren dann die Ausgabematrix.
  # Die Bytes muessen trotzdem konsumiert werden, sonst verrutscht alles
  # Folgende. Ziel ist g_w_out: exakt vocab*dim gross, bereits alloziert, und zu
  # Beginn jedes Trainingsschritts ohnehin genullt. Kein Zusatzspeicher, und die
  # Abbruchpruefung in rd() bleibt wirksam.
  if weight_tying == 0:
    blast(s.w_out.data, s.m_w_out.data, s.v_w_out.data, embSize)
  else:
    blast(s.g_w_out.data, s.g_w_out.data, s.g_w_out.data, embSize)
  blast(s.rms_final_weight, s.m_rms_final_weight, s.v_rms_final_weight, nDim)

  # --- LAYER TENSOREN ---
  for l in 0 ..< nLayers:
    blast(s.rms_att_weights[l], s.m_rms_att_weights[l], s.v_rms_att_weights[l], nDim)
    blast(s.rms_ffn_weights[l], s.m_rms_ffn_weights[l], s.v_rms_ffn_weights[l], nDim)

    blast(s.w_q[l].data, s.m_w_q[l].data, s.v_w_q[l].data, dim2)
    blast(s.w_k[l].data, s.m_w_k[l].data, s.v_w_k[l].data, dim2)
    blast(s.w_v[l].data, s.m_w_v[l].data, s.v_w_v[l].data, dim2)
    blast(s.w_o[l].data, s.m_w_o[l].data, s.v_w_o[l].data, dim2)

    blast(s.w_up[l].data, s.m_w_up[l].data, s.v_w_up[l].data, ffnSize)
    blast(s.w_down[l].data, s.m_w_down[l].data, s.v_w_down[l].data, ffnSize)

  if withT:
    var tRaw: int32 = 0
    if f.readBuffer(addr tRaw, 4) != 4:
      last_load_error = LOAD_ERR_TRUNC
      f.close()
      cortex_free(s)
      return nil
    s.adam_t = tRaw.cint

  f.close()

  # --- POST-CHECK 1: LAYOUT-BEWEIS -------------------------------------------
  # v ist das zweite Moment (Summe von Quadraten) und kann mathematisch
  # NIEMALS negativ sein. Negative Werte = die Datei liegt anders im Speicher
  # als hier gelesen. Das faengt genau den Fehler, den du befuerchtet hast.
  if withAdam:
    var layoutBroken = scanBad(s.v_embeddings.data, embSize, true) or
                       scanBad(s.v_w_out.data, embSize, true) or
                       scanBad(s.v_rms_final_weight, nDim, true)
    if not layoutBroken:
      for l in 0 ..< nLayers:
        if scanBad(s.v_w_q[l].data, dim2, true) or
           scanBad(s.v_w_o[l].data, dim2, true) or
           scanBad(s.v_w_down[l].data, ffnSize, true):
          layoutBroken = true
          break
    if layoutBroken:
      last_load_error = LOAD_ERR_LAYOUT
      stderr.writeLine("[cortex] FATAL: v-Tensor enthaelt negative Werte. Das File-Layout ",
                       "ist NICHT [W|m|v] pro Tensor. Writer-Seite pruefen.")
      cortex_free(s)
      return nil

  # --- POST-CHECK 2: NaN/Inf in den Gewichten --------------------------------
  if scanBad(s.embeddings.data, embSize, false) or
     scanBad(s.w_out.data, embSize, false) or
     scanBad(s.rms_final_weight, nDim, false):
    last_load_error = LOAD_ERR_NAN
    stderr.writeLine("[cortex] FATAL: NaN/Inf in den geladenen Gewichten.")
    cortex_free(s)
    return nil

  # --- POST-CHECK 3: ADAM-ZEITSCHRITT ----------------------------------------
  # KORREKTUR V11.2b: adam_t = 0 auf warmem m/v sprengt NICHTS, es bremst.
  #   t=1: m_hat = m/0.1 = 10m, v_hat = v/0.001 = 1000v -> sqrt = 31.6*sqrt(v)
  #        Schrittweite = 0.32x der eingeschwungenen Groesse, nicht 10x.
  #   Der Schaden ist ein ungewollter Re-Warmup ueber ~1000 Steps nach JEDEM
  #   Resume - schleichend, nicht explosiv. Trotzdem falsch.
  # Wichtig: nur bei WARMEM Adam-Zustand neutralisieren. Bei m = v = 0
  # (echter Cold Start) ist die Bias-Correction korrekt; adam_t = 1000 wuerde
  # den ersten Schritt dort um Faktor ~3.2 zu gross machen.
  if withAdam and s.adam_t <= 0:
    if isAllZero(s.v_embeddings.data, embSize):
      stderr.writeLine("[cortex] INFO: m/v sind leer -> Cold Start, adam_t bleibt 0.")
    else:
      s.adam_t = 1000
      stderr.writeLine("[cortex] WARN: adam_t fehlt im Checkpoint, m/v sind aber warm ",
                       "-> adam_t = 1000 (Bias-Correction neutralisiert). ",
                       "Besser: adam_t writer-seitig persistieren.")

  # --- POST-CHECK 4: NET2NET GAIN GUARD --------------------------------------
  # Faengt Checkpoints ab, die schon mit toten Kanaelen auf die Platte kamen.
  checkGains(cast[ptr cfloat](s.rms_final_weight), s.hiddenDim, "rms_final")
  for l in 0 ..< s.numLayers:
    checkGains(cast[ptr cfloat](s.rms_att_weights[l]), s.hiddenDim,
               "L" & $l & ".rms_att")
    checkGains(cast[ptr cfloat](s.rms_ffn_weights[l]), s.hiddenDim,
               "L" & $l & ".rms_ffn")

  return s

proc cortex_init_gpu*() {.ffi, exportc, dynlib, cdecl.} =
  # Update 2026-08-08: intelligenter Backend-Switch. CUDA zuerst versuchen -
  # aber NUR nachdem cuda_lib_loadable() (std/dynlib-basierter sicherer
  # Vortest, siehe cortex_cuda_bridge.nim) bestaetigt hat, dass
  # libcortex_cuda.so ueberhaupt ladbar ist. Ohne diesen Vortest wuerde
  # init_cuda_engine() auf einem Knoten ohne CUDA-Runtime (die meisten
  # Knoten im Schwarm, siehe docs/ROADMAP.md) den gesamten Prozess hart
  # abstuerzen lassen (empirisch verifiziert). Danach OpenCL, sonst CPU.
  # CUDA bleibt damit explizit opt-in pro Knoten, kein Ersatz fuer OpenCL.
  if cuda_lib_loadable() and init_cuda_engine() == 1:
    gpu_backend = gbCuda
    gpu_initialized = true
  elif init_opencl_engine() == 1:
    gpu_backend = gbOpenCL
    gpu_initialized = true
  else:
    gpu_backend = gbNone
    gpu_initialized = false

proc get_vocab_size*(state: ptr CortexState): cint {.ffi, exportc, dynlib, cdecl.} =
  if state.isNil: return 0
  return state.vocabSize

proc get_ctx_size*(state: ptr CortexState): cint {.ffi, exportc, dynlib, cdecl.} =
  if state.isNil: return 0
  return state.ctxSize

# Update 2026-08-09 (docs/ROADMAP.md, "Engine-/Framework-Verbesserungen",
# Punkt 1): Lese-Seite fuer den optionalen int32-Adam-Schritt-Trailer
# existiert seit Laengerem (siehe cortex_load_stream weiter unten, "withT"),
# adam_t war aber nie ueber FFI abrufbar - extract_state() in
# tensor_ffi_adapter.php konnte es folglich nie mitschreiben. Jeder
# gespeicherte Checkpoint verlor dadurch seinen Bias-Correction-Fortschritt
# (Fallback beim naechsten Laden: adam_t=1000, neutralisiert die Korrektur,
# sprengt nichts, bremst nur - siehe V11.2b-Kommentar unten).
proc cortex_get_adam_t*(state: ptr CortexState): cint {.ffi, exportc, dynlib, cdecl.} =
  if state.isNil: return 0
  return state.adam_t

proc cortex_read_j_space*(state: ptr CortexState, layer_idx: cint, raw_logits: ptr cfloat) {.ffi, exportc, dynlib, cdecl.} =
  if state.isNil or layer_idx < 0 or layer_idx >= state.numLayers: return
  let logits = cast[ptr FloatArray](raw_logits)
  
  applyRMSNorm(state.j_space_cache[layer_idx], state.norm_buf, state.hiddenDim, state.rms_final_weight)
  
  matVecMul(state.w_out, state.norm_buf, logits)