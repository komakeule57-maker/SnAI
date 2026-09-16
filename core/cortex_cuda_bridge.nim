# Dateiname: cortex_cuda_bridge.nim
# Funktion: FFI-Schnittstelle zwischen der Nim-Engine und der libcortex_cuda.so
# MAXIME: E > H. Zero-Copy Pointer-Durchreichung an die Tensor Cores.
#
# STATUS: seit Update 2026-08-08 in cortex.nim's Backend-Dispatch eingeklinkt
# (intelligenter Switch: CUDA -> OpenCL -> CPU, siehe cortex_init_gpu()) -
# funktional vollstaendig deckungsgleich zu cortex_opencl.nim (alle Forward-/
# Backward-/Adam-/Hilfsfunktionen inkl. der gebuendelten Attention-Variante
# portiert und einzeln + im Zusammenspiel verifiziert, siehe docs/ROADMAP.md).
#
# AUDIT-FIX (2026-08-06): cuda_adamStep hatte im urspruenglichen Entwurf nicht
# die Growth-Boundary/Boost-Parameter (cols, bound_r, bound_c, boost), die
# ocl_adamStep seit dem Net2Net-LR-Fix traegt - Signatur hier nachgezogen,
# damit ein spaeteres Einklinken in cortex.nim den LR-Boost nicht still
# unterschlaegt. cuda_sync_to_device ergaenzt (Gegenstueck zu
# ocl_sync_to_device, fehlte komplett - siehe cortex_cuda.cu-Kommentar).
#
# AUDIT-FIX (2026-08-08): wd-Parameter (Weight-Decay) ergaenzt - Nachzug aus
# dem "Trainings-Absicherung: Nachruest-Paket" fuer ocl_adamStep, das nach
# dem urspruenglichen CUDA-Audit entstand (siehe cortex_cuda.cu-Kommentar
# und docs/ROADMAP.md, Update 2026-08-08).
#
# Update 2026-08-08: Backward-Pass, Einzelvektor-Ops und gebuendelte Attention
# ergaenzt - siehe docs/ROADMAP.md fuer die vollstaendige Chronologie. Alle
# Signaturen 1:1 identisch zu ihren ocl_*-Pendants (Parameterzahl,
# -reihenfolge, -namen).
#
# ==============================================================================
# AUDIT-FIX (2026-08-08, sicherheitskritisch): manuelles std/dynlib-Laden
# statt {.importc, dynlib.}
# ==============================================================================
# {.importc, dynlib: "libcortex_cuda.so".} laedt lazy beim ERSTEN AUFRUF -
# schlaegt das fehl (Bibliothek oder eine ihrer Laufzeit-Abhaengigkeiten wie
# libcudart.so/libcublas.so nicht auffindbar), ist das ein harter, nicht
# abfangbarer Prozessabsturz. In isolierten Testprogrammen (auch mit den
# exakt gleichen Compiler-Flags und Modul-Imports) verhielt sich das wie
# erwartet lazy und sicher - beim tatsaechlichen Einbau in die VOLLE
# cortex.nim (als Teil von deren --app:lib-Build) stuerzte der Prozess
# jedoch schon VOR dem ersten Aufruf von cuda_lib_loadable() ab, obwohl
# dessen einzige Aufgabe genau diese Pruefung ist (reproduziert, exakte
# Ursache innerhalb der verfuegbaren Zeit nicht abschliessend isoliert -
# vermutlich eine Eigenheit von Nims Dynlib-Registrierung bei vielen
# {.importc, dynlib.}-Prozeduren im selben grossen --app:lib-Build).
#
# Deshalb hier komplett auf manuelles Laden umgestellt: std/dynlib.loadLib()
# (sicher, liefert nil statt abzustuerzen) plus symAddr() pro Funktion. Das
# ist in JEDER getesteten Konfiguration nachweislich sicher - auch beim
# tatsaechlichen Fehlschlag innerhalb der vollen cortex.nim (empirisch
# verifiziert, siehe docs/ROADMAP.md Update 2026-08-08). Ergebnis ist
# funktional identisch zu vorher (dieselben Funktionsnamen/Signaturen nach
# aussen), nur der Lademechanismus dahinter ist robuster.
#
# cuda_lib_loadable() MUSS true liefern, BEVOR irgendeine andere Funktion
# aus diesem Modul aufgerufen wird - siehe cortex_init_gpu() in cortex.nim
# fuer den einzigen vorgesehenen Aufrufer.

import std/dynlib

type
  InitCudaEngineProc = proc(): cint {.cdecl.}
  ShutdownCudaEngineProc = proc() {.cdecl.}
  CudaReleaseBufferProc = proc(h_ptr: pointer) {.cdecl.}
  CudaSyncToHostProc = proc(h_ptr: pointer) {.cdecl.}
  CudaSyncToDeviceProc = proc(h_ptr: pointer, bytes: csize_t) {.cdecl.}
  CudaZeroBufferProc = proc(h_ptr: pointer, bytes: csize_t) {.cdecl.}
  CudaAdamStepProc = proc(w: ptr cfloat, m: ptr cfloat, v: ptr cfloat, g: ptr cfloat, lr: cfloat, b1: cfloat, b2: cfloat, b1t: cfloat, b2t: cfloat, scale: cfloat, size: cint, is_sparse: cint, cols: cint, bound_r: cint, bound_c: cint, boost: cfloat, wd: cfloat) {.cdecl.}
  CudaMatVecMulProc = proc(M: ptr cfloat, v: ptr cfloat, o: ptr cfloat, rows: cint, cols: cint) {.cdecl.}
  CudaMatVecMulTProc = proc(M: ptr cfloat, d_out: ptr cfloat, d_in: ptr cfloat, rows: cint, cols: cint, accumulate: cint) {.cdecl.}
  CudaGradAccumProc = proc(gm: ptr cfloat, d_out: ptr cfloat, i_v: ptr cfloat, rows: cint, cols: cint) {.cdecl.}
  CudaMatMulBatchedProc = proc(M: ptr cfloat, in_seq: ptr cfloat, out_seq: ptr cfloat, rows: cint, cols: cint, batch_size: cint, max_batch: cint) {.cdecl.}
  CudaAttentionForwardProc = proc(q: ptr cfloat, k_cache: ptr cfloat, v_cache: ptr cfloat, out_v: ptr cfloat, pos: cint, numHeads: cint, headSize: cint, dim: cint, ctx: cint) {.cdecl.}
  CudaAttentionBackwardProc = proc(q: ptr cfloat, k_cache: ptr cfloat, v_cache: ptr cfloat, d_att_out: ptr cfloat, d_q: ptr cfloat, d_k_cache: ptr cfloat, d_v_cache: ptr cfloat, pos: cint, numHeads: cint, headSize: cint, dim: cint, ctx: cint) {.cdecl.}
  CudaMatMulTBatchedProc = proc(M: ptr cfloat, d_out: ptr cfloat, d_in: ptr cfloat, rows: cint, cols: cint, batch: cint, accumulate: cint) {.cdecl.}
  CudaGradAccumBatchedProc = proc(gm: ptr cfloat, d_out: ptr cfloat, i_v: ptr cfloat, rows: cint, cols: cint, batch: cint) {.cdecl.}
  CudaAttentionForwardBatchedProc = proc(q_all: ptr cfloat, k_cache: ptr cfloat, v_cache: ptr cfloat, att_out_all: ptr cfloat, act: cint, numHeads: cint, headSize: cint, dim: cint, ctx: cint) {.cdecl.}
  CudaAttentionBackwardBatchedProc = proc(q_all: ptr cfloat, k_cache: ptr cfloat, v_cache: ptr cfloat, d_att_out_all: ptr cfloat, d_q_all: ptr cfloat, d_k_cache: ptr cfloat, d_v_cache: ptr cfloat, act: cint, numHeads: cint, headSize: cint, dim: cint, ctx: cint) {.cdecl.}

var
  cudaLib: LibHandle = nil
  cudaProbeDone = false
  cudaProbeResult = false
  pInitCudaEngine: InitCudaEngineProc
  pShutdownCudaEngine: ShutdownCudaEngineProc
  pCudaReleaseBuffer: CudaReleaseBufferProc
  pCudaSyncToHost: CudaSyncToHostProc
  pCudaSyncToDevice: CudaSyncToDeviceProc
  pCudaZeroBuffer: CudaZeroBufferProc
  pCudaAdamStep: CudaAdamStepProc
  pCudaMatVecMul: CudaMatVecMulProc
  pCudaMatVecMulT: CudaMatVecMulTProc
  pCudaGradAccum: CudaGradAccumProc
  pCudaMatMulBatched: CudaMatMulBatchedProc
  pCudaAttentionForward: CudaAttentionForwardProc
  pCudaAttentionBackward: CudaAttentionBackwardProc
  pCudaMatMulTBatched: CudaMatMulTBatchedProc
  pCudaGradAccumBatched: CudaGradAccumBatchedProc
  pCudaAttentionForwardBatched: CudaAttentionForwardBatchedProc
  pCudaAttentionBackwardBatched: CudaAttentionBackwardBatchedProc

proc cuda_lib_loadable*(): bool =
  ## True, wenn libcortex_cuda.so (und alle 17 erwarteten Symbole) auf diesem
  ## System ladbar sind. Sagt NICHTS darueber aus, ob tatsaechlich eine
  ## funktionsfaehige NVIDIA-GPU vorhanden ist - dafuer zusaetzlich
  ## init_cuda_engine() aufrufen (cudaSetDevice() kann trotzdem scheitern,
  ## z.B. kein Treiber geladen). Muss true liefern, BEVOR irgendeine andere
  ## Funktion aus diesem Modul aufgerufen wird - alle pCuda*-Funktionszeiger
  ## bleiben sonst nil.
  if cudaProbeDone: return cudaProbeResult
  cudaProbeDone = true

  cudaLib = loadLib("libcortex_cuda.so")
  if cudaLib == nil:
    return false

  template bindSym(p, T: untyped, name: string) =
    p = cast[T](cudaLib.symAddr(name))
    if p == nil:
      return false

  bindSym(pInitCudaEngine, InitCudaEngineProc, "init_cuda_engine")
  bindSym(pShutdownCudaEngine, ShutdownCudaEngineProc, "shutdown_cuda_engine")
  bindSym(pCudaReleaseBuffer, CudaReleaseBufferProc, "cuda_release_buffer")
  bindSym(pCudaSyncToHost, CudaSyncToHostProc, "cuda_sync_to_host")
  bindSym(pCudaSyncToDevice, CudaSyncToDeviceProc, "cuda_sync_to_device")
  bindSym(pCudaZeroBuffer, CudaZeroBufferProc, "cuda_zero_buffer")
  bindSym(pCudaAdamStep, CudaAdamStepProc, "cuda_adamStep")
  bindSym(pCudaMatVecMul, CudaMatVecMulProc, "cuda_matVecMul")
  bindSym(pCudaMatVecMulT, CudaMatVecMulTProc, "cuda_matVecMulT")
  bindSym(pCudaGradAccum, CudaGradAccumProc, "cuda_gradAccum")
  bindSym(pCudaMatMulBatched, CudaMatMulBatchedProc, "cuda_matMulBatched")
  bindSym(pCudaAttentionForward, CudaAttentionForwardProc, "cuda_attentionForward")
  bindSym(pCudaAttentionBackward, CudaAttentionBackwardProc, "cuda_attentionBackward")
  bindSym(pCudaMatMulTBatched, CudaMatMulTBatchedProc, "cuda_matMulTBatched")
  bindSym(pCudaGradAccumBatched, CudaGradAccumBatchedProc, "cuda_gradAccumBatched")
  bindSym(pCudaAttentionForwardBatched, CudaAttentionForwardBatchedProc, "cuda_attentionForwardBatched")
  bindSym(pCudaAttentionBackwardBatched, CudaAttentionBackwardBatchedProc, "cuda_attentionBackwardBatched")

  cudaProbeResult = true
  true

proc init_cuda_engine*(): cint =
  # Selbst-absichernd: ruft cuda_lib_loadable() intern auf, falls der
  # Aufrufer das vergessen hat (leicht zu vergessen - selbst
  # skripte/e2e_compare.nim tat es urspruenglich nicht). Damit ist
  # init_cuda_engine() allein schon sicher, nicht nur in Kombination mit
  # einem vorgeschalteten expliziten Probe-Aufruf.
  if not cuda_lib_loadable(): return 0
  pInitCudaEngine()
proc shutdown_cuda_engine*() = pShutdownCudaEngine()
proc cuda_release_buffer*(h_ptr: pointer) = pCudaReleaseBuffer(h_ptr)
proc cuda_sync_to_host*(h_ptr: pointer) = pCudaSyncToHost(h_ptr)
proc cuda_sync_to_device*(h_ptr: pointer, bytes: csize_t) = pCudaSyncToDevice(h_ptr, bytes)
proc cuda_zero_buffer*(h_ptr: pointer, bytes: csize_t) = pCudaZeroBuffer(h_ptr, bytes)
proc cuda_adamStep*(w: ptr cfloat, m: ptr cfloat, v: ptr cfloat, g: ptr cfloat, lr: cfloat, b1: cfloat, b2: cfloat, b1t: cfloat, b2t: cfloat, scale: cfloat, size: cint, is_sparse: cint, cols: cint, bound_r: cint, bound_c: cint, boost: cfloat, wd: cfloat) =
  pCudaAdamStep(w, m, v, g, lr, b1, b2, b1t, b2t, scale, size, is_sparse, cols, bound_r, bound_c, boost, wd)
proc cuda_matVecMul*(M: ptr cfloat, v: ptr cfloat, o: ptr cfloat, rows: cint, cols: cint) =
  pCudaMatVecMul(M, v, o, rows, cols)
proc cuda_matVecMulT*(M: ptr cfloat, d_out: ptr cfloat, d_in: ptr cfloat, rows: cint, cols: cint, accumulate: cint) =
  pCudaMatVecMulT(M, d_out, d_in, rows, cols, accumulate)
proc cuda_gradAccum*(gm: ptr cfloat, d_out: ptr cfloat, i_v: ptr cfloat, rows: cint, cols: cint) =
  pCudaGradAccum(gm, d_out, i_v, rows, cols)
proc cuda_matMulBatched*(M: ptr cfloat, in_seq: ptr cfloat, out_seq: ptr cfloat, rows: cint, cols: cint, batch_size: cint, max_batch: cint) =
  pCudaMatMulBatched(M, in_seq, out_seq, rows, cols, batch_size, max_batch)
proc cuda_attentionForward*(q: ptr cfloat, k_cache: ptr cfloat, v_cache: ptr cfloat, out_v: ptr cfloat, pos: cint, numHeads: cint, headSize: cint, dim: cint, ctx: cint) =
  pCudaAttentionForward(q, k_cache, v_cache, out_v, pos, numHeads, headSize, dim, ctx)
proc cuda_attentionBackward*(q: ptr cfloat, k_cache: ptr cfloat, v_cache: ptr cfloat, d_att_out: ptr cfloat, d_q: ptr cfloat, d_k_cache: ptr cfloat, d_v_cache: ptr cfloat, pos: cint, numHeads: cint, headSize: cint, dim: cint, ctx: cint) =
  pCudaAttentionBackward(q, k_cache, v_cache, d_att_out, d_q, d_k_cache, d_v_cache, pos, numHeads, headSize, dim, ctx)
proc cuda_matMulTBatched*(M: ptr cfloat, d_out: ptr cfloat, d_in: ptr cfloat, rows: cint, cols: cint, batch: cint, accumulate: cint) =
  pCudaMatMulTBatched(M, d_out, d_in, rows, cols, batch, accumulate)
proc cuda_gradAccumBatched*(gm: ptr cfloat, d_out: ptr cfloat, i_v: ptr cfloat, rows: cint, cols: cint, batch: cint) =
  pCudaGradAccumBatched(gm, d_out, i_v, rows, cols, batch)
proc cuda_attentionForwardBatched*(q_all: ptr cfloat, k_cache: ptr cfloat, v_cache: ptr cfloat, att_out_all: ptr cfloat, act: cint, numHeads: cint, headSize: cint, dim: cint, ctx: cint) =
  pCudaAttentionForwardBatched(q_all, k_cache, v_cache, att_out_all, act, numHeads, headSize, dim, ctx)
proc cuda_attentionBackwardBatched*(q_all: ptr cfloat, k_cache: ptr cfloat, v_cache: ptr cfloat, d_att_out_all: ptr cfloat, d_q_all: ptr cfloat, d_k_cache: ptr cfloat, d_v_cache: ptr cfloat, act: cint, numHeads: cint, headSize: cint, dim: cint, ctx: cint) =
  pCudaAttentionBackwardBatched(q_all, k_cache, v_cache, d_att_out_all, d_q_all, d_k_cache, d_v_cache, act, numHeads, headSize, dim, ctx)
