# Dateiname: snai_tensor_utils.nim
# Funktion: Native Merge-/Upscale-Bibliothek fuer royal_distillery.php.
#           Ersetzt PHP-seitiges pack()/unpack() (langsam, kein FP16-Support)
#           durch Zero-Copy-Pointer-Arithmetik ueber FFI.
# MAXIME: E > H.
#
# BUILD: braucht --panics:on zusaetzlich zu den ueblichen Flags (siehe
# docs/INSTALL.md fuer cortex.nim). Ohne das schlaegt der C-Compiler mit
# "ungueltiger Zweig zu/von strukturiertem OpenMP-Block" fehl, sobald
# innerhalb einer "||"-parallelisierten Schleife eine eigene Prozedur
# aufgerufen wird (hier: f16_to_f32/f32_to_f16 im FP16-Zweig von
# snai_tensor_merge) - Nims normales Exception-Modell fuegt dafuer einen
# "goto BeforeRet_"-Fehlerpfad ein, den OpenMP in einem strukturierten Block
# nicht erlaubt. --panics:on wandelt Defects in harte Aborts ohne diese
# Rueckabwicklungs-Scaffolding um. cortex.nim brauchte das nie, weil seine
# "||"-Schleifen nie eine separate Prozedur aufrufen, nur Inline-Arithmetik.
#   nim c -d:danger --panics:on --threads:on --passC:-fopenmp --passL:-fopenmp \
#         --app:lib --out:libtensorutils.so snai_tensor_utils.nim
#
# HERKUNFT: aus einem schnellen Entwurf des Nutzers hervorgegangen
# (2026-08-05) - hier mit zwei behobenen Luecken:
#   1. f16_to_f32() behandelte echte Subnormals (e==0, m!=0) nicht korrekt
#      (nur exaktes 0.0 wurde geflusht, das Original-Kommentar sagte aber
#      "Flush to zero fuer Subnormals").
#   2. snai_tensor_upscale() kannte nur zwei Fuellwerte (0.0 fuer Adam-Zustand,
#      Rauschen fuer alles andere) - der bestehende PHP-Code unterscheidet
#      aber DREI Faelle: Adam->0.0, Gewichte->Rauschen-um-0.0,
#      RMSNorm-Gains->Rauschen-um-1.0 (siehe pad_center in
#      royal_distillery.php::upscale_matrix_stream()). Ohne diese dritte
#      Unterscheidung wuerden frisch expandierte RMSNorm-Gains bei 0 statt 1
#      landen - der schon einmal gefixte "Net2Net-Gain-Bug" (Gain=0 ist ein
#      Sattelpunkt ohne Gradient, der Kanal bleibt fuer immer stumm).

import std/random
import std/math

{.pragma: ffi, exportc, dynlib, cdecl.}

type
  Float32Array = UncheckedArray[float32]
  Float16Array = UncheckedArray[uint16] # FP16 wird als roher 16-Bit Int durchgereicht
  PointerArray = UncheckedArray[pointer]

const
  SNAI_TYPE_FP32*: cint = 1
  SNAI_TYPE_FP16*: cint = 2

randomize()

# -------------------------------------------------------------------------
# FP16 <-> FP32 BIT-TWIDDLING (PHP kennt kein FP16, Nim macht die Magie)
# -------------------------------------------------------------------------

proc f16_to_f32(h: uint16): float32 {.inline, raises: [].} =
  let s = (h.uint32 and 0x8000'u32) shl 16
  let e = (h.uint32 and 0x7C00'u32) shr 10
  let m = (h.uint32 and 0x03FF'u32)

  # AUDIT-FIX: vorher nur e==0 UND m==0 (exakte Null) geflusht - echte
  # Subnormals (e==0, m!=0, Betrag < ~6.1e-5) fielen durch die allgemeine
  # Formel unten und ergaben eine falsche Groessenordnung. Fuer
  # KI-Gewichte/Adam-Momentum/-Varianz nahe Null (durchaus real, gerade
  # frueh im Training oder bei kaum aktiven Parametern) ist Flush-to-Zero
  # fuer den gesamten Subnormal-Bereich eine uebliche, vernachlaessigbare
  # Vereinfachung - kein Praezisionsverlust, der hier relevant waere.
  if e == 0: return cast[float32](s)

  let e32 = (e + 112'u32) shl 23
  let m32 = m shl 13
  return cast[float32](s or e32 or m32)

proc f32_to_f16(f: float32): uint16 {.inline, raises: [].} =
  let b = cast[uint32](f)
  let s = (b and 0x80000000'u32) shr 16
  let e = (b and 0x7F800000'u32) shr 23
  let m = (b and 0x007FFFFF'u32)

  # AUDIT-FIX: NaN (e==255, m!=0) fiel vorher in denselben Zweig wie
  # Overflow und wurde zu Inf - NaN-Werte sollten aber ohnehin nie hier
  # ankommen (cortex.nim::adamStep() faengt NaN-Gradienten schon vorher ab,
  # snai_xray.php meldet NaN in Gewichten als kritischen Gesundheits-Fehler),
  # aber falls doch: NaN bleibt jetzt NaN statt still zu Inf zu werden.
  if e == 0xFF'u32 and m != 0: return (s or 0x7E00'u32).uint16

  if e <= 112: return s.uint16 # Underflow -> 0.0 (deckt auch Subnormals ab)
  if e >= 143: return (s or 0x7C00'u32).uint16 # Overflow -> Inf

  let e16 = (e - 112'u32) shl 10
  let m16 = m shr 13
  return (s or e16 or m16).uint16

# -------------------------------------------------------------------------
# OPERATION 1: FEDERATED AVERAGING (OOM-Safe Chunk Merging)
# -------------------------------------------------------------------------

proc snai_tensor_merge*(out_buf: pointer, in_buffers: ptr pointer, num_models: cint, elements: csize_t, dtype: cint) {.ffi.} =
  if out_buf == nil or in_buffers == nil or num_models <= 0 or elements == 0: return
  let p_in = cast[ptr PointerArray](in_buffers)
  let n = num_models.int

  # AUDIT-FIX: der urspruengliche Entwurf nutzte das Pragma
  # "{.omp: parallel for.}" - das ist keine gueltige Nim-Syntax und liess den
  # Build fehlschlagen. Im Rest des Projekts (cortex.nim) wird OpenMP ueber
  # den eingebauten "||"-Parallel-Operator angesteuert (kompiliert bei
  # --passC:-fopenmp automatisch zu einer OpenMP-parallel-for-Schleife) -
  # dasselbe, bewaehrte Muster hier uebernommen.
  if dtype == SNAI_TYPE_FP32:
    let p_out = cast[ptr Float32Array](out_buf)
    # AUDIT-FIX: die Pointer-Umwandlung MUSS ausserhalb der parallelen
    # Schleife passieren - ein cast() im Schleifenkoerper liess den C-
    # Compiler mit "ungueltiger Zweig zu/von strukturiertem OpenMP-Block"
    # scheitern. Ein newSeq() dafuer half NICHT (GC-verwaltete Typen bringen
    # trotz -d:danger noch ORC-Fehlerpfade mit, die denselben Konflikt
    # ausloesen) - deshalb hier rohes alloc()/dealloc() ohne jeden
    # GC-Bezug, wie im Rest des Projekts (cortex.nim/cortex_opencl.nim)
    # ueblich. Die Modell-Pointer aendern sich ohnehin nicht pro Element i,
    # nur einmal vorab umwandeln.
    let typed_in = cast[ptr UncheckedArray[ptr Float32Array]](alloc(n * sizeof(ptr Float32Array)))
    for m in 0 ..< n: typed_in[m] = cast[ptr Float32Array](p_in[m])

    for i in 0 || (elements.int - 1):
      var sum: float32 = 0.0
      for m in 0 ..< n:
        sum += typed_in[m][i]
      p_out[i] = sum / n.float32
    dealloc(typed_in)

  elif dtype == SNAI_TYPE_FP16:
    let p_out = cast[ptr Float16Array](out_buf)
    let typed_in = cast[ptr UncheckedArray[ptr Float16Array]](alloc(n * sizeof(ptr Float16Array)))
    for m in 0 ..< n: typed_in[m] = cast[ptr Float16Array](p_in[m])

    for i in 0 || (elements.int - 1):
      var sum: float32 = 0.0
      for m in 0 ..< n:
        sum += f16_to_f32(typed_in[m][i])
      p_out[i] = f32_to_f16(sum / n.float32)
    dealloc(typed_in)

# -------------------------------------------------------------------------
# OPERATION 2: NET2NET MATRIX UPSCALING
# -------------------------------------------------------------------------

proc snai_tensor_upscale*(in_buf: pointer, out_buf: pointer, old_r, old_c, new_r, new_c: cint,
                          scale: float32, noise_factor: float32, pad_center: float32,
                          is_adam: cint, dtype: cint) {.ffi.} =
  # Expandiert eine Matrix. Alte Werte erhalten 'scale', neue Bereiche:
  #   is_adam == 1            -> exakt 0.0 (Adam-Zustand darf kein Rauschen bekommen)
  #   sonst                   -> pad_center + kleines Rauschen (0.0 fuer Gewichte,
  #                               1.0 fuer RMSNorm-Gains - siehe AUDIT-FIX oben)
  if in_buf == nil or out_buf == nil or new_r <= 0 or new_c <= 0: return
  let old_r_i = old_r.int
  let old_c_i = old_c.int
  let new_r_i = new_r.int
  let new_c_i = new_c.int

  # AUDIT: bewusst NICHT mit OpenMP parallelisiert (anders als
  # snai_tensor_merge oben) - rand() aus std/random nutzt einen globalen,
  # nicht nachweislich thread-sicheren Zufallszustand. Der Gewinn durch
  # Parallelisierung waere hier ohnehin klein gegen den Sprung von
  # PHP-pack()/unpack() pro einzelnem Float auf einen nativen Bulk-Durchlauf.
  # Sicherheit vor der letzten moeglichen Beschleunigung.
  if dtype == SNAI_TYPE_FP32:
    let p_in = cast[ptr Float32Array](in_buf)
    let p_out = cast[ptr Float32Array](out_buf)

    for r in 0 ..< new_r_i:
      for c in 0 ..< new_c_i:
        let out_idx = r * new_c_i + c
        if r < old_r_i and c < old_c_i:
          let in_idx = r * old_c_i + c
          p_out[out_idx] = p_in[in_idx] * scale
        elif is_adam == 1:
          p_out[out_idx] = 0.0'f32
        else:
          let noise = (rand(2.0) - 1.0) * noise_factor
          p_out[out_idx] = pad_center + noise.float32

  elif dtype == SNAI_TYPE_FP16:
    let p_in = cast[ptr Float16Array](in_buf)
    let p_out = cast[ptr Float16Array](out_buf)

    for r in 0 ..< new_r_i:
      for c in 0 ..< new_c_i:
        let out_idx = r * new_c_i + c
        if r < old_r_i and c < old_c_i:
          let in_idx = r * old_c_i + c
          let val = f16_to_f32(p_in[in_idx]) * scale
          p_out[out_idx] = f32_to_f16(val)
        elif is_adam == 1:
          p_out[out_idx] = f32_to_f16(0.0'f32)
        else:
          let noise = (rand(2.0) - 1.0) * noise_factor
          p_out[out_idx] = f32_to_f16(pad_center + noise.float32)
