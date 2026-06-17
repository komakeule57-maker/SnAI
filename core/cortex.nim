# ==============================================================================
# Dateiname: cortex.nim
# Funktion: Symbio Nano-AI Framework - Cortex Core Backend (FFI, Inference, BPTT)
# ==============================================================================
# SYMBIO NANO-AI FRAMEWORK (Code-4372)
# Modul: Cortex Core (Nim FFI Backend) - PRODUCTION V9.9 (LOBOTOMIE-FIX)
# ------------------------------------------------------------------------------
# MAXIME: E > H. Der absolute VRAM Sweetspot.
# FIX: Multi-Head Attention Indexing Bug gefixt! Die Heads haben sich vorher 
#      gegenseitig überschrieben / falsch aus dem Value-Cache gelesen. Das 
#      Netzwerk war effektiv "blind".
# FIX: Semicolon-Artefakte entfernt. Idiomatisches Nim-Spacing.
# ==============================================================================

import std/math
import cortex_opencl

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
    act_hidden_final*: ptr FloatArray

    hidden_state_buf*, norm_buf*, att_out_buf*, ffn_buf*, q_buf*: ptr FloatArray
    d_logits_buf*, scores_buf*, d_w_buf*: ptr FloatArray
    d_hidden*, d_norm*, d_att_out*, d_q*, d_k*, d_v*, d_ffn*: ptr FloatArray

var gpu_initialized: bool = false 

proc allocMatrix(r, c: cint): Matrix =
  result.rows = r
  result.cols = c
  result.data = cast[ptr FloatArray](allocShared0(cast[int](r.uint64 * c.uint64 * sizeof(cfloat).uint64)))

proc freeMatrix(m: var Matrix) =
  if not m.data.isNil:
    deallocShared(m.data)
  m.data = nil

# --- MATH CORE ---
proc applyRoPE(vec: ptr FloatArray, dim, head_size, pos: cint, backward: bool = false) =
  let num_heads = dim div head_size
  let sign: cfloat = if backward: -1.0 else: 1.0
  for h in 0 ..< num_heads:
    let h_off = h * head_size
    for i in countup(0, head_size - 1, 2):
      let freq = 1.0 / pow(10000.0, i.float32 / head_size.float32)
      let theta = pos.float32 * freq
      let cos_t = cos(theta).cfloat
      let sin_t = sign * sin(theta).cfloat
      let v0 = vec[h_off + i]
      let v1 = vec[h_off + i + 1]
      vec[h_off + i] = v0 * cos_t - v1 * sin_t
      vec[h_off + i + 1] = v0 * sin_t + v1 * cos_t

proc matVecMul*(m: Matrix, i_v, o_v: ptr FloatArray) =
  if m.rows >= 128 and m.cols >= 128 and gpu_initialized:
    ocl_matVecMul(cast[ptr cfloat](m.data), cast[ptr cfloat](i_v), cast[ptr cfloat](o_v), m.rows, m.cols)
  else:
    for r in 0 ..< m.rows:
      var s: cfloat = 0.0
      let off = r * m.cols
      for c in 0 ..< m.cols: 
        s += m.data[off + c] * i_v[c]
      o_v[r] = s

proc matVecMulBackwardAcc*(m, gm: Matrix, i_v, d_out, d_in: ptr FloatArray, accumulate: bool = false) =
  if not d_in.isNil:
    if not accumulate:
      for c in 0 ..< m.cols: 
        d_in[c] = 0.0
    for r in 0 ..< m.rows:
      let do_v = d_out[r]
      let off = r * m.cols
      for c in 0 ..< m.cols:
        d_in[c] += m.data[off + c] * do_v
        
  for r in 0 ..< m.rows:
    let do_v = d_out[r]
    let off = r * m.cols
    for c in 0 ..< m.cols: 
      gm.data[off + c] += do_v * i_v[c]

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
proc adamStep(w, m, v, g: ptr cfloat, lr, b1, b2, b1t, b2t, scale: cfloat, is_sparse: bool = false) {.inline.} =
  if scale == 0.0: return 
  var grad = g[] / scale
  if is_sparse and grad == 0.0: return
  if grad.isNaN or grad > 9999.0 or grad < -9999.0:
    g[] = 0.0
    return
 
  m[] = b1 * m[] + (1.0 - b1) * grad
  v[] = b2 * v[] + (1.0 - b2) * grad * grad
  let m_hat = m[] / b1t
  let v_hat = v[] / b2t 
  w[] -= lr * m_hat / (sqrt(v_hat) + 1e-8)
  g[] = 0.0

proc applyGlobalAdamUpdate(state: ptr CortexState, lr: cfloat, scale: cfloat) =
  state.adam_t += 1
  let b1: cfloat = 0.9
  let b2: cfloat = 0.999
  let b1t = 1.0 - pow(b1, state.adam_t.float32)
  let b2t = 1.0 - pow(b2, state.adam_t.float32)
  
  if gpu_initialized:
    template oclUpM(w, m, v, g) = ocl_adamStep(cast[ptr cfloat](w.data), cast[ptr cfloat](m.data), cast[ptr cfloat](v.data), cast[ptr cfloat](g.data), lr, b1, b2, b1t, b2t, scale, w.rows * w.cols, 0.cint)
    template oclUpEmb(w, m, v, g) = ocl_adamStep(cast[ptr cfloat](w.data), cast[ptr cfloat](m.data), cast[ptr cfloat](v.data), cast[ptr cfloat](g.data), lr, b1, b2, b1t, b2t, scale, w.rows * w.cols, 1.cint)
    template oclUpA(w, m, v, g, size) = ocl_adamStep(cast[ptr cfloat](w), cast[ptr cfloat](m), cast[ptr cfloat](v), cast[ptr cfloat](g), lr, b1, b2, b1t, b2t, scale, size, 0.cint)

    oclUpEmb(state.embeddings, state.m_embeddings, state.v_embeddings, state.g_embeddings)
  
    oclUpM(state.w_out, state.m_w_out, state.v_w_out, state.g_w_out)
    oclUpA(state.rms_final_weight, state.m_rms_final_weight, state.v_rms_final_weight, state.g_rms_final_weight, state.hiddenDim)

    for l in 0 ..< state.numLayers:
      oclUpA(state.rms_att_weights[l], state.m_rms_att_weights[l], state.v_rms_att_weights[l], state.g_rms_att_weights[l], state.hiddenDim)
      oclUpA(state.rms_ffn_weights[l], state.m_rms_ffn_weights[l], state.v_rms_ffn_weights[l], state.g_rms_ffn_weights[l], state.hiddenDim)
      oclUpM(state.w_q[l], state.m_w_q[l], state.v_w_q[l], state.g_w_q[l])
      oclUpM(state.w_k[l], state.m_w_k[l], state.v_w_k[l], state.g_w_k[l])
      oclUpM(state.w_v[l], state.m_w_v[l], state.v_w_v[l], state.g_w_v[l])
      oclUpM(state.w_o[l], state.m_w_o[l], state.v_w_o[l], state.g_w_o[l])
      oclUpM(state.w_up[l], state.m_w_up[l], state.v_w_up[l], state.g_w_up[l])
      oclUpM(state.w_down[l], state.m_w_down[l], state.v_w_down[l], state.g_w_down[l])
  else:
    template upM(w, m, v, g) = 
      for i in 0 ..< (w.rows * w.cols): adamStep(addr w.data[i], addr m.data[i], addr v.data[i], addr g.data[i], lr, b1, b2, b1t, b2t, scale)
    template upEmb(w, m, v, g) = 
      for i in 0 ..< (w.rows * w.cols): adamStep(addr w.data[i], addr m.data[i], addr v.data[i], addr g.data[i], lr, b1, b2, b1t, b2t, scale, true)
    template upA(w, m, v, g, size) = 
      for i in 0 ..< size: adamStep(addr w[i], addr m[i], addr v[i], addr g[i], lr, b1, b2, b1t, b2t, scale)

    upEmb(state.embeddings, state.m_embeddings, state.v_embeddings, state.g_embeddings)
  
    upM(state.w_out, state.m_w_out, state.v_w_out, state.g_w_out)
    upA(state.rms_final_weight, state.m_rms_final_weight, state.v_rms_final_weight, state.g_rms_final_weight, state.hiddenDim)

    for l in 0 ..< state.numLayers:
      upA(state.rms_att_weights[l], state.m_rms_att_weights[l], state.v_rms_att_weights[l], state.g_rms_att_weights[l], state.hiddenDim)
      upA(state.rms_ffn_weights[l], state.m_rms_ffn_weights[l], state.v_rms_ffn_weights[l], state.g_rms_ffn_weights[l], state.hiddenDim)
      upM(state.w_q[l], state.m_w_q[l], state.v_w_q[l], state.g_w_q[l])
      upM(state.w_k[l], state.m_w_k[l], state.v_w_k[l], state.g_w_k[l])
      upM(state.w_v[l], state.m_w_v[l], state.v_w_v[l], state.g_w_v[l])
      upM(state.w_o[l], state.m_w_o[l], state.v_w_o[l], state.g_w_o[l])
      upM(state.w_up[l], state.m_w_up[l], state.v_w_up[l], state.g_w_up[l])
      upM(state.w_down[l], state.m_w_down[l], state.v_w_down[l], state.g_w_down[l])

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

proc clipGlobalGradientNorm(s: ptr CortexState, max_norm: cfloat) =
  var total_sq: cfloat = 0.0
  total_sq += calcL2Sq(s.g_embeddings)
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
  if global_norm.isNaN or global_norm > max_norm:
    let scale = if global_norm.isNaN or global_norm == 0.0: 0.0 else: max_norm / global_norm
    scaleMat(s.g_embeddings, scale)
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

proc cortex_init*(vocab, dim, ctx, layers, heads: cint): ptr CortexState {.ffi.} =
  var s = cast[ptr CortexState](allocShared0(sizeof(CortexState)))
  s.vocabSize = vocab
  s.hiddenDim = dim
  s.ctxSize = ctx
  s.numLayers = layers
  s.ffnDim = dim * 4
  s.numHeads = max(1, heads)
  s.headSize = dim div s.numHeads
  if (s.headSize mod 2) != 0: s.headSize -= 1 
  if s.headSize < 2: s.headSize = 2

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
  
  initA(s.act_hidden_final, ctxDim)

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
  initA(s.d_k, dim)
  initA(s.d_v, dim)
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

proc cortex_load_layer*(s: ptr CortexState, layer_idx: cint, tensor_type: cint, raw_weights: ptr cfloat, size: cint) {.ffi.} =
  if size != getExpectedSize(s, tensor_type): return 
  let p = getTensorPtr(s, layer_idx, tensor_type)
  if not p.isNil: copyMem(p, raw_weights, size * sizeof(cfloat))

proc cortex_extract_layer*(s: ptr CortexState, layer_idx: cint, tensor_type: cint, raw_weights: ptr cfloat, size: cint) {.ffi.} =
  if size != getExpectedSize(s, tensor_type): return
  let p = getTensorPtr(s, layer_idx, tensor_type)
  if not p.isNil: copyMem(raw_weights, p, size * sizeof(cfloat))

proc cortex_load_adam_layer*(s: ptr CortexState, layer_idx: cint, tensor_type: cint, is_v: bool, raw: ptr cfloat, size: cint) {.ffi.} =
  if size != getExpectedSize(s, tensor_type): return
  let p = getTensorPtr(s, layer_idx, tensor_type, not is_v, is_v)
  if not p.isNil: copyMem(p, raw, size * sizeof(cfloat))

proc cortex_extract_adam_layer*(s: ptr CortexState, layer_idx: cint, tensor_type: cint, is_v: bool, raw: ptr cfloat, size: cint) {.ffi.} =
  if size != getExpectedSize(s, tensor_type): return
  let p = getTensorPtr(s, layer_idx, tensor_type, not is_v, is_v)
  if not p.isNil: copyMem(raw, p, size * sizeof(cfloat))

proc cortex_reset_cache*(s: ptr CortexState) {.ffi.} =
  s.cachedTokens = 0

proc cortex_forward*(state: ptr CortexState, raw_in: ptr cint, num_tok: cint, raw_logits: ptr cfloat) {.ffi.} =
  let tokens = cast[ptr IntArray](raw_in)
  let logits = cast[ptr FloatArray](raw_logits)
  
  for i in 0 ..< num_tok:
    var tid = tokens[i]
    if tid < 0 or tid >= state.vocabSize: tid = 0 
    let pos = state.cachedTokens
    if pos >= state.ctxSize: break 
    
    for d in 0 ..< state.hiddenDim: 
      state.hidden_state_buf[d] = state.embeddings.data[tid * state.hiddenDim + d]
    
    for l in 0 ..< state.numLayers:
      applyRMSNorm(state.hidden_state_buf, state.norm_buf, state.hiddenDim, state.rms_att_weights[l])
      
      matVecMul(state.w_q[l], state.norm_buf, state.q_buf)
      applyRoPE(state.q_buf, state.hiddenDim, state.headSize, pos)
      
      let kp = cast[ptr FloatArray](addr state.k_cache[l].data[pos * state.hiddenDim])
      let vp = cast[ptr FloatArray](addr state.v_cache[l].data[pos * state.hiddenDim])
      matVecMul(state.w_k[l], state.norm_buf, kp)
      applyRoPE(kp, state.hiddenDim, state.headSize, pos)
      matVecMul(state.w_v[l], state.norm_buf, vp)
      
      for d in 0 ..< state.hiddenDim: 
        state.att_out_buf[d] = 0.0
        
      for h in 0 ..< state.numHeads:
        let h_off = h * state.headSize
        var max_s: cfloat = -9999.0
        for t in 0 .. pos:
          let tk = cast[ptr FloatArray](addr state.k_cache[l].data[t * state.hiddenDim])
          var sc: cfloat = 0.0
          for d in 0 ..< state.headSize: 
            # SCORE BERECHNUNG IST HIER KORREKT: Nutzt h_off + d
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
            # 🚨 FIX: Nutzt jetzt korrekterweise h_off + d für das Lesen aus tv!
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
    
    applyRMSNorm(state.hidden_state_buf, state.norm_buf, state.hiddenDim, state.rms_final_weight)
    
    if i == num_tok - 1:
      for v in 0 ..< state.vocabSize:
        var s: cfloat = 0.0
        let off = v * state.hiddenDim
        for d in 0 ..< state.hiddenDim: 
          s += state.norm_buf[d] * state.w_out.data[off + d]
        logits[v] = s
    
    state.cachedTokens += 1

proc cortex_train_step*(state: ptr CortexState, raw_in: ptr cint, raw_tg: ptr cint, num_tok: cint, lr: cfloat): cfloat {.ffi.} =
  let act = min(num_tok, state.ctxSize)
  if act <= 0: return 0.0
  let t_in = cast[ptr IntArray](raw_in)
  let t_tg = cast[ptr IntArray](raw_tg)
  var total_loss: cfloat = 0.0

  template zeroM(m) = 
    for i in 0 ..< (m.rows * m.cols): m.data[i] = 0.0
  template zeroA(a, s) = 
    for i in 0 ..< s: a[i] = 0.0
  
  zeroM(state.g_embeddings)
  zeroM(state.g_w_out)
  zeroA(state.g_rms_final_weight, state.hiddenDim)
  
  for l in 0 ..< state.numLayers:
    zeroM(state.g_w_q[l])
    zeroM(state.g_w_k[l])
    zeroM(state.g_w_v[l])
    zeroM(state.g_w_o[l])
    zeroM(state.g_w_up[l])
    zeroM(state.g_w_down[l])
    zeroA(state.g_rms_att_weights[l], state.hiddenDim)
    zeroA(state.g_rms_ffn_weights[l], state.hiddenDim)
    
    for i in 0 ..< (act * state.hiddenDim): 
      state.d_k_cache[l].data[i] = 0.0
      state.d_v_cache[l].data[i] = 0.0
    
    if gpu_initialized:
      let full_cache_bytes = cast[csize_t](state.ctxSize * state.hiddenDim * sizeof(cfloat))
      ocl_zero_buffer(cast[ptr cfloat](state.d_k_cache[l].data), full_cache_bytes)
      ocl_zero_buffer(cast[ptr cfloat](state.d_v_cache[l].data), full_cache_bytes)

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
      applyRoPE(q_p, state.hiddenDim, state.headSize, pos)
      
      let kp = cast[ptr FloatArray](addr state.k_cache[l].data[off_dim])
      let vp = cast[ptr FloatArray](addr state.v_cache[l].data[off_dim])
      matVecMul(state.w_k[l], state.norm_buf, kp)
      applyRoPE(kp, state.hiddenDim, state.headSize, pos)
      matVecMul(state.w_v[l], state.norm_buf, vp)

      for d in 0 ..< state.hiddenDim: 
        state.att_out_buf[d] = 0.0
        
      for h in 0 ..< state.numHeads:
        let h_off = h * state.headSize
        var max_s: cfloat = -9999.0
        for t in 0 .. pos:
          let tk = cast[ptr FloatArray](addr state.k_cache[l].data[t * state.hiddenDim])
          var sc: cfloat = 0.0
          for d in 0 ..< state.headSize: 
            # SCORE BERECHNUNG: Korrekt h_off + d
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
            # 🚨 FIX: Nutzt jetzt korrekterweise h_off + d für das Lesen aus tv!
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
      for d in 0 ..< state.ffnDim: 
        state.act_ffn_out[l][off_ffn + d] = state.ffn_buf[d]
      matVecMul(state.w_down[l], state.ffn_buf, state.att_out_buf)
      for d in 0 ..< state.hiddenDim: 
        state.hidden_state_buf[d] += state.att_out_buf[d]

    for d in 0 ..< state.hiddenDim: 
      state.act_hidden_final[off_dim + d] = state.hidden_state_buf[d]
    applyRMSNorm(state.hidden_state_buf, state.norm_buf, state.hiddenDim, state.rms_final_weight)
    
    var max_l: cfloat = -9999.0
    let logits = cast[ptr FloatArray](addr state.logits_cache.data[pos * state.vocabSize])
    for v in 0 ..< state.vocabSize:
      var s: cfloat = 0.0
      let off = v * state.hiddenDim
      for d in 0 ..< state.hiddenDim: 
        s += state.norm_buf[d] * state.w_out.data[off + d]
      logits[v] = s
      if s > max_l: max_l = s
    var sum_p: cfloat = 0.0
    for v in 0 ..< state.vocabSize: 
      logits[v] = exp(logits[v] - max_l)
      sum_p += logits[v]
    for v in 0 ..< state.vocabSize: 
      logits[v] /= sum_p
    total_loss += -ln(logits[tgid] + 1e-7)

  for pos in countdown(act - 1, 0):
    var tid = t_in[pos]
    var tgid = t_tg[pos]
    if tid < 0 or tid >= state.vocabSize: tid = 0 
    if tgid < 0 or tgid >= state.vocabSize: tgid = 0
    let off_dim = pos * state.hiddenDim
    let off_ffn = pos * state.ffnDim
    let logits = cast[ptr FloatArray](addr state.logits_cache.data[pos * state.vocabSize])

    for v in 0 ..< state.vocabSize: 
      state.d_logits_buf[v] = logits[v] - (if v == tgid: 1.0.cfloat else: 0.0.cfloat)
    for d in 0 ..< state.hiddenDim: 
      state.d_hidden[d] = 0.0

    let final_hidden = cast[ptr FloatArray](addr state.act_hidden_final[off_dim])
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
            # BPTT SCORE BERECHNUNG: Korrekt h_off + d
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
            # 🚨 FIX BPTT: Nutzt jetzt korrekterweise h_off + d für das Lesen aus v_t!
            dw += state.d_att_out[h_off + d] * v_t[h_off + d]
            
          let d_v_t = cast[ptr FloatArray](addr state.d_v_cache[l].data[t * state.hiddenDim])
          for d in 0 ..< state.headSize: 
            # 🚨 FIX BPTT: Nutzt jetzt korrekterweise h_off + d für das Schreiben nach d_v_t!
            d_v_t[h_off + d] += state.d_att_out[h_off + d] * w
            
          state.d_w_buf[t] = dw
          sum_w_dw += w * dw
          
        for t in 0 .. pos:
          let w = state.scores_buf[t]
          let ds = w * (state.d_w_buf[t] - sum_w_dw) * scale
          let k_t = cast[ptr FloatArray](addr state.k_cache[l].data[t * state.hiddenDim])
          let d_k_t = cast[ptr FloatArray](addr state.d_k_cache[l].data[t * state.hiddenDim])
          for d in 0 ..< state.headSize:
            # 🚨 FIX BPTT: Nutzt jetzt korrekterweise h_off + d!
            state.d_q[h_off + d] += ds * k_t[h_off + d]
            d_k_t[h_off + d] += ds * q_p[h_off + d] 

      let current_d_k = cast[ptr FloatArray](addr state.d_k_cache[l].data[off_dim])
      let current_d_v = cast[ptr FloatArray](addr state.d_v_cache[l].data[off_dim])

      applyRoPE(state.d_q, state.hiddenDim, state.headSize, pos, true)
      applyRoPE(current_d_k, state.hiddenDim, state.headSize, pos, true)

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

  clipGlobalGradientNorm(state, 1.0)
  applyGlobalAdamUpdate(state, lr, act.cfloat)
  return total_loss / act.cfloat

proc cortex_free*(state: ptr CortexState) {.ffi.} =
  if state.isNil: return
  
  template freeM(m) = 
    freeMatrix(m)
  
  template freeA(a) = 
    if not a.isNil: 
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
  freeA(state.d_k)
  freeA(state.d_v)
  freeA(state.d_ffn)
  
  freeHist(state.act_hidden, state.numLayers)
  freeHist(state.act_norm_att, state.numLayers)
  freeHist(state.act_att_out, state.numLayers)
  freeHist(state.act_hidden_ffn, state.numLayers)
  freeHist(state.act_norm_ffn, state.numLayers)
  freeHist(state.act_ffn_in, state.numLayers)
  freeHist(state.act_ffn_out, state.numLayers)
  
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
# FFI EXPORT: STREAM LOADER (Unverändert)
# ==============================================================================

type
  ModelHeader {.packed.} = object
    magic: array[8, char]
    vocab_size: uint32
    hidden_dim: uint32
    ffn_dim: uint32
    ctx_size: uint32
    num_layers: uint32

proc cortex_load_stream*(filepath_c: cstring, num_heads: cint): ptr CortexState {.ffi, exportc, dynlib, cdecl.} =
  let filepath = $filepath_c
  var f: File
  if not open(f, filepath, fmRead): 
    return nil
  
  var header: ModelHeader
  if f.readBuffer(addr header, sizeof(ModelHeader)) != sizeof(ModelHeader):
    f.close()
    return nil
    
  var magicStr = ""
  for c in header.magic:
    if c != '\0': magicStr.add(c)
  if magicStr != "SNAI_V4":
    f.close()
    return nil

  let s = cortex_init(header.vocab_size.cint, header.hidden_dim.cint, header.ctx_size.cint, header.num_layers.cint, num_heads)
  
  proc blastBytes(file: File, dest: pointer, count: int): bool =
    let bytesToRead = count * sizeof(cfloat)
    return file.readBuffer(dest, bytesToRead) == bytesToRead

  if not blastBytes(f, s.embeddings.data, header.vocab_size.int * header.hidden_dim.int): 
    f.close()
    return nil
  
  if not blastBytes(f, s.w_out.data, header.vocab_size.int * header.hidden_dim.int): 
    f.close()
    return nil
    
  if not blastBytes(f, s.rms_final_weight, header.hidden_dim.int): 
    f.close()
    return nil

  let dim2 = header.hidden_dim.int * header.hidden_dim.int
  let ffn_hidden = header.ffn_dim.int * header.hidden_dim.int

  for l in 0 ..< header.num_layers.int:
    if not blastBytes(f, s.rms_att_weights[l], header.hidden_dim.int): 
      f.close()
      return nil
    if not blastBytes(f, s.rms_ffn_weights[l], header.hidden_dim.int): 
      f.close()
      return nil
    if not blastBytes(f, s.w_q[l].data, dim2): 
      f.close()
      return nil
    if not blastBytes(f, s.w_k[l].data, dim2): 
      f.close()
      return nil
    if not blastBytes(f, s.w_v[l].data, dim2): 
      f.close()
      return nil
    if not blastBytes(f, s.w_o[l].data, dim2): 
      f.close()
      return nil
    if not blastBytes(f, s.w_up[l].data, ffn_hidden): 
      f.close()
      return nil
    if not blastBytes(f, s.w_down[l].data, ffn_hidden): 
      f.close()
      return nil

  f.close()
  return s

proc cortex_init_gpu*() {.ffi, exportc, dynlib, cdecl.} =
  init_opencl_engine()
  gpu_initialized = true 

proc get_vocab_size*(state: ptr CortexState): cint {.ffi, exportc, dynlib, cdecl.} =
  if state.isNil: return 0
  return state.vocabSize

proc get_ctx_size*(state: ptr CortexState): cint {.ffi, exportc, dynlib, cdecl.} =
  if state.isNil: return 0
  return state.ctxSize