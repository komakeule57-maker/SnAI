# Dateiname: cortex_opencl.nim
# Funktion: OpenCL Compute Engine (Bare-Metal JIT) - PHASE VIII.4 (Memory Audit Fixes)
# ==============================================================================
# SYMBIO NANO-AI FRAMEWORK (Code-4372)
# -> VRAM Leaks geschlossen (ocl_release_buffer).
# -> Sparse Adam GPU Angleichung (fabs).
# -> Tote PCIe-Transfers eliminiert (E > H).
# ==============================================================================

{.passL: "-lOpenCL".}

{.emit: """
#define CL_TARGET_OPENCL_VERSION 120 
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <math.h>
#include <string.h>

cl_context ocl_ctx = NULL;
cl_command_queue ocl_queue = NULL;
cl_kernel ocl_kernel_matvec = NULL;
cl_kernel ocl_kernel_adam = NULL; 
cl_kernel ocl_kernel_att_fwd = NULL;
cl_kernel ocl_kernel_att_bwd = NULL;
cl_kernel ocl_kernel_matvec_t = NULL;   // V8.5: Backward d_in = M^T * d_out
cl_kernel ocl_kernel_grad_acc = NULL;   // V8.5: Backward gm += d_out (x) i_v
cl_kernel ocl_kernel_gemm_f = NULL;     // V8.6: Batched Forward
cl_kernel ocl_kernel_gemm_t = NULL;     // V8.6: Batched Backward d_in
cl_kernel ocl_kernel_gemm_g = NULL;     // V8.6: Batched Gradienten-Akkumulation
cl_kernel ocl_kernel_att_fwd_b = NULL;  // AUDIT: Attention Forward, ueber alle Positionen gebuendelt
cl_kernel ocl_kernel_att_bwd_q = NULL;  // AUDIT: Attention Backward Stufe A (query-parallel: d_q, weights, ds)
cl_kernel ocl_kernel_att_bwd_k = NULL;  // AUDIT: Attention Backward Stufe B (key-parallel: d_k_cache, d_v_cache)

size_t ocl_max_work_group_size = 64; 
_Atomic int ocl_initialized = 0;     

#define MAX_OCL_BUFFERS 2048

// AUDIT-FIX: max. Positionen pro Attention-Kernel-Dispatch (siehe Einsatzstellen
// in ocl_attentionForwardBatched/ocl_attentionBackwardBatched weiter unten).
#define ATT_POS_CHUNK 256

typedef struct { 
    void* h_ptr; 
    cl_mem d_buf; 
    size_t bytes; 
} OclBufMap;

OclBufMap ocl_buf_map[MAX_OCL_BUFFERS];
int ocl_buf_count = 0;

cl_mem ocl_eph_v = NULL; size_t ocl_eph_v_size = 0;
cl_mem ocl_eph_out = NULL; size_t ocl_eph_out_size = 0;
cl_mem ocl_eph_dq = NULL; size_t ocl_eph_dq_size = 0;
cl_mem ocl_eph_scores = NULL; size_t ocl_eph_scores_size = 0;
cl_mem ocl_eph_dw = NULL; size_t ocl_eph_dw_size = 0;
cl_mem ocl_eph_bt_a = NULL; size_t ocl_eph_bt_a_size = 0;   // V8.5: d_out
cl_mem ocl_eph_bt_b = NULL; size_t ocl_eph_bt_b_size = 0;   // V8.5: d_in / i_v
cl_mem ocl_eph_bm_r = NULL; size_t ocl_eph_bm_r_size = 0;   // V8.6: batch x rows
cl_mem ocl_eph_bm_c = NULL; size_t ocl_eph_bm_c_size = 0;   // V8.6: batch x cols

// AUDIT: gebuendelte Attention (Query-Batch statt Pro-Position-Dispatch)
cl_mem ocl_eph_att_q = NULL;      size_t ocl_eph_att_q_size = 0;       // act x dim
cl_mem ocl_eph_att_out = NULL;    size_t ocl_eph_att_out_size = 0;     // act x dim
cl_mem ocl_eph_att_datt = NULL;   size_t ocl_eph_att_datt_size = 0;    // act x dim
cl_mem ocl_eph_att_dq = NULL;     size_t ocl_eph_att_dq_size = 0;      // act x dim
cl_mem ocl_eph_att_scores = NULL; size_t ocl_eph_att_scores_size = 0;  // act x numHeads x ctx
cl_mem ocl_eph_att_ds = NULL;     size_t ocl_eph_att_ds_size = 0;      // act x numHeads x ctx

// FIX 1: Stale Pointer-Registry (Memory Leak)
void ocl_release_buffer(void* h_ptr) {
    if (!h_ptr) return;
    for(int i = 0; i < ocl_buf_count; i++) {
        if(ocl_buf_map[i].h_ptr == h_ptr) {
            clReleaseMemObject(ocl_buf_map[i].d_buf);
            for(int j = i; j < ocl_buf_count - 1; j++) {
                ocl_buf_map[j] = ocl_buf_map[j+1];
            }
            ocl_buf_count--;
            return;
        }
    }
}

cl_mem ocl_get_buffer(void* h_ptr, size_t bytes) {
    if (!h_ptr || bytes == 0) return NULL;

    for(int i = 0; i < ocl_buf_count; i++) {
        if(ocl_buf_map[i].h_ptr == h_ptr) {
            if (ocl_buf_map[i].bytes != bytes) {
                fprintf(stderr, "🚨 [SYMBIO GPU] Buffer Size Mismatch für Pointer %p!\n", h_ptr);
                return NULL;
            }
            return ocl_buf_map[i].d_buf;
        }
    }
    
    if(ocl_buf_count >= MAX_OCL_BUFFERS) {
        fprintf(stderr, "🚨 [SYMBIO GPU] VRAM Registry Overflow!\n");
        return NULL; 
    }
    
    cl_int err;
    cl_mem d_buf = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, bytes, NULL, &err);
    if(err == CL_SUCCESS) {
        clEnqueueWriteBuffer(ocl_queue, d_buf, CL_TRUE, 0, bytes, h_ptr, 0, NULL, NULL);
        ocl_buf_map[ocl_buf_count].h_ptr = h_ptr;
        ocl_buf_map[ocl_buf_count].d_buf = d_buf;
        ocl_buf_map[ocl_buf_count].bytes = bytes;
        ocl_buf_count++;
        return d_buf;
    }
    return NULL;
}

void ocl_sync_to_host(void* h_ptr) {
    if (!ocl_ctx || !h_ptr) return;
    for(int i = 0; i < ocl_buf_count; i++) {
        if(ocl_buf_map[i].h_ptr == h_ptr) {
            clEnqueueReadBuffer(ocl_queue, ocl_buf_map[i].d_buf, CL_TRUE, 0, ocl_buf_map[i].bytes, h_ptr, 0, NULL, NULL);
            return;
        }
    }
}

void ocl_sync_to_device(void* h_ptr, size_t bytes) {
    if (!ocl_ctx || !h_ptr || bytes == 0) return;
    cl_mem d_buf = ocl_get_buffer(h_ptr, bytes);
    if (d_buf) {
        clEnqueueWriteBuffer(ocl_queue, d_buf, CL_TRUE, 0, bytes, h_ptr, 0, NULL, NULL);
    }
}

void ocl_zero_buffer(void* h_ptr, size_t bytes) {
    if (!ocl_ctx) return;
    cl_mem d_buf = ocl_get_buffer(h_ptr, bytes);
    if (d_buf) {
        float zero = 0.0f;
        clEnqueueFillBuffer(ocl_queue, d_buf, &zero, sizeof(float), 0, bytes, 0, NULL, NULL);
        clFinish(ocl_queue);
    }
}

const char *kernel_source = 
"__kernel void matVecMul(__global const float* M, __global const float* v, __global float* out, int rows, int cols) {\n"
"    size_t r = get_global_id(0);\n"
"    if (r < (size_t)rows) {\n"
"        float sum = 0.0f;\n"
"        int c = 0;\n"
"        #pragma unroll 4\n"
"        for (; c <= cols - 4; c += 4) {\n"
"            sum += M[r * cols + c] * v[c];\n"
"            sum += M[r * cols + c + 1] * v[c + 1];\n"
"            sum += M[r * cols + c + 2] * v[c + 2];\n"
"            sum += M[r * cols + c + 3] * v[c + 3];\n"
"        }\n"
"        for (; c < cols; ++c) {\n"
"            sum += M[r * cols + c] * v[c];\n"
"        }\n"
"        out[r] = sum;\n"
"    }\n"
"}\n"
"\n"
"__kernel void matVecMulT(__global const float* M, __global const float* d_out, __global float* d_in,\n"
"                         int rows, int cols, int accumulate) {\n"
"    size_t c = get_global_id(0);\n"
"    if (c < (size_t)cols) {\n"
"        float sum = 0.0f;\n"
"        size_t idx = c;\n"
"        for (int r = 0; r < rows; ++r) {\n"
"            sum += M[idx] * d_out[r];\n"
"            idx += (size_t)cols;\n"
"        }\n"
"        d_in[c] = (accumulate == 1) ? (d_in[c] + sum) : sum;\n"
"    }\n"
"}\n"
"\n"
"__kernel void gradAccum(__global float* gm, __global const float* d_out, __global const float* i_v,\n"
"                        int rows, int cols) {\n"
"    size_t c = get_global_id(0);\n"
"    size_t r = get_global_id(1);\n"
"    if (c < (size_t)cols && r < (size_t)rows) {\n"
"        gm[r * (size_t)cols + c] += d_out[r] * i_v[c];\n"
"    }\n"
"}\n"
"\n"
"__kernel void matMulBatched(__global const float* M, __global const float* IN, __global float* OUT,\n"
"                            int rows, int cols, int batch) {\n"
"    size_t r = get_global_id(0);\n"
"    size_t p = get_global_id(1);\n"
"    if (r < (size_t)rows && p < (size_t)batch) {\n"
"        __global const float* v  = IN + p * (size_t)cols;\n"
"        __global const float* mr = M  + r * (size_t)cols;\n"
"        float sum = 0.0f;\n"
"        int c = 0;\n"
"        for (; c <= cols - 4; c += 4) {\n"
"            sum += mr[c] * v[c];         sum += mr[c+1] * v[c+1];\n"
"            sum += mr[c+2] * v[c+2];     sum += mr[c+3] * v[c+3];\n"
"        }\n"
"        for (; c < cols; ++c) sum += mr[c] * v[c];\n"
"        OUT[p * (size_t)rows + r] = sum;\n"
"    }\n"
"}\n"
"\n"
"__kernel void matMulTBatched(__global const float* M, __global const float* D_OUT, __global float* D_IN,\n"
"                             int rows, int cols, int batch, int accumulate) {\n"
"    size_t c = get_global_id(0);\n"
"    size_t p = get_global_id(1);\n"
"    if (c < (size_t)cols && p < (size_t)batch) {\n"
"        __global const float* d = D_OUT + p * (size_t)rows;\n"
"        float sum = 0.0f;\n"
"        size_t idx = c;\n"
"        for (int r = 0; r < rows; ++r) { sum += M[idx] * d[r]; idx += (size_t)cols; }\n"
"        size_t o = p * (size_t)cols + c;\n"
"        D_IN[o] = (accumulate == 1) ? (D_IN[o] + sum) : sum;\n"
"    }\n"
"}\n"
"\n"
"// Ein Thread pro (r,c), Schleife ueber die Batch-Achse INNERHALB des Threads.\n"
"// Dadurch schreibt jeder Thread genau ein Element - keine Atomics noetig.\n"
"__kernel void gradAccumBatched(__global float* gm, __global const float* D_OUT, __global const float* I_V,\n"
"                               int rows, int cols, int batch) {\n"
"    size_t c = get_global_id(0);\n"
"    size_t r = get_global_id(1);\n"
"    if (c < (size_t)cols && r < (size_t)rows) {\n"
"        float sum = 0.0f;\n"
"        for (int p = 0; p < batch; ++p) {\n"
"            sum += D_OUT[(size_t)p * (size_t)rows + r] * I_V[(size_t)p * (size_t)cols + c];\n"
"        }\n"
"        gm[r * (size_t)cols + c] += sum;\n"
"    }\n"
"}\n"
"\n"
"__kernel void adamUpdate(__global float* w, __global float* m, __global float* v, __global float* g, \n"
"                         float lr, float b1, float b2, float b1t, float b2t, float scale, int size, int is_sparse,\n"
"                         int cols, int bound_r, int bound_c, float boost, float wd) {\n"
"    size_t i = get_global_id(0);\n"
"    if (i < (size_t)size) {\n"
"        float grad = g[i] / scale;\n"
"        if (is_sparse == 1 && fabs(grad) < 1e-8f) return;\n" // FIX 2: GPU/CPU Inkonsistenz (fabs)
"        if (isnan(grad) || grad > 9999.0f || grad < -9999.0f) {\n"
"            g[i] = 0.0f;\n"
"            return;\n"
"        }\n"
// AUDIT: LR-Boost fuer frisch Net2Net-expandierte Bereiche (siehe
// docs/ROADMAP.md, "Differenzierte Learning-Rate"). cols/bound_r/bound_c
// beschreiben, ab welcher Zeile/Spalte eine Matrix "neu" ist - identische
// Logik wie im CPU-Pfad (elemBoost() in cortex.nim).
"        int row = (int)(i / (size_t)cols);\n"
"        int col = (int)(i % (size_t)cols);\n"
"        float eff_lr = (row >= bound_r || col >= bound_c) ? (lr * boost) : lr;\n"
"        m[i] = b1 * m[i] + (1.0f - b1) * grad;\n"
"        v[i] = b2 * v[i] + (1.0f - b2) * grad * grad;\n"
"        float m_hat = m[i] / b1t;\n"
"        float v_hat = v[i] / b2t;\n"
"        w[i] -= eff_lr * m_hat / (sqrt(v_hat) + 1e-8f);\n"
// Weight-Decay (AdamW-Style, dekorreliert) - siehe docs/ROADMAP.md,
// "Trainings-Absicherung: Nachruest-Paket" Punkt 1. wd=0.0 (Default) macht
// das bitidentisch zum bisherigen reinen-Adam-Verhalten. Muss exakt zum
// CPU-Pfad (adamStep() in cortex.nim) passen.
"        if (wd > 0.0f) { w[i] -= lr * wd * w[i]; }\n"
"        g[i] = 0.0f;\n"
"    }\n"
"}\n"
"\n"
"__kernel void attentionForward(__global const float* q, __global const float* k_cache, __global const float* v_cache, \n"
"                               __global float* att_out, __global float* scores, int pos, int numHeads, int headSize, int dim, int ctx) {\n"
"    int h = get_global_id(0);\n"
"    if (h >= numHeads) return;\n"
"    int h_off = h * headSize;\n"
"    float scale = 1.0f / sqrt((float)headSize);\n"
"    float max_s = -9999.0f;\n"
"    for (int t = 0; t <= pos; t++) {\n"
"        float sc = 0.0f;\n"
"        int k_off = t * dim + h_off;\n"
"        int d = 0;\n"
"        for (; d <= headSize - 4; d += 4) {\n"
"            sc += q[h_off + d] * k_cache[k_off + d]; sc += q[h_off + d + 1] * k_cache[k_off + d + 1];\n"
"            sc += q[h_off + d + 2] * k_cache[k_off + d + 2]; sc += q[h_off + d + 3] * k_cache[k_off + d + 3];\n"
"        }\n"
"        for (; d < headSize; d++) sc += q[h_off + d] * k_cache[k_off + d];\n"
"        sc *= scale; scores[h * ctx + t] = sc;\n"
"        if (sc > max_s) max_s = sc;\n"
"    }\n"
"    float sum_e = 0.0f;\n"
"    for (int t = 0; t <= pos; t++) { float e = exp(scores[h * ctx + t] - max_s); scores[h * ctx + t] = e; sum_e += e; }\n"
"    for (int d = 0; d < headSize; d++) att_out[h_off + d] = 0.0f;\n"
"    for (int t = 0; t <= pos; t++) {\n"
"        float w = scores[h * ctx + t] / sum_e;\n"
"        int v_off = t * dim + h_off;\n"
"        int d = 0;\n"
"        for (; d <= headSize - 4; d += 4) {\n"
"            att_out[h_off + d] += v_cache[v_off + d] * w; att_out[h_off + d + 1] += v_cache[v_off + d + 1] * w;\n"
"            att_out[h_off + d + 2] += v_cache[v_off + d + 2] * w; att_out[h_off + d + 3] += v_cache[v_off + d + 3] * w;\n"
"        }\n"
"        for (; d < headSize; d++) att_out[h_off + d] += v_cache[v_off + d] * w;\n"
"    }\n"
"}\n"
"\n"
"__kernel void attentionBackward(__global const float* q, __global const float* k_cache, __global const float* v_cache, \n"
"                                __global const float* d_att_out, __global float* d_q, __global float* d_k_cache, __global float* d_v_cache, \n"
"                                __global float* scores, __global float* d_w, int pos, int numHeads, int headSize, int dim, int ctx) {\n"
"    int h = get_global_id(0);\n"
"    if (h >= numHeads) return;\n"
"    int h_off = h * headSize;\n"
"    float scale = 1.0f / sqrt((float)headSize);\n"
"    float max_s = -9999.0f;\n"
"    for (int d = 0; d < headSize; d++) d_q[h_off + d] = 0.0f;\n"
"    for (int t = 0; t <= pos; t++) {\n"
"        float sc = 0.0f; int k_off = t * dim + h_off;\n"
"        for (int d = 0; d < headSize; d++) sc += q[h_off + d] * k_cache[k_off + d];\n"
"        sc *= scale; scores[h * ctx + t] = sc;\n"
"        if (sc > max_s) max_s = sc;\n"
"    }\n"
"    float sum_e = 0.0f;\n"
"    for (int t = 0; t <= pos; t++) { float e = exp(scores[h * ctx + t] - max_s); scores[h * ctx + t] = e; sum_e += e; }\n"
"    float sum_w_dw = 0.0f;\n"
"    for (int t = 0; t <= pos; t++) {\n"
"        float w = scores[h * ctx + t] / sum_e; scores[h * ctx + t] = w;\n"
"        int v_off = t * dim + h_off; float dw = 0.0f;\n"
"        for (int d = 0; d < headSize; d++) {\n"
"            dw += d_att_out[h_off + d] * v_cache[v_off + d];\n"
"            d_v_cache[v_off + d] += d_att_out[h_off + d] * w;\n"
"        }\n"
"        d_w[h * ctx + t] = dw; sum_w_dw += w * dw;\n"
"    }\n"
"    for (int t = 0; t <= pos; t++) {\n"
"        float w = scores[h * ctx + t];\n"
"        float ds = w * (d_w[h * ctx + t] - sum_w_dw) * scale;\n"
"        int k_off = t * dim + h_off;\n"
"        for (int d = 0; d < headSize; d++) {\n"
"            d_q[h_off + d] += ds * k_cache[k_off + d];\n"
"            d_k_cache[k_off + d] += ds * q[h_off + d];\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"// AUDIT: gebuendelte Attention ueber die Positionsachse - ein Dispatch statt\n"
"// 'act' einzelner Aufrufe. Arbeitsgruppe (h, p): jede Position p bekommt ihr\n"
"// eigenes Work-Item, das exakt dieselbe Kausal-Maske (t <= p) und dieselbe\n"
"// Softmax-Mathematik wie der alte Pro-Position-Kernel rechnet.\n"
"__kernel void attentionForwardBatched(__global const float* q_all, __global const float* k_cache, __global const float* v_cache,\n"
"                                       __global float* att_out_all, __global float* scores,\n"
"                                       int act, int numHeads, int headSize, int dim, int ctx) {\n"
"    int h = get_global_id(0);\n"
"    int p = get_global_id(1);\n"
"    if (h >= numHeads || p >= act) return;\n"
"    int h_off = h * headSize;\n"
"    __global const float* q = q_all + (size_t)p * (size_t)dim;\n"
"    __global float* att_out = att_out_all + (size_t)p * (size_t)dim;\n"
"    __global float* my_scores = scores + ((size_t)p * (size_t)numHeads + h) * (size_t)ctx;\n"
"    float scale = 1.0f / sqrt((float)headSize);\n"
"    float max_s = -9999.0f;\n"
"    for (int t = 0; t <= p; t++) {\n"
"        float sc = 0.0f; int k_off = t * dim + h_off;\n"
"        for (int d = 0; d < headSize; d++) sc += q[h_off + d] * k_cache[k_off + d];\n"
"        sc *= scale; my_scores[t] = sc;\n"
"        if (sc > max_s) max_s = sc;\n"
"    }\n"
"    float sum_e = 0.0f;\n"
"    for (int t = 0; t <= p; t++) { float e = exp(my_scores[t] - max_s); my_scores[t] = e; sum_e += e; }\n"
"    for (int d = 0; d < headSize; d++) att_out[h_off + d] = 0.0f;\n"
"    for (int t = 0; t <= p; t++) {\n"
"        float w = my_scores[t] / sum_e;\n"
"        int v_off = t * dim + h_off;\n"
"        for (int d = 0; d < headSize; d++) att_out[h_off + d] += v_cache[v_off + d] * w;\n"
"    }\n"
"}\n"
"\n"
"// AUDIT: Attention-Backward Stufe A (query-parallel, ein Work-Item pro (h,p)).\n"
"// d_q ist ausschliesslich von Position p abhaengig -> race-frei direkt geschrieben.\n"
"// weights/ds werden fuer Stufe B zwischengespeichert (bleiben GPU-resident).\n"
"__kernel void attentionBackwardBatchedQuery(__global const float* q_all, __global const float* k_cache, __global const float* v_cache,\n"
"                                             __global const float* d_att_out_all, __global float* d_q_all,\n"
"                                             __global float* weights, __global float* ds,\n"
"                                             int act, int numHeads, int headSize, int dim, int ctx) {\n"
"    int h = get_global_id(0);\n"
"    int p = get_global_id(1);\n"
"    if (h >= numHeads || p >= act) return;\n"
"    int h_off = h * headSize;\n"
"    __global const float* q = q_all + (size_t)p * (size_t)dim;\n"
"    __global const float* d_att_out = d_att_out_all + (size_t)p * (size_t)dim;\n"
"    __global float* d_q = d_q_all + (size_t)p * (size_t)dim;\n"
"    __global float* my_w  = weights + ((size_t)p * (size_t)numHeads + h) * (size_t)ctx;\n"
"    __global float* my_ds = ds      + ((size_t)p * (size_t)numHeads + h) * (size_t)ctx;\n"
"    float scale = 1.0f / sqrt((float)headSize);\n"
"    float max_s = -9999.0f;\n"
"    for (int d = 0; d < headSize; d++) d_q[h_off + d] = 0.0f;\n"
"    for (int t = 0; t <= p; t++) {\n"
"        float sc = 0.0f; int k_off = t * dim + h_off;\n"
"        for (int d = 0; d < headSize; d++) sc += q[h_off + d] * k_cache[k_off + d];\n"
"        sc *= scale; my_w[t] = sc;\n"
"        if (sc > max_s) max_s = sc;\n"
"    }\n"
"    float sum_e = 0.0f;\n"
"    for (int t = 0; t <= p; t++) { float e = exp(my_w[t] - max_s); my_w[t] = e; sum_e += e; }\n"
"    float sum_w_dw = 0.0f;\n"
"    for (int t = 0; t <= p; t++) {\n"
"        float w = my_w[t] / sum_e; my_w[t] = w;\n"
"        int v_off = t * dim + h_off; float dw = 0.0f;\n"
"        for (int d = 0; d < headSize; d++) dw += d_att_out[h_off + d] * v_cache[v_off + d];\n"
"        my_ds[t] = dw; sum_w_dw += w * dw;\n"
"    }\n"
"    for (int t = 0; t <= p; t++) {\n"
"        float w = my_w[t];\n"
"        float dw = my_ds[t];\n"
"        float ds_val = w * (dw - sum_w_dw) * scale;\n"
"        my_ds[t] = ds_val;\n"
"        int k_off = t * dim + h_off;\n"
"        for (int d = 0; d < headSize; d++) d_q[h_off + d] += ds_val * k_cache[k_off + d];\n"
"    }\n"
"}\n"
"\n"
"// AUDIT: Attention-Backward Stufe B (key-parallel, ein Work-Item pro (h,t)).\n"
"// d_k_cache/d_v_cache an Position t erhalten Beitraege von JEDER Query p >= t -\n"
"// ueber die Query-Achse zu parallelisieren waere ein Data-Race (mehrere p\n"
"// schreiben in dasselbe t). Deshalb hier die Key-Achse als Parallel-Dimension:\n"
"// jedes Work-Item besitzt exklusiv genau ein (h,t)-Ausgabe-Segment und summiert\n"
"// darin seriell ueber alle p >= t - race-frei, ohne Atomics.\n"
"__kernel void attentionBackwardBatchedKey(__global const float* q_all, __global const float* d_att_out_all,\n"
"                                           __global const float* weights, __global const float* ds,\n"
"                                           __global float* d_k_cache, __global float* d_v_cache,\n"
"                                           int act, int numHeads, int headSize, int dim, int ctx) {\n"
"    int h = get_global_id(0);\n"
"    int t = get_global_id(1);\n"
"    if (h >= numHeads || t >= act) return;\n"
"    int h_off = h * headSize;\n"
"    int kv_off = t * dim + h_off;\n"
"    for (int d = 0; d < headSize; d++) { d_k_cache[kv_off + d] = 0.0f; d_v_cache[kv_off + d] = 0.0f; }\n"
"    for (int p = t; p < act; p++) {\n"
"        size_t idx = ((size_t)p * (size_t)numHeads + h) * (size_t)ctx + t;\n"
"        float w  = weights[idx];\n"
"        float dsv = ds[idx];\n"
"        __global const float* d_att_out = d_att_out_all + (size_t)p * (size_t)dim;\n"
"        __global const float* q = q_all + (size_t)p * (size_t)dim;\n"
"        for (int d = 0; d < headSize; d++) {\n"
"            d_v_cache[kv_off + d] += d_att_out[h_off + d] * w;\n"
"            d_k_cache[kv_off + d] += dsv * q[h_off + d];\n"
"        }\n"
"    }\n"
"}\n";

void shutdown_opencl_engine() {
    for(int i=0; i<ocl_buf_count; i++) {
        if(ocl_buf_map[i].d_buf) clReleaseMemObject(ocl_buf_map[i].d_buf);
    }
    ocl_buf_count = 0;
    
    if (ocl_eph_v) { clReleaseMemObject(ocl_eph_v); ocl_eph_v = NULL; ocl_eph_v_size = 0; }
    if (ocl_eph_out) { clReleaseMemObject(ocl_eph_out); ocl_eph_out = NULL; ocl_eph_out_size = 0; }
    if (ocl_eph_dq) { clReleaseMemObject(ocl_eph_dq); ocl_eph_dq = NULL; ocl_eph_dq_size = 0; }
    if (ocl_eph_scores) { clReleaseMemObject(ocl_eph_scores); ocl_eph_scores = NULL; ocl_eph_scores_size = 0; }
    if (ocl_eph_dw) { clReleaseMemObject(ocl_eph_dw); ocl_eph_dw = NULL; ocl_eph_dw_size = 0; }
    if (ocl_eph_bt_a) { clReleaseMemObject(ocl_eph_bt_a); ocl_eph_bt_a = NULL; ocl_eph_bt_a_size = 0; }
    if (ocl_eph_bt_b) { clReleaseMemObject(ocl_eph_bt_b); ocl_eph_bt_b = NULL; ocl_eph_bt_b_size = 0; }
    if (ocl_eph_bm_r) { clReleaseMemObject(ocl_eph_bm_r); ocl_eph_bm_r = NULL; ocl_eph_bm_r_size = 0; }
    if (ocl_eph_bm_c) { clReleaseMemObject(ocl_eph_bm_c); ocl_eph_bm_c = NULL; ocl_eph_bm_c_size = 0; }
    if (ocl_eph_att_q) { clReleaseMemObject(ocl_eph_att_q); ocl_eph_att_q = NULL; ocl_eph_att_q_size = 0; }
    if (ocl_eph_att_out) { clReleaseMemObject(ocl_eph_att_out); ocl_eph_att_out = NULL; ocl_eph_att_out_size = 0; }
    if (ocl_eph_att_datt) { clReleaseMemObject(ocl_eph_att_datt); ocl_eph_att_datt = NULL; ocl_eph_att_datt_size = 0; }
    if (ocl_eph_att_dq) { clReleaseMemObject(ocl_eph_att_dq); ocl_eph_att_dq = NULL; ocl_eph_att_dq_size = 0; }
    if (ocl_eph_att_scores) { clReleaseMemObject(ocl_eph_att_scores); ocl_eph_att_scores = NULL; ocl_eph_att_scores_size = 0; }
    if (ocl_eph_att_ds) { clReleaseMemObject(ocl_eph_att_ds); ocl_eph_att_ds = NULL; ocl_eph_att_ds_size = 0; }

    if (ocl_kernel_gemm_f) clReleaseKernel(ocl_kernel_gemm_f);
    if (ocl_kernel_gemm_t) clReleaseKernel(ocl_kernel_gemm_t);
    if (ocl_kernel_gemm_g) clReleaseKernel(ocl_kernel_gemm_g);
    if (ocl_kernel_matvec_t) clReleaseKernel(ocl_kernel_matvec_t);
    if (ocl_kernel_grad_acc) clReleaseKernel(ocl_kernel_grad_acc);
    if (ocl_kernel_matvec) clReleaseKernel(ocl_kernel_matvec);
    if (ocl_kernel_adam) clReleaseKernel(ocl_kernel_adam);
    if (ocl_kernel_att_fwd) clReleaseKernel(ocl_kernel_att_fwd);
    if (ocl_kernel_att_bwd) clReleaseKernel(ocl_kernel_att_bwd);
    if (ocl_kernel_att_fwd_b) clReleaseKernel(ocl_kernel_att_fwd_b);
    if (ocl_kernel_att_bwd_q) clReleaseKernel(ocl_kernel_att_bwd_q);
    if (ocl_kernel_att_bwd_k) clReleaseKernel(ocl_kernel_att_bwd_k);
    if (ocl_queue) clReleaseCommandQueue(ocl_queue);
    if (ocl_ctx) clReleaseContext(ocl_ctx);

    ocl_kernel_matvec = NULL; ocl_kernel_adam = NULL; ocl_kernel_att_fwd = NULL; ocl_kernel_att_bwd = NULL;
    ocl_kernel_att_fwd_b = NULL; ocl_kernel_att_bwd_q = NULL; ocl_kernel_att_bwd_k = NULL;
    ocl_queue = NULL; ocl_ctx = NULL;
    atomic_store(&ocl_initialized, 0); 
    fprintf(stderr, "⚡ [SYMBIO VRAM] OpenCL-Ressourcen & Registry freigegeben.\n");
}

int init_opencl_engine() {
    if (atomic_load(&ocl_initialized)) return 1; 
    int expected = 0;
    if (!atomic_compare_exchange_strong(&ocl_initialized, &expected, 1)) return 0; 

    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) return 0;

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);

    cl_device_id target_device = NULL;
    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &target_device, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0 && target_device != NULL) break;
    }

    // AUDIT-FIX: manche Treiber (z.B. Mesa RustiCL fuer AMD-APUs) melden ihr
    // GPU-Geraet als CL_DEVICE_TYPE_CUSTOM statt CL_DEVICE_TYPE_GPU - der obige
    // Filter liefert dann still err=-1/count=0 und init_opencl_engine() gibt
    // ohne jede Meldung 0 zurueck ("GPU wird nicht erkannt", verifiziert per
    // eigenem clGetDeviceInfo-Test: raw_type=CL_DEVICE_TYPE_CUSTOM auf diesem
    // System). Fallback: erstes Nicht-CPU-Geraet nehmen, wenn kein offiziell
    // deklariertes CL_DEVICE_TYPE_GPU gefunden wurde.
    if (target_device == NULL) {
        cl_platform_id* fb_platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
        clGetPlatformIDs(num_platforms, fb_platforms, NULL);
        for (cl_uint i = 0; i < num_platforms && target_device == NULL; i++) {
            cl_uint num_devices = 0;
            clGetDeviceIDs(fb_platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
            if (num_devices == 0) continue;
            cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
            clGetDeviceIDs(fb_platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
            for (cl_uint j = 0; j < num_devices; j++) {
                cl_device_type t;
                clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(t), &t, NULL);
                if (!(t & CL_DEVICE_TYPE_CPU)) {
                    target_device = devices[j];
                    fprintf(stderr, "⚡ [SYMBIO VRAM] Kein CL_DEVICE_TYPE_GPU gemeldet, nutze Nicht-CPU-Geraet als Fallback (Treiber-Quirk, z.B. Mesa RustiCL).\n");
                    break;
                }
            }
            free(devices);
        }
        free(fb_platforms);
    }

    free(platforms);

    if (target_device == NULL) return 0;
    
    ocl_ctx = clCreateContext(NULL, 1, &target_device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return 0;

    ocl_queue = clCreateCommandQueue(ocl_ctx, target_device, 0, &err); 
    if (err != CL_SUCCESS) { shutdown_opencl_engine(); return 0; }
    
    cl_program program = clCreateProgramWithSource(ocl_ctx, 1, (const char **)&kernel_source, NULL, &err);
    // FIX 6: Program Null Check
    if (err != CL_SUCCESS || !program) { shutdown_opencl_engine(); return 0; }

    cl_int build_status = clBuildProgram(program, 1, &target_device, NULL, NULL, NULL);
    
    if (build_status != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, target_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(program, target_device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        fprintf(stderr, "🚨 [SYMBIO GPU] JIT Kernel Kompilierung fehlgeschlagen!\n%s\n", log);
        free(log);
        clReleaseProgram(program);
        shutdown_opencl_engine();
        return 0;
    }

    ocl_kernel_matvec = clCreateKernel(program, "matVecMul", &err);
    ocl_kernel_adam = clCreateKernel(program, "adamUpdate", &err);
    ocl_kernel_att_fwd = clCreateKernel(program, "attentionForward", &err);
    ocl_kernel_att_bwd = clCreateKernel(program, "attentionBackward", &err);
    ocl_kernel_matvec_t = clCreateKernel(program, "matVecMulT", &err);
    ocl_kernel_grad_acc = clCreateKernel(program, "gradAccum", &err);
    ocl_kernel_gemm_f = clCreateKernel(program, "matMulBatched", &err);
    ocl_kernel_gemm_t = clCreateKernel(program, "matMulTBatched", &err);
    ocl_kernel_gemm_g = clCreateKernel(program, "gradAccumBatched", &err);
    ocl_kernel_att_fwd_b = clCreateKernel(program, "attentionForwardBatched", &err);
    ocl_kernel_att_bwd_q = clCreateKernel(program, "attentionBackwardBatchedQuery", &err);
    ocl_kernel_att_bwd_k = clCreateKernel(program, "attentionBackwardBatchedKey", &err);
    clReleaseProgram(program);
    
    if (err != CL_SUCCESS) { shutdown_opencl_engine(); return 0; }

    size_t hw_max_wg;
    if (clGetDeviceInfo(target_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &hw_max_wg, NULL) == CL_SUCCESS) {
        ocl_max_work_group_size = (hw_max_wg < 64) ? hw_max_wg : 64;
    }
    
    char deviceName[256];
    clGetDeviceInfo(target_device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    fprintf(stderr, "⚡ [SYMBIO VRAM] Stateful OpenCL JIT eingerastet auf: [%s]\n", deviceName);
    return 1;
}

void ocl_matVecMul(const float* h_M, const float* h_v, float* h_out, int rows, int cols) {
    if (!ocl_ctx || !h_M || !h_v || !h_out || rows == 0 || cols == 0) return;

    size_t size_M = (size_t)rows * (size_t)cols * sizeof(float);
    size_t size_v = (size_t)cols * sizeof(float);
    size_t size_out = (size_t)rows * sizeof(float);

    cl_mem d_M = ocl_get_buffer((void*)h_M, size_M);
    if (!d_M) return;

    cl_int err = CL_SUCCESS;
    // FIX 4: Robust Buffer Resize
    if (ocl_eph_v_size < size_v) {
        if (ocl_eph_v) clReleaseMemObject(ocl_eph_v);
        ocl_eph_v = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, size_v, NULL, &err);
        ocl_eph_v_size = (err == CL_SUCCESS) ? size_v : 0;
    }
    if (ocl_eph_out_size < size_out) {
        if (ocl_eph_out) clReleaseMemObject(ocl_eph_out);
        ocl_eph_out = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, size_out, NULL, &err);
        ocl_eph_out_size = (err == CL_SUCCESS) ? size_out : 0;
    }
    if (!ocl_eph_v || !ocl_eph_out) return;

    clEnqueueWriteBuffer(ocl_queue, ocl_eph_v, CL_FALSE, 0, size_v, h_v, 0, NULL, NULL);

    err  = clSetKernelArg(ocl_kernel_matvec, 0, sizeof(cl_mem), (void *)&d_M);
    err |= clSetKernelArg(ocl_kernel_matvec, 1, sizeof(cl_mem), (void *)&ocl_eph_v);
    err |= clSetKernelArg(ocl_kernel_matvec, 2, sizeof(cl_mem), (void *)&ocl_eph_out);
    err |= clSetKernelArg(ocl_kernel_matvec, 3, sizeof(int), (void *)&rows);
    err |= clSetKernelArg(ocl_kernel_matvec, 4, sizeof(int), (void *)&cols);
    if (err != CL_SUCCESS) return;

    size_t local_item_size = ocl_max_work_group_size; 
    size_t global_item_size = (((size_t)rows + local_item_size - 1) / local_item_size) * local_item_size;

    clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_matvec, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl_queue, ocl_eph_out, CL_TRUE, 0, size_out, h_out, 0, NULL, NULL);
}

void ocl_attentionForward(const float* h_q, const float* h_k_cache, const float* h_v_cache, float* h_out, int pos, int numHeads, int headSize, int dim, int ctx) {
    if (!ocl_ctx || pos >= ctx) return; 
    
    size_t size_cache = (size_t)ctx * dim * sizeof(float);
    size_t size_vec = (size_t)dim * sizeof(float);
    size_t size_scores = (size_t)numHeads * ctx * sizeof(float);
    
    cl_mem d_k = ocl_get_buffer((void*)h_k_cache, size_cache);
    cl_mem d_v = ocl_get_buffer((void*)h_v_cache, size_cache);
    if (!d_k || !d_v) return; 
    
    size_t offset = (size_t)pos * dim * sizeof(float);
    clEnqueueWriteBuffer(ocl_queue, d_k, CL_FALSE, offset, size_vec, h_k_cache + (pos * dim), 0, NULL, NULL);
    clEnqueueWriteBuffer(ocl_queue, d_v, CL_FALSE, offset, size_vec, h_v_cache + (pos * dim), 0, NULL, NULL);
    
    cl_int err = CL_SUCCESS;
    // FIX 4: Robust Buffer Resize
    if (ocl_eph_v_size < size_vec) {
        if (ocl_eph_v) clReleaseMemObject(ocl_eph_v);
        ocl_eph_v = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, size_vec, NULL, &err); 
        ocl_eph_v_size = (err == CL_SUCCESS) ? size_vec : 0;
    }
    if (ocl_eph_out_size < size_vec) {
        if (ocl_eph_out) clReleaseMemObject(ocl_eph_out);
        ocl_eph_out = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, size_vec, NULL, &err); 
        ocl_eph_out_size = (err == CL_SUCCESS) ? size_vec : 0;
    }
    if (ocl_eph_scores_size < size_scores) {
        if (ocl_eph_scores) clReleaseMemObject(ocl_eph_scores);
        ocl_eph_scores = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, size_scores, NULL, &err); 
        ocl_eph_scores_size = (err == CL_SUCCESS) ? size_scores : 0;
    }
    if (!ocl_eph_v || !ocl_eph_out || !ocl_eph_scores) return;
    
    clEnqueueWriteBuffer(ocl_queue, ocl_eph_v, CL_FALSE, 0, size_vec, h_q, 0, NULL, NULL);
    
    err  = clSetKernelArg(ocl_kernel_att_fwd, 0, sizeof(cl_mem), &ocl_eph_v);
    err |= clSetKernelArg(ocl_kernel_att_fwd, 1, sizeof(cl_mem), &d_k);
    err |= clSetKernelArg(ocl_kernel_att_fwd, 2, sizeof(cl_mem), &d_v);
    err |= clSetKernelArg(ocl_kernel_att_fwd, 3, sizeof(cl_mem), &ocl_eph_out);
    err |= clSetKernelArg(ocl_kernel_att_fwd, 4, sizeof(cl_mem), &ocl_eph_scores);
    err |= clSetKernelArg(ocl_kernel_att_fwd, 5, sizeof(int), &pos);
    err |= clSetKernelArg(ocl_kernel_att_fwd, 6, sizeof(int), &numHeads);
    err |= clSetKernelArg(ocl_kernel_att_fwd, 7, sizeof(int), &headSize);
    err |= clSetKernelArg(ocl_kernel_att_fwd, 8, sizeof(int), &dim);
    err |= clSetKernelArg(ocl_kernel_att_fwd, 9, sizeof(int), &ctx);
    if (err != CL_SUCCESS) return;
    
    size_t global_item_size = numHeads;
    clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_att_fwd, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl_queue, ocl_eph_out, CL_TRUE, 0, size_vec, h_out, 0, NULL, NULL);
}

void ocl_attentionBackward(const float* h_q, const float* h_k_cache, const float* h_v_cache, const float* h_d_att_out,
                           float* h_d_q, float* h_d_k_cache, float* h_d_v_cache, 
                           int pos, int numHeads, int headSize, int dim, int ctx) {
    if (!ocl_ctx || pos >= ctx) return; 
    
    size_t size_cache = (size_t)ctx * dim * sizeof(float);
    size_t size_vec = (size_t)dim * sizeof(float);
    size_t size_scores = (size_t)numHeads * ctx * sizeof(float);
    
    cl_mem d_k = ocl_get_buffer((void*)h_k_cache, size_cache);
    cl_mem d_v = ocl_get_buffer((void*)h_v_cache, size_cache);
    cl_mem d_dk = ocl_get_buffer((void*)h_d_k_cache, size_cache);
    cl_mem d_dv = ocl_get_buffer((void*)h_d_v_cache, size_cache);
    if (!d_k || !d_v || !d_dk || !d_dv) return;
    
    cl_int err = CL_SUCCESS;
    // FIX 4: Robust Buffer Resize
    if (ocl_eph_v_size < size_vec) {
        if (ocl_eph_v) clReleaseMemObject(ocl_eph_v);
        ocl_eph_v = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, size_vec, NULL, &err); 
        ocl_eph_v_size = (err == CL_SUCCESS) ? size_vec : 0;
    }
    if (ocl_eph_out_size < size_vec) {
        if (ocl_eph_out) clReleaseMemObject(ocl_eph_out);
        ocl_eph_out = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, size_vec, NULL, &err); 
        ocl_eph_out_size = (err == CL_SUCCESS) ? size_vec : 0;
    }
    if (ocl_eph_dq_size < size_vec) {
        if (ocl_eph_dq) clReleaseMemObject(ocl_eph_dq);
        ocl_eph_dq = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, size_vec, NULL, &err); 
        ocl_eph_dq_size = (err == CL_SUCCESS) ? size_vec : 0;
    }
    if (ocl_eph_scores_size < size_scores) {
        if (ocl_eph_scores) clReleaseMemObject(ocl_eph_scores);
        ocl_eph_scores = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, size_scores, NULL, &err); 
        ocl_eph_scores_size = (err == CL_SUCCESS) ? size_scores : 0;
    }
    if (ocl_eph_dw_size < size_scores) {
        if (ocl_eph_dw) clReleaseMemObject(ocl_eph_dw);
        ocl_eph_dw = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, size_scores, NULL, &err); 
        ocl_eph_dw_size = (err == CL_SUCCESS) ? size_scores : 0;
    }
    if (!ocl_eph_v || !ocl_eph_out || !ocl_eph_dq || !ocl_eph_scores || !ocl_eph_dw) return;
    
    clEnqueueWriteBuffer(ocl_queue, ocl_eph_v, CL_FALSE, 0, size_vec, h_q, 0, NULL, NULL);
    clEnqueueWriteBuffer(ocl_queue, ocl_eph_out, CL_FALSE, 0, size_vec, h_d_att_out, 0, NULL, NULL);
    
    // FIX 5: Toter Host-to-Device Copy für h_d_q entfernt (wird ohnehin im Kernel überschrieben)
    
    err  = clSetKernelArg(ocl_kernel_att_bwd, 0, sizeof(cl_mem), &ocl_eph_v);
    err |= clSetKernelArg(ocl_kernel_att_bwd, 1, sizeof(cl_mem), &d_k);
    err |= clSetKernelArg(ocl_kernel_att_bwd, 2, sizeof(cl_mem), &d_v);
    err |= clSetKernelArg(ocl_kernel_att_bwd, 3, sizeof(cl_mem), &ocl_eph_out);
    err |= clSetKernelArg(ocl_kernel_att_bwd, 4, sizeof(cl_mem), &ocl_eph_dq);
    err |= clSetKernelArg(ocl_kernel_att_bwd, 5, sizeof(cl_mem), &d_dk);
    err |= clSetKernelArg(ocl_kernel_att_bwd, 6, sizeof(cl_mem), &d_dv);
    err |= clSetKernelArg(ocl_kernel_att_bwd, 7, sizeof(cl_mem), &ocl_eph_scores);
    err |= clSetKernelArg(ocl_kernel_att_bwd, 8, sizeof(cl_mem), &ocl_eph_dw);
    err |= clSetKernelArg(ocl_kernel_att_bwd, 9, sizeof(int), &pos);
    err |= clSetKernelArg(ocl_kernel_att_bwd, 10, sizeof(int), &numHeads);
    err |= clSetKernelArg(ocl_kernel_att_bwd, 11, sizeof(int), &headSize);
    err |= clSetKernelArg(ocl_kernel_att_bwd, 12, sizeof(int), &dim);
    err |= clSetKernelArg(ocl_kernel_att_bwd, 13, sizeof(int), &ctx);
    if (err != CL_SUCCESS) return;
    
    size_t global_item_size = numHeads;
    clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_att_bwd, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
    
    clEnqueueReadBuffer(ocl_queue, ocl_eph_dq, CL_FALSE, 0, size_vec, h_d_q, 0, NULL, NULL);
    size_t offset = (size_t)pos * dim * sizeof(float);
    clEnqueueReadBuffer(ocl_queue, d_dk, CL_FALSE, offset, size_vec, h_d_k_cache + (pos * dim), 0, NULL, NULL);
    clEnqueueReadBuffer(ocl_queue, d_dv, CL_FALSE, offset, size_vec, h_d_v_cache + (pos * dim), 0, NULL, NULL);

    clFinish(ocl_queue);
}

// ==============================================================================
// V8.5 BACKWARD OFFLOAD
// d_in = M^T * d_out   und   gm += d_out (x) i_v
// gm bleibt persistent im VRAM und akkumuliert ueber alle Token eines Steps.
// Der Host holt sie einmal pro Step per ocl_sync_to_host ab - kein Transfer
// pro Token. Nur die kleinen Vektoren wandern hin und her.
// ==============================================================================
static int ocl_eph_bt_ensure(cl_mem* buf, size_t* cur, size_t need) {
    if (*cur >= need) return 1;
    cl_int err;
    if (*buf) clReleaseMemObject(*buf);
    *buf = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, need, NULL, &err);
    *cur = (err == CL_SUCCESS) ? need : 0;
    return (err == CL_SUCCESS);
}

void ocl_matVecMulT(const float* h_M, const float* h_d_out, float* h_d_in, int rows, int cols, int accumulate) {
    if (!ocl_ctx || !h_M || !h_d_out || !h_d_in || rows == 0 || cols == 0) return;

    size_t size_M   = (size_t)rows * (size_t)cols * sizeof(float);
    size_t size_out = (size_t)rows * sizeof(float);
    size_t size_in  = (size_t)cols * sizeof(float);

    cl_mem d_M = ocl_get_buffer((void*)h_M, size_M);
    if (!d_M) return;
    if (!ocl_eph_bt_ensure(&ocl_eph_bt_a, &ocl_eph_bt_a_size, size_out)) return;
    if (!ocl_eph_bt_ensure(&ocl_eph_bt_b, &ocl_eph_bt_b_size, size_in)) return;

    clEnqueueWriteBuffer(ocl_queue, ocl_eph_bt_a, CL_FALSE, 0, size_out, h_d_out, 0, NULL, NULL);
    if (accumulate == 1) {
        clEnqueueWriteBuffer(ocl_queue, ocl_eph_bt_b, CL_FALSE, 0, size_in, h_d_in, 0, NULL, NULL);
    }

    cl_int err;
    err  = clSetKernelArg(ocl_kernel_matvec_t, 0, sizeof(cl_mem), (void *)&d_M);
    err |= clSetKernelArg(ocl_kernel_matvec_t, 1, sizeof(cl_mem), (void *)&ocl_eph_bt_a);
    err |= clSetKernelArg(ocl_kernel_matvec_t, 2, sizeof(cl_mem), (void *)&ocl_eph_bt_b);
    err |= clSetKernelArg(ocl_kernel_matvec_t, 3, sizeof(int), (void *)&rows);
    err |= clSetKernelArg(ocl_kernel_matvec_t, 4, sizeof(int), (void *)&cols);
    err |= clSetKernelArg(ocl_kernel_matvec_t, 5, sizeof(int), (void *)&accumulate);
    if (err != CL_SUCCESS) return;

    size_t local_item_size = ocl_max_work_group_size;
    size_t global_item_size = (((size_t)cols + local_item_size - 1) / local_item_size) * local_item_size;
    clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_matvec_t, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl_queue, ocl_eph_bt_b, CL_TRUE, 0, size_in, h_d_in, 0, NULL, NULL);
}

void ocl_gradAccum(float* h_gm, const float* h_d_out, const float* h_i_v, int rows, int cols) {
    if (!ocl_ctx || !h_gm || !h_d_out || !h_i_v || rows == 0 || cols == 0) return;

    size_t size_gm  = (size_t)rows * (size_t)cols * sizeof(float);
    size_t size_out = (size_t)rows * sizeof(float);
    size_t size_in  = (size_t)cols * sizeof(float);

    cl_mem d_gm = ocl_get_buffer((void*)h_gm, size_gm);
    if (!d_gm) return;
    if (!ocl_eph_bt_ensure(&ocl_eph_bt_a, &ocl_eph_bt_a_size, size_out)) return;
    if (!ocl_eph_bt_ensure(&ocl_eph_bt_b, &ocl_eph_bt_b_size, size_in)) return;

    // CL_TRUE ist hier Pflicht: gradAccum hat am Ende KEINEN blockierenden Read.
    // Mit CL_FALSE koennte der Host d_out/i_v ueberschreiben, bevor die Kopie lief.
    clEnqueueWriteBuffer(ocl_queue, ocl_eph_bt_a, CL_TRUE, 0, size_out, h_d_out, 0, NULL, NULL);
    clEnqueueWriteBuffer(ocl_queue, ocl_eph_bt_b, CL_TRUE, 0, size_in, h_i_v, 0, NULL, NULL);

    cl_int err;
    err  = clSetKernelArg(ocl_kernel_grad_acc, 0, sizeof(cl_mem), (void *)&d_gm);
    err |= clSetKernelArg(ocl_kernel_grad_acc, 1, sizeof(cl_mem), (void *)&ocl_eph_bt_a);
    err |= clSetKernelArg(ocl_kernel_grad_acc, 2, sizeof(cl_mem), (void *)&ocl_eph_bt_b);
    err |= clSetKernelArg(ocl_kernel_grad_acc, 3, sizeof(int), (void *)&rows);
    err |= clSetKernelArg(ocl_kernel_grad_acc, 4, sizeof(int), (void *)&cols);
    if (err != CL_SUCCESS) return;

    size_t local2[2]  = { ocl_max_work_group_size, 1 };
    size_t global2[2] = { (((size_t)cols + local2[0] - 1) / local2[0]) * local2[0], (size_t)rows };
    clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_grad_acc, 2, NULL, global2, local2, 0, NULL, NULL);
    // Kein Readback: gm bleibt im VRAM und akkumuliert weiter.
}

// ==============================================================================
// V8.6 ETAPPE 1 - BATCHING UEBER DIE POSITIONSACHSE
// Layout durchgehend row-major: IN[p*cols + c], OUT[p*rows + r].
// Das ist exakt die Geometrie der bestehenden act_*-Caches (pos * dim).
// Bewusst ephemer: Upload rein, Kernel, Download raus. Kein Buffer bleibt
// zwischen den Aufrufen gueltig -> keine Host/Device-Staleness moeglich.
// ==============================================================================
void ocl_matMulBatched(const float* h_M, const float* h_in, float* h_out, int rows, int cols, int batch) {
    if (!ocl_ctx || !h_M || !h_in || !h_out || rows <= 0 || cols <= 0 || batch <= 0) return;
    size_t size_M = (size_t)rows * (size_t)cols * sizeof(float);
    size_t size_i = (size_t)batch * (size_t)cols * sizeof(float);
    size_t size_o = (size_t)batch * (size_t)rows * sizeof(float);

    cl_mem d_M = ocl_get_buffer((void*)h_M, size_M);
    if (!d_M) return;
    if (!ocl_eph_bt_ensure(&ocl_eph_bm_c, &ocl_eph_bm_c_size, size_i)) return;
    if (!ocl_eph_bt_ensure(&ocl_eph_bm_r, &ocl_eph_bm_r_size, size_o)) return;

    clEnqueueWriteBuffer(ocl_queue, ocl_eph_bm_c, CL_FALSE, 0, size_i, h_in, 0, NULL, NULL);

    cl_int err;
    err  = clSetKernelArg(ocl_kernel_gemm_f, 0, sizeof(cl_mem), (void*)&d_M);
    err |= clSetKernelArg(ocl_kernel_gemm_f, 1, sizeof(cl_mem), (void*)&ocl_eph_bm_c);
    err |= clSetKernelArg(ocl_kernel_gemm_f, 2, sizeof(cl_mem), (void*)&ocl_eph_bm_r);
    err |= clSetKernelArg(ocl_kernel_gemm_f, 3, sizeof(int), (void*)&rows);
    err |= clSetKernelArg(ocl_kernel_gemm_f, 4, sizeof(int), (void*)&cols);
    err |= clSetKernelArg(ocl_kernel_gemm_f, 5, sizeof(int), (void*)&batch);
    if (err != CL_SUCCESS) return;

    size_t local2[2]  = { ocl_max_work_group_size, 1 };
    size_t global2[2] = { (((size_t)rows + local2[0] - 1) / local2[0]) * local2[0], (size_t)batch };
    clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_gemm_f, 2, NULL, global2, local2, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl_queue, ocl_eph_bm_r, CL_TRUE, 0, size_o, h_out, 0, NULL, NULL);
}

void ocl_matMulTBatched(const float* h_M, const float* h_d_out, float* h_d_in, int rows, int cols, int batch, int accumulate) {
    if (!ocl_ctx || !h_M || !h_d_out || !h_d_in || rows <= 0 || cols <= 0 || batch <= 0) return;
    size_t size_M = (size_t)rows * (size_t)cols * sizeof(float);
    size_t size_o = (size_t)batch * (size_t)rows * sizeof(float);
    size_t size_i = (size_t)batch * (size_t)cols * sizeof(float);

    cl_mem d_M = ocl_get_buffer((void*)h_M, size_M);
    if (!d_M) return;
    if (!ocl_eph_bt_ensure(&ocl_eph_bm_r, &ocl_eph_bm_r_size, size_o)) return;
    if (!ocl_eph_bt_ensure(&ocl_eph_bm_c, &ocl_eph_bm_c_size, size_i)) return;

    clEnqueueWriteBuffer(ocl_queue, ocl_eph_bm_r, CL_FALSE, 0, size_o, h_d_out, 0, NULL, NULL);
    if (accumulate == 1) {
        clEnqueueWriteBuffer(ocl_queue, ocl_eph_bm_c, CL_FALSE, 0, size_i, h_d_in, 0, NULL, NULL);
    }

    cl_int err;
    err  = clSetKernelArg(ocl_kernel_gemm_t, 0, sizeof(cl_mem), (void*)&d_M);
    err |= clSetKernelArg(ocl_kernel_gemm_t, 1, sizeof(cl_mem), (void*)&ocl_eph_bm_r);
    err |= clSetKernelArg(ocl_kernel_gemm_t, 2, sizeof(cl_mem), (void*)&ocl_eph_bm_c);
    err |= clSetKernelArg(ocl_kernel_gemm_t, 3, sizeof(int), (void*)&rows);
    err |= clSetKernelArg(ocl_kernel_gemm_t, 4, sizeof(int), (void*)&cols);
    err |= clSetKernelArg(ocl_kernel_gemm_t, 5, sizeof(int), (void*)&batch);
    err |= clSetKernelArg(ocl_kernel_gemm_t, 6, sizeof(int), (void*)&accumulate);
    if (err != CL_SUCCESS) return;

    size_t local2[2]  = { ocl_max_work_group_size, 1 };
    size_t global2[2] = { (((size_t)cols + local2[0] - 1) / local2[0]) * local2[0], (size_t)batch };
    clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_gemm_t, 2, NULL, global2, local2, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl_queue, ocl_eph_bm_c, CL_TRUE, 0, size_i, h_d_in, 0, NULL, NULL);
}

void ocl_gradAccumBatched(float* h_gm, const float* h_d_out, const float* h_i_v, int rows, int cols, int batch) {
    if (!ocl_ctx || !h_gm || !h_d_out || !h_i_v || rows <= 0 || cols <= 0 || batch <= 0) return;
    size_t size_gm = (size_t)rows * (size_t)cols * sizeof(float);
    size_t size_o  = (size_t)batch * (size_t)rows * sizeof(float);
    size_t size_i  = (size_t)batch * (size_t)cols * sizeof(float);

    cl_mem d_gm = ocl_get_buffer((void*)h_gm, size_gm);
    if (!d_gm) return;
    if (!ocl_eph_bt_ensure(&ocl_eph_bm_r, &ocl_eph_bm_r_size, size_o)) return;
    if (!ocl_eph_bt_ensure(&ocl_eph_bm_c, &ocl_eph_bm_c_size, size_i)) return;

    // CL_TRUE: dieser Aufruf endet ohne blockierenden Read, der Host darf die
    // Quellpuffer erst nach der Kopie wieder anfassen.
    clEnqueueWriteBuffer(ocl_queue, ocl_eph_bm_r, CL_TRUE, 0, size_o, h_d_out, 0, NULL, NULL);
    clEnqueueWriteBuffer(ocl_queue, ocl_eph_bm_c, CL_TRUE, 0, size_i, h_i_v, 0, NULL, NULL);

    cl_int err;
    err  = clSetKernelArg(ocl_kernel_gemm_g, 0, sizeof(cl_mem), (void*)&d_gm);
    err |= clSetKernelArg(ocl_kernel_gemm_g, 1, sizeof(cl_mem), (void*)&ocl_eph_bm_r);
    err |= clSetKernelArg(ocl_kernel_gemm_g, 2, sizeof(cl_mem), (void*)&ocl_eph_bm_c);
    err |= clSetKernelArg(ocl_kernel_gemm_g, 3, sizeof(int), (void*)&rows);
    err |= clSetKernelArg(ocl_kernel_gemm_g, 4, sizeof(int), (void*)&cols);
    err |= clSetKernelArg(ocl_kernel_gemm_g, 5, sizeof(int), (void*)&batch);
    if (err != CL_SUCCESS) return;

    size_t local2[2]  = { ocl_max_work_group_size, 1 };
    size_t global2[2] = { (((size_t)cols + local2[0] - 1) / local2[0]) * local2[0], (size_t)rows };
    clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_gemm_g, 2, NULL, global2, local2, 0, NULL, NULL);
    // gm bleibt persistent im VRAM, Abholung wie bisher per ocl_sync_to_host.
}

// ==============================================================================
// AUDIT: GEBUENDELTE ATTENTION - ein Dispatch pro Layer statt 'act' einzelner.
// Ursache der urspruenglichen 20x-Verlangsamung auf schwachen/Treiber-limitierten
// GPUs: jede Position zahlte vollen Kernel-Launch + Transfer-Overhead inklusive
// eines BLOCKIERENDEN clEnqueueReadBuffer. Hier genau ein Upload, ein/zwei
// Kernel-Dispatches, ein Download fuer die GESAMTE Positionsachse.
// Erfordert layer-major statt position-major Aufrufreihenfolge im Trainings-Loop
// (siehe cortex.nim::cortex_train_step, batched_attention-Zweig): Q/K/V muessen
// fuer ALLE Positionen eines Layers feststehen, bevor Attention fuer eine davon
// berechnet wird.
// ==============================================================================
void ocl_attentionForwardBatched(const float* h_q_all, const float* h_k_cache, const float* h_v_cache,
                                  float* h_att_out_all, int act, int numHeads, int headSize, int dim, int ctx) {
    if (!ocl_ctx || !h_q_all || !h_k_cache || !h_v_cache || !h_att_out_all || act <= 0) return;

    size_t size_cache  = (size_t)ctx * (size_t)dim * sizeof(float);
    size_t size_qout   = (size_t)act * (size_t)dim * sizeof(float);
    size_t size_scores = (size_t)act * (size_t)numHeads * (size_t)ctx * sizeof(float);

    cl_mem d_k = ocl_get_buffer((void*)h_k_cache, size_cache);
    cl_mem d_v = ocl_get_buffer((void*)h_v_cache, size_cache);
    if (!d_k || !d_v) return;

    // AUDIT-FIX: ocl_get_buffer() laedt nur beim ERSTEN Aufruf fuer einen Host-
    // Pointer hoch, danach kommt stur der gecachte Device-Buffer zurueck - auch
    // wenn der Host-Inhalt sich (durch neue QKV-Berechnung im naechsten Trainings-
    // Schritt) laengst geaendert hat. Der alte Pro-Position-Pfad glich das mit
    // einem Write pro Aufruf aus (dort nur 1 Position gross); hier explizit fuer
    // alle 'act' Positionen. Ohne das driftet der Loss ab Schritt 2 (empirisch
    // verifiziert: 0.000000 Differenz mit diesem Fix, bis zu 0.013 ohne).
    clEnqueueWriteBuffer(ocl_queue, d_k, CL_FALSE, 0, size_qout, h_k_cache, 0, NULL, NULL);
    clEnqueueWriteBuffer(ocl_queue, d_v, CL_FALSE, 0, size_qout, h_v_cache, 0, NULL, NULL);

    if (!ocl_eph_bt_ensure(&ocl_eph_att_q, &ocl_eph_att_q_size, size_qout)) return;
    if (!ocl_eph_bt_ensure(&ocl_eph_att_out, &ocl_eph_att_out_size, size_qout)) return;
    if (!ocl_eph_bt_ensure(&ocl_eph_att_scores, &ocl_eph_att_scores_size, size_scores)) return;

    clEnqueueWriteBuffer(ocl_queue, ocl_eph_att_q, CL_FALSE, 0, size_qout, h_q_all, 0, NULL, NULL);

    cl_int err;
    err  = clSetKernelArg(ocl_kernel_att_fwd_b, 0, sizeof(cl_mem), &ocl_eph_att_q);
    err |= clSetKernelArg(ocl_kernel_att_fwd_b, 1, sizeof(cl_mem), &d_k);
    err |= clSetKernelArg(ocl_kernel_att_fwd_b, 2, sizeof(cl_mem), &d_v);
    err |= clSetKernelArg(ocl_kernel_att_fwd_b, 3, sizeof(cl_mem), &ocl_eph_att_out);
    err |= clSetKernelArg(ocl_kernel_att_fwd_b, 4, sizeof(cl_mem), &ocl_eph_att_scores);
    err |= clSetKernelArg(ocl_kernel_att_fwd_b, 5, sizeof(int), &act);
    err |= clSetKernelArg(ocl_kernel_att_fwd_b, 6, sizeof(int), &numHeads);
    err |= clSetKernelArg(ocl_kernel_att_fwd_b, 7, sizeof(int), &headSize);
    err |= clSetKernelArg(ocl_kernel_att_fwd_b, 8, sizeof(int), &dim);
    err |= clSetKernelArg(ocl_kernel_att_fwd_b, 9, sizeof(int), &ctx);
    if (err != CL_SUCCESS) return;

    // AUDIT-FIX: ein einzelner Dispatch ueber alle 'act' Positionen liess bei
    // grossen Kontextlaengen (ctx=2048+) den GPU-Job so lange laufen, dass der
    // AMDGPU-Watchdog die Compute-Ring als haengengeblieben eingestuft und den
    // Context gekillt hat ("ring timeout", Absturz beim Matriarchin-Fine-Tuning,
    // 2026-08-03). Work-Items sind pro Position unabhaengig (schreiben nur in
    // ihr eigenes p-Segment von att_out/scores) - der Dispatch laesst sich daher
    // gefahrlos in kurze Positions-Bloecke aufteilen.
    for (size_t p0 = 0; p0 < (size_t)act; p0 += ATT_POS_CHUNK) {
        size_t chunkLen = ((size_t)act - p0) < ATT_POS_CHUNK ? ((size_t)act - p0) : ATT_POS_CHUNK;
        size_t offset2b[2] = { 0, p0 };
        size_t global2b[2] = { (size_t)numHeads, chunkLen };
        clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_att_fwd_b, 2, offset2b, global2b, NULL, 0, NULL, NULL);
    }
    clEnqueueReadBuffer(ocl_queue, ocl_eph_att_out, CL_TRUE, 0, size_qout, h_att_out_all, 0, NULL, NULL);
}

void ocl_attentionBackwardBatched(const float* h_q_all, const float* h_k_cache, const float* h_v_cache,
                                   const float* h_d_att_out_all, float* h_d_q_all,
                                   float* h_d_k_cache, float* h_d_v_cache,
                                   int act, int numHeads, int headSize, int dim, int ctx) {
    if (!ocl_ctx || !h_q_all || !h_k_cache || !h_v_cache || !h_d_att_out_all ||
        !h_d_q_all || !h_d_k_cache || !h_d_v_cache || act <= 0) return;

    size_t size_cache  = (size_t)ctx * (size_t)dim * sizeof(float);
    size_t size_qout   = (size_t)act * (size_t)dim * sizeof(float);
    size_t size_scores = (size_t)act * (size_t)numHeads * (size_t)ctx * sizeof(float);

    cl_mem d_k  = ocl_get_buffer((void*)h_k_cache, size_cache);
    cl_mem d_v  = ocl_get_buffer((void*)h_v_cache, size_cache);
    cl_mem d_dk = ocl_get_buffer((void*)h_d_k_cache, size_cache);
    cl_mem d_dv = ocl_get_buffer((void*)h_d_v_cache, size_cache);
    if (!d_k || !d_v || !d_dk || !d_dv) return;

    // AUDIT-FIX: siehe ocl_attentionForwardBatched - d_k/d_v sind Lese-Inputs und
    // muessen bei jedem Aufruf frisch vom Host geholt werden (ocl_get_buffer()
    // laedt nur beim allerersten Mal automatisch hoch). d_dk/d_dv sind reine
    // Schreib-Ziele des Kernels (der sie selbst nullt und neu befuellt) und
    // brauchen deshalb KEIN Pre-Upload.
    clEnqueueWriteBuffer(ocl_queue, d_k, CL_FALSE, 0, size_qout, h_k_cache, 0, NULL, NULL);
    clEnqueueWriteBuffer(ocl_queue, d_v, CL_FALSE, 0, size_qout, h_v_cache, 0, NULL, NULL);

    if (!ocl_eph_bt_ensure(&ocl_eph_att_q, &ocl_eph_att_q_size, size_qout)) return;
    if (!ocl_eph_bt_ensure(&ocl_eph_att_datt, &ocl_eph_att_datt_size, size_qout)) return;
    if (!ocl_eph_bt_ensure(&ocl_eph_att_dq, &ocl_eph_att_dq_size, size_qout)) return;
    if (!ocl_eph_bt_ensure(&ocl_eph_att_scores, &ocl_eph_att_scores_size, size_scores)) return;
    if (!ocl_eph_bt_ensure(&ocl_eph_att_ds, &ocl_eph_att_ds_size, size_scores)) return;

    clEnqueueWriteBuffer(ocl_queue, ocl_eph_att_q, CL_FALSE, 0, size_qout, h_q_all, 0, NULL, NULL);
    clEnqueueWriteBuffer(ocl_queue, ocl_eph_att_datt, CL_FALSE, 0, size_qout, h_d_att_out_all, 0, NULL, NULL);

    cl_int err;
    // --- Stufe A: query-parallel (d_q, weights, ds) ---
    err  = clSetKernelArg(ocl_kernel_att_bwd_q, 0, sizeof(cl_mem), &ocl_eph_att_q);
    err |= clSetKernelArg(ocl_kernel_att_bwd_q, 1, sizeof(cl_mem), &d_k);
    err |= clSetKernelArg(ocl_kernel_att_bwd_q, 2, sizeof(cl_mem), &d_v);
    err |= clSetKernelArg(ocl_kernel_att_bwd_q, 3, sizeof(cl_mem), &ocl_eph_att_datt);
    err |= clSetKernelArg(ocl_kernel_att_bwd_q, 4, sizeof(cl_mem), &ocl_eph_att_dq);
    err |= clSetKernelArg(ocl_kernel_att_bwd_q, 5, sizeof(cl_mem), &ocl_eph_att_scores);
    err |= clSetKernelArg(ocl_kernel_att_bwd_q, 6, sizeof(cl_mem), &ocl_eph_att_ds);
    err |= clSetKernelArg(ocl_kernel_att_bwd_q, 7, sizeof(int), &act);
    err |= clSetKernelArg(ocl_kernel_att_bwd_q, 8, sizeof(int), &numHeads);
    err |= clSetKernelArg(ocl_kernel_att_bwd_q, 9, sizeof(int), &headSize);
    err |= clSetKernelArg(ocl_kernel_att_bwd_q, 10, sizeof(int), &dim);
    err |= clSetKernelArg(ocl_kernel_att_bwd_q, 11, sizeof(int), &ctx);
    if (err != CL_SUCCESS) return;

    // AUDIT-FIX: siehe Kommentar in ocl_attentionForwardBatched - Stufe A ist
    // pro Position (p) unabhaengig (d_q/weights/ds werden exklusiv fuer das
    // eigene p geschrieben), deshalb ebenfalls in Positions-Bloecke aufgeteilt,
    // um den AMDGPU-Watchdog nicht ueber einen einzelnen Riesen-Dispatch
    // auszuloesen.
    for (size_t p0 = 0; p0 < (size_t)act; p0 += ATT_POS_CHUNK) {
        size_t chunkLen = ((size_t)act - p0) < ATT_POS_CHUNK ? ((size_t)act - p0) : ATT_POS_CHUNK;
        size_t offset_q[2] = { 0, p0 };
        size_t global_q[2] = { (size_t)numHeads, chunkLen };
        clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_att_bwd_q, 2, offset_q, global_q, NULL, 0, NULL, NULL);
    }

    // --- Stufe B: key-parallel (d_k_cache, d_v_cache) - siehe Kernel-Kommentar
    // fuer die Begruendung, warum das NICHT ueber die Query-Achse laufen darf. ---
    err  = clSetKernelArg(ocl_kernel_att_bwd_k, 0, sizeof(cl_mem), &ocl_eph_att_q);
    err |= clSetKernelArg(ocl_kernel_att_bwd_k, 1, sizeof(cl_mem), &ocl_eph_att_datt);
    err |= clSetKernelArg(ocl_kernel_att_bwd_k, 2, sizeof(cl_mem), &ocl_eph_att_scores);
    err |= clSetKernelArg(ocl_kernel_att_bwd_k, 3, sizeof(cl_mem), &ocl_eph_att_ds);
    err |= clSetKernelArg(ocl_kernel_att_bwd_k, 4, sizeof(cl_mem), &d_dk);
    err |= clSetKernelArg(ocl_kernel_att_bwd_k, 5, sizeof(cl_mem), &d_dv);
    err |= clSetKernelArg(ocl_kernel_att_bwd_k, 6, sizeof(int), &act);
    err |= clSetKernelArg(ocl_kernel_att_bwd_k, 7, sizeof(int), &numHeads);
    err |= clSetKernelArg(ocl_kernel_att_bwd_k, 8, sizeof(int), &headSize);
    err |= clSetKernelArg(ocl_kernel_att_bwd_k, 9, sizeof(int), &dim);
    err |= clSetKernelArg(ocl_kernel_att_bwd_k, 10, sizeof(int), &ctx);
    if (err != CL_SUCCESS) return;

    // AUDIT-FIX: siehe oben - Stufe B ist pro Key-Position (t) unabhaengig
    // (d_k_cache/d_v_cache werden exklusiv fuer das eigene t geschrieben),
    // deshalb ebenfalls in Bloecke aufgeteilt.
    for (size_t t0 = 0; t0 < (size_t)act; t0 += ATT_POS_CHUNK) {
        size_t chunkLen = ((size_t)act - t0) < ATT_POS_CHUNK ? ((size_t)act - t0) : ATT_POS_CHUNK;
        size_t offset_k[2] = { 0, t0 };
        size_t global_k[2] = { (size_t)numHeads, chunkLen };
        clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_att_bwd_k, 2, offset_k, global_k, NULL, 0, NULL, NULL);
    }

    // Nur die tatsaechlich beschriebenen 'act' Positionen zurueckholen (nicht den
    // gesamten ctx-grossen Cache) - Positionen >= act liest ohnehin niemand.
    clEnqueueReadBuffer(ocl_queue, ocl_eph_att_dq, CL_FALSE, 0, size_qout, h_d_q_all, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl_queue, d_dk, CL_FALSE, 0, size_qout, h_d_k_cache, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl_queue, d_dv, CL_TRUE, 0, size_qout, h_d_v_cache, 0, NULL, NULL);
}

void ocl_adamStep(float* h_w, float* h_m, float* h_v, float* h_g, float lr, float b1, float b2, float b1t, float b2t, float scale, int size, int is_sparse,
                   int cols, int bound_r, int bound_c, float boost, float wd) {
    if (!ocl_ctx || !h_w || !h_m || !h_v || !h_g || size == 0 || scale == 0.0f) return;
    size_t bytes = (size_t)size * sizeof(float);
    cl_mem d_w = ocl_get_buffer((void*)h_w, bytes); cl_mem d_m = ocl_get_buffer((void*)h_m, bytes);
    cl_mem d_v = ocl_get_buffer((void*)h_v, bytes); cl_mem d_g = ocl_get_buffer((void*)h_g, bytes);
    if (!d_w || !d_m || !d_v || !d_g) return;

    clEnqueueWriteBuffer(ocl_queue, d_g, CL_TRUE, 0, bytes, h_g, 0, NULL, NULL);
    cl_int err;
    err  = clSetKernelArg(ocl_kernel_adam, 0, sizeof(cl_mem), (void *)&d_w); err |= clSetKernelArg(ocl_kernel_adam, 1, sizeof(cl_mem), (void *)&d_m);
    err |= clSetKernelArg(ocl_kernel_adam, 2, sizeof(cl_mem), (void *)&d_v); err |= clSetKernelArg(ocl_kernel_adam, 3, sizeof(cl_mem), (void *)&d_g);
    err |= clSetKernelArg(ocl_kernel_adam, 4, sizeof(float), (void *)&lr);  err |= clSetKernelArg(ocl_kernel_adam, 5, sizeof(float), (void *)&b1);
    err |= clSetKernelArg(ocl_kernel_adam, 6, sizeof(float), (void *)&b2);  err |= clSetKernelArg(ocl_kernel_adam, 7, sizeof(float), (void *)&b1t);
    err |= clSetKernelArg(ocl_kernel_adam, 8, sizeof(float), (void *)&b2t); err |= clSetKernelArg(ocl_kernel_adam, 9, sizeof(float), (void *)&scale);
    err |= clSetKernelArg(ocl_kernel_adam, 10, sizeof(int), (void *)&size); err |= clSetKernelArg(ocl_kernel_adam, 11, sizeof(int), (void *)&is_sparse);
    err |= clSetKernelArg(ocl_kernel_adam, 12, sizeof(int), (void *)&cols); err |= clSetKernelArg(ocl_kernel_adam, 13, sizeof(int), (void *)&bound_r);
    err |= clSetKernelArg(ocl_kernel_adam, 14, sizeof(int), (void *)&bound_c); err |= clSetKernelArg(ocl_kernel_adam, 15, sizeof(float), (void *)&boost);
    err |= clSetKernelArg(ocl_kernel_adam, 16, sizeof(float), (void *)&wd);

    if (err != CL_SUCCESS) return;

    size_t local_item_size = ocl_max_work_group_size; 
    size_t global_item_size = (((size_t)size + local_item_size - 1) / local_item_size) * local_item_size;
    clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_adam, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    
    clEnqueueReadBuffer(ocl_queue, d_w, CL_TRUE, 0, bytes, h_w, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl_queue, d_m, CL_TRUE, 0, bytes, h_m, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl_queue, d_v, CL_TRUE, 0, bytes, h_v, 0, NULL, NULL); 
    
    memset(h_g, 0, bytes); 
}
""" .}

proc init_opencl_engine*(): cint {.importc: "init_opencl_engine", cdecl.}
proc shutdown_opencl_engine*() {.importc: "shutdown_opencl_engine", cdecl.}
proc ocl_sync_to_host*(h_ptr: pointer) {.importc: "ocl_sync_to_host", cdecl.}
proc ocl_sync_to_device*(h_ptr: pointer, bytes: csize_t) {.importc: "ocl_sync_to_device", cdecl.}
proc ocl_release_buffer*(h_ptr: pointer) {.importc: "ocl_release_buffer", cdecl.}
proc ocl_matVecMul*(m: ptr cfloat, v: ptr cfloat, out_v: ptr cfloat, rows: cint, cols: cint) {.importc: "ocl_matVecMul", cdecl.}
proc ocl_attentionForward*(q: ptr cfloat, k_cache: ptr cfloat, v_cache: ptr cfloat, out_v: ptr cfloat, pos: cint, numHeads: cint, headSize: cint, dim: cint, ctx: cint) {.importc: "ocl_attentionForward", cdecl.}
proc ocl_attentionBackward*(q: ptr cfloat, k_cache: ptr cfloat, v_cache: ptr cfloat, d_att_out: ptr cfloat, d_q: ptr cfloat, d_k_cache: ptr cfloat, d_v_cache: ptr cfloat, pos: cint, numHeads: cint, headSize: cint, dim: cint, ctx: cint) {.importc: "ocl_attentionBackward", cdecl.}
proc ocl_adamStep*(w: ptr cfloat, m: ptr cfloat, v: ptr cfloat, g: ptr cfloat, lr: cfloat, b1: cfloat, b2: cfloat, b1t: cfloat, b2t: cfloat, scale: cfloat, size: cint, is_sparse: cint, cols: cint, bound_r: cint, bound_c: cint, boost: cfloat, wd: cfloat) {.importc: "ocl_adamStep", cdecl.}
proc ocl_zero_buffer*(h_ptr: ptr cfloat, bytes: csize_t) {.importc: "ocl_zero_buffer", cdecl.}
proc ocl_matVecMulT*(m: ptr cfloat, d_out: ptr cfloat, d_in: ptr cfloat, rows: cint, cols: cint, accumulate: cint) {.importc: "ocl_matVecMulT", cdecl.}
proc ocl_gradAccum*(gm: ptr cfloat, d_out: ptr cfloat, i_v: ptr cfloat, rows: cint, cols: cint) {.importc: "ocl_gradAccum", cdecl.}
proc ocl_matMulBatched*(m: ptr cfloat, i_v: ptr cfloat, o_v: ptr cfloat, rows: cint, cols: cint, batch: cint) {.importc: "ocl_matMulBatched", cdecl.}
proc ocl_matMulTBatched*(m: ptr cfloat, d_out: ptr cfloat, d_in: ptr cfloat, rows: cint, cols: cint, batch: cint, accumulate: cint) {.importc: "ocl_matMulTBatched", cdecl.}
proc ocl_gradAccumBatched*(gm: ptr cfloat, d_out: ptr cfloat, i_v: ptr cfloat, rows: cint, cols: cint, batch: cint) {.importc: "ocl_gradAccumBatched", cdecl.}
proc ocl_attentionForwardBatched*(q_all: ptr cfloat, k_cache: ptr cfloat, v_cache: ptr cfloat, att_out_all: ptr cfloat, act: cint, numHeads: cint, headSize: cint, dim: cint, ctx: cint) {.importc: "ocl_attentionForwardBatched", cdecl.}
proc ocl_attentionBackwardBatched*(q_all: ptr cfloat, k_cache: ptr cfloat, v_cache: ptr cfloat, d_att_out_all: ptr cfloat, d_q_all: ptr cfloat, d_k_cache: ptr cfloat, d_v_cache: ptr cfloat, act: cint, numHeads: cint, headSize: cint, dim: cint, ctx: cint) {.importc: "ocl_attentionBackwardBatched", cdecl.}