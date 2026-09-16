/*
 * Dateiname: cortex_cuda.cu
 * Funktion: Native CUDA Compute Engine
 * ==============================================================================
 * KOMPILIERUNG: nvcc -shared -Xcompiler -fPIC -lcublas -o libcortex_cuda.so cortex_cuda.cu
 *
 * HERKUNFT: Nutzer-Entwurf (2026-08-06), hier mit folgenden Fixes/Ergaenzungen
 * gegen das etablierte OpenCL-Pendant (cortex_opencl.nim) auditiert:
 *   1. #include <cstring> fehlte - memset() in cuda_adamStep() war ein
 *      unportabler impliziter Include (haette je nach nvcc/libstdc++-Version
 *      scheitern koennen).
 *   2. cuda_sync_to_device() fehlte komplett - ocl_get_buffer() UND
 *      cuda_get_buffer() cachen den Device-Buffer ab dem ersten Aufruf pro
 *      Host-Pointer und laden ihn bei Cache-Hit NICHT automatisch neu (siehe
 *      ocl_sync_to_device()-Kommentar in cortex_opencl.nim). Ohne Gegenstueck
 *      gab es keinen Weg, eine ausserhalb dieser Bibliothek veraenderte
 *      Host-Matrix (z.B. nach Net2Net-Upscale) vor dem naechsten CUDA-Aufruf
 *      gezielt neu hochzuladen - einzige Alternative waere der teure Umweg
 *      ueber cuda_release_buffer() (kompletter Realloc statt Overwrite).
 *   3. cuda_adamStep()/adamUpdateKernel fehlten die Growth-Boundary/Boost-
 *      Parameter (cols, bound_r, bound_c, boost), die der OpenCL-Adam-Kernel
 *      seit dem Net2Net-LR-Fix traegt (siehe docs/ROADMAP.md, "Differenzierte
 *      Learning-Rate"). Ohne das haette ein spaeteres Einklinken dieses
 *      Pfads in cortex.nim den LR-Boost fuer frisch expandierte Parameter
 *      stillschweigend abgeschaltet. Logik 1:1 aus dem OpenCL-Kernel
 *      uebernommen (elemBoost()-Aequivalent).
 *   4. matMulBatchedForwardKernel war ein von Hand geschriebener Kernel -
 *      trotz des cublasSetMathMode(TF32)-Aufrufs beim Init wurde cuBLAS an
 *      keiner Stelle tatsaechlich benutzt, die Tensor Cores blieben also
 *      komplett ungenutzt (die eine Sache, die CUDA gegenueber OpenCL auf
 *      NVIDIA ueberhaupt einen Vorteil verschaffen koennte - siehe Roadmap-
 *      Notiz vom 2026-08-03). Ersetzt durch echtes cublasSgemm() mit
 *      Row-Major-auf-Column-Major-Transpositionstrick (siehe Kommentar an
 *      der Aufrufstelle). Der alte Kernel wurde entfernt, nicht daneben
 *      liegen gelassen.
 *   5. matMulBatchedBackwardWeightsKernel war deklariert, aber nirgendwo
 *      aufrufbar (keine Host-Wrapper-Funktion, keine Nim-FFI-Deklaration) -
 *      toter Code. Entfernt: Backward-Pass ist fuer diese Integrationsstufe
 *      bewusst nicht im Scope (siehe Roadmap - erst nach/mit dem FP16-Umbau
 *      sinnvoll vollstaendig nachzuziehen, sonst doppelte Kernel-Pflege ohne
 *      aktuellen Nutzen).
 *   6. cuda_attentionForward() allozierte/befreite d_q/d_out/d_scores bei
 *      JEDEM Aufruf per cudaMalloc()/cudaFree() - bei der eigentlich
 *      vorgesehenen Nutzung (ein Aufruf pro Token waehrend der Generierung,
 *      hunderte/tausende Male) waere das genau das Alloc/Free-pro-Aufruf-
 *      Problem, das im OpenCL-Pendant schon einmal gefixt wurde (siehe "FIX
 *      4: Robust Buffer Resize" in cortex_opencl.nim). Ersetzt durch
 *      persistente, bei Bedarf wachsende Ephemeral-Buffer nach demselben
 *      Muster.
 *   7. cuda_matMulBatched()/cuda_adamStep() prueften die Rueckgabe von
 *      cuda_get_buffer() nicht auf NULL (Buffer-Groessen-Mismatch oder
 *      Registry-Overflow) - waere ein Nullpointer-Crash statt eines
 *      kontrollierten Abbruchs gewesen. Guards ergaenzt (Muster aus
 *      cuda_attentionForward()/ocl_attentionForward() uebernommen).
 *
 * AUDIT-FIX (2026-08-08): adamUpdateKernel/cuda_adamStep() fehlte der
 * `wd`-Parameter (Weight-Decay, AdamW-Style), den ocl_adamStep seit dem
 * "Trainings-Absicherung: Nachruest-Paket" traegt (siehe docs/ROADMAP.md).
 * Der CUDA-Port stammt von vor diesem Nachruest-Paket und wurde beim
 * ersten Audit nicht nachgezogen, weil es damals noch nicht existierte -
 * durch den Funktionstest (skripte/test_cuda_functional.nim, Update
 * 2026-08-08) aufgefallen. Signatur und Kernel-Logik 1:1 aus ocl_adamStep
 * uebernommen: `if (wd > 0.0f) { w[i] -= lr * wd * w[i]; }`, nach dem
 * eigentlichen Adam-Update, vor dem Gradient-Reset. wd=0.0 ist weiterhin
 * ein exaktes No-Op (identisch zum bisherigen Verhalten).
 * ==============================================================================
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstring>
#include <iostream>
#include <cmath>

cublasHandle_t cublas_handle;
bool cuda_initialized = false;

// Simples VRAM-Tracking (Analog zur OpenCL Registry)
#define MAX_CUDA_BUFFERS 2048
struct CudaBufMap {
    void* h_ptr;
    void* d_buf;
    size_t bytes;
};

CudaBufMap cuda_buf_map[MAX_CUDA_BUFFERS];
int cuda_buf_count = 0;

// AUDIT-FIX (Punkt 6): persistente Ephemeral-Scratch-Buffer fuer
// cuda_attentionForward(), analog ocl_eph_v/ocl_eph_out/ocl_eph_scores -
// werden nur bei Bedarf (neu) alloziert, sonst wiederverwendet.
static float* cuda_eph_q = nullptr;      size_t cuda_eph_q_size = 0;
static float* cuda_eph_out = nullptr;    size_t cuda_eph_out_size = 0;
static float* cuda_eph_scores = nullptr; size_t cuda_eph_scores_size = 0;

// Update 2026-08-08 (Backward-Pass): weitere Ephemeral-Scratch-Buffer, analog
// ocl_eph_dq/ocl_eph_dw (Attention-Backward) und ocl_eph_bm_r/ocl_eph_bm_c
// (Batched-Matmul Vorwaerts+Rueckwaerts, in cortex_opencl.nim ueber alle drei
// Operationen hinweg geteilt). cuda_matMulBatched() (Forward) bleibt bewusst
// unangetastet bei seinen eigenen, per Host-Pointer gecachten in/out-Buffern
// (siehe dort) - die beiden neuen bm_r/bm_c-Puffer sind ausschliesslich fuer
// die hier neu hinzukommenden Backward-Funktionen, um das bereits verifizierte
// Forward-Verhalten nicht anzufassen.
static float* cuda_eph_dq = nullptr;     size_t cuda_eph_dq_size = 0;
static float* cuda_eph_dw = nullptr;     size_t cuda_eph_dw_size = 0;
static float* cuda_eph_bm_r = nullptr;   size_t cuda_eph_bm_r_size = 0;  // batch x rows
static float* cuda_eph_bm_c = nullptr;   size_t cuda_eph_bm_c_size = 0;  // batch x cols

// Update 2026-08-08 (Einzelvektor-Ops): analog ocl_eph_bt_a/ocl_eph_bt_b
// ("V8.5 BACKWARD OFFLOAD" in cortex_opencl.nim) - bewusst NICHT mit
// cuda_eph_bm_r/cuda_eph_bm_c (V8.6, batched) geteilt, obwohl beide Paare
// aehnlich (row-/spaltengroesse) sind: OpenCL haelt diese beiden Familien
// ebenfalls getrennt, hier 1:1 uebernommen statt eigenmaechtig zusammenzulegen.
static float* cuda_eph_bt_a = nullptr;   size_t cuda_eph_bt_a_size = 0;  // rows (d_out)
static float* cuda_eph_bt_b = nullptr;   size_t cuda_eph_bt_b_size = 0;  // cols (d_in / i_v)

// Update 2026-08-08 (gebuendelte Attention): analog ocl_eph_att_q/-out/-datt/
// -dq/-scores/-ds. Eigene Scratch-Familie, nicht mit cuda_eph_q/cuda_eph_out/
// cuda_eph_scores (Einzelposition) geteilt, obwohl aehnlich benannt - andere
// Groessen (act*dim bzw. act*numHeads*ctx statt dim bzw. numHeads*ctx) und
// andere Lebensdauer-Anforderungen; OpenCL haelt sie ebenfalls getrennt.
static float* cuda_eph_att_q = nullptr;      size_t cuda_eph_att_q_size = 0;
static float* cuda_eph_att_out = nullptr;    size_t cuda_eph_att_out_size = 0;
static float* cuda_eph_att_datt = nullptr;   size_t cuda_eph_att_datt_size = 0;
static float* cuda_eph_att_dq = nullptr;     size_t cuda_eph_att_dq_size = 0;
static float* cuda_eph_att_scores = nullptr; size_t cuda_eph_att_scores_size = 0;
static float* cuda_eph_att_ds = nullptr;     size_t cuda_eph_att_ds_size = 0;

// AUDIT-FIX-Analogie: siehe ocl_attentionForwardBatched/-BackwardBatched in
// cortex_opencl.nim - ein einzelner Dispatch ueber grosse Positionsmengen
// riskiert einen GPU-Watchdog-Timeout (dort empirisch auf AMDGPU aufgetreten).
// Selbe Konstante wie im OpenCL-Original, aus Konsistenzgruenden auch fuer
// NVIDIA/CUDA uebernommen (TDR-Mechanismus existiert dort ebenfalls).
#define ATT_POS_CHUNK 256

extern "C" {

void cuda_release_buffer(void* h_ptr) {
    if (!h_ptr) return;
    for(int i = 0; i < cuda_buf_count; i++) {
        if(cuda_buf_map[i].h_ptr == h_ptr) {
            cudaFree(cuda_buf_map[i].d_buf);
            for(int j = i; j < cuda_buf_count - 1; j++) {
                cuda_buf_map[j] = cuda_buf_map[j+1];
            }
            cuda_buf_count--;
            return;
        }
    }
}

void* cuda_get_buffer(void* h_ptr, size_t bytes) {
    if (!h_ptr || bytes == 0) return nullptr;

    for(int i = 0; i < cuda_buf_count; i++) {
        if(cuda_buf_map[i].h_ptr == h_ptr) {
            if (cuda_buf_map[i].bytes != bytes) {
                std::cerr << "🚨 [SYMBIO CUDA] Buffer Size Mismatch!" << std::endl;
                return nullptr;
            }
            // AUDIT: Cache-Hit liefert den bisherigen Device-Buffer OHNE
            // erneuten Upload zurueck - identisch zu ocl_get_buffer(). Wer
            // die Host-Seite ausserhalb dieser Bibliothek veraendert (z.B.
            // Net2Net-Upscale, frisches Laden von Platte), MUSS danach
            // cuda_sync_to_device() aufrufen, sonst rechnet die GPU mit
            // veralteten Daten weiter.
            return cuda_buf_map[i].d_buf;
        }
    }

    if(cuda_buf_count >= MAX_CUDA_BUFFERS) {
        std::cerr << "🚨 [SYMBIO CUDA] VRAM Registry Overflow!" << std::endl;
        return nullptr;
    }

    void* d_buf;
    cudaError_t err = cudaMalloc(&d_buf, bytes);
    if(err == cudaSuccess) {
        cudaMemcpy(d_buf, h_ptr, bytes, cudaMemcpyHostToDevice);
        cuda_buf_map[cuda_buf_count].h_ptr = h_ptr;
        cuda_buf_map[cuda_buf_count].d_buf = d_buf;
        cuda_buf_map[cuda_buf_count].bytes = bytes;
        cuda_buf_count++;
        return d_buf;
    }
    return nullptr;
}

void cuda_sync_to_host(void* h_ptr) {
    if (!cuda_initialized || !h_ptr) return;
    for(int i = 0; i < cuda_buf_count; i++) {
        if(cuda_buf_map[i].h_ptr == h_ptr) {
            cudaMemcpy(h_ptr, cuda_buf_map[i].d_buf, cuda_buf_map[i].bytes, cudaMemcpyDeviceToHost);
            return;
        }
    }
}

// AUDIT-FIX (Punkt 2): Gegenstueck zu ocl_sync_to_device() - erzwingt einen
// Re-Upload eines schon gecachten Device-Buffers, falls die Host-Seite
// zwischenzeitlich ausserhalb dieser Bibliothek veraendert wurde. Legt den
// Buffer an, falls er noch nicht existiert (dann ist es ohnehin ein
// frischer Upload via cuda_get_buffer()).
void cuda_sync_to_device(void* h_ptr, size_t bytes) {
    if (!cuda_initialized || !h_ptr || bytes == 0) return;
    void* d_buf = cuda_get_buffer(h_ptr, bytes);
    if (d_buf) {
        cudaMemcpy(d_buf, h_ptr, bytes, cudaMemcpyHostToDevice);
    }
}

// Update 2026-08-08: Port von ocl_zero_buffer(). cudaMemset() statt
// clEnqueueFillBuffer() - CUDA hat dafuer keine eigene Kernel-Dispatch noetig.
// cudaDeviceSynchronize() danach, um exakt das blockierende Verhalten von
// OpenCLs clFinish() nachzubilden (Aufrufer darf sich auf "fertig, wenn diese
// Funktion zurueckkehrt" verlassen).
void cuda_zero_buffer(void* h_ptr, size_t bytes) {
    if (!cuda_initialized) return;
    void* d_buf = cuda_get_buffer(h_ptr, bytes);
    if (d_buf) {
        cudaMemset(d_buf, 0, bytes);
        cudaDeviceSynchronize();
    }
}

__global__ void adamUpdateKernel(float* w, float* m, float* v, float* g,
                                 float lr, float b1, float b2, float b1t, float b2t, float scale, int size, int is_sparse,
                                 int cols, int bound_r, int bound_c, float boost, float wd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float grad = g[i] / scale;
        if (is_sparse == 1 && fabsf(grad) < 1e-8f) return;
        if (isnan(grad) || grad > 9999.0f || grad < -9999.0f) {
            g[i] = 0.0f;
            return;
        }
        // AUDIT-FIX (Punkt 3): LR-Boost fuer frisch Net2Net-expandierte
        // Bereiche - identische Logik wie im OpenCL-Kernel/CPU-Pfad
        // (elemBoost() in cortex.nim). cols/bound_r/bound_c beschreiben, ab
        // welcher Zeile/Spalte eine Matrix "neu" ist.
        int row = i / cols;
        int col = i % cols;
        float eff_lr = (row >= bound_r || col >= bound_c) ? (lr * boost) : lr;
        m[i] = b1 * m[i] + (1.0f - b1) * grad;
        v[i] = b2 * v[i] + (1.0f - b2) * grad * grad;
        float m_hat = m[i] / b1t;
        float v_hat = v[i] / b2t;
        w[i] -= eff_lr * m_hat / (sqrtf(v_hat) + 1e-8f);
        // AUDIT-FIX (2026-08-08): Weight-Decay (AdamW-Style, dekorreliert) -
        // nachgezogen aus ocl_adamStep. wd=0.0 (Default) ist ein exaktes
        // No-Op. Bewusst mit lr (nicht eff_lr) skaliert, identisch zum
        // OpenCL-Kernel.
        if (wd > 0.0f) { w[i] -= lr * wd * w[i]; }
        g[i] = 0.0f; // Reset Gradient
    }
}

__global__ void attentionForwardKernel(const float* q, const float* k_cache, const float* v_cache,
                                       float* att_out, float* scores, int pos, int numHeads, int headSize, int dim, int ctx) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= numHeads) return;

    int h_off = h * headSize;
    float scale = 1.0f / sqrtf((float)headSize);
    float max_s = -9999.0f;

    for (int t = 0; t <= pos; t++) {
        float sc = 0.0f;
        int k_off = t * dim + h_off;
        for (int d = 0; d < headSize; d++) {
            sc += q[h_off + d] * k_cache[k_off + d];
        }
        sc *= scale;
        scores[h * ctx + t] = sc;
        if (sc > max_s) max_s = sc;
    }

    float sum_e = 0.0f;
    for (int t = 0; t <= pos; t++) {
        float e = expf(scores[h * ctx + t] - max_s);
        scores[h * ctx + t] = e;
        sum_e += e;
    }

    for (int d = 0; d < headSize; d++) att_out[h_off + d] = 0.0f;
    for (int t = 0; t <= pos; t++) {
        float w = scores[h * ctx + t] / sum_e;
        int v_off = t * dim + h_off;
        for (int d = 0; d < headSize; d++) {
            att_out[h_off + d] += v_cache[v_off + d] * w;
        }
    }
}

// Update 2026-08-08 (Backward-Pass): 1:1-Uebernahme aus attentionBackward
// (cortex_opencl.nim) - gleiche Softmax-Rueckrechnung (Jacobian einer
// Softmax-gewichteten Summe), gleiche Zwischenspeicherung von w in scores[]
// und dw in d_w[] fuer die zweite Ableitungs-Passe. d_q wird pro Aufruf
// komplett ueberschrieben (=0 am Anfang, dann akkumuliert), d_k_cache/
// d_v_cache werden NUR akkumuliert (+=), NIE genullt - identisch zum
// OpenCL-Kernel muss der Aufrufer (bzw. spaeter cortex.nim) sie vor einer
// frischen Rueckwaerts-Passe selbst auf 0 setzen (z.B. per memset vor dem
// ersten Aufruf einer Sequenz).
__global__ void attentionBackwardKernel(const float* q, const float* k_cache, const float* v_cache,
                                        const float* d_att_out, float* d_q, float* d_k_cache, float* d_v_cache,
                                        float* scores, float* d_w, int pos, int numHeads, int headSize, int dim, int ctx) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= numHeads) return;

    int h_off = h * headSize;
    float scale = 1.0f / sqrtf((float)headSize);
    float max_s = -9999.0f;

    for (int d = 0; d < headSize; d++) d_q[h_off + d] = 0.0f;

    for (int t = 0; t <= pos; t++) {
        float sc = 0.0f;
        int k_off = t * dim + h_off;
        for (int d = 0; d < headSize; d++) sc += q[h_off + d] * k_cache[k_off + d];
        sc *= scale;
        scores[h * ctx + t] = sc;
        if (sc > max_s) max_s = sc;
    }

    float sum_e = 0.0f;
    for (int t = 0; t <= pos; t++) {
        float e = expf(scores[h * ctx + t] - max_s);
        scores[h * ctx + t] = e;
        sum_e += e;
    }

    float sum_w_dw = 0.0f;
    for (int t = 0; t <= pos; t++) {
        float w = scores[h * ctx + t] / sum_e;
        scores[h * ctx + t] = w;
        int v_off = t * dim + h_off;
        float dw = 0.0f;
        for (int d = 0; d < headSize; d++) {
            dw += d_att_out[h_off + d] * v_cache[v_off + d];
            d_v_cache[v_off + d] += d_att_out[h_off + d] * w;
        }
        d_w[h * ctx + t] = dw;
        sum_w_dw += w * dw;
    }

    for (int t = 0; t <= pos; t++) {
        float w = scores[h * ctx + t];
        float ds = w * (d_w[h * ctx + t] - sum_w_dw) * scale;
        int k_off = t * dim + h_off;
        for (int d = 0; d < headSize; d++) {
            d_q[h_off + d] += ds * k_cache[k_off + d];
            d_k_cache[k_off + d] += ds * q[h_off + d];
        }
    }
}

// Update 2026-08-08 (Einzelvektor-Ops): 1:1 aus den gleichnamigen OpenCL-
// Kernel-Strings in cortex_opencl.nim uebernommen (matVecMul, matVecMulT,
// gradAccum). Der manuelle 4er-Unroll aus dem OpenCL-Kernel wird hier NICHT
// nachgebaut - reine OpenCL-Compiler-Mikrooptimierung ohne Bedeutung fuer die
// Korrektheit, nvcc entrollt einfache Schleifen wie diese ohnehin selbst.
__global__ void matVecMulKernel(const float* M, const float* v, float* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += M[(size_t)r * cols + c] * v[c];
        }
        out[r] = sum;
    }
}

__global__ void matVecMulTKernel(const float* M, const float* d_out, float* d_in, int rows, int cols, int accumulate) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols) {
        float sum = 0.0f;
        size_t idx = c;
        for (int r = 0; r < rows; r++) {
            sum += M[idx] * d_out[r];
            idx += (size_t)cols;
        }
        d_in[c] = (accumulate == 1) ? (d_in[c] + sum) : sum;
    }
}

__global__ void gradAccumKernel(float* gm, const float* d_out, const float* i_v, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < cols && r < rows) {
        gm[(size_t)r * cols + c] += d_out[r] * i_v[c];
    }
}

// Update 2026-08-08 (gebuendelte Attention): 1:1 aus attentionForwardBatched/
// attentionBackwardBatchedQuery/attentionBackwardBatchedKey (cortex_opencl.nim)
// uebernommen. OpenCL parallelisiert ueber ein 2D-NDRange (h, p) mit einem
// Chunk-Offset auf der zweiten Achse (siehe ocl_attentionForwardBatched-
// Kommentar zum AMDGPU-Watchdog) - CUDA kennt kein natives Launch-Offset,
// daher hier stattdessen ein FLACHER Thread-Index ueber (h, lokaler Index
// innerhalb des Chunks), p0 als zusaetzlicher Kernel-Parameter fuer den
// tatsaechlichen Positions-Offset. Mathematisch identisch, nur andere
// Grid/Thread-Aufteilung.
__global__ void attentionForwardBatchedKernel(const float* q_all, const float* k_cache, const float* v_cache,
                                              float* att_out_all, float* scores,
                                              int p0, int chunkLen, int numHeads, int headSize, int dim, int ctx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numHeads * chunkLen) return;
    int h = idx / chunkLen;
    int p = p0 + (idx % chunkLen);

    int h_off = h * headSize;
    const float* q = q_all + (size_t)p * (size_t)dim;
    float* att_out = att_out_all + (size_t)p * (size_t)dim;
    float* my_scores = scores + ((size_t)p * (size_t)numHeads + h) * (size_t)ctx;
    float scale = 1.0f / sqrtf((float)headSize);
    float max_s = -9999.0f;

    for (int t = 0; t <= p; t++) {
        float sc = 0.0f; int k_off = t * dim + h_off;
        for (int d = 0; d < headSize; d++) sc += q[h_off + d] * k_cache[k_off + d];
        sc *= scale; my_scores[t] = sc;
        if (sc > max_s) max_s = sc;
    }
    float sum_e = 0.0f;
    for (int t = 0; t <= p; t++) { float e = expf(my_scores[t] - max_s); my_scores[t] = e; sum_e += e; }
    for (int d = 0; d < headSize; d++) att_out[h_off + d] = 0.0f;
    for (int t = 0; t <= p; t++) {
        float w = my_scores[t] / sum_e;
        int v_off = t * dim + h_off;
        for (int d = 0; d < headSize; d++) att_out[h_off + d] += v_cache[v_off + d] * w;
    }
}

// Stufe A (query-parallel): d_q, weights, ds - siehe attentionBackwardBatchedQuery
// (cortex_opencl.nim) fuer die Herleitung, 1:1 uebernommen.
__global__ void attentionBackwardBatchedQueryKernel(const float* q_all, const float* k_cache, const float* v_cache,
                                                     const float* d_att_out_all, float* d_q_all,
                                                     float* weights, float* ds,
                                                     int p0, int chunkLen, int numHeads, int headSize, int dim, int ctx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numHeads * chunkLen) return;
    int h = idx / chunkLen;
    int p = p0 + (idx % chunkLen);

    int h_off = h * headSize;
    const float* q = q_all + (size_t)p * (size_t)dim;
    const float* d_att_out = d_att_out_all + (size_t)p * (size_t)dim;
    float* d_q = d_q_all + (size_t)p * (size_t)dim;
    float* my_w  = weights + ((size_t)p * (size_t)numHeads + h) * (size_t)ctx;
    float* my_ds = ds      + ((size_t)p * (size_t)numHeads + h) * (size_t)ctx;
    float scale = 1.0f / sqrtf((float)headSize);
    float max_s = -9999.0f;

    for (int d = 0; d < headSize; d++) d_q[h_off + d] = 0.0f;
    for (int t = 0; t <= p; t++) {
        float sc = 0.0f; int k_off = t * dim + h_off;
        for (int d = 0; d < headSize; d++) sc += q[h_off + d] * k_cache[k_off + d];
        sc *= scale; my_w[t] = sc;
        if (sc > max_s) max_s = sc;
    }
    float sum_e = 0.0f;
    for (int t = 0; t <= p; t++) { float e = expf(my_w[t] - max_s); my_w[t] = e; sum_e += e; }
    float sum_w_dw = 0.0f;
    for (int t = 0; t <= p; t++) {
        float w = my_w[t] / sum_e; my_w[t] = w;
        int v_off = t * dim + h_off; float dw = 0.0f;
        for (int d = 0; d < headSize; d++) dw += d_att_out[h_off + d] * v_cache[v_off + d];
        my_ds[t] = dw; sum_w_dw += w * dw;
    }
    for (int t = 0; t <= p; t++) {
        float w = my_w[t];
        float dw = my_ds[t];
        float ds_val = w * (dw - sum_w_dw) * scale;
        my_ds[t] = ds_val;
        int k_off = t * dim + h_off;
        for (int d = 0; d < headSize; d++) d_q[h_off + d] += ds_val * k_cache[k_off + d];
    }
}

// Stufe B (key-parallel): d_k_cache, d_v_cache - siehe attentionBackwardBatchedKey
// (cortex_opencl.nim) fuer die Race-Analyse (warum Key- statt Query-Achse).
// Nullt sein eigenes (h,t)-Segment selbst - kein Pre-Zero durch den Aufrufer
// noetig (anders als bei der Einzelpositions-Variante).
__global__ void attentionBackwardBatchedKeyKernel(const float* q_all, const float* d_att_out_all,
                                                   const float* weights, const float* ds,
                                                   float* d_k_cache, float* d_v_cache,
                                                   int t0, int chunkLen, int act, int numHeads, int headSize, int dim, int ctx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numHeads * chunkLen) return;
    int h = idx / chunkLen;
    int t = t0 + (idx % chunkLen);

    int h_off = h * headSize;
    int kv_off = t * dim + h_off;
    for (int d = 0; d < headSize; d++) { d_k_cache[kv_off + d] = 0.0f; d_v_cache[kv_off + d] = 0.0f; }
    for (int p = t; p < act; p++) {
        size_t idx2 = ((size_t)p * (size_t)numHeads + h) * (size_t)ctx + t;
        float w  = weights[idx2];
        float dsv = ds[idx2];
        const float* d_att_out = d_att_out_all + (size_t)p * (size_t)dim;
        const float* q = q_all + (size_t)p * (size_t)dim;
        for (int d = 0; d < headSize; d++) {
            d_v_cache[kv_off + d] += d_att_out[h_off + d] * w;
            d_k_cache[kv_off + d] += dsv * q[h_off + d];
        }
    }
}

int init_cuda_engine() {
    if (cuda_initialized) return 1;

    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) return 0;

    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) return 0;

    // Aktiviert TF32 auf Tensor Cores. Nim und PHP schicken normales FP32,
    // die GPU komprimiert es intern rasant auf Tensor-Geschwindigkeit.
    // AUDIT-FIX (Punkt 4): dieser Modus wurde vorher gesetzt, aber nie
    // tatsaechlich von einem cuBLAS-Aufruf genutzt - siehe cuda_matMulBatched().
    cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cerr << "⚡ [SYMBIO VRAM] CUDA Engine eingerastet: [" << prop.name << "] (TF32 Tensor Cores aktiv)" << std::endl;

    cuda_initialized = true;
    return 1;
}

void shutdown_cuda_engine() {
    for(int i=0; i < cuda_buf_count; i++) {
        cudaFree(cuda_buf_map[i].d_buf);
    }
    cuda_buf_count = 0;

    if (cuda_eph_q) { cudaFree(cuda_eph_q); cuda_eph_q = nullptr; cuda_eph_q_size = 0; }
    if (cuda_eph_out) { cudaFree(cuda_eph_out); cuda_eph_out = nullptr; cuda_eph_out_size = 0; }
    if (cuda_eph_scores) { cudaFree(cuda_eph_scores); cuda_eph_scores = nullptr; cuda_eph_scores_size = 0; }
    // Update 2026-08-08 (Backward-Pass): Gegenstuecke zu den obigen drei -
    // ohne diese Ergaenzung waeren das VRAM-Lecks bei jedem shutdown/init-Zyklus.
    if (cuda_eph_dq) { cudaFree(cuda_eph_dq); cuda_eph_dq = nullptr; cuda_eph_dq_size = 0; }
    if (cuda_eph_dw) { cudaFree(cuda_eph_dw); cuda_eph_dw = nullptr; cuda_eph_dw_size = 0; }
    if (cuda_eph_bm_r) { cudaFree(cuda_eph_bm_r); cuda_eph_bm_r = nullptr; cuda_eph_bm_r_size = 0; }
    if (cuda_eph_bm_c) { cudaFree(cuda_eph_bm_c); cuda_eph_bm_c = nullptr; cuda_eph_bm_c_size = 0; }
    if (cuda_eph_bt_a) { cudaFree(cuda_eph_bt_a); cuda_eph_bt_a = nullptr; cuda_eph_bt_a_size = 0; }
    if (cuda_eph_bt_b) { cudaFree(cuda_eph_bt_b); cuda_eph_bt_b = nullptr; cuda_eph_bt_b_size = 0; }
    if (cuda_eph_att_q) { cudaFree(cuda_eph_att_q); cuda_eph_att_q = nullptr; cuda_eph_att_q_size = 0; }
    if (cuda_eph_att_out) { cudaFree(cuda_eph_att_out); cuda_eph_att_out = nullptr; cuda_eph_att_out_size = 0; }
    if (cuda_eph_att_datt) { cudaFree(cuda_eph_att_datt); cuda_eph_att_datt = nullptr; cuda_eph_att_datt_size = 0; }
    if (cuda_eph_att_dq) { cudaFree(cuda_eph_att_dq); cuda_eph_att_dq = nullptr; cuda_eph_att_dq_size = 0; }
    if (cuda_eph_att_scores) { cudaFree(cuda_eph_att_scores); cuda_eph_att_scores = nullptr; cuda_eph_att_scores_size = 0; }
    if (cuda_eph_att_ds) { cudaFree(cuda_eph_att_ds); cuda_eph_att_ds = nullptr; cuda_eph_att_ds_size = 0; }

    if (cublas_handle) cublasDestroy(cublas_handle);
    cuda_initialized = false;
    std::cerr << "⚡ [SYMBIO VRAM] CUDA-Ressourcen freigegeben." << std::endl;
}

void cuda_adamStep(float* h_w, float* h_m, float* h_v, float* h_g, float lr, float b1, float b2, float b1t, float b2t, float scale, int size, int is_sparse,
                   int cols, int bound_r, int bound_c, float boost, float wd) {
    if (!cuda_initialized) return;

    size_t bytes = (size_t)size * sizeof(float);
    float* d_w = (float*)cuda_get_buffer(h_w, bytes);
    float* d_m = (float*)cuda_get_buffer(h_m, bytes);
    float* d_v = (float*)cuda_get_buffer(h_v, bytes);
    float* d_g = (float*)cuda_get_buffer(h_g, bytes);
    // AUDIT-FIX (Punkt 7): ohne diesen Guard waere ein Registry-Overflow
    // oder Size-Mismatch ein Nullpointer-Crash statt eines kontrollierten
    // Abbruchs gewesen.
    if (!d_w || !d_m || !d_v || !d_g) return;

    cudaMemcpy(d_g, h_g, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    adamUpdateKernel<<<blocksPerGrid, threadsPerBlock>>>(d_w, d_m, d_v, d_g, lr, b1, b2, b1t, b2t, scale, size, is_sparse, cols, bound_r, bound_c, boost, wd);

    cudaMemcpy(h_w, d_w, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_m, d_m, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost);

    memset(h_g, 0, bytes); // Reset Host-Gradient
}

void cuda_matMulBatched(const float* h_M, const float* h_in, float* h_out, int rows, int cols, int batch_size, int max_batch) {
    if (!cuda_initialized) return;

    size_t size_M = (size_t)rows * cols * sizeof(float);
    size_t active_in = (size_t)batch_size * cols * sizeof(float);
    size_t active_out = (size_t)batch_size * rows * sizeof(float);

    float* d_M = (float*)cuda_get_buffer((void*)h_M, size_M);
    float* d_in = (float*)cuda_get_buffer((void*)h_in, (size_t)max_batch * cols * sizeof(float));
    float* d_out = (float*)cuda_get_buffer((void*)h_out, (size_t)max_batch * rows * sizeof(float));
    if (!d_M || !d_in || !d_out) return;

    cudaMemcpy(d_in, h_in, active_in, cudaMemcpyHostToDevice);

    // AUDIT-FIX (Punkt 4): echtes cuBLAS-GEMM statt Handkernel, damit der
    // beim Init gesetzte TF32-Tensor-Core-Modus ueberhaupt greift.
    //
    // Gesucht (Row-Major-Semantik, identisch zum alten Handkernel):
    //   out_seq[b][r] = sum_c M[r][c] * in_seq[b][c]        (b < batch_size)
    //   M:       rows x cols
    //   in_seq:  batch x cols
    //   out_seq: batch x rows
    //
    // cuBLAS ist Column-Major. Trick: dieselbe Speicherregion, die Row-Major
    // als (p x q) gelesen wird, ist Column-Major identisch zu (q x p)^T.
    // Damit gilt (Herleitung siehe Commit-Kommentar):
    //   out_seq_cm (rows x batch) = M_cm^T (rows x cols) @ in_seq_cm (cols x batch)
    // -> transA = T (M_cm liegt als cols x rows vor, wird transponiert),
    //    transB = N, m=rows, n=batch, k=cols, lda=cols, ldb=cols, ldc=rows.
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                rows, batch_size, cols,
                &alpha,
                d_M, cols,
                d_in, cols,
                &beta,
                d_out, rows);

    cudaMemcpy(h_out, d_out, active_out, cudaMemcpyDeviceToHost);
}

void cuda_attentionForward(const float* h_q, const float* h_k_cache, const float* h_v_cache, float* h_out, int pos, int numHeads, int headSize, int dim, int ctx) {
    if (!cuda_initialized || pos >= ctx) return;

    size_t size_cache = (size_t)ctx * dim * sizeof(float);
    float* d_k = (float*)cuda_get_buffer((void*)h_k_cache, size_cache);
    float* d_v = (float*)cuda_get_buffer((void*)h_v_cache, size_cache);
    if (!d_k || !d_v) return;

    size_t vec_bytes = (size_t)dim * sizeof(float);
    cudaMemcpy(d_k + ((size_t)pos * dim), h_k_cache + ((size_t)pos * dim), vec_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v + ((size_t)pos * dim), h_v_cache + ((size_t)pos * dim), vec_bytes, cudaMemcpyHostToDevice);

    // AUDIT-FIX (Punkt 6): persistente, wachsende Scratch-Buffer statt
    // cudaMalloc()/cudaFree() bei jedem einzelnen Aufruf (analog
    // ocl_eph_v/ocl_eph_out/ocl_eph_scores in cortex_opencl.nim, "FIX 4:
    // Robust Buffer Resize"). Bei der vorgesehenen Nutzung - ein Aufruf pro
    // generiertem Token - waere die alte Variante ein Alloc/Free pro Token.
    size_t size_scores = (size_t)numHeads * ctx * sizeof(float);
    if (cuda_eph_q_size < vec_bytes) {
        if (cuda_eph_q) cudaFree(cuda_eph_q);
        cuda_eph_q_size = (cudaMalloc(&cuda_eph_q, vec_bytes) == cudaSuccess) ? vec_bytes : 0;
    }
    if (cuda_eph_out_size < vec_bytes) {
        if (cuda_eph_out) cudaFree(cuda_eph_out);
        cuda_eph_out_size = (cudaMalloc(&cuda_eph_out, vec_bytes) == cudaSuccess) ? vec_bytes : 0;
    }
    if (cuda_eph_scores_size < size_scores) {
        if (cuda_eph_scores) cudaFree(cuda_eph_scores);
        cuda_eph_scores_size = (cudaMalloc(&cuda_eph_scores, size_scores) == cudaSuccess) ? size_scores : 0;
    }
    if (!cuda_eph_q || !cuda_eph_out || !cuda_eph_scores) return;

    cudaMemcpy(cuda_eph_q, h_q, vec_bytes, cudaMemcpyHostToDevice);

    int threads = 64;
    int blocks = (numHeads + threads - 1) / threads;
    attentionForwardKernel<<<blocks, threads>>>(cuda_eph_q, d_k, d_v, cuda_eph_out, cuda_eph_scores, pos, numHeads, headSize, dim, ctx);

    cudaMemcpy(h_out, cuda_eph_out, vec_bytes, cudaMemcpyDeviceToHost);
}

// ==============================================================================
// Update 2026-08-08: BACKWARD-PASS (Matmul + Attention)
// Ports von ocl_attentionBackward, ocl_matMulTBatched, ocl_gradAccumBatched
// (cortex_opencl.nim) - siehe docs/ROADMAP.md fuer die Herleitung der
// cuBLAS-Parametrisierung der beiden Matmul-Backward-Funktionen. Attention-
// Backward bleibt bewusst die EINZELPOSITIONS-Variante (analog zum bereits
// vorhandenen cuda_attentionForward) - die gebuendelte
// attentionBackwardBatched-Variante (ocl_attentionBackwardBatched, zweistufig
// query-/key-parallel) ist weiterhin nicht Teil dieses Ports, siehe
// "keine batched Attention" im bestehenden Scope-Abschnitt oben.
// ==============================================================================

void cuda_attentionBackward(const float* h_q, const float* h_k_cache, const float* h_v_cache, const float* h_d_att_out,
                            float* h_d_q, float* h_d_k_cache, float* h_d_v_cache,
                            int pos, int numHeads, int headSize, int dim, int ctx) {
    if (!cuda_initialized || pos >= ctx) return;

    size_t size_cache = (size_t)ctx * dim * sizeof(float);
    float* d_k = (float*)cuda_get_buffer((void*)h_k_cache, size_cache);
    float* d_v = (float*)cuda_get_buffer((void*)h_v_cache, size_cache);
    // AUDIT: d_dk/d_dv sind eigene, per Host-Pointer gecachte Buffer (nicht
    // dieselben wie d_k/d_v) - der Kernel akkumuliert NUR in sie hinein,
    // liest sie aber nie. Erster Touch pro Pointer via cuda_get_buffer laedt
    // den aktuellen Host-Inhalt hoch (i.d.R. vom Aufrufer vorher genullt).
    float* d_dk = (float*)cuda_get_buffer((void*)h_d_k_cache, size_cache);
    float* d_dv = (float*)cuda_get_buffer((void*)h_d_v_cache, size_cache);
    if (!d_k || !d_v || !d_dk || !d_dv) return;

    size_t vec_bytes = (size_t)dim * sizeof(float);
    size_t size_scores = (size_t)numHeads * ctx * sizeof(float);

    // Ephemere Scratch-Buffer analog cuda_attentionForward (AUDIT-FIX Punkt 6).
    // cuda_eph_q/cuda_eph_out/cuda_eph_scores werden mit der Forward-Funktion
    // geteilt (gleiche Groessen: dim bzw. numHeads*ctx) - hier fuer q bzw.
    // d_att_out bzw. scores/weights genutzt. cuda_eph_dq/cuda_eph_dw sind
    // backward-exklusiv (analog ocl_eph_dq/ocl_eph_dw).
    if (cuda_eph_q_size < vec_bytes) {
        if (cuda_eph_q) cudaFree(cuda_eph_q);
        cuda_eph_q_size = (cudaMalloc(&cuda_eph_q, vec_bytes) == cudaSuccess) ? vec_bytes : 0;
    }
    if (cuda_eph_out_size < vec_bytes) {
        if (cuda_eph_out) cudaFree(cuda_eph_out);
        cuda_eph_out_size = (cudaMalloc(&cuda_eph_out, vec_bytes) == cudaSuccess) ? vec_bytes : 0;
    }
    if (cuda_eph_dq_size < vec_bytes) {
        if (cuda_eph_dq) cudaFree(cuda_eph_dq);
        cuda_eph_dq_size = (cudaMalloc(&cuda_eph_dq, vec_bytes) == cudaSuccess) ? vec_bytes : 0;
    }
    if (cuda_eph_scores_size < size_scores) {
        if (cuda_eph_scores) cudaFree(cuda_eph_scores);
        cuda_eph_scores_size = (cudaMalloc(&cuda_eph_scores, size_scores) == cudaSuccess) ? size_scores : 0;
    }
    if (cuda_eph_dw_size < size_scores) {
        if (cuda_eph_dw) cudaFree(cuda_eph_dw);
        cuda_eph_dw_size = (cudaMalloc(&cuda_eph_dw, size_scores) == cudaSuccess) ? size_scores : 0;
    }
    if (!cuda_eph_q || !cuda_eph_out || !cuda_eph_dq || !cuda_eph_scores || !cuda_eph_dw) return;

    cudaMemcpy(cuda_eph_q, h_q, vec_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_eph_out, h_d_att_out, vec_bytes, cudaMemcpyHostToDevice);

    int threads = 64;
    int blocks = (numHeads + threads - 1) / threads;
    attentionBackwardKernel<<<blocks, threads>>>(cuda_eph_q, d_k, d_v, cuda_eph_out, cuda_eph_dq, d_dk, d_dv,
                                                   cuda_eph_scores, cuda_eph_dw, pos, numHeads, headSize, dim, ctx);

    cudaMemcpy(h_d_q, cuda_eph_dq, vec_bytes, cudaMemcpyDeviceToHost);
    // AUDIT: wie im OpenCL-Original wird nur die Position `pos` zurueckkopiert
    // - d_dk/d_dv akkumulieren am Device weiter ueber t=0..pos JEDES Aufrufs,
    // die frueheren Positionen werden erst beim jeweils eigenen `pos`-Durchlauf
    // (oder gar nicht, falls der Aufrufer das anders orchestriert) abgeholt.
    cudaMemcpy(h_d_k_cache + ((size_t)pos * dim), d_dk + ((size_t)pos * dim), vec_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d_v_cache + ((size_t)pos * dim), d_dv + ((size_t)pos * dim), vec_bytes, cudaMemcpyDeviceToHost);
}

void cuda_matMulTBatched(const float* h_M, const float* h_d_out, float* h_d_in, int rows, int cols, int batch, int accumulate) {
    if (!cuda_initialized || !h_M || !h_d_out || !h_d_in || rows <= 0 || cols <= 0 || batch <= 0) return;

    size_t size_M = (size_t)rows * cols * sizeof(float);
    size_t size_o = (size_t)batch * rows * sizeof(float);
    size_t size_i = (size_t)batch * cols * sizeof(float);

    // d_M wird mit cuda_matMulBatched() (Forward) geteilt - gleicher
    // Host-Pointer, gleiche Groesse, kein erneuter Upload noetig.
    float* d_M = (float*)cuda_get_buffer((void*)h_M, size_M);
    if (!d_M) return;

    if (cuda_eph_bm_r_size < size_o) {
        if (cuda_eph_bm_r) cudaFree(cuda_eph_bm_r);
        cuda_eph_bm_r_size = (cudaMalloc(&cuda_eph_bm_r, size_o) == cudaSuccess) ? size_o : 0;
    }
    if (cuda_eph_bm_c_size < size_i) {
        if (cuda_eph_bm_c) cudaFree(cuda_eph_bm_c);
        cuda_eph_bm_c_size = (cudaMalloc(&cuda_eph_bm_c, size_i) == cudaSuccess) ? size_i : 0;
    }
    if (!cuda_eph_bm_r || !cuda_eph_bm_c) return;

    cudaMemcpy(cuda_eph_bm_r, h_d_out, size_o, cudaMemcpyHostToDevice);
    if (accumulate == 1) {
        cudaMemcpy(cuda_eph_bm_c, h_d_in, size_i, cudaMemcpyHostToDevice);
    }

    // Gesucht (Row-Major-Semantik, identisch zu matMulTBatched-Kernel):
    //   d_in[p][c] = sum_r M[r][c] * d_out[p][r]             (p < batch)
    //   M:      rows x cols   |   d_out: batch x rows   |   d_in: batch x cols
    //
    // Herleitung analog zum Forward-Kommentar (Row-Major-als-Column-Major-
    // Transpositionstrick): M_cm (cols x rows) = M_rm^T liegt bereits so vor
    // (ld=cols), d_out_cm (rows x batch) = d_out_rm^T liegt bereits so vor
    // (ld=rows). Gesucht ist d_in_cm (cols x batch) = d_in_rm^T:
    //   d_in_rm = d_out_rm @ M_rm  =>  d_in_rm^T = M_rm^T @ d_out_rm^T
    //   => d_in_cm = M_cm @ d_out_cm      (beides schon in der richtigen Form!)
    // -> transA = N, transB = N, m=cols, n=batch, k=rows, lda=cols, ldb=rows,
    //    ldc=cols. beta=1 bei accumulate (addiert auf vorhandenes d_in),
    //    sonst 0 (Ueberschreiben) - identisch zum `accumulate`-Flag im
    //    OpenCL-Kernel.
    const float alpha = 1.0f;
    const float beta = (accumulate == 1) ? 1.0f : 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                cols, batch, rows,
                &alpha,
                d_M, cols,
                cuda_eph_bm_r, rows,
                &beta,
                cuda_eph_bm_c, cols);

    cudaMemcpy(h_d_in, cuda_eph_bm_c, size_i, cudaMemcpyDeviceToHost);
}

void cuda_gradAccumBatched(float* h_gm, const float* h_d_out, const float* h_i_v, int rows, int cols, int batch) {
    if (!cuda_initialized || !h_gm || !h_d_out || !h_i_v || rows <= 0 || cols <= 0 || batch <= 0) return;

    size_t size_gm = (size_t)rows * cols * sizeof(float);
    size_t size_o = (size_t)batch * rows * sizeof(float);
    size_t size_i = (size_t)batch * cols * sizeof(float);

    // AUDIT: gm bleibt persistent im VRAM und akkumuliert ueber viele Aufrufe
    // (z.B. ein Aufruf pro Token eines Trainingsschritts) - analog
    // ocl_gradAccumBatched. Abholung erfolgt spaeter separat per
    // cuda_sync_to_host(h_gm), kein Readback hier.
    float* d_gm = (float*)cuda_get_buffer((void*)h_gm, size_gm);
    if (!d_gm) return;

    if (cuda_eph_bm_r_size < size_o) {
        if (cuda_eph_bm_r) cudaFree(cuda_eph_bm_r);
        cuda_eph_bm_r_size = (cudaMalloc(&cuda_eph_bm_r, size_o) == cudaSuccess) ? size_o : 0;
    }
    if (cuda_eph_bm_c_size < size_i) {
        if (cuda_eph_bm_c) cudaFree(cuda_eph_bm_c);
        cuda_eph_bm_c_size = (cudaMalloc(&cuda_eph_bm_c, size_i) == cudaSuccess) ? size_i : 0;
    }
    if (!cuda_eph_bm_r || !cuda_eph_bm_c) return;

    cudaMemcpy(cuda_eph_bm_r, h_d_out, size_o, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_eph_bm_c, h_i_v, size_i, cudaMemcpyHostToDevice);

    // Gesucht (Row-Major-Semantik, identisch zu gradAccumBatched-Kernel):
    //   gm[r][c] += sum_p d_out[p][r] * i_v[p][c]
    //   d_out: batch x rows   |   i_v: batch x cols   |   gm: rows x cols
    //
    // i_v_cm (cols x batch) = i_v_rm^T liegt bereits so vor (ld=cols).
    // Gesucht: gm_cm (cols x rows) = gm_rm^T:
    //   gm_rm += d_out_rm^T @ i_v_rm  =>  gm_rm^T += i_v_rm^T @ d_out_rm
    //   => gm_cm += i_v_cm @ d_out_rm
    // d_out_rm (batch x rows) ist NICHT direkt die Column-Major-Ansicht von
    // d_out (die waere d_out_cm = d_out_rm^T, rows x batch, ld=rows) - wird
    // daher ueber transB=T zurueckgeholt: op(B) = d_out_cm^T = d_out_rm.
    // -> transA = N, transB = T, m=cols, n=rows, k=batch, lda=cols, ldb=rows,
    //    ldc=cols, beta=1 (gm akkumuliert immer, kein Overwrite-Modus -
    //    identisch zum OpenCL-Kernel, der ausschliesslich `+=` kennt).
    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                cols, rows, batch,
                &alpha,
                cuda_eph_bm_c, cols,
                cuda_eph_bm_r, rows,
                &beta,
                d_gm, cols);
}

// ==============================================================================
// Update 2026-08-08: EINZELVEKTOR-OPS (matVecMul/matVecMulT/gradAccum)
// Ports von ocl_matVecMul, ocl_matVecMulT, ocl_gradAccum (cortex_opencl.nim).
// Bewusst Handkernel statt cuBLAS (cublasSgemv/cublasSger waeren die
// GEMM-Bibliotheks-Entsprechung) - konsistent mit der bestehenden
// Attention-Forward/Backward-Portierung, die aus demselben Grund ebenfalls
// Handkernel statt cuBLAS nutzt: bei Einzelvektor-Aufrufen (ein Token, kein
// Batch) ueberwiegt der cuBLAS-Handle-Dispatch-Overhead den moeglichen
// Tensor-Core-Vorteil, der laut obiger Einordnung ohnehin nur bei den
// batched GEMM-Operationen realistisch zum Tragen kommt.
// ==============================================================================

void cuda_matVecMul(const float* h_M, const float* h_v, float* h_out, int rows, int cols) {
    if (!cuda_initialized || !h_M || !h_v || !h_out || rows == 0 || cols == 0) return;

    size_t size_M = (size_t)rows * cols * sizeof(float);
    size_t size_v = (size_t)cols * sizeof(float);
    size_t size_out = (size_t)rows * sizeof(float);

    float* d_M = (float*)cuda_get_buffer((void*)h_M, size_M);
    if (!d_M) return;

    // Ephemere Scratch-Buffer geteilt mit cuda_attentionForward/Backward
    // (analog ocl_eph_v/ocl_eph_out, die OpenCL ebenfalls ueber matVecMul UND
    // Attention hinweg wiederverwendet) - wachsen bei Bedarf auf die jeweils
    // groesste bisher angefragte Groesse.
    if (cuda_eph_q_size < size_v) {
        if (cuda_eph_q) cudaFree(cuda_eph_q);
        cuda_eph_q_size = (cudaMalloc(&cuda_eph_q, size_v) == cudaSuccess) ? size_v : 0;
    }
    if (cuda_eph_out_size < size_out) {
        if (cuda_eph_out) cudaFree(cuda_eph_out);
        cuda_eph_out_size = (cudaMalloc(&cuda_eph_out, size_out) == cudaSuccess) ? size_out : 0;
    }
    if (!cuda_eph_q || !cuda_eph_out) return;

    cudaMemcpy(cuda_eph_q, h_v, size_v, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (rows + threads - 1) / threads;
    matVecMulKernel<<<blocks, threads>>>(d_M, cuda_eph_q, cuda_eph_out, rows, cols);

    cudaMemcpy(h_out, cuda_eph_out, size_out, cudaMemcpyDeviceToHost);
}

void cuda_matVecMulT(const float* h_M, const float* h_d_out, float* h_d_in, int rows, int cols, int accumulate) {
    if (!cuda_initialized || !h_M || !h_d_out || !h_d_in || rows == 0 || cols == 0) return;

    size_t size_M = (size_t)rows * cols * sizeof(float);
    size_t size_out = (size_t)rows * sizeof(float);
    size_t size_in = (size_t)cols * sizeof(float);

    float* d_M = (float*)cuda_get_buffer((void*)h_M, size_M);
    if (!d_M) return;

    if (cuda_eph_bt_a_size < size_out) {
        if (cuda_eph_bt_a) cudaFree(cuda_eph_bt_a);
        cuda_eph_bt_a_size = (cudaMalloc(&cuda_eph_bt_a, size_out) == cudaSuccess) ? size_out : 0;
    }
    if (cuda_eph_bt_b_size < size_in) {
        if (cuda_eph_bt_b) cudaFree(cuda_eph_bt_b);
        cuda_eph_bt_b_size = (cudaMalloc(&cuda_eph_bt_b, size_in) == cudaSuccess) ? size_in : 0;
    }
    if (!cuda_eph_bt_a || !cuda_eph_bt_b) return;

    cudaMemcpy(cuda_eph_bt_a, h_d_out, size_out, cudaMemcpyHostToDevice);
    if (accumulate == 1) {
        cudaMemcpy(cuda_eph_bt_b, h_d_in, size_in, cudaMemcpyHostToDevice);
    }

    int threads = 256;
    int blocks = (cols + threads - 1) / threads;
    matVecMulTKernel<<<blocks, threads>>>(d_M, cuda_eph_bt_a, cuda_eph_bt_b, rows, cols, accumulate);

    cudaMemcpy(h_d_in, cuda_eph_bt_b, size_in, cudaMemcpyDeviceToHost);
}

void cuda_gradAccum(float* h_gm, const float* h_d_out, const float* h_i_v, int rows, int cols) {
    if (!cuda_initialized || !h_gm || !h_d_out || !h_i_v || rows == 0 || cols == 0) return;

    size_t size_gm = (size_t)rows * cols * sizeof(float);
    size_t size_out = (size_t)rows * sizeof(float);
    size_t size_in = (size_t)cols * sizeof(float);

    // AUDIT: gm bleibt persistent im VRAM (analog ocl_gradAccum) - kein
    // Readback hier, Abholung ueber das bestehende cuda_sync_to_host().
    float* d_gm = (float*)cuda_get_buffer((void*)h_gm, size_gm);
    if (!d_gm) return;

    if (cuda_eph_bt_a_size < size_out) {
        if (cuda_eph_bt_a) cudaFree(cuda_eph_bt_a);
        cuda_eph_bt_a_size = (cudaMalloc(&cuda_eph_bt_a, size_out) == cudaSuccess) ? size_out : 0;
    }
    if (cuda_eph_bt_b_size < size_in) {
        if (cuda_eph_bt_b) cudaFree(cuda_eph_bt_b);
        cuda_eph_bt_b_size = (cudaMalloc(&cuda_eph_bt_b, size_in) == cudaSuccess) ? size_in : 0;
    }
    if (!cuda_eph_bt_a || !cuda_eph_bt_b) return;

    cudaMemcpy(cuda_eph_bt_a, h_d_out, size_out, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_eph_bt_b, h_i_v, size_in, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);
    gradAccumKernel<<<blocks, threads>>>(d_gm, cuda_eph_bt_a, cuda_eph_bt_b, rows, cols);
    // Kein Readback: gm bleibt im VRAM und akkumuliert weiter (wie ocl_gradAccum).
}

// ==============================================================================
// Update 2026-08-08: GEBUENDELTE ATTENTION (Forward + Backward)
// Ports von ocl_attentionForwardBatched/ocl_attentionBackwardBatched
// (cortex_opencl.nim) - letzte verbliebene Luecke aus der urspruenglichen
// Lueckenliste vom 2026-08-06. Siehe Kernel-Kommentare oben fuer die
// Grid/Thread-Aufteilung (flacher Index statt 2D-NDRange-Offset).
// ==============================================================================

void cuda_attentionForwardBatched(const float* h_q_all, const float* h_k_cache, const float* h_v_cache,
                                   float* h_att_out_all, int act, int numHeads, int headSize, int dim, int ctx) {
    if (!cuda_initialized || !h_q_all || !h_k_cache || !h_v_cache || !h_att_out_all || act <= 0) return;

    size_t size_cache = (size_t)ctx * dim * sizeof(float);
    size_t size_qout = (size_t)act * dim * sizeof(float);
    size_t size_scores = (size_t)act * numHeads * ctx * sizeof(float);

    float* d_k = (float*)cuda_get_buffer((void*)h_k_cache, size_cache);
    float* d_v = (float*)cuda_get_buffer((void*)h_v_cache, size_cache);
    if (!d_k || !d_v) return;

    // AUDIT-FIX-Analogie (siehe ocl_attentionForwardBatched): cuda_get_buffer()
    // laedt nur beim allerersten Aufruf fuer einen Host-Pointer automatisch
    // hoch. Bei Cache-Hit muss der aktuelle Stand (die gerade frisch
    // berechneten ersten act*dim Floats von k_cache/v_cache) explizit
    // nachgeladen werden, sonst rechnet die GPU im naechsten Trainingsschritt
    // mit veralteten K/V-Werten - OpenCL-Kommentar dazu: empirisch bis zu
    // 0.013 Loss-Differenz ohne diesen Fix, 0.0 mit ihm.
    cudaMemcpy(d_k, h_k_cache, size_qout, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v_cache, size_qout, cudaMemcpyHostToDevice);

    if (cuda_eph_att_q_size < size_qout) {
        if (cuda_eph_att_q) cudaFree(cuda_eph_att_q);
        cuda_eph_att_q_size = (cudaMalloc(&cuda_eph_att_q, size_qout) == cudaSuccess) ? size_qout : 0;
    }
    if (cuda_eph_att_out_size < size_qout) {
        if (cuda_eph_att_out) cudaFree(cuda_eph_att_out);
        cuda_eph_att_out_size = (cudaMalloc(&cuda_eph_att_out, size_qout) == cudaSuccess) ? size_qout : 0;
    }
    if (cuda_eph_att_scores_size < size_scores) {
        if (cuda_eph_att_scores) cudaFree(cuda_eph_att_scores);
        cuda_eph_att_scores_size = (cudaMalloc(&cuda_eph_att_scores, size_scores) == cudaSuccess) ? size_scores : 0;
    }
    if (!cuda_eph_att_q || !cuda_eph_att_out || !cuda_eph_att_scores) return;

    cudaMemcpy(cuda_eph_att_q, h_q_all, size_qout, cudaMemcpyHostToDevice);

    // ATT_POS_CHUNK-Bloecke statt eines Riesen-Dispatch (siehe Konstanten-
    // Kommentar oben).
    int threads = 256;
    for (int p0 = 0; p0 < act; p0 += ATT_POS_CHUNK) {
        int chunkLen = (act - p0) < ATT_POS_CHUNK ? (act - p0) : ATT_POS_CHUNK;
        int total = numHeads * chunkLen;
        int blocks = (total + threads - 1) / threads;
        attentionForwardBatchedKernel<<<blocks, threads>>>(cuda_eph_att_q, d_k, d_v, cuda_eph_att_out,
                                                             cuda_eph_att_scores, p0, chunkLen, numHeads, headSize, dim, ctx);
    }

    cudaMemcpy(h_att_out_all, cuda_eph_att_out, size_qout, cudaMemcpyDeviceToHost);
}

void cuda_attentionBackwardBatched(const float* h_q_all, const float* h_k_cache, const float* h_v_cache,
                                    const float* h_d_att_out_all, float* h_d_q_all,
                                    float* h_d_k_cache, float* h_d_v_cache,
                                    int act, int numHeads, int headSize, int dim, int ctx) {
    if (!cuda_initialized || !h_q_all || !h_k_cache || !h_v_cache || !h_d_att_out_all ||
        !h_d_q_all || !h_d_k_cache || !h_d_v_cache || act <= 0) return;

    size_t size_cache = (size_t)ctx * dim * sizeof(float);
    size_t size_qout = (size_t)act * dim * sizeof(float);
    size_t size_scores = (size_t)act * numHeads * ctx * sizeof(float);

    float* d_k = (float*)cuda_get_buffer((void*)h_k_cache, size_cache);
    float* d_v = (float*)cuda_get_buffer((void*)h_v_cache, size_cache);
    float* d_dk = (float*)cuda_get_buffer((void*)h_d_k_cache, size_cache);
    float* d_dv = (float*)cuda_get_buffer((void*)h_d_v_cache, size_cache);
    if (!d_k || !d_v || !d_dk || !d_dv) return;

    // d_k/d_v sind Lese-Inputs, muessen bei jedem Aufruf frisch hochgeladen
    // werden (siehe cuda_attentionForwardBatched). d_dk/d_dv sind reine
    // Schreib-Ziele - Stufe B nullt und befuellt ihr (h,t)-Segment selbst,
    // kein Pre-Upload noetig.
    cudaMemcpy(d_k, h_k_cache, size_qout, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v_cache, size_qout, cudaMemcpyHostToDevice);

    if (cuda_eph_att_q_size < size_qout) {
        if (cuda_eph_att_q) cudaFree(cuda_eph_att_q);
        cuda_eph_att_q_size = (cudaMalloc(&cuda_eph_att_q, size_qout) == cudaSuccess) ? size_qout : 0;
    }
    if (cuda_eph_att_datt_size < size_qout) {
        if (cuda_eph_att_datt) cudaFree(cuda_eph_att_datt);
        cuda_eph_att_datt_size = (cudaMalloc(&cuda_eph_att_datt, size_qout) == cudaSuccess) ? size_qout : 0;
    }
    if (cuda_eph_att_dq_size < size_qout) {
        if (cuda_eph_att_dq) cudaFree(cuda_eph_att_dq);
        cuda_eph_att_dq_size = (cudaMalloc(&cuda_eph_att_dq, size_qout) == cudaSuccess) ? size_qout : 0;
    }
    if (cuda_eph_att_scores_size < size_scores) {
        if (cuda_eph_att_scores) cudaFree(cuda_eph_att_scores);
        cuda_eph_att_scores_size = (cudaMalloc(&cuda_eph_att_scores, size_scores) == cudaSuccess) ? size_scores : 0;
    }
    if (cuda_eph_att_ds_size < size_scores) {
        if (cuda_eph_att_ds) cudaFree(cuda_eph_att_ds);
        cuda_eph_att_ds_size = (cudaMalloc(&cuda_eph_att_ds, size_scores) == cudaSuccess) ? size_scores : 0;
    }
    if (!cuda_eph_att_q || !cuda_eph_att_datt || !cuda_eph_att_dq || !cuda_eph_att_scores || !cuda_eph_att_ds) return;

    cudaMemcpy(cuda_eph_att_q, h_q_all, size_qout, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_eph_att_datt, h_d_att_out_all, size_qout, cudaMemcpyHostToDevice);

    int threads = 256;

    // Stufe A: query-parallel (d_q, weights, ds).
    for (int p0 = 0; p0 < act; p0 += ATT_POS_CHUNK) {
        int chunkLen = (act - p0) < ATT_POS_CHUNK ? (act - p0) : ATT_POS_CHUNK;
        int total = numHeads * chunkLen;
        int blocks = (total + threads - 1) / threads;
        attentionBackwardBatchedQueryKernel<<<blocks, threads>>>(cuda_eph_att_q, d_k, d_v, cuda_eph_att_datt,
                                                                   cuda_eph_att_dq, cuda_eph_att_scores, cuda_eph_att_ds,
                                                                   p0, chunkLen, numHeads, headSize, dim, ctx);
    }

    // Stufe B: key-parallel (d_k_cache, d_v_cache).
    for (int t0 = 0; t0 < act; t0 += ATT_POS_CHUNK) {
        int chunkLen = (act - t0) < ATT_POS_CHUNK ? (act - t0) : ATT_POS_CHUNK;
        int total = numHeads * chunkLen;
        int blocks = (total + threads - 1) / threads;
        attentionBackwardBatchedKeyKernel<<<blocks, threads>>>(cuda_eph_att_q, cuda_eph_att_datt,
                                                                 cuda_eph_att_scores, cuda_eph_att_ds,
                                                                 d_dk, d_dv, t0, chunkLen, act, numHeads, headSize, dim, ctx);
    }

    // Nur die tatsaechlich beschriebenen 'act' Positionen zurueckholen (nicht
    // den gesamten ctx-grossen Cache).
    cudaMemcpy(h_d_q_all, cuda_eph_att_dq, size_qout, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d_k_cache, d_dk, size_qout, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d_v_cache, d_dv, size_qout, cudaMemcpyDeviceToHost);
}

} // extern "C"
