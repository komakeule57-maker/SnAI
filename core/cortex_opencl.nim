# ==============================================================================
# Dateiname: cortex_opencl.nim
# Funktion: OpenCL Compute Engine (Bare-Metal JIT) - PHASE XII (Syntax-Harden)
# ==============================================================================
# This file is part of SnAI.
#
# SnAI is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
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

size_t ocl_max_work_group_size = 64; 
_Atomic int ocl_initialized = 0;     

#define MAX_OCL_BUFFERS 4096

typedef struct { 
    void* h_ptr; 
    cl_mem d_buf; 
    size_t bytes; 
} OclBufMap;

OclBufMap ocl_buf_map[MAX_OCL_BUFFERS];
int ocl_buf_count = 0;

cl_mem ocl_eph_v = NULL; size_t ocl_eph_v_size = 0;
cl_mem ocl_eph_out = NULL; size_t ocl_eph_out_size = 0;

cl_mem ocl_get_buffer(void* h_ptr, size_t bytes) {
    for(int i = 0; i < ocl_buf_count; i++) {
        if(ocl_buf_map[i].h_ptr == h_ptr) return ocl_buf_map[i].d_buf;
    }
    
    if(ocl_buf_count >= MAX_OCL_BUFFERS) {
        printf("🚨 [SYMBIO GPU] VRAM Registry Overflow (Cap: %d)!\n", MAX_OCL_BUFFERS);
        return NULL; 
    }
    
    cl_int err;
    cl_mem d_buf = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, bytes, NULL, &err);
    if(err == CL_SUCCESS) {
        if(bytes > 0 && h_ptr != NULL) {
            clEnqueueWriteBuffer(ocl_queue, d_buf, CL_TRUE, 0, bytes, h_ptr, 0, NULL, NULL);
        }
        ocl_buf_map[ocl_buf_count].h_ptr = h_ptr;
        ocl_buf_map[ocl_buf_count].d_buf = d_buf;
        ocl_buf_map[ocl_buf_count].bytes = bytes;
        ocl_buf_count++;
        return d_buf;
    }
    return NULL;
}

// --- FFI IMPLEMENTATIONS ---
void ocl_sync_to_device(void* h_ptr, size_t bytes) {
    if (!ocl_ctx || !h_ptr || bytes == 0) return;
    cl_mem d_buf = ocl_get_buffer(h_ptr, bytes);
    if (d_buf) {
        clEnqueueWriteBuffer(ocl_queue, d_buf, CL_TRUE, 0, bytes, h_ptr, 0, NULL, NULL);
    }
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

void ocl_zero_buffer(void* h_ptr, size_t bytes) {
    if (!ocl_ctx) return;
    cl_mem d_buf = ocl_get_buffer(h_ptr, bytes);
    if (d_buf) {
        float zero = 0.0f;
        clEnqueueFillBuffer(ocl_queue, d_buf, &zero, sizeof(float), 0, bytes, 0, NULL, NULL);
        clFinish(ocl_queue);
    }
}

void ocl_release_buffer(void* h_ptr) {
    if (!ocl_ctx || !h_ptr) return;
    for(int i = 0; i < ocl_buf_count; i++) {
        if(ocl_buf_map[i].h_ptr == h_ptr) {
            clReleaseMemObject(ocl_buf_map[i].d_buf);
            // Defragmentierung: Letztes Element in die Lücke verschieben
            ocl_buf_map[i] = ocl_buf_map[ocl_buf_count - 1];
            ocl_buf_count--;
            return;
        }
    }
}

// -----------------------------------------------------------------------------
// JIT KERNELS - E > H OPTIMIZED
// -----------------------------------------------------------------------------
const char *kernel_source = 
"__kernel void matVecMul(__global const float* M, __global const float* v, __global float* out, int rows, int cols) {\n"
"    __local float scratch[64];\n"
"    int r = get_group_id(0);\n"
"    int tid = get_local_id(0);\n"
"    int lsize = get_local_size(0);\n"
"    if (r < rows) {\n"
"        float sum = 0.0f;\n"
"        for (int c = tid; c < cols; c += lsize) {\n"
"            sum += M[r * cols + c] * v[c];\n"
"        }\n"
"        scratch[tid] = sum;\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        for (int offset = lsize / 2; offset > 0; offset /= 2) {\n"
"            if (tid < offset) scratch[tid] += scratch[tid + offset];\n"
"            barrier(CLK_LOCAL_MEM_FENCE);\n"
"        }\n"
"        if (tid == 0) out[r] = scratch[0];\n"
"    }\n"
"}\n"
"\n"
"__kernel void adamUpdate(__global float* w, __global float* m, __global float* v, __global const float* g, \n"
"                         float lr, float b1, float b2, float b1t, float b2t, float scale, int size, int is_sparse) {\n"
"    size_t i = get_global_id(0);\n"
"    if (i < (size_t)size) {\n"
"        float grad = g[i] / scale;\n"
"        if (is_sparse == 1 && grad == 0.0f) return;\n"
"        if (isnan(grad) || grad > 9999.0f || grad < -9999.0f) return;\n"
"        \n"
"        m[i] = b1 * m[i] + (1.0f - b1) * grad;\n"
"        v[i] = b2 * v[i] + (1.0f - b2) * grad * grad;\n"
"        float m_hat = m[i] / b1t;\n"
"        float v_hat = v[i] / b2t;\n"
"        w[i] -= lr * m_hat / (sqrt(v_hat) + 1e-8f);\n"
"    }\n"
"}\n";

void shutdown_opencl_engine() {
    for(int i=0; i<ocl_buf_count; i++) {
        if(ocl_buf_map[i].d_buf) clReleaseMemObject(ocl_buf_map[i].d_buf);
    }
    ocl_buf_count = 0;
    
    if (ocl_eph_v) { clReleaseMemObject(ocl_eph_v); ocl_eph_v = NULL; ocl_eph_v_size = 0; }
    if (ocl_eph_out) { clReleaseMemObject(ocl_eph_out); ocl_eph_out = NULL; ocl_eph_out_size = 0; }
    
    if (ocl_kernel_matvec) clReleaseKernel(ocl_kernel_matvec);
    if (ocl_kernel_adam) clReleaseKernel(ocl_kernel_adam);
    if (ocl_queue) clReleaseCommandQueue(ocl_queue);
    if (ocl_ctx) clReleaseContext(ocl_ctx);
    
    ocl_kernel_matvec = NULL; ocl_kernel_adam = NULL;
    ocl_queue = NULL; ocl_ctx = NULL;
    atomic_store(&ocl_initialized, 0); 
    printf("⚡ [SYMBIO VRAM] OpenCL-Ressourcen & Registry freigegeben.\n");
}

void init_opencl_engine() {
    if (atomic_load(&ocl_initialized)) return; 
    int expected = 0;
    if (!atomic_compare_exchange_strong(&ocl_initialized, &expected, 1)) return; 

    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        printf("🚨 [SYMBIO GPU] Fatal: Keine OpenCL-Plattformen installiert!\n");
        atomic_store(&ocl_initialized, 0);
        return;
    }

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);

    cl_device_id target_device = NULL;
    
    // STAGE 1: Die noble dedizierte GPU suchen (NVIDIA / AMD)
    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &target_device, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0 && target_device != NULL) break; 
    }
    
    // STAGE 2: Kaskade auf DEFAULT (Fängt Laptop-Chips wie Intel Iris / Apple M-Series)
    if (target_device == NULL) {
        for (cl_uint i = 0; i < num_platforms; i++) {
            cl_uint num_devices = 0;
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_DEFAULT, 1, &target_device, &num_devices);
            if (err == CL_SUCCESS && num_devices > 0 && target_device != NULL) break; 
        }
    }

    // STAGE 3: Letzte Verzweiflung auf ALL
    if (target_device == NULL) {
        for (cl_uint i = 0; i < num_platforms; i++) {
            cl_uint num_devices = 0;
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 1, &target_device, &num_devices);
            if (err == CL_SUCCESS && num_devices > 0 && target_device != NULL) break; 
        }
    }
    
    free(platforms);

    if (target_device == NULL) {
        printf("🚨 [SYMBIO GPU] OpenCL Discovery fehlgeschlagen. Harter Fallback auf CPU.\n");
        atomic_store(&ocl_initialized, 0);
        return;
    }
    
    ocl_ctx = clCreateContext(NULL, 1, &target_device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        atomic_store(&ocl_initialized, 0);
        return;
    }

    ocl_queue = clCreateCommandQueue(ocl_ctx, target_device, 0, &err); 
    if (err != CL_SUCCESS) { shutdown_opencl_engine(); return; }
    
    cl_program program = clCreateProgramWithSource(ocl_ctx, 1, (const char **)&kernel_source, NULL, &err);
    cl_int build_status = clBuildProgram(program, 1, &target_device, NULL, NULL, NULL);
    
    if (build_status != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, target_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, target_device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("🚨 [SYMBIO GPU] JIT Kernel Kompilierung fehlgeschlagen!\n--- BUILD LOG ---\n%s\n-----------------\n", log);
        free(log);
        clReleaseProgram(program);
        shutdown_opencl_engine();
        return;
    }

    ocl_kernel_matvec = clCreateKernel(program, "matVecMul", &err);
    ocl_kernel_adam = clCreateKernel(program, "adamUpdate", &err);
    clReleaseProgram(program); 
    
    if (err != CL_SUCCESS) { shutdown_opencl_engine(); return; }

    size_t hw_max_wg;
    if (clGetDeviceInfo(target_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &hw_max_wg, NULL) == CL_SUCCESS) {
        ocl_max_work_group_size = 64;
        if (hw_max_wg < 64) ocl_max_work_group_size = 32;
        if (hw_max_wg < 32) ocl_max_work_group_size = 16;
        if (hw_max_wg < 16) ocl_max_work_group_size = 8;
        if (hw_max_wg < 8)  ocl_max_work_group_size = 4;
        if (hw_max_wg < 4)  ocl_max_work_group_size = 2;
        if (hw_max_wg < 2)  ocl_max_work_group_size = 1;
    }
    
    char deviceName[256];
    clGetDeviceInfo(target_device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    printf("⚡ [SYMBIO VRAM] Stateful OpenCL JIT eingerastet auf: [%s] (WG-Size: %zu)\n", deviceName, ocl_max_work_group_size);
}

void ocl_matVecMul(const float* h_M, const float* h_v, float* h_out, int rows, int cols) {
    if (!ocl_ctx || !h_M || !h_v || !h_out || rows == 0 || cols == 0) return;

    size_t size_M = (size_t)rows * (size_t)cols * sizeof(float);
    size_t size_v = (size_t)cols * sizeof(float);
    size_t size_out = (size_t)rows * sizeof(float);

    cl_mem d_M = ocl_get_buffer((void*)h_M, size_M);
    if (!d_M) return;

    if (ocl_eph_v_size < size_v) {
        if (ocl_eph_v) clReleaseMemObject(ocl_eph_v);
        ocl_eph_v = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, size_v, NULL, NULL);
        ocl_eph_v_size = size_v;
    }
    if (ocl_eph_out_size < size_out) {
        if (ocl_eph_out) clReleaseMemObject(ocl_eph_out);
        ocl_eph_out = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE, size_out, NULL, NULL);
        ocl_eph_out_size = size_out;
    }

    clEnqueueWriteBuffer(ocl_queue, ocl_eph_v, CL_FALSE, 0, size_v, h_v, 0, NULL, NULL);

    cl_int err;
    err  = clSetKernelArg(ocl_kernel_matvec, 0, sizeof(cl_mem), (void *)&d_M);
    err |= clSetKernelArg(ocl_kernel_matvec, 1, sizeof(cl_mem), (void *)&ocl_eph_v);
    err |= clSetKernelArg(ocl_kernel_matvec, 2, sizeof(cl_mem), (void *)&ocl_eph_out);
    err |= clSetKernelArg(ocl_kernel_matvec, 3, sizeof(int), (void *)&rows);
    err |= clSetKernelArg(ocl_kernel_matvec, 4, sizeof(int), (void *)&cols);

    size_t local_item_size = ocl_max_work_group_size; 
    size_t global_item_size = (size_t)rows * local_item_size;

    clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_matvec, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl_queue, ocl_eph_out, CL_TRUE, 0, size_out, h_out, 0, NULL, NULL);
}

void ocl_adamStep(float* h_w, float* h_m, float* h_v, float* h_g, float lr, float b1, float b2, float b1t, float b2t, float scale, int size, int is_sparse) {
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
    
    size_t local_item_size = ocl_max_work_group_size; 
    size_t global_item_size = (((size_t)size + local_item_size - 1) / local_item_size) * local_item_size;
    clEnqueueNDRangeKernel(ocl_queue, ocl_kernel_adam, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    clEnqueueReadBuffer(ocl_queue, d_w, CL_TRUE, 0, bytes, h_w, 0, NULL, NULL);
    
    memset(h_g, 0, bytes); 
}
""" .}

proc init_opencl_engine*() {.importc: "init_opencl_engine", cdecl.}
proc shutdown_opencl_engine*() {.importc: "shutdown_opencl_engine", cdecl.}
proc ocl_sync_to_host*(h_ptr: pointer) {.importc: "ocl_sync_to_host", cdecl.}
proc ocl_sync_to_device*(h_ptr: pointer, bytes: csize_t) {.importc: "ocl_sync_to_device", cdecl.}
proc ocl_matVecMul*(m: ptr cfloat, v: ptr cfloat, out_v: ptr cfloat, rows: cint, cols: cint) {.importc: "ocl_matVecMul", cdecl.}
proc ocl_adamStep*(w: ptr cfloat, m: ptr cfloat, v: ptr cfloat, g: ptr cfloat, lr: cfloat, b1: cfloat, b2: cfloat, b1t: cfloat, b2t: cfloat, scale: cfloat, size: cint, is_sparse: cint) {.importc: "ocl_adamStep", cdecl.}
proc ocl_zero_buffer*(h_ptr: ptr cfloat, bytes: csize_t) {.importc: "ocl_zero_buffer", cdecl.}
proc ocl_release_buffer*(h_ptr: pointer) {.importc: "ocl_release_buffer", cdecl.}
