#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "symbolic.h"
#include <cmath>

using namespace std;


__global__ void RL(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned* __restrict__ csr_r_ptr_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const int* __restrict__ level_idx_dev,
        REAL* __restrict__ tmpMem,
        const unsigned n,
        const int levelHead,
        const int inLevPos)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = threadIdx.x / 32;

    const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    extern __shared__ REAL s[];

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
    __syncthreads();

    //broadcast to submatrix
    const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
    const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
    unsigned subCol;
    const int tidInWarp = threadIdx.x % 32;
    unsigned subColElem = 0;

    int woffset = 0;
    while (subMatSize > woffset)
    {
        if (wid + woffset < subMatSize)
        {
            offset = 0;
            subCol = csr_c_idx_dev[subColPos + woffset];
            while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
            {
                if (tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
                {

                    subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
                    unsigned ridx = sym_r_idx_dev[subColElem];

                    if (ridx == currentCol)
                    {
                        s[wid] = val_dev[subColElem];
                    }
                    //Threads in a warp are always synchronized
                    //__syncthreads();
                    if (ridx > currentCol)
                    {
                        //elem in currentCol same row with subColElem might be 0, so
                        //clearing tmpMem is necessary
                        atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*s[wid]);
                    }
                }
                offset += 32;
            }
        }
        woffset += blockDim.x/32;
    }

    __syncthreads();
    //Clear tmpMem
    offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];
            tmpMem[bid*n + ridx]= 0;
        }
        offset += blockDim.x;
    }
}

__global__ void RL_perturb(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned* __restrict__ csr_r_ptr_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const int* __restrict__ level_idx_dev,
        REAL* __restrict__ tmpMem,
        const unsigned n,
        const int levelHead,
        const int inLevPos,
        const float pert)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = threadIdx.x / 32;

    const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    extern __shared__ REAL s[];

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            if (abs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
                val_dev[l_col_ptr_dev[currentCol]] = pert;

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[bid*n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
    __syncthreads();

    //broadcast to submatrix
    const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
    const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
    unsigned subCol;
    const int tidInWarp = threadIdx.x % 32;
    unsigned subColElem = 0;

    int woffset = 0;
    while (subMatSize > woffset)
    {
        if (wid + woffset < subMatSize)
        {
            offset = 0;
            subCol = csr_c_idx_dev[subColPos + woffset];
            while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
            {
                if (tidInWarp + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
                {

                    subColElem = sym_c_ptr_dev[subCol] + tidInWarp + offset;
                    unsigned ridx = sym_r_idx_dev[subColElem];

                    if (ridx == currentCol)
                    {
                        s[wid] = val_dev[subColElem];
                    }
                    //Threads in a warp are always synchronized
                    //__syncthreads();
                    if (ridx > currentCol)
                    {
                        //elem in currentCol same row with subColElem might be 0, so
                        //clearing tmpMem is necessary
                        atomicAdd(&val_dev[subColElem], -tmpMem[ridx+n*bid]*s[wid]);
                    }
                }
                offset += 32;
            }
        }
        woffset += blockDim.x/32;
    }

    __syncthreads();
    //Clear tmpMem
    offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];
            tmpMem[bid*n + ridx]= 0;
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_factorizeCurrentCol(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[stream * n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_factorizeCurrentCol_perturb(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n,
        const float pert)
{
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            if (abs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
                val_dev[l_col_ptr_dev[currentCol]] = pert;

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[stream * n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_updateSubmat(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    __shared__ REAL s;

    //broadcast to submatrix
    const unsigned subColPos = csr_diag_ptr_dev[currentCol] + bid + 1;
    unsigned subCol;
    unsigned subColElem = 0;

    int offset = 0;
    subCol = csr_c_idx_dev[subColPos];
    while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
    {
        if (tid + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
        {
            subColElem = sym_c_ptr_dev[subCol] + tid + offset;
            unsigned ridx = sym_r_idx_dev[subColElem];

            if (ridx == currentCol)
            {
                s = val_dev[subColElem];
            }
            __syncthreads();
            if (ridx > currentCol)
            {
                atomicAdd(&val_dev[subColElem], -tmpMem[stream * n + ridx] * s);
            }
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_cleartmpMem(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    unsigned offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];
            tmpMem[stream * n + ridx]= 0;
        }
        offset += blockDim.x;
    }
}





void LUonDevice(Symbolic_Matrix &A_sym, ostream &out, ostream &err, bool PERTURB) {
    // 1. 初始化 GPU
    int deviceCount, dev = 0;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaSetDevice(dev);
    out << "Device " << dev << ": " << deviceProp.name << " has been selected." << endl;

    // 2. 创建事件和流
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;

    const int Nstreams = 4;  // 流数量
    cudaStream_t streams[Nstreams];
    for (int j = 0; j < Nstreams; ++j) {
        cudaStreamCreate(&streams[j]);
    }

    // 3. 分配设备内存（一次性分配）
    unsigned n = A_sym.n, nnz = A_sym.nnz, num_lev = A_sym.num_lev;
    unsigned *sym_c_ptr_dev, *sym_r_idx_dev, *l_col_ptr_dev;
    REAL *val_dev;
    unsigned *csr_r_ptr_dev, *csr_c_idx_dev, *csr_diag_ptr_dev;
    int *level_idx_dev;
    REAL *tmpMem;
    unsigned TMPMEMNUM;

    cudaMalloc((void**)&sym_c_ptr_dev, (n + 1) * sizeof(unsigned));
    cudaMalloc((void**)&sym_r_idx_dev, nnz * sizeof(unsigned));
    cudaMalloc((void**)&val_dev, nnz * sizeof(REAL));
    cudaMalloc((void**)&l_col_ptr_dev, n * sizeof(unsigned));
    cudaMalloc((void**)&csr_r_ptr_dev, (n + 1) * sizeof(unsigned));
    cudaMalloc((void**)&csr_c_idx_dev, nnz * sizeof(unsigned));
    cudaMalloc((void**)&csr_diag_ptr_dev, n * sizeof(unsigned));
    cudaMalloc((void**)&level_idx_dev, n * sizeof(int));

    // 4. 异步内存拷贝（使用流0）
    cudaMemcpyAsync(sym_c_ptr_dev, &A_sym.sym_c_ptr[0], (n + 1) * sizeof(unsigned), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(sym_r_idx_dev, &A_sym.sym_r_idx[0], nnz * sizeof(unsigned), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(val_dev, &A_sym.val[0], nnz * sizeof(REAL), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(l_col_ptr_dev, &A_sym.l_col_ptr[0], n * sizeof(unsigned), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(csr_r_ptr_dev, &A_sym.csr_r_ptr[0], (n + 1) * sizeof(unsigned), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(csr_c_idx_dev, &A_sym.csr_c_idx[0], nnz * sizeof(unsigned), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(csr_diag_ptr_dev, &A_sym.csr_diag_ptr[0], n * sizeof(unsigned), cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(level_idx_dev, &A_sym.level_idx[0], n * sizeof(int), cudaMemcpyHostToDevice, streams[0]);

    // 5. 计算临时内存大小
    size_t MaxtmpMemSize;
    {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        MaxtmpMemSize = free - 4ull * 1024ull * 1024ull * 1024ull;  // 保留4GB
    }
    const size_t GoodtmpMemChoice = sizeof(REAL) * size_t(n) * size_t(A_sym.level_ptr[1]);
    TMPMEMNUM = (GoodtmpMemChoice < MaxtmpMemSize) ? A_sym.level_ptr[1] : MaxtmpMemSize / n / sizeof(REAL);
    cudaMalloc((void**)&tmpMem, TMPMEMNUM * n * sizeof(REAL));
    cudaMemsetAsync(tmpMem, 0, TMPMEMNUM * n * sizeof(REAL), streams[0]);

    // 6. 计算扰动值（CPU计算，与GPU异步）
    float pert = 0;
    if (PERTURB) {
        float norm_A = 0;
        for (int i = 0; i < n; ++i) {
            float tmp = 0;
            for (int j = A_sym.sym_c_ptr[i]; j < A_sym.sym_c_ptr[i+1]; ++j) {
                tmp += abs(A_sym.val[j]);
            }
            norm_A = max(norm_A, tmp);
        }
        pert = 3.45e-4 * norm_A;
        out << "Perturbation value: " << pert << endl;
    }

    // 7. 等待初始化完成
    cudaStreamSynchronize(streams[0]);

    // 8. 多流并行处理所有层级
    cudaEventRecord(start, 0);
    for (int i = 0; i < num_lev; ++i) {
        int lev_size = A_sym.level_ptr[i + 1] - A_sym.level_ptr[i];
        int num_batches = (lev_size + Nstreams - 1) / Nstreams;  // 计算批次

        for (int batch = 0; batch < num_batches; ++batch) {
            // 当前批次处理的列数
            int cols_in_batch = min(Nstreams, lev_size - batch * Nstreams);

            // 为每个流分配任务
            for (int j = 0; j < cols_in_batch; ++j) {
                int global_col_idx = A_sym.level_ptr[i] + batch * Nstreams + j;
                const unsigned currentCol = A_sym.level_idx[global_col_idx];
                const unsigned subMatSize = A_sym.csr_r_ptr[currentCol + 1] - A_sym.csr_diag_ptr[currentCol] - 1;

                // 使用流 j 异步执行
                if (!PERTURB) {
                    RL_onecol_factorizeCurrentCol<<<1, 1024, 0, streams[j]>>>(
                        sym_c_ptr_dev, sym_r_idx_dev, val_dev, l_col_ptr_dev,
                        currentCol, tmpMem, j, n);
                } else {
                    RL_onecol_factorizeCurrentCol_perturb<<<1, 1024, 0, streams[j]>>>(
                        sym_c_ptr_dev, sym_r_idx_dev, val_dev, l_col_ptr_dev,
                        currentCol, tmpMem, j, n, pert);
                }

                if (subMatSize > 0) {
                    RL_onecol_updateSubmat<<<subMatSize, 1024, 0, streams[j]>>>(
                        sym_c_ptr_dev, sym_r_idx_dev, val_dev, csr_c_idx_dev,
                        csr_diag_ptr_dev, currentCol, tmpMem, j, n);
                }

                RL_onecol_cleartmpMem<<<1, 1024, 0, streams[j]>>>(
                    sym_c_ptr_dev, sym_r_idx_dev, l_col_ptr_dev,
                    currentCol, tmpMem, j, n);
            }

            // 同步当前批次的所有流
            for (int j = 0; j < cols_in_batch; ++j) {
                cudaStreamSynchronize(streams[j]);
            }
        }
    }

    // 9. 异步回传结果并记录时间
    cudaMemcpyAsync(&A_sym.val[0], val_dev, nnz * sizeof(REAL), cudaMemcpyDeviceToHost, streams[0]);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    // 10. 释放资源
    for (int j = 0; j < Nstreams; ++j) {
        cudaStreamDestroy(streams[j]);
    }
    cudaFree(sym_c_ptr_dev);
    cudaFree(sym_r_idx_dev);
    cudaFree(val_dev);
    cudaFree(l_col_ptr_dev);
    cudaFree(csr_r_ptr_dev);
    cudaFree(csr_c_idx_dev);
    cudaFree(csr_diag_ptr_dev);
    cudaFree(level_idx_dev);
    cudaFree(tmpMem);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    out << "Total GPU time: " << time << " ms" << endl;

    cudaError_t cudaRet = cudaGetLastError();
    if (cudaRet != cudaSuccess) {
        out << cudaGetErrorName(cudaRet) << endl;
        out << cudaGetErrorString(cudaRet) << endl;
    }

#ifdef GLU_DEBUG
    //check NaN elements
    unsigned err_find = 0;
    for(unsigned i = 0; i < nnz; i++)
        if(isnan(A_sym.val[i]) || isinf(A_sym.val[i])) 
            err_find++;

    if (err_find != 0)
        err << "LU data check: " << " NaN found!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
#endif

    cudaFree(sym_c_ptr_dev);
    cudaFree(sym_r_idx_dev);
    cudaFree(val_dev);

    cudaFree(l_col_ptr_dev);
    cudaFree(csr_c_idx_dev);
    cudaFree(csr_r_ptr_dev);
    cudaFree(csr_diag_ptr_dev);

    cudaFree(level_idx_dev);
}
