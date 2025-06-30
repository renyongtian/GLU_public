#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "symbolic.h"
#include <cmath>
#include <cusparse.h>
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
            // printf("tmpMem len: %d, bid: %d, n: %d, ridx: %d, offset: %d, bid*n+ridx: %d", sizeof(tmpMem)/sizeof(REAL), bid, n, ridx, offset, bid*n+ridx);
        }
        offset += blockDim.x;
    }
    // printf("tid: %d , bid: %d ", tid, bid);
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
                        // val_dev[subColElem] -= tmpMem[ridx+n*bid]*s[wid];
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
                        // val_dev[subColElem] -= tmpMem[ridx+n*bid]*s[wid];
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
                // val_dev[subColElem] -= tmpMem[stream * n + ridx] * s;
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

__global__ void permuteScaleKernel(
    const unsigned* rp_dev, const REAL* rows_dev, 
    const REAL* rhs_dev, REAL* b_dev, 
    unsigned n, bool mc64_scale) 
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (unsigned j = 0; j < n; ++j) {
            unsigned p = rp_dev[j];
            b_dev[j] = mc64_scale ? rhs_dev[p] * rows_dev[p] : rhs_dev[p];
            // printf("j: %d , b_dev: %f , rhs_dev: %f\n", j, b_dev[j], rhs_dev[j]);
        }
    }
}

// 下三角求解 (前代)
// __global__ void lowerTriSolveKernel(
//     const unsigned* sym_c_ptr_dev, const unsigned* l_col_ptr_dev,
//     const unsigned* sym_r_idx_dev, const REAL* val_dev,
//     REAL* b_dev, unsigned n)
// {
//     for (int row = n - 1 - blockIdx.x * blockDim.x - threadIdx.x; 
//          row >= 0; 
//          row -= blockDim.x * gridDim.x) {
//         REAL sum = 0.0;
        
//         // 处理非对角元素
//         for (unsigned p = l_col_ptr_dev[row] + 1; p < sym_c_ptr_dev[row + 1]; ++p) {
//             sum += val_dev[p] * b_dev[sym_r_idx_dev[p]];
//         }
        
//         // 对角元素处理
//         b_dev[row] = (b_dev[row] - sum) / val_dev[l_col_ptr_dev[row]];
//     }
//     // if(threadIdx.x == 0){
//     //     for(int i = 0; i < n; ++i){
//     //         printf("i: %d, b: %f\n", i, b_dev[i]);
//     //     }
//     // }
// }

// 上三角求解 (回代)
// __global__ void upperTriSolveKernel(
//     const unsigned* sym_c_ptr_dev, const unsigned* l_col_ptr_dev,
//     const unsigned* sym_r_idx_dev, const REAL* val_dev,
//     REAL* b_dev, unsigned n)
// {
//     for (unsigned row = blockIdx.x * blockDim.x + threadIdx.x; 
//          row < n; 
//          row += blockDim.x * gridDim.x) {

//         REAL sum = 0.0;
//         unsigned diag = l_col_ptr_dev[row];
        
//         // 处理非对角元素
//         for (unsigned p = sym_c_ptr_dev[row]; p < diag; ++p) {
//             sum += val_dev[p] * b_dev[sym_r_idx_dev[p]];
//         }
        
//         // 对角元素处理
//         b_dev[row] = (b_dev[row] - sum) / val_dev[diag];
//     }
//     if(threadIdx.x == 0){
//         for(int i = 0; i < n; ++i){
//             printf("i: %d, b: %f\n", i, b_dev[i]);
//         }
//     }
// }

__global__ void lowerTriSolveKernel(
    const unsigned* sym_c_ptr_dev, const unsigned* l_col_ptr_dev,
    const unsigned* sym_r_idx_dev, const REAL* val_dev,
    REAL* b_dev, unsigned n)
{
    // 使用单个线程处理整个求解过程
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (unsigned j = 0; j < n; ++j) {
            REAL sum = 0.0;
            unsigned diag = l_col_ptr_dev[j];
            
            for (unsigned p = sym_c_ptr_dev[j]; p < l_col_ptr_dev[j]; ++p) {
                sum += val_dev[p] * b_dev[sym_r_idx_dev[p]];
            }
            
            b_dev[j] = (b_dev[j] - sum) / val_dev[diag];

            // printf("j: %d, b: %f\n", j, b_dev[j]);

        }
    }
}

__global__ void upperTriSolveKernel(
    const unsigned* sym_c_ptr_dev, const unsigned* l_col_ptr_dev,
    const unsigned* sym_r_idx_dev, const REAL* val_dev,
    REAL* b_dev, unsigned n)
{
    // 使用单个线程处理整个求解过程
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int jj = n - 1; jj >= 0; --jj) {
            REAL sum = 0.0;
            
            for (unsigned p = l_col_ptr_dev[jj] + 1; p < sym_c_ptr_dev[jj + 1]; ++p) {
                sum += val_dev[p] * b_dev[sym_r_idx_dev[p]];
            }
            
            b_dev[jj] -= sum;
        }
    }
}


__global__ void finalPermuteScaleKernel(
    const unsigned* cp_dev, const int* piv_dev,
    const REAL* cols_dev, REAL* b_dev,
    REAL* x_dev, unsigned n, bool mc64_scale)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (unsigned j = 0; j < n; ++j) {
            unsigned p = cp_dev[j];
            x_dev[j] = mc64_scale ? b_dev[piv_dev[p]] * cols_dev[p] : b_dev[piv_dev[p]];
        }
    }
}

void LUonDevice(Symbolic_Matrix &A_sym, ostream &out, ostream &err, bool PERTURB)
{
    int deviceCount, dev;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp deviceProp;
    dev = 5;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaSetDevice(dev);
    out << "Device " << dev << ": " << deviceProp.name << " has been selected." << endl;

    cudaEvent_t start, stop;
    // cudaEvent_t start_h2d, stop_h2d, start_d2h, stop_d2h;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time, t_h2d, t_d2h;
    unsigned n = A_sym.n;
    unsigned nnz = A_sym.nnz;
    unsigned num_lev = A_sym.num_lev;
    // float time2[num_lev];
    unsigned *sym_c_ptr_dev, *sym_r_idx_dev, *l_col_ptr_dev;
    REAL *val_dev;
    unsigned *csr_r_ptr_dev, *csr_c_idx_dev, *csr_diag_ptr_dev;
    int *level_idx_dev;

    cudaEventRecord(start, 0);

    cudaMalloc((void**)&sym_c_ptr_dev, (n + 1) * sizeof(unsigned));
    cudaMalloc((void**)&sym_r_idx_dev, nnz * sizeof(unsigned));
    cudaMalloc((void**)&val_dev, nnz * sizeof(REAL));
    cudaMalloc((void**)&l_col_ptr_dev, n * sizeof(unsigned));
    cudaMalloc((void**)&csr_r_ptr_dev, (n + 1) * sizeof(unsigned));
    cudaMalloc((void**)&csr_c_idx_dev, nnz * sizeof(unsigned));
    cudaMalloc((void**)&csr_diag_ptr_dev, n * sizeof(unsigned));
    cudaMalloc((void**)&level_idx_dev, n * sizeof(int));
    // cudaEventCreate(&start_h2d);
    // cudaEventCreate(&stop_h2d);
    // cudaEventRecord(start_h2d, 0);
    cudaMemcpy(sym_c_ptr_dev, &(A_sym.sym_c_ptr[0]), (n + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(sym_r_idx_dev, &(A_sym.sym_r_idx[0]), nnz * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(val_dev, &(A_sym.val[0]), nnz * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(l_col_ptr_dev, &(A_sym.l_col_ptr[0]), n * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(csr_r_ptr_dev, &(A_sym.csr_r_ptr[0]), (n + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(csr_c_idx_dev, &(A_sym.csr_c_idx[0]), nnz * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(csr_diag_ptr_dev, &(A_sym.csr_diag_ptr[0]), n * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(level_idx_dev, &(A_sym.level_idx[0]), n * sizeof(int), cudaMemcpyHostToDevice);
    // cudaEventRecord(stop_h2d, 0);
    // cudaEventSynchronize(stop_h2d);
    // cudaEventElapsedTime(&t_h2d, start_h2d, stop_h2d);
    // out << "cudaMemcpyHostToDevice time: " << t_h2d << " ms" << endl;
    REAL* tmpMem;
    unsigned TMPMEMNUM;
    size_t MaxtmpMemSize;
    {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        //Leave at least 4GB free, smaller size does not work for unknown reason
        MaxtmpMemSize = free - 4ull * 1024ull * 1024ull * 1024ull;
    }
    // Use size of first level to estimate a good tmpMem size
    const size_t GoodtmpMemChoice = sizeof(REAL) * size_t(n) * size_t(A_sym.level_ptr[1]);
    if (GoodtmpMemChoice < MaxtmpMemSize)
        TMPMEMNUM = A_sym.level_ptr[1];
    else
        TMPMEMNUM = MaxtmpMemSize / n / sizeof(REAL);
    // out << "GoodtmpMemChoice: " << GoodtmpMemChoice << 
    //      "\nMaxtmpMemSize:    " << MaxtmpMemSize << 
    //      "\nTMPMEMNUM:        " << TMPMEMNUM <<endl;

    cudaMalloc((void**)&tmpMem, TMPMEMNUM*n*sizeof(REAL));
    cudaMemset(tmpMem, 0, TMPMEMNUM*n*sizeof(REAL));

    int Nstreams = 16;
    cudaStream_t streams[Nstreams];
    for (int j = 0; j < Nstreams; ++j)
        cudaStreamCreate(&streams[j]);
    // out << TMPMEMNUM*n << endl;
    // calculate 1-norm of A and perturbation value for perturbation
    float pert = 0;
    if (PERTURB)
    {
        float norm_A = 0;
        for (int i = 0; i < n; ++i)
        {
            float tmp = 0;
            for (int j = A_sym.sym_c_ptr[i]; j < A_sym.sym_c_ptr[i+1]; ++j)
                tmp += abs(A_sym.val[j]);
            if (norm_A < tmp)
                norm_A = tmp;
        }
        pert = 3.45e-4 * norm_A;
        out << "Gaussian elimination with static pivoting (GESP)..." << endl;
        out << "1-Norm of A matrix is " << norm_A << ", Perturbation value is " << pert << endl;
    }

    // #pragma unroll 32
    // cudaEvent_t start2[num_lev], stop2[num_lev];
    
    // out << num_lev << endl;
    for (int i = 0; i < num_lev; ++i)
    {
        // cudaEventCreate(&start2[i]);
        // cudaEventCreate(&stop2[i]);
        // cudaEventRecord(start2[i], 0);
        int lev_size = A_sym.level_ptr[i + 1] - A_sym.level_ptr[i];

        if (lev_size > 896) { //3584 / 4
            unsigned WarpsPerBlock = 2;
            dim3 dimBlock(WarpsPerBlock * 32, 1);
            size_t MemSize = WarpsPerBlock * sizeof(REAL);

            unsigned j = 0;
            while(lev_size > 0) {
                unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
                dim3 dimGrid(restCol, 1);
                if (!PERTURB)
                    RL<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM);
                else
                    RL_perturb<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM,
                                        pert);
                j++;
                lev_size -= TMPMEMNUM;
            }
        }
        else if (lev_size > 448) {
            unsigned WarpsPerBlock = 4;
            dim3 dimBlock(WarpsPerBlock * 32, 1);
            size_t MemSize = WarpsPerBlock * sizeof(REAL);

            unsigned j = 0;
            while(lev_size > 0) {
                unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
                dim3 dimGrid(restCol, 1);
                if (!PERTURB)
                    RL<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM);
                else
                    RL_perturb<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM,
                                        pert);
                j++;
                lev_size -= TMPMEMNUM;
            }
        }
        else if (lev_size > Nstreams) {
            dim3 dimBlock(1024, 1);
            size_t MemSize = 32 * sizeof(REAL);
            unsigned j = 0;
            while(lev_size > 0) {
                unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
                dim3 dimGrid(restCol, 1);
                if (!PERTURB)
                    RL<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM);
                else
                    RL_perturb<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM,
                                        pert);
                j++;
                lev_size -= TMPMEMNUM;
            }
        }
        else { // "Big" levels
            for (unsigned offset = 0; offset < lev_size; offset += Nstreams) {
                for (int j = 0; j < Nstreams; j++) {
                    if (j + offset < lev_size) {
                        const unsigned currentCol = A_sym.level_idx[A_sym.level_ptr[i] + j + offset];
                        const unsigned currentLColSize = A_sym.sym_c_ptr[currentCol + 1]
                            - A_sym.l_col_ptr[currentCol] - 1;
                        const unsigned subMatSize = A_sym.csr_r_ptr[currentCol + 1]
                            - A_sym.csr_diag_ptr[currentCol] - 1;

                        if (!PERTURB)
                            RL_onecol_factorizeCurrentCol<<<1, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                    sym_r_idx_dev,
                                                    val_dev,
                                                    l_col_ptr_dev,
                                                    currentCol,
                                                    tmpMem,
                                                    j,
                                                    n);
                        else
                            RL_onecol_factorizeCurrentCol_perturb<<<1, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                    sym_r_idx_dev,
                                                    val_dev,
                                                    l_col_ptr_dev,
                                                    currentCol,
                                                    tmpMem,
                                                    j,
                                                    n,
                                                    pert);
                        if (subMatSize > 0)
                            RL_onecol_updateSubmat<<<subMatSize, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                        sym_r_idx_dev,
                                                        val_dev,
                                                        csr_c_idx_dev,
                                                        csr_diag_ptr_dev,
                                                        currentCol,
                                                        tmpMem,
                                                        j,
                                                        n);
                        RL_onecol_cleartmpMem<<<1, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                    sym_r_idx_dev,
                                                    l_col_ptr_dev,
                                                    currentCol,
                                                    tmpMem,
                                                    j,
                                                    n);
                    }
                }
            }
        }
        // cudaEventRecord(stop2[i], 0);
        // cudaEventSynchronize(stop2[i]);
        // cudaEventElapsedTime(&time2[i], start2[i], stop2[i]);
        // out << "num_lev: " << i << ",  Total copy time: " << time2[i] << " ms" << endl;
        cudaDeviceSynchronize();
        // fopen("./data.txt", "w")

    }

    // cudaEvent_t start2, stop2;
    // cudaEventCreate(&start2);
    // cudaEventCreate(&stop2);
    // cudaEventRecord(start2, 0);

    //copy LU val back to main mem
    // cudaEventCreate(&start_d2h);
    // cudaEventCreate(&stop_d2h);
    // cudaEventRecord(start_d2h, 0);
    cudaMemcpy(&(A_sym.val[0]), val_dev, nnz * sizeof(REAL), cudaMemcpyDeviceToHost);
    // cudaEventRecord(stop_d2h, 0);
    // cudaEventSynchronize(stop_d2h);
    // cudaEventElapsedTime(&t_d2h, start_d2h, stop_d2h);
    // out << "cudaMemcpyDeviceToHost time: " << t_d2h << " ms" << endl;

    // cudaEventRecord(stop2, 0);
    // cudaEventSynchronize(stop2);
    // cudaEventElapsedTime(&time2, start2, stop2);
    // out << "Total copy time: " << time2 << " ms" << endl;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

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

void LUonDevice2(Symbolic_Matrix &A_sym, SNicsLU *nicslu, ostream &out, ostream &err,
                bool PERTURB, const REAL* rhs = nullptr, REAL* x = nullptr)
{
    int deviceCount, dev;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp deviceProp;
    dev = 5;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaSetDevice(dev);
    out << "Device " << dev << ": " << deviceProp.name << " has been selected." << endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;
    unsigned n = A_sym.n;
    unsigned nnz = A_sym.nnz;
    unsigned num_lev = A_sym.num_lev;
    // float time2[num_lev];
    unsigned *sym_c_ptr_dev, *sym_r_idx_dev, *l_col_ptr_dev;
    REAL *val_dev;
    unsigned *csr_r_ptr_dev, *csr_c_idx_dev, *csr_diag_ptr_dev;
    int *level_idx_dev;

    cudaEventRecord(start, 0);

    cudaMalloc((void**)&sym_c_ptr_dev, (n + 1) * sizeof(unsigned));
    cudaMalloc((void**)&sym_r_idx_dev, nnz * sizeof(unsigned));
    cudaMalloc((void**)&val_dev, nnz * sizeof(REAL));
    cudaMalloc((void**)&l_col_ptr_dev, n * sizeof(unsigned));
    cudaMalloc((void**)&csr_r_ptr_dev, (n + 1) * sizeof(unsigned));
    cudaMalloc((void**)&csr_c_idx_dev, nnz * sizeof(unsigned));
    cudaMalloc((void**)&csr_diag_ptr_dev, n * sizeof(unsigned));
    cudaMalloc((void**)&level_idx_dev, n * sizeof(int));

    cudaMemcpy(sym_c_ptr_dev, &(A_sym.sym_c_ptr[0]), (n + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(sym_r_idx_dev, &(A_sym.sym_r_idx[0]), nnz * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(val_dev, &(A_sym.val[0]), nnz * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(l_col_ptr_dev, &(A_sym.l_col_ptr[0]), n * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(csr_r_ptr_dev, &(A_sym.csr_r_ptr[0]), (n + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(csr_c_idx_dev, &(A_sym.csr_c_idx[0]), nnz * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(csr_diag_ptr_dev, &(A_sym.csr_diag_ptr[0]), n * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(level_idx_dev, &(A_sym.level_idx[0]), n * sizeof(int), cudaMemcpyHostToDevice);

    REAL* tmpMem;
    unsigned TMPMEMNUM;
    size_t MaxtmpMemSize;
    {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        //Leave at least 4GB free, smaller size does not work for unknown reason
        MaxtmpMemSize = free - 4ull * 1024ull * 1024ull * 1024ull;
    }
    // Use size of first level to estimate a good tmpMem size
    const size_t GoodtmpMemChoice = sizeof(REAL) * size_t(n) * size_t(A_sym.level_ptr[1]);
    if (GoodtmpMemChoice < MaxtmpMemSize)
        TMPMEMNUM = A_sym.level_ptr[1];
    else
        TMPMEMNUM = MaxtmpMemSize / n / sizeof(REAL);
    // out << "GoodtmpMemChoice: " << GoodtmpMemChoice << 
    //      "\nMaxtmpMemSize:    " << MaxtmpMemSize << 
    //      "\nTMPMEMNUM:        " << TMPMEMNUM <<endl;

    cudaMalloc((void**)&tmpMem, TMPMEMNUM*n*sizeof(REAL));
    cudaMemset(tmpMem, 0, TMPMEMNUM*n*sizeof(REAL));

    int Nstreams = 16;
    cudaStream_t streams[Nstreams];
    for (int j = 0; j < Nstreams; ++j)
        cudaStreamCreate(&streams[j]);
    // out << TMPMEMNUM*n << endl;
    // calculate 1-norm of A and perturbation value for perturbation
    float pert = 0;
    if (PERTURB)
    {
        float norm_A = 0;
        for (int i = 0; i < n; ++i)
        {
            float tmp = 0;
            for (int j = A_sym.sym_c_ptr[i]; j < A_sym.sym_c_ptr[i+1]; ++j)
                tmp += abs(A_sym.val[j]);
            if (norm_A < tmp)
                norm_A = tmp;
        }
        pert = 3.45e-4 * norm_A;
        out << "Gaussian elimination with static pivoting (GESP)..." << endl;
        out << "1-Norm of A matrix is " << norm_A << ", Perturbation value is " << pert << endl;
    }

    // #pragma unroll 32
    // cudaEvent_t start2[num_lev], stop2[num_lev];
    
    // out << num_lev << endl;
    for (int i = 0; i < num_lev; ++i)
    {
        // cudaEventCreate(&start2[i]);
        // cudaEventCreate(&stop2[i]);
        // cudaEventRecord(start2[i], 0);
        int lev_size = A_sym.level_ptr[i + 1] - A_sym.level_ptr[i];

        if (lev_size > 896) { //3584 / 4
            unsigned WarpsPerBlock = 2;
            dim3 dimBlock(WarpsPerBlock * 32, 1);
            size_t MemSize = WarpsPerBlock * sizeof(REAL);

            unsigned j = 0;
            while(lev_size > 0) {
                unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
                dim3 dimGrid(restCol, 1);
                if (!PERTURB)
                    RL<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM);
                else
                    RL_perturb<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM,
                                        pert);
                j++;
                lev_size -= TMPMEMNUM;
            }
        }
        else if (lev_size > 448) {
            unsigned WarpsPerBlock = 4;
            dim3 dimBlock(WarpsPerBlock * 32, 1);
            size_t MemSize = WarpsPerBlock * sizeof(REAL);

            unsigned j = 0;
            while(lev_size > 0) {
                unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
                dim3 dimGrid(restCol, 1);
                if (!PERTURB)
                    RL<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM);
                else
                    RL_perturb<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM,
                                        pert);
                j++;
                lev_size -= TMPMEMNUM;
            }
        }
        else if (lev_size > Nstreams) {
            dim3 dimBlock(1024, 1);
            size_t MemSize = 32 * sizeof(REAL);
            unsigned j = 0;
            while(lev_size > 0) {
                unsigned restCol = lev_size > TMPMEMNUM ? TMPMEMNUM : lev_size;
                dim3 dimGrid(restCol, 1);
                if (!PERTURB)
                    RL<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM);
                else
                    RL_perturb<<<dimGrid, dimBlock, MemSize>>>(sym_c_ptr_dev,
                                        sym_r_idx_dev,
                                        val_dev,
                                        l_col_ptr_dev,
                                        csr_r_ptr_dev,
                                        csr_c_idx_dev,
                                        csr_diag_ptr_dev,
                                        level_idx_dev,
                                        tmpMem,
                                        n,
                                        A_sym.level_ptr[i],
                                        j*TMPMEMNUM,
                                        pert);
                j++;
                lev_size -= TMPMEMNUM;
            }
        }
        else { // "Big" levels
            for (unsigned offset = 0; offset < lev_size; offset += Nstreams) {
                for (int j = 0; j < Nstreams; j++) {
                    if (j + offset < lev_size) {
                        const unsigned currentCol = A_sym.level_idx[A_sym.level_ptr[i] + j + offset];
                        const unsigned currentLColSize = A_sym.sym_c_ptr[currentCol + 1]
                            - A_sym.l_col_ptr[currentCol] - 1;
                        const unsigned subMatSize = A_sym.csr_r_ptr[currentCol + 1]
                            - A_sym.csr_diag_ptr[currentCol] - 1;

                        if (!PERTURB)
                            RL_onecol_factorizeCurrentCol<<<1, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                    sym_r_idx_dev,
                                                    val_dev,
                                                    l_col_ptr_dev,
                                                    currentCol,
                                                    tmpMem,
                                                    j,
                                                    n);
                        else
                            RL_onecol_factorizeCurrentCol_perturb<<<1, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                    sym_r_idx_dev,
                                                    val_dev,
                                                    l_col_ptr_dev,
                                                    currentCol,
                                                    tmpMem,
                                                    j,
                                                    n,
                                                    pert);
                        if (subMatSize > 0)
                            RL_onecol_updateSubmat<<<subMatSize, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                        sym_r_idx_dev,
                                                        val_dev,
                                                        csr_c_idx_dev,
                                                        csr_diag_ptr_dev,
                                                        currentCol,
                                                        tmpMem,
                                                        j,
                                                        n);
                        RL_onecol_cleartmpMem<<<1, 1024, 0, streams[j]>>>(sym_c_ptr_dev,
                                                    sym_r_idx_dev,
                                                    l_col_ptr_dev,
                                                    currentCol,
                                                    tmpMem,
                                                    j,
                                                    n);
                    }
                }
            }
        }
        // cudaEventRecord(stop2[i], 0);
        // cudaEventSynchronize(stop2[i]);
        // cudaEventElapsedTime(&time2[i], start2[i], stop2[i]);
        // out << "num_lev: " << i << ",  Total copy time: " << time2[i] << " ms" << endl;
        cudaDeviceSynchronize();
        // fopen("./data.txt", "w")

    }

    // cudaEvent_t start2, stop2;
    // cudaEventCreate(&start2);
    // cudaEventCreate(&stop2);
    // cudaEventRecord(start2, 0);

    //copy LU val back to main mem
    cudaMemcpy(&(A_sym.val[0]), val_dev, nnz * sizeof(REAL), cudaMemcpyDeviceToHost);

    // cudaEventRecord(stop2, 0);
    // cudaEventSynchronize(stop2);
    // cudaEventElapsedTime(&time2, start2, stop2);
    // out << "Total copy time: " << time2 << " ms" << endl;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    out << "Total GPU time: " << time << " ms" << endl;
    cudaDeviceSynchronize();
    // 如果传入了右端项，则进行求解
    if (rhs != nullptr && x != nullptr) {
        cudaEvent_t solve_start, solve_stop;
        cudaEventCreate(&solve_start);
        cudaEventCreate(&solve_stop);
        float solve_time;
        cudaEventRecord(solve_start, 0);

        // 分配求解所需的设备内存
        REAL *b_dev, *rhs_dev, *x_dev;
        unsigned *rp_dev, *cp_dev;
        int *piv_dev;
        REAL *rows_dev, *cols_dev;
        cudaMalloc(&rhs_dev, n * sizeof(REAL));
        cudaMalloc(&x_dev, n * sizeof(REAL));

        cudaMalloc((void**)&rp_dev, n * sizeof(unsigned));
        cudaMalloc((void**)&cp_dev, n * sizeof(unsigned));
        cudaMalloc((void**)&piv_dev, n * sizeof(int));
        cudaMalloc((void**)&rows_dev, n * sizeof(REAL));
        cudaMalloc((void**)&cols_dev, n * sizeof(REAL));
        
        // 从nicslu结构体复制数据到设备
        cudaMemcpy(rp_dev, nicslu->row_perm, n * sizeof(unsigned), cudaMemcpyHostToDevice);
        cudaMemcpy(cp_dev, nicslu->col_perm_inv, n * sizeof(unsigned), cudaMemcpyHostToDevice);
        cudaMemcpy(piv_dev, nicslu->pivot_inv, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(rows_dev, nicslu->row_scale, n * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(cols_dev, nicslu->col_scale_perm, n * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(rhs_dev, rhs, n * sizeof(REAL), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&b_dev, n * sizeof(REAL));
        cudaMemset(b_dev, 0, n*sizeof(REAL));
        cudaMemset(x_dev, 0, n*sizeof(REAL));
        unsigned blockSize = 1;
        unsigned gridSize = 1;//(n + blockSize - 1) / blockSize;

        // 配置内核启动参数
        // dim3 blockSize(256);
        // dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
        // 获取配置标志
        unsigned mc64_scale = nicslu->cfgi[1];
        // 1. 应用行置换和缩放
        permuteScaleKernel<<<gridSize, blockSize>>>(
            rp_dev, rows_dev, rhs_dev, b_dev, n, mc64_scale);
        // cudaDeviceSynchronize();
        // cudaMemcpy(x, b_dev, n * sizeof(REAL), cudaMemcpyDeviceToHost);
         // 串行调用kernel，每次处理一行
        // for (unsigned i = 0; i < n; ++i) {
        //     lowerTriSolveKernel<<<1, 1>>>(sym_c_ptr_dev, l_col_ptr_dev, sym_r_idx_dev, val_dev, b_dev, n, i);
        //     cudaDeviceSynchronize();
        // }
        // for (unsigned i = 0; i < n; ++i) {
        //     upperTriSolveKernel<<<1, 1>>>(sym_c_ptr_dev, l_col_ptr_dev, sym_r_idx_dev, val_dev, b_dev, n, i);
        //     cudaDeviceSynchronize();
        // }

    
        // 2. 下三角求解
        lowerTriSolveKernel<<<gridSize, blockSize>>>(
            sym_c_ptr_dev, l_col_ptr_dev, sym_r_idx_dev, val_dev, b_dev, n);
            // cudaDeviceSynchronize();
            // 3. 上三角求解
        upperTriSolveKernel<<<gridSize, blockSize>>>(
                sym_c_ptr_dev, l_col_ptr_dev, sym_r_idx_dev, val_dev, b_dev, n);
        // 4. 应用列置换和缩放
        finalPermuteScaleKernel<<<gridSize, blockSize>>>(
            cp_dev, piv_dev, cols_dev, b_dev, x_dev, n, mc64_scale);

        cudaDeviceSynchronize();
        cudaMemcpy(x, x_dev, n * sizeof(REAL), cudaMemcpyDeviceToHost);
 
        // 记录求解时间
        cudaEventRecord(solve_stop, 0);
        cudaEventSynchronize(solve_stop);
        cudaEventElapsedTime(&solve_time, solve_start, solve_stop);
        out << "GPU solve time: " << solve_time << " ms" << endl;

        // 释放临时内存
        cudaFree(rhs_dev);
        cudaFree(x_dev);
        cudaFree(b_dev);
        cudaFree(rp_dev);
        cudaFree(cp_dev);
        cudaFree(piv_dev);
        cudaFree(rows_dev);
        cudaFree(cols_dev);
    }

    cudaError_t cudaRet = cudaGetLastError();
    if (cudaRet != cudaSuccess) {
        out << cudaGetErrorName(cudaRet) << endl;
        out << cudaGetErrorString(cudaRet) << endl;
    }

    cudaFree(sym_c_ptr_dev);
    cudaFree(sym_r_idx_dev);
    cudaFree(val_dev);

    cudaFree(l_col_ptr_dev);
    cudaFree(csr_c_idx_dev);
    cudaFree(csr_r_ptr_dev);
    cudaFree(csr_diag_ptr_dev);

    cudaFree(level_idx_dev);
}
