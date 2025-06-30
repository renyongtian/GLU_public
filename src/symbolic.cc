#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "Timer.h"
#include "symbolic.h"
#include <cmath>

using namespace std;
extern size_t g_sp;

void Symbolic_Matrix :: fill_in(unsigned *ai, unsigned *ap, unsigned int nnz1)
{
    sym_c_ptr.push_back(0);
    vector<unsigned> :: iterator it;

    for (unsigned i = 0; i < n; ++i)
    {
        vector<unsigned> tmpcol1;
        vector<unsigned> tmpcol2(ai + ap[i], ai + ap[i+1]);
        sort(tmpcol2.begin(), tmpcol2.end());

        for (unsigned j = 0; j < tmpcol2.size(); ++j)
        {
            unsigned nz_idx = tmpcol2[j];
            if (nz_idx < i)
            {
                auto colhead = sym_r_idx.begin() + l_col_ptr[nz_idx];
                auto colend = sym_r_idx.begin() + sym_c_ptr[nz_idx+1];
                tmpcol1.resize(distance(colhead, colend) + tmpcol2.size());
                it = set_union(colhead, colend, tmpcol2.begin(), tmpcol2.end(), tmpcol1.begin());
                tmpcol1.resize(it - tmpcol1.begin());
                tmpcol2.clear();
                swap(tmpcol1, tmpcol2);
            }
            else
                break;
        }

        for (unsigned j = 0; j < tmpcol2.size(); ++j)
        {
            if (tmpcol2[j] == i)
            {
                l_col_ptr.push_back(j + sym_c_ptr.back());
            }
        }
        sym_c_ptr.push_back(tmpcol2.size() + sym_c_ptr.back());
        sym_r_idx.insert(sym_r_idx.end(), tmpcol2.begin(), tmpcol2.end());
    }

    nnz = sym_c_ptr.back();

    m_out << "Symbolic nonzero: " << nnz << endl;
}

void Symbolic_Matrix :: csr()
{
    csr_r_ptr.push_back(0);
    vector<vector<unsigned> > buffer(n, vector<unsigned>());

    for (unsigned i = 0; i < n; ++i)
    {
        for (unsigned j = sym_c_ptr[i]; j < sym_c_ptr[i+1]; ++j)
        {
            unsigned row_idx = sym_r_idx[j];
            buffer[row_idx].push_back(i);
        }
    }

    for (unsigned i = 0; i < n; ++i)
    {
        for (unsigned j = 0; j < buffer[i].size(); ++j)
        {
            csr_c_idx.push_back(buffer[i][j]);
            if(buffer[i][j] == i)
                csr_diag_ptr.push_back(csr_c_idx.size()-1);
        }
        csr_r_ptr.push_back(csr_r_ptr.back() + buffer[i].size());
    }
}

//Construct val vector (including filled in zeros) of symbolic_matrix
void Symbolic_Matrix :: predictLU (unsigned *ai, unsigned *ap, double *ax)
{
    val.reserve(nnz);
    for (unsigned i = 0; i < n; ++i)
    {
        unsigned *start = ai + ap[i];
        unsigned *end = ai + ap[i+1];
        for (unsigned j = sym_c_ptr[i]; j < sym_c_ptr[i+1]; ++j)
        {
            unsigned *idx = find(start, end, sym_r_idx[j]);
            if (idx != end)
            {
                double* x_idx = ax + distance(ai, idx);
                val.push_back((REAL)*x_idx);
            }
            else
            {
                val.push_back(0);
            }
        }
    }
}


void Symbolic_Matrix :: leveling()
{
    vector<int> inlevel, level_size(n, 0);
    inlevel.reserve(n);

    for (unsigned i = 0; i < n; ++i)
    {
        int max_lv = -1, lv;
        //search dependent columns on the left
        for (unsigned j = sym_c_ptr[i]; j < l_col_ptr[i]; ++j) {
            unsigned nz_idx = sym_r_idx[j]; //Nonzero row in col i, U part

            //L part of col nz_idx exists , U-dependency found
            if (l_col_ptr[nz_idx] + 1 != sym_c_ptr[nz_idx+1]) {
                lv = inlevel[nz_idx];
                if (lv > max_lv)
                    max_lv = lv;
            }
        }
        for (unsigned j = csr_r_ptr[i]; j < csr_diag_ptr[i]; ++j) {
            unsigned nz_idx = csr_c_idx[j];
            lv = inlevel[nz_idx];
            if (lv > max_lv)
                max_lv = lv;
        }        
        lv = max_lv + 1;
        inlevel.push_back(lv);
        ++level_size[lv];
        if (lv > num_lev)
            num_lev = lv;
    }

    ++num_lev;

    level_ptr.reserve(num_lev);
    level_ptr.push_back(0);
    for (int i = 0; i < num_lev; ++i)
        level_ptr.push_back(level_ptr[i] + level_size[i]);

    level_idx.resize(n);
    vector<int> tlen(level_ptr);
    for (unsigned i = 0; i < n; ++i)
        level_idx[tlen[inlevel[i]]++] = i;

    m_out << "Number of levels: " << num_lev << endl;
}

#if GLU_DEBUG
void Symbolic_Matrix::PrintLevel()
{
    for (int i = 0; i < num_lev; ++i) {
        for (int j = level_ptr[i]; j < level_ptr[i + 1]; ++j)
            m_out << level_idx[j] << ' ';
        m_out << '\n';
    }
}

void Symbolic_Matrix::ABFTCalculateCCA()
{
    CCA.assign(n, 0.0);
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = sym_c_ptr[i]; j < sym_c_ptr[i + 1]; ++j)
            CCA[i] += val[j];
    }
}

void Symbolic_Matrix::ABFTCheckResult()
{
     vector<REAL> CCL(n, 1.0);
     for(unsigned i = 0; i < n; i++)
        for(unsigned j = l_col_ptr[i] + 1; j < sym_c_ptr[i + 1]; j++)
            CCL[i] += val[j];

     vector<REAL> CCA_ABFT(n, 0.0);
     for(unsigned i = 0; i < n; i++)
     {
        CCA_ABFT[i] = 0;
        for(unsigned j = sym_c_ptr[i]; j <= l_col_ptr[i]; j++)
            CCA_ABFT[i] += CCL[sym_r_idx[j]] * val[j];
     }

     //Compare CCA and CCA_ABFT
     for(unsigned i = 0; i < n; ++i) {
         if (abs(CCA_ABFT[i] - CCA[i]) > 1e-5) {
             m_err << "Column " << i << ": CCA = " << CCA[i] <<
                 ", CCA_ABFT[i] = " << CCA_ABFT[i] << endl;
             m_err << "More to come..." << endl;
             return;
         }
     }
     m_out << "Results passed ABFT check." << endl;
}
#endif

vector<REAL> Symbolic_Matrix::solve(SNicsLU *nicslu, const vector<REAL> &rhs)
{
    vector<REAL> b(n);
    vector<REAL> x(n);
    unsigned mc64_scale = nicslu->cfgi[1];

    unsigned *rp = nicslu->col_perm;
    unsigned *cp = nicslu->row_perm_inv;
    int *piv = nicslu->pivot;
    double *rows = nicslu->col_scale_perm;
    double *cols = nicslu->row_scale;

    //apply row permutation and row scaling to rhs
    if (mc64_scale)
        for (unsigned j = 0; j < n; ++j) {
            unsigned p = piv[j];
            b[j] = rhs[rp[piv[j]]] * rows[p];
        }
    else
        for (unsigned j = 0; j < n; ++j)
            b[j] = rhs[rp[piv[j]]];

    //left-multiply inv(L)
    for (unsigned j = 0; j < n; ++j) {
        for (unsigned p = l_col_ptr[j] + 1; p < sym_c_ptr[j + 1]; ++p)
            b[sym_r_idx[p]] -= val[p] * b[j];
    }

    //left-multiply inv(U)
    for (int jj = n - 1; jj >= 0; --jj) {
        unsigned diag = l_col_ptr[jj];
        b[jj] /= val[diag];

        for (unsigned p = sym_c_ptr[jj]; p < l_col_ptr[jj]; ++p) {
            b[sym_r_idx[p]] -= val[p] * b[jj];
        }
    }

    //apply col permutation and scaling
    if (mc64_scale)
        for (unsigned j = 0; j < n; ++j)
            x[j] = b[cp[j]] * cols[j];
    else
        for (unsigned j = 0; j < n; ++j)
            x[j] = b[cp[j]];


    return x;
}

vector<REAL> Symbolic_Matrix::solve_CSR(SNicsLU *nicslu, const vector<REAL> &rhs)
{
    vector<REAL> b(n);
    vector<REAL> x(n);
    unsigned mc64_scale = nicslu->cfgi[1];

    unsigned *rp = nicslu->row_perm;
    unsigned *cp = nicslu->col_perm_inv;
    int *piv = nicslu->pivot_inv;
    double *rows = nicslu->row_scale;
    double *cols = nicslu->col_scale_perm;

    // Apply row permutation and scaling to rhs
    if (mc64_scale) {
        for (unsigned j = 0; j < n; ++j) {
            unsigned p = rp[j];
            b[j] = rhs[p] * rows[p];
        }
    } else {
        for (unsigned j = 0; j < n; ++j) {
            b[j] = rhs[rp[j]];
        }
    }

    // Left-multiply inv(L) - lower triangular solve
    for (unsigned j = 0; j < n; ++j) {
        REAL sum = 0.0;
        unsigned diag = l_col_ptr[j];
        
        for (unsigned p = sym_c_ptr[j]; p < l_col_ptr[j]; ++p) {
            sum += val[p] * b[sym_r_idx[p]];
        }
        b[j] = (b[j] - sum) / val[diag];
        // m_out << b[j] << " " << l_col_ptr[j] << " " << sum << endl;
    }

    // Left-multiply inv(U) - upper triangular solve
    for (int jj = n - 1; jj >= 0; --jj) {
        REAL sum = 0.0;
        
        for (unsigned p = l_col_ptr[jj] + 1; p < sym_c_ptr[jj + 1]; ++p) {
            sum += val[p] * b[sym_r_idx[p]];
        }
        
        b[jj] -= sum;
    }

    // Apply column permutation and scaling
    if (mc64_scale) {
        for (unsigned j = 0; j < n; ++j) {
            unsigned p = cp[j];
            x[j] = b[piv[p]] * cols[p];
        }
    } else {
        for (unsigned j = 0; j < n; ++j) {
            x[j] = b[piv[cp[j]]];
        }
    }

    return x;
}

int Symbolic_Matrix::updateNicsLUFromSymbolicMatrix(SNicsLU* nicslu, const Symbolic_Matrix& A_sym) {
    uint__t n, nnz;
    n = A_sym.n;
    nnz = A_sym.nnz;

    // 检查nicslu结构是否已初始化
    if (nicslu == NULL) {
        return -1;
    }

    // 如果nicslu中已有矩阵数据，先释放
    if (nicslu->ax != NULL) {
        free(nicslu->ax);
        nicslu->ax = NULL;
    }
    if (nicslu->ai != NULL) {
        free(nicslu->ai);
        nicslu->ai = NULL;
    }
    if (nicslu->ap != NULL) {
        free(nicslu->ap);
        nicslu->ap = NULL;
    }
    
    // 分配新的内存空间
    nicslu->ax = (double*)malloc(sizeof(double) * nnz);
    nicslu->ai = (unsigned int*)malloc(sizeof(unsigned int) * nnz);
    nicslu->ap = (unsigned int*)malloc(sizeof(unsigned int) * (n + 1));
    
    if (nicslu->ax == NULL || nicslu->ai == NULL || nicslu->ap == NULL) {
        // 分配失败，清理内存
        if (nicslu->ax != NULL) free(nicslu->ax);
        if (nicslu->ai != NULL) free(nicslu->ai);
        if (nicslu->ap != NULL) free(nicslu->ap);
        return -2;
    }

    // 从Symbolic_Matrix中获取CSR格式数据
    // 注意：这里假设sym_c_ptr和sym_r_idx存储的是对称矩阵的CSC格式
    // 需要转换为CSR格式存入nicslu
    
    // 首先填充ap数组（行指针）
    memcpy(nicslu->ap, A_sym.csr_r_ptr.data(), sizeof(unsigned int) * (n + 1));
    
    // 然后填充ai和ax数组（列索引和值）
    if (!A_sym.csr_c_idx.empty() && !A_sym.val.empty()) {
        // 如果有现成的CSR格式数据，直接使用
        memcpy(nicslu->ai, A_sym.csr_c_idx.data(), sizeof(unsigned int) * nnz);
        memcpy(nicslu->ax, A_sym.val.data(), sizeof(double) * nnz);
    } else {
        // 否则需要从对称格式转换
        unsigned int idx = 0;
        for (unsigned int i = 0; i < n; ++i) {
            for (unsigned int p = A_sym.sym_c_ptr[i]; p < A_sym.sym_c_ptr[i + 1]; ++p) {
                nicslu->ai[idx] = A_sym.sym_r_idx[p];
                nicslu->ax[idx] = A_sym.val[p];
                idx++;
            }
        }
    }

    // 更新nicslu的基本信息
    nicslu->n = n;
    nicslu->nnz = nnz;

    return 0;
}

void Symbolic_Matrix::ExtractLUToSymbolicMatrix(SNicsLU *nicslu) {
    sym_c_ptr.clear();
    sym_r_idx.clear();
    l_col_ptr.clear();
    sym_c_ptr.push_back(0);

    uint__t n = nicslu->n;
    uint__t *ulen = nicslu->ulen;
    uint__t *llen = nicslu->llen;
    size_t *up = nicslu->up;
    void *lu_array = nicslu->lu_array;

    for (uint__t i = 0; i < n; ++i) {
        // 收集U部分列索引（已排序）
        uint__t *u_idx = (uint__t *)((byte__t *)lu_array + up[i]);
        vector<uint__t> row(u_idx, u_idx + ulen[i]);

        // 收集L部分列索引（需要排序）
        uint__t *l_idx = (uint__t *)((byte__t *)lu_array + up[i] + ulen[i] * g_sp);
        row.insert(row.end(), l_idx, l_idx + llen[i]);
        sort(row.begin(), row.end());  // 关键：按fill_in要求排序

        // 查找对角线位置
        auto diag_pos = find(row.begin(), row.end(), i);
        if (diag_pos != row.end()) {
            l_col_ptr.push_back(sym_r_idx.size() + (diag_pos - row.begin()));
        }

        // 添加到全局数组
        sym_r_idx.insert(sym_r_idx.end(), row.begin(), row.end());
        sym_c_ptr.push_back(sym_r_idx.size());
    }

    nnz = sym_c_ptr.back();
    m_out << "Symbolic nonzero: " << nnz << endl;
}