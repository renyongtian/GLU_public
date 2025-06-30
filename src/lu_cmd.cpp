#include <iostream>
#include <vector>
#include <set>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <fstream>
#include "symbolic.h"
#include "numeric.h"
#include "Timer.h"
#include "preprocess.h"
#include "nicslu.h"
#include <sstream>

using namespace std;

void help_message()
{
    cout << endl;
    cout << "GLU program V3.0" << endl;
    cout << "Usage: ./lu_cmd -i inputfile" << endl;
    cout << "Additional usage: ./lu_cmd -i inputfile -p" << endl;
    cout << "-p to enable perturbation" << endl;
}

vector<REAL> read_vector_from_file(const std::string& filename, int n) {
    std::vector<REAL> b1(n, 0.); // 初始化为全 0
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int row, col;
        REAL value;
        char comma; // 用于读取逗号（如果文件是 CSV 格式）

        // 解析方式取决于文件格式：
        // 1. 如果是逗号分隔（如 "1, 1, 1.76"）：
        if (iss >> col >> comma >> row >> comma >> value) {
            if (col == 1) { // 只处理列索引为 1 的数据
                if (row >= 1 && row <= n) {
                    b1[row - 1] = value; // C++ 索引从 0 开始
                } else {
                    throw std::runtime_error("行索引超出范围: " + std::to_string(row));
                }
            }
        }

    }
    
    return b1;
}

int main(int argc, char** argv)
{
    Timer t;
    double utime;
    SNicsLU *nicslu;

    char *matrixName = NULL;
    bool PERTURB = false;

    double *ax = NULL, *ax_backup = NULL;
    unsigned int *ai = NULL, *ai_backup = NULL;
    unsigned int *ap = NULL, *ap_backup = NULL;
    unsigned int n, nnz_backup;

    if (argc < 3) {
        help_message();
        return -1;
    }

    for (int i = 1; i < argc;) {
        if (strcmp(argv[i], "-i") == 0) {
            if(i+1 > argc) {
                help_message();
                return -1;
            }
            matrixName = argv[i+1];
            i += 2;
        }
        else if (strcmp(argv[i], "-p") == 0) {
            PERTURB = true;
            i += 1;
        }        
        else {
            help_message();
            return -1;
        }
    }

    nicslu = (SNicsLU *)malloc(sizeof(SNicsLU));

    int err = preprocess(matrixName, nicslu, &ax, &ai, &ap, &ax_backup, &ai_backup, &ap_backup);
    if (err)
    {
        // cout << "Reading matrix error" << endl;
        exit(1);
    }

    n = nicslu->n;
    nnz_backup = nicslu->nnz;

    cout << "Matrix Row: " << n << endl;
    cout << "Original nonzero: " << nicslu->nnz << endl;

    t.start();

    Symbolic_Matrix A_sym(n, cout, cerr);
    // A_sym.fill_in(ai, ap, nnz_backup);
    NicsLU_Factorize2(nicslu);
    A_sym.ExtractLUToSymbolicMatrix(nicslu); 
    t.elapsedUserTime(utime);
    cout << "Symbolic time: " << utime << " ms" << endl;

    t.start();
    A_sym.csr();
    t.elapsedUserTime(utime);
    cout << "CSR time: " << utime << " ms" << endl;

    t.start();
    A_sym.predictLU(ai, ap, ax);
    t.elapsedUserTime(utime);
    cout << "PredictLU time: " << utime << " ms" << endl;

    t.start();
    A_sym.leveling();
    t.elapsedUserTime(utime);
    cout << "Leveling time: " << utime << " ms" << endl;

    LUonDevice(A_sym, cout, cerr, PERTURB);
    vector<REAL> b(n, 1.);
    // b = read_vector_from_file("matrix11_b.txt", n);
    // if  (b.size() != n){
    //     cerr << "Error: Dimension mismatch! b.size() = " << b.size() << ", but n = " << n << endl;
    //     return 1;
    // }

    // int ret = A_sym.updateNicsLUFromSymbolicMatrix(nicslu, A_sym);
    // if (ret != 0) {
    //     cerr << "Failed to update nicslu from symbolic matrix" << endl;
    //     return -1;
    // }
    t.start();
    vector<REAL> x = A_sym.solve_CSR(nicslu, b);
    t.elapsedUserTime(utime);
    cout << "solve_CSR time: " << utime << " ms" << endl;

    // REAL x[n];
    // vector<REAL> tmp_b(n, 1.);
    // const REAL * const b = tmp_b.data();
    // for (int i = 0; i < n; ++i){
    //     x[i] = 0.;
    // }
    // LUonDevice2(A_sym, nicslu, cout, cerr, PERTURB, b, x);

    {
        ofstream x_f("x.dat");
        for (double xx: x)
        x_f << xx << '\n';
    }
    // for(int i = 0; i < A_sym.val.size(); ++i){
    //     cout << A_sym.val[i] << " ";
    // }
    // cout << endl;
    // 残差计算
    // real__t err1 = -1;
    // NicsLU_Residual(n, ax_backup, ai_backup, ap_backup, x.data(), b.data(), &err1, 1, 0);
    // cout << "NicsLU_Residual: " << err1 << endl;
    // 验证计算
    uint__t i, j, end;
    real__t sum, tmp, n1, n2, ni;
    n1 = 0.;
	n2 = 0.;
	ni = 0.;
    for (i = 0; i < n; ++i) {
        sum = 0;
        end = ap_backup[i+1];
        for (j = ap_backup[i]; j < end; ++j) {
            sum += ax_backup[j] * x[ai_backup[j]];
        }
        tmp = sum - b[i];
        if (tmp < 0.) tmp = -tmp;
        n1 += tmp;
        n2 += tmp*tmp;
		if (tmp > ni) ni = tmp;
    }
    
    cout << "Ax-b (1-norm): " << n1 << endl;
    cout << "Ax-b (2-norm): " << sqrt(n2) << endl;
    cout << "Ax-b (infinite-norm): " << ni << endl;
}

