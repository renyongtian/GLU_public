#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"
#include "nicslu.h"
#include "nicslu_util.h"

int my_DumpA(SNicsLU *nicslu, double **ax, unsigned int **ai, unsigned int **ap)
{
    uint__t n, nnz;
    double *ax0;
    unsigned int *ai0, *ap0;
    uint__t *rowperm, *pinv, *piv, oldrow, start, end;
    uint__t i, j, p;

    if (NULL == nicslu || NULL == ax || NULL == ai || NULL == ap)
    {
        return -1;
    }

    if (*ax != NULL)
    {
        free(*ax);
        *ax = NULL;
    }
    if (*ai != NULL)
    {
        free(*ai);
        *ai = NULL;
    }
    if (*ap != NULL)
    {
        free(*ap);
        *ap = NULL;
    }

    // printf("row_perm: ");
    // for(uint__t i=0; i<nicslu->n; ++i){
    //     printf("%d ", nicslu->row_perm[i]);
    // }
    // printf("\nrow_perm_inv: ");
    // for(uint__t i=0; i<nicslu->n; ++i){
    //     printf("%d ", nicslu->row_perm_inv[i]);
    // }
    // printf("\ncol_perm: ");
    // for(uint__t i=0; i<nicslu->n; ++i){
    //     printf("%d ", nicslu->col_perm[i]);
    // }
    // printf("\ncol_perm_inv: ");
    // for(uint__t i=0; i<nicslu->n; ++i){
    //     printf("%d ", nicslu->col_perm_inv[i]);
    // }
    // printf("\n");
    n = nicslu->n;
    nnz = nicslu->nnz;
    ax0 = nicslu->ax;
    ai0 = nicslu->ai;
    ap0 = nicslu->ap;
    rowperm = nicslu->row_perm;/*row_perm[i]=j-->row i in the permuted matrix is row j in the original matrix*/
    pinv = (uint__t *)nicslu->pivot_inv;/*pivot_inv[i]=j-->column i is the jth pivot column*/
    piv = (uint__t *)nicslu->pivot;

    //generate pivot and pivot_inv for function NicsLU_DumpA
    for (i = 0; i < n; ++i)
    {
        pinv[i] = i;
        piv[i] = i;
    }

    *ax = (double *)malloc(sizeof(double)*nnz);
    *ai = (unsigned int *)malloc(sizeof(unsigned int)*nnz);
    *ap = (unsigned int *)malloc(sizeof(unsigned int)*(n+1));
    // *ax = (real__t *)malloc(sizeof(real__t)*nnz);
    // *ai = (uint__t *)malloc(sizeof(uint__t)*nnz);
    // *ap = (uint__t *)malloc(sizeof(uint__t)*(n+1));

    if (NULL == *ax || NULL == *ai || NULL == *ap)
    {
        goto FAIL;
    }
    (*ap)[0] = 0;

    p = 0;
    for (i=0; i<n; ++i)
    {
        oldrow = rowperm[i];
        start = ap0[oldrow];
        end = ap0[oldrow+1];
        (*ap)[i+1] = (*ap)[i] + end - start;

        for (j=start; j<end; ++j)
        {
            (*ax)[p] = ax0[j];
            (*ai)[p++] = pinv[ai0[j]];
        }
    }

    return 0;

FAIL:
    if (*ax != NULL)
    {
        free(*ax);
        *ax = NULL;
    }
    if (*ai != NULL)
    {
        free(*ai);
        *ai = NULL;
    }
    if (*ap != NULL)
    {
        free(*ap);
        *ap = NULL;
    }
    return -2;
}

int preprocess(char *matrixName, SNicsLU *nicslu, double **ax, unsigned int **ai, unsigned int **ap, \
               double **ax_backup, unsigned int **ai_backup, unsigned int **ap_backup)
{
    int ret;
    uint__t *n, *nnz;

    // nicslu = (SNicsLU *)malloc(sizeof(SNicsLU));
    NicsLU_Initialize(nicslu);

    n = (uint__t *)malloc(sizeof(uint__t));
    nnz = (uint__t *)malloc(sizeof(uint__t));

    printf("Reading matrix...\n");

    ret = NicsLU_ReadTripletRowToSparse(matrixName, n, nnz, ax, ai, ap);
    if (ret == NICSLU_MATRIX_INVALID)
    {    
        printf("Read invalid matrix\n");
        goto EXIT;
    }
    else if (ret == NICSLU_FILE_CANNOT_OPEN) 
    {    
        printf("File cannot open\n");
        goto EXIT;
    }
    else if (ret != NICS_OK) 
    {    
        printf("Open file error\n");
        goto EXIT;
    }

    // 备份原始数据（深拷贝）
    *ax_backup = (double *)malloc((*nnz) * sizeof(double));
    *ai_backup = (unsigned int *)malloc((*nnz) * sizeof(unsigned int));
    *ap_backup = (unsigned int *)malloc((*n + 1) * sizeof(unsigned int));
    
    memcpy(*ax_backup, *ax, *nnz * sizeof(double));
    memcpy(*ai_backup, *ai, *nnz * sizeof(unsigned int));
    memcpy(*ap_backup, *ap, (*n + 1) * sizeof(unsigned int));

    NicsLU_CreateMatrix(nicslu, *n, *nnz, *ax, *ai, *ap);
    nicslu->cfgi[0] = 1;
    nicslu->cfgf[1] = 0;

    printf("Preprocessing matrix...\n");

    // for(uint__t i=0; i < nicslu->nnz; ++i){
    //     printf("%lf ", nicslu->ax[i]);
    // }
    // printf("\n");
    NicsLU_Analyze(nicslu);
    printf("Preprocessing time: %f ms\n", nicslu->stat[0] * 1000);

    my_DumpA(nicslu, ax, ai, ap);
    //rp = nicslu->col_perm;
    //cp = nicslu->row_perm_inv;
    //piv = nicslu->pivot;
    //rows = nicslu->col_scale_perm;
    //cols = nicslu->row_scale;
    //cscale = nicslu->cscale;

    return 0;
EXIT:
    NicsLU_Destroy(nicslu);
    free(*ax);
    free(*ai);
    free(*ap);
    free(nicslu);
    return -1;

}
