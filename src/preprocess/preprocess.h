#ifndef __PREPROCESS__
#define __PREPROCESS__

#include "nics_config.h"
#include "nicslu.h"
#include "type.h"

#define IN__
#define OUT__

#ifdef __cplusplus
extern "C" {
#endif

int preprocess( \
	IN__ char *matrixName, \
    IN__ SNicsLU *nicslu,\
	OUT__ double **ax, \
	OUT__ unsigned int **ai, \
	OUT__ unsigned int **ap, \
	OUT__ double **ax_back, \
	OUT__ unsigned int **ai_back, \
	OUT__ unsigned int **ap_back);


#ifdef __cplusplus
}
#endif

#endif
