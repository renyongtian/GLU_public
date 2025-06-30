#ifndef _NUMERIC_H_
#define _NUMERIC_H_
#include "symbolic.h"

using namespace std;
void LUonDevice(Symbolic_Matrix &, ostream &, ostream &, bool);
void LUonDevice2(Symbolic_Matrix &, SNicsLU *, ostream &, ostream &, bool , const REAL* rhs = nullptr, REAL* x = nullptr);

#endif
