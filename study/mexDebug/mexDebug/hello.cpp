# include "mex.h"

// number/pointer left hand side
// number/pointer right hand side
void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[]) {
	//mexErrMsgTxt("error message\n");
	mexPrintf("hello\n");
}

//#include <stdio.h>
//
//int main() {
//	printf("hello\n");
//	return 0;
//}