#include "mex.h"

int discrete(double d);


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[]) {

	// Check the input parameter number
	if (nrhs != 1) {
		mexErrMsgTxt("Wrong number of input arguments.\n");
	}

	if (nlhs > 1) {
		mexErrMsgTxt("Too many output arguments.\n");
	}

#define A_IN prhs[0]
#define B_OUT plhs[0]

	int M = mxGetM(A_IN);
	int N = mxGetN(A_IN);

	// allocate memory for output
	B_OUT = mxCreateDoubleMatrix(M, N, mxREAL);

	// get the pointer of input A and output B
	double *A = mxGetPr(A_IN);
	double *B = mxGetPr(B_OUT);

	for (int i = 0; i < M * N; ++i) {
		B[i] = discrete(A[i]);
	}
}


int discrete(double d) {
	if (d < 1.0 / 3.0) {
		return 0;
	}
	else if (d < 2.0 / 3.0) {
		return 1;
	}
	return 2;
}