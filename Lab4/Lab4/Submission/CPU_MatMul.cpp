#include <wb.h>
#include <stdlib.h>

int main(int argc, char **argv) {
	wbArg_t args;
	float *hostA;		// The A matrix
	float *hostB;		// The B matrix
	float *hostC;		// The output C matrix
	int numARows;		// number of rows in the matrix A
	int numAColumns;	// number of columns in the matrix A
	int numBRows;		// number of rows in the matrix B
	int numBColumns;	// number of columns in the matrix B
	int numCRows;
	int numCColumns;

	args = wbArg_read(argc, argv);

#if LAB_DEBUG
	std::cout << "Running CPU Matrix Multiplicaion ..." << std::endl;
#endif

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
		&numAColumns);
	hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
		&numBColumns);
	// TODO: Allocate the hostC matrix
	hostC = (float *)malloc(numARows*sizeof(float) * numBColumns*sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");
	// TODO: Set numCRows and numCColumns
	numCRows = numARows;
	numCColumns = numBColumns;

	wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
	wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
	wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

	// TODO: Write matrix multiplication

	for (int i = 0; i < numARows; i++){
		for (int j = 0; j < numBColumns; j++){
			float sum = 0.0;
			for (int k = 0; k < numBRows; k++){
				sum = sum + hostA[i* numAColumns + k] * hostB[k * numBColumns + j];
				
			}
			hostC[i*numCColumns + j] = sum;
		}
		
		

	}

	wbSolution(args, hostC, numARows, numBColumns);

	free(hostA);
	free(hostB);
	free(hostC);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}
