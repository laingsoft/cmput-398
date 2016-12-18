#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(
	float *A, float *B, float *C, 
	int numARows, int numAColumns, 
	int numBRows, int numBColumns, 
	int numCRows, int numCColumns) 
{
	// TODO: Insert code to implement matrix multiplication 
	// here you have to use shared memory for this lab.
	// Take a the tiled matrix multiplication. Also we 
	// will be testing the speed up between a basic
	// matrix multiplication and this kernel. To pass 
	// the tests for the tiled matrix multiplication
	// you will need to have the correct output and
	// have a significant speed up over a basic matrix
	// multiplication.
	//
	// HINT: Take a look at the slides
	// HINT: Look at TILE_WIDTH defined above
	__shared__ float sm_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float sm_B[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	float p_value = 0;

	for (int t = 0; t < (numAColumns - 1) / TILE_WIDTH + 1; t++) {
		if (Row < numARows && t * TILE_WIDTH + tx < numAColumns)
			sm_A[ty][tx] = A[Row * numAColumns + t * TILE_WIDTH + tx];
		else
			sm_A[ty][tx] = 0;

		if (t * TILE_WIDTH + ty < numBRows && Col < numBColumns)
			sm_B[ty][tx] = B[(t * TILE_WIDTH + ty) * numBColumns + Col];
		else
			sm_B[ty][tx] = 0;

		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; k++)
			p_value += sm_A[ty][k] * sm_B[k][tx];

		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns)
		C[Row * numCColumns + Col] = p_value;
}

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
	    }                                                                     \
    } while (0)

int main(int argc, char **argv) {
	wbArg_t args;
	float *hostA; // The A matrix
	float *hostB; // The B matrix
	float *hostC; // The output C matrix
	float *deviceA;
	float *deviceB;
	float *deviceC;
	int numARows;    // number of rows in the matrix A
	int numAColumns; // number of columns in the matrix A
	int numBRows;    // number of rows in the matrix B
	int numBColumns; // number of columns in the matrix B
	int numCRows;
	int numCColumns;

	args = wbArg_read(argc, argv);

#if LAB_DEBUG
	std::cout << "Running Tiled Matrix Multiplicaion ..." << std::endl;
#endif

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
		&numAColumns);
	hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
		&numBColumns);
	// TODO: Allocate the hostC matrix
	hostC = (float *)malloc(numARows*numBColumns *sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	// TODO: Set numCRows and numCColumns
	numCRows = numARows;
	numCColumns = numBColumns;

	wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
	wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
	wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

	wbTime_start(GPU, "Allocating GPU memory.");
	// TODO: Allocate GPU memory here
	cudaMalloc(&deviceA, numARows*numAColumns*sizeof(float));
	cudaMalloc(&deviceB, numBRows*numBColumns*sizeof(float));
	cudaMalloc(&deviceC, numCRows*numCColumns*sizeof(float));

	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	// TODO: Copy memory to the GPU here
	cudaMemcpy(deviceA, hostA, (numARows*numAColumns*sizeof(float)), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, (numBRows*numBColumns*sizeof(float)), cudaMemcpyHostToDevice);

	wbTime_stop(GPU, "Copying input memory to the GPU.");

	// TODO: Initialize the grid and block dimensions here
	// Here you will have to use dim3
	//
	// HINT: Take a look at the slides
	// HINT: Look at TILE_WIDTH defined at the top
	//
	// dim3 blockDim( ... );
	// dim3 gridDim( ... );
	int block_size = 16;
	dim3 dimGrid((numCColumns - 1) / block_size + 1, (numCRows - 1) / block_size + 1, 1);
	dim3 dimBlock(block_size, block_size, 1);


	// wbLog(TRACE, "The block dimensions are ", blockDim.x, " x ", blockDim.y);
	// wbLog(TRACE, "The grid dimensions are ", gridDim.x, " x ", gridDim.y);

	wbTime_start(Compute, "Performing CUDA computation");
	// TODO:: Launch the GPU Kernel here
	matrixMultiplyShared << <dimGrid, dimBlock >> >(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	// TODO:: Copy the GPU memory back to the CPU here
	cudaMemcpy(hostC, deviceC, (numCRows*numCColumns*sizeof(float)), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	// TODO:: Free the GPU memory here
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
	
	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostC, numCRows, numCColumns);

	free(hostA);
	free(hostB);
	free(hostC);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}
