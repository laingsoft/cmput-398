#include <wb.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__device__ int binarySearch(const int value, const int *A, const int N)
{
	// TODO: Implement a binary search that returns
	// the index where all values in A are less than
	// the given values

	int first = 0;
	int last = N;
	int middle = (int) (first+last) / 2;
	
	if (value < A[0]){
		return 0;
	}
	else if (value > A[last-1]){
		return last;
	}
	
	while (first <= last){
		middle = (int)(first + last) / 2;

		if (value > A[middle]){
			first = middle+1;
		}
		else if (value < A[middle]){
			last = middle-1;
		}
		else if (value == A[middle]){
			return middle;
		}

	}
	if (value > A[middle]){
		return middle+1;
	}
	else if (value <= A[middle]){
		return middle;
	}
	return 999;



}

__global__ void merge(int *C, const int *A, const int *B, const int N)
{
	// TODO: Merge arrays A and B into C. To make it
	// easier you can assume the following:
	// 
	// 1) A and B are both size N
	//
	// 2) C is size 2N
	//
	// 3) Both A and B are sorted arrays
	//
	// The algorithm should work as follows:
	// Given inputs A and B as follows:
	// A = [0 2 4 10]
	// B = [1 5 7 9]
	//
	// Step 1:
	// Find for each element in array A the index i that
	// would A[i] be inserted in array B or in other 
	// words find the smallest j where A[i] < B[j].
	//
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	// Step 3:
	// Since we know how many elements come before
	// A[i] in array A and we know how many elements 
	// come before A[i] in array B, which is given by
	// are calculation of j. We should know where A[i]
	// is inserted into C, given i and j.
	//
	if (id < N){
		int aindex;
		aindex = binarySearch(A[id], B, N);
		int bindex;
		bindex = binarySearch(B[id]+1, A, N);
		
		C[id + aindex] = A[id];
		while (C[bindex + id] != 0){
			bindex--;
		}
		C[bindex + id] = B[id];
	}
	//C[id+bindex] = B[id];
	// This same logic can be used to find where B[i]
	// should be inserted into C. Although you will have
	// to make a minor change to handle duplicates in A 
	// and B. Or in other words if A and B intersect at 
	// all some values in C will be incorrect. This 
	// occurs because A and B will want to put the values 
	// in the same place in C.
	
}

int main(int argc, char **argv) {
	wbArg_t args;
	int N;
	int* A;
	int* B;
	int* C;
	int* deviceA;
	int* deviceB;
	int* deviceC;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	A = (int *)wbImport(wbArg_getInputFile(args, 0), &N, NULL, "Integer");
	B = (int *)wbImport(wbArg_getInputFile(args, 1), &N, NULL, "Integer");
	C = (int *)malloc(2 * N * sizeof(int));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", N);

	int threads = 256;
	int blocks = N / threads + ((N%threads == 0) ? 0 : 1);

	wbTime_start(GPU, "Allocating GPU memory.");
	cudaMalloc((void **)&deviceA, N * sizeof(int));
	cudaMalloc((void **)&deviceB, N * sizeof(int));
	cudaMalloc((void **)&deviceC, 2 * N * sizeof(int));
	wbTime_stop(GPU, "Allocating GPU memory.");


	wbTime_start(GPU, "Copying input memory to the GPU.");
	cudaMemcpy(deviceA, A, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, B, N * sizeof(int), cudaMemcpyHostToDevice);
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	// Perform on CUDA.
	const dim3 blockSize(threads, 1, 1);
	const dim3 gridSize(blocks, 1, 1);

	wbTime_start(Compute, "Performing CUDA computation");
	merge << < gridSize, blockSize >> >(deviceC, deviceA, deviceB, N);
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	cudaMemcpy(C, deviceC, 2 * N * sizeof(int), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, C, 2*N);

	free(A);
	free(B);
	free(C);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}
