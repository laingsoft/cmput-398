// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// + lst[n-1]}


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

#define BLOCK_SIZE 512 //TODO: You can change this

#define wbCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__global__ void blockadd(float* g_aux, float* g_odata, int n){
	int id = blockIdx.x*blockDim.x+threadIdx.x; //Id of the thread within the block

	if (id < n && blockIdx.x > 0){
		g_odata[id] += g_aux[blockIdx.x-1];
	}

}
__global__ void scan(float *g_odata, float *g_idata, float *g_aux, int n){

	int i = blockIdx.x*blockDim.x+threadIdx.x; //id of the thread within the block
	__shared__ float temp[BLOCK_SIZE]; //create the temporary array
	//copy the elements into the temp array
	
	if (i < n){
		temp[threadIdx.x] = g_idata[i];
	}

	for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2){
		__syncthreads();
		float in1 = 0.0;

		if (threadIdx.x >= stride){
			in1 = temp[threadIdx.x - stride];
		}
		__syncthreads();
		temp[threadIdx.x] += in1;
	}

	__syncthreads();

	if (i < n) g_odata[i] = temp[threadIdx.x];


	if (g_aux != NULL && threadIdx.x == blockDim.x - 1){
		g_aux[blockIdx.x] = g_odata[i];
	}

}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
void recursive_scan(float* deviceOutput, float* deviceInput, int numElements){
	int numBlocks = (numElements / BLOCK_SIZE) + 1;
	if (numBlocks == 1){ //If one block, do the scan
		dim3 block(BLOCK_SIZE, 1);
		dim3 grid(numBlocks, 1);

		scan << <grid, block >> >(deviceOutput, deviceInput, NULL, numElements);
		wbCheck(cudaDeviceSynchronize());
	}
	else{ //if more than one, cut the num elements and start again
		float* deviceAux;
		cudaMalloc((void**)&deviceAux, (numBlocks*sizeof(float)));

		float *deviceAuxPass;
		cudaMalloc((void**)&deviceAuxPass, (numBlocks*sizeof(float)));

		dim3 block(BLOCK_SIZE, 1);
		dim3 grid(numBlocks, 1);

		scan << <grid, block >> >(deviceOutput, deviceInput, deviceAux, numElements);
		wbCheck(cudaDeviceSynchronize());


		dim3 grid2(1, 1);
		dim3 block2(numBlocks, 1, 1);

		scan << <grid2, block2 >> >(deviceAuxPass, deviceAux, NULL, numBlocks);
		wbCheck(cudaDeviceSynchronize());

		recursive_scan(deviceAuxPass, deviceAux, numBlocks);

		blockadd << <block2, block >> >(deviceAuxPass, deviceOutput, numElements);
		wbCheck(cudaDeviceSynchronize());

		cudaFree(deviceAux);
		cudaFree(deviceAuxPass);
	}

}

int main(int argc, char **argv) {
	wbArg_t args;
	float *hostInput;  // The input 1D list
	float *hostOutput; // The output list
	float *deviceInput;
	float *deviceOutput;
	int numElements; // number of elements in the list

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
	cudaHostAlloc(&hostOutput, numElements * sizeof(float),
		cudaHostAllocDefault);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The number of input elements in the input is ",
		numElements);

	wbTime_start(GPU, "Allocating GPU memory.");
	wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));

	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Clearing output memory.");
	wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
	wbTime_stop(GPU, "Clearing output memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
		cudaMemcpyHostToDevice));
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	wbTime_start(Compute, "Performing CUDA computation");
	//TODO: Modify this to complete the functionality of the scan on the deivce
	// You should call wbCheck(cudaDeviceSynchronize()); after you finished launching
	// a kernel
	recursive_scan(deviceOutput, deviceInput, numElements);


	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
		cudaMemcpyDeviceToHost));

	

	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostOutput, numElements);

	free(hostInput);
	cudaFreeHost(hostOutput);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}
