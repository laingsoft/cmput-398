#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

#define BLOCK_SIZE 512 //TODO: You can change this

#define wbCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void blockadd(int* g_aux, int* g_odata, int n){
	int id = blockIdx.x*blockDim.x + threadIdx.x; //Id of the thread within the block

	if (id < n && blockIdx.x > 0){
		g_odata[id] += g_aux[blockIdx.x];
	}

}
__global__ void scan(int*g_odata, int *g_idata, int *g_aux, int n){

	int i = blockIdx.x*blockDim.x + threadIdx.x; //id of the thread within the block
	__shared__ int temp[BLOCK_SIZE]; //create the temporary array
	//copy the elements into the temp array

	if (i < n){
		temp[threadIdx.x] = g_idata[i];
	}

	for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2){
		__syncthreads();
		int in1 = 0;

		if (threadIdx.x >= stride){
			in1 = temp[threadIdx.x - stride];
		}
		__syncthreads();
		temp[threadIdx.x] += in1;
	}

	__syncthreads();

	if (i + 1 < n) g_odata[i + 1] = temp[threadIdx.x];


	if (g_aux != NULL && threadIdx.x == blockDim.x - 1){

		g_aux[blockIdx.x] = g_odata[i + 1];
		g_odata[i + 1] = 0;
	}
}

__global__ void split(int*in_d,int *out_d,  int in_size, int bit_shift) {
	 int index = threadIdx.x + blockDim.x * blockIdx.x;
	int bit = 0;
	if (index < in_size) {
		bit = in_d[index] & (1 << bit_shift);
		bit = (bit > 0) ? 1 : 0;
		__syncthreads();
		out_d[index] = 1 - bit;
	}

}

__global__ void indef( int *in_d,  int *rev_bit_d,  int in_size,  int last_input) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	int total_falses = in_d[in_size - 1] + rev_bit_d[in_size-1];
	__syncthreads();
	if (index < in_size) {
		if (rev_bit_d[index] == 0) {
			__syncthreads();
			int val = in_d[index];
			in_d[index] = index - val + total_falses;
		}
	}

}

__global__ void scatter( int *in_d,  int *index_d,  int *out_d,  int in_size) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if (index < in_size) {
		int val = index_d[index];
		__syncthreads();
		if (val < in_size){
			out_d[val] = in_d[index];
		}
	}

}
void sort(int* d_deviceInput, int *d_deviceOutput, int numElements, int* hostinput)
{	
	int numBlocks = (numElements / BLOCK_SIZE) + 1;
	dim3 block(BLOCK_SIZE, 1);
	dim3 grid(numBlocks, 1);
	int *temparr;
	cudaMalloc(&temparr, sizeof(int)*numElements);
	int *temparr1;
	cudaMalloc(&temparr1, sizeof(int)*numElements);
	int *swap;
	int bit;
	for (bit = 0; bit < 15; bit++){
		split<<<grid, block>>>(d_deviceInput, d_deviceOutput, numElements, bit);
		cudaDeviceSynchronize();

		scan << <grid, block >> >(temparr1, d_deviceOutput, NULL, numElements);
		cudaDeviceSynchronize();

		indef << <grid, block >> >(temparr1, d_deviceOutput, numElements, hostinput[numElements - 1]);
		cudaDeviceSynchronize();

		scatter<<<grid, block>>>(d_deviceInput, temparr1, d_deviceOutput, numElements);
		cudaDeviceSynchronize();

		//circle swap
		swap = d_deviceInput;
		d_deviceInput = d_deviceOutput;
		d_deviceOutput = swap;

	}
}



int main(int argc, char **argv) {
	wbArg_t args;
	int *hostInput;  // The input 1D list
	int *hostOutput; // The output list
	int *deviceInput;
	int *deviceOutput;
	int numElements; // number of elements in the list

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (int *)wbImport(wbArg_getInputFile(args, 0), &numElements, "integral_vector");
	cudaHostAlloc(&hostOutput, numElements * sizeof(int), cudaHostAllocDefault);
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The number of input elements in the input is ", numElements);

	wbTime_start(GPU, "Allocating GPU memory.");
	wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(int)));
	wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(int)));
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Clearing output memory.");
	wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(int)));
	wbTime_stop(GPU, "Clearing output memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(int),
		cudaMemcpyHostToDevice));
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	wbTime_start(Compute, "Performing CUDA computation");

	sort(deviceInput, deviceOutput, numElements, hostInput);
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(int),
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
