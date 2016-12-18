#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
		    }                                                                     \
      } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define clamp(x) (min(max((x), 0.0), 1.0))

__global__ void convolution(float *I, const float *M,
	float *P, int channels, int width, int height) {
	int Col = blockIdx.x * blockDim.x + threadIdx.x; //Get the Columns
	int Row = blockIdx.y * blockDim.y + threadIdx.y; //Get the Rows

	int maskWidth = Mask_width;
	int maskRadius = Mask_radius;

	if (Col < width  && Row < height){ //If the Threads are within the image (eg 64x64)
		for (int channel = 0; channel < channels; channel++){ //for each integer in the channel
			float accum = 0;

			int start_col = Col +maskRadius;
			int start_row = Row +maskRadius;	

			for (int j = 0; j < maskWidth; ++j){ //Y aspect of the mask offset
				for (int k = 0; k < maskWidth; ++k){ //X aspect of the mask offset
					int curRow = start_row - j;
					int curCol = start_col - k;
					if (curRow >= -1 && curRow < height &&	//If the X is within the mask
						curCol >= -1 && curCol < width){	//If the Y is within the mask
						float pixel = I[(curRow*width + curCol)*channels + channel];
						float mask = M[j*maskWidth + k];
						accum += pixel * mask;

					}
				}
			}
			P[(Row*width + Col)*channels + channel] = clamp(accum);
		}

		


	}
}

int main(int argc, char *argv[]) {
	wbArg_t arg;
	int maskRows;
	int maskColumns;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	char *inputMaskFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *hostMaskData;
	float *deviceInputImageData;
	float *deviceOutputImageData;
	float *deviceMaskData;

	arg = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(arg, 0);
	inputMaskFile = wbArg_getInputFile(arg, 1);

	inputImage = wbImport(inputImageFile);
	hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

	assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
	assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);

	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	//TODO: INSERT CODE HERE
	cudaMalloc(&deviceInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float));
	cudaMalloc(&deviceOutputImageData, imageWidth*imageHeight*imageChannels*sizeof(float));
	cudaMalloc(&deviceMaskData, maskRows*maskColumns*sizeof(float));

	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	//TODO: INSERT CODE HERE
	cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData, hostMaskData, maskRows*maskColumns*sizeof(float) , cudaMemcpyHostToDevice);

	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");
	//TODO: INSERT CODE HERE
	int blocksize = 16;
	dim3 dimGrid(ceil(imageWidth+1 / blocksize + 1), ceil(imageHeight+1 / blocksize + 1));
	dim3 dimBlock(blocksize, blocksize, 1);

	convolution << <dimGrid, dimBlock>> >(deviceInputImageData, deviceMaskData,
											deviceOutputImageData, imageChannels,
											imageWidth, imageHeight);


	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Doing the computation on the GPU");

	wbTime_start(Copy, "Copying data from the GPU");
	//TODO: INSERT CODE HERE
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyDeviceToHost);
	
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(arg, outputImage);

	//TODO: RELEASE CUDA MEMORY
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	cudaFree(deviceMaskData);

	free(hostMaskData);
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}
