#include <stdio.h>
#include <cfloat>

#define TILE_WIDTH 32
#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};
float checkCorrectFloat(float * a1, float* a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)	
		err += abs(a1[i] - a2[i]);
	err /= n;
	return err;
}
void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("****************************\n\n");

}
// ===================== ReLu =====================
__global__ void reluKernelV1(float *input, float *output, int numCols, int numRows)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows && col < numCols)
    {
        int index = row * numCols + col;
        output[index] = fmaxf(0.0f, input[index]);
    }
}

__global__ void reluKernelV2(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
    }
}

__global__ void reluKernelV3(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // Khoảng cách giữa các luồng khi xử lý

    #pragma unroll
    for (int i = 0; i < 4; ++i) // Unrolling thủ công với 4 phần tử mỗi luồng
    {
        int currentIdx = idx + i * stride;
        if (currentIdx < size)
        {
            output[currentIdx] = (input[currentIdx] > 0.0f) ? input[currentIdx] : 0.0f;
        }
    }
}

void reLuHost(float* in, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
    }
}

void reLu(float * in, float * out, int length, int width,
    int verDevice = 0, dim3 blockSize = dim3(1))
{
    GpuTimer timer;
    timer.Start();
	if (verDevice == 0)
	{
		reLuHost(in, out, length * width);
	}
	else if (verDevice == 1) // Use device
	{
		// Allocate device memories
		float * d_in, * d_out;
		dim3 gridSize((width - 1) / blockSize.x + 1, 
					(length - 1) / blockSize.y + 1); // TODO: Compute gridSize from n and blockSize
		
		// TODO: Allocate device memories
        CHECK(cudaMalloc(&d_in, width * length * sizeof(float)));
        CHECK(cudaMalloc(&d_out, width * length * sizeof(float)));
		// TODO: Copy data to device memories
        CHECK(cudaMemcpy(d_in, in, width * length * sizeof(float), cudaMemcpyHostToDevice));
		

		// Call kernel
		reluKernelV1<<<gridSize, blockSize>>>(d_in, d_out, length, width);

		cudaDeviceSynchronize();

		CHECK(cudaGetLastError());
		
		// TODO: Copy result from device memories
        CHECK(cudaMemcpy(out, d_out, width * length * sizeof(float), cudaMemcpyDeviceToHost));
		// TODO: Free device memories
        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));
		// Print info
		printf("Kernel version 1, Grid size: %d, block size: %d\n", gridSize.x, blockSize.x);
	} else if (verDevice == 2) {
        // Allocate device memories
		float * d_in, * d_out;
		
		// TODO: Allocate device memories
        CHECK(cudaMalloc(&d_in, width * length * sizeof(float)));
        CHECK(cudaMalloc(&d_out, width * length * sizeof(float)));
		// TODO: Copy data to device memories
        CHECK(cudaMemcpy(d_in, in, width * length * sizeof(float), cudaMemcpyHostToDevice));
		

		// Call kernel
		int size = length * width;
        int threadPerBlock = blockSize.x * blockSize.y;
        dim3 gridSize((size - 1) / threadPerBlock + 1);
        // Call kernel function
        reluKernelV2<<<gridSize, threadPerBlock>>>(d_in, d_out, size);

		cudaDeviceSynchronize();

		CHECK(cudaGetLastError());
		
		// TODO: Copy result from device memories
        CHECK(cudaMemcpy(out, d_out, width * length * sizeof(float), cudaMemcpyDeviceToHost));
		// TODO: Free device memories
        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));
		// Print info
		printf("Kernel version 2, Grid size: %d, block size: %d\n", gridSize.x, blockSize.x);
    } else {
        // Allocate device memories
		float * d_in, * d_out;
		
		// TODO: Allocate device memories
        CHECK(cudaMalloc(&d_in, width * length * sizeof(float)));
        CHECK(cudaMalloc(&d_out, width * length * sizeof(float)));
		// TODO: Copy data to device memories
        CHECK(cudaMemcpy(d_in, in, width * length * sizeof(float), cudaMemcpyHostToDevice));
		

		// Call kernel
		int size = length * width;
        int threadPerBlock = blockSize.x * blockSize.y;
        dim3 gridSize((size - 1) / threadPerBlock + 1);
        // Call kernel function
        reluKernelV3<<<gridSize, threadPerBlock>>>(d_in, d_out, size);

		cudaDeviceSynchronize();

		CHECK(cudaGetLastError());
		
		// TODO: Copy result from device memories
        CHECK(cudaMemcpy(out, d_out, width * length * sizeof(float), cudaMemcpyDeviceToHost));
		// TODO: Free device memories
        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));
		// Print info
		printf("Kernel version 3, Grid size: %d, block size: %d\n", gridSize.x, blockSize.x);
    }

    timer.Stop();
    float time = timer.Elapsed();
    if (verDevice == 0) {
        printf("Host time : %f ms\n\n", time);
    }
    else if (verDevice == 1) {
        printf("Kernel version 1 time : %f ms\n", time);
    }
    else if (verDevice == 2) {
        printf("Kernel version 2 time : %f ms\n", time);
    }
    else {
        printf("Kernel version 3 time : %f ms\n", time);
    }
}

void checkReLu(int argc, char ** argv){
    printf("************************ ReLU ************************\n");
    
    //Declare variables
    int length = (1 <<12);
    int width = (1 << 12);

    float * in = (float *) malloc(width * length * sizeof(float));
    float * outHost = (float *) malloc(width * length * sizeof(float));
    float * outDevice1 = (float *) malloc(width * length * sizeof(float));
    float * outDevice2 = (float *) malloc(width * length * sizeof(float));
    float * outDevice3 = (float *) malloc(width * length * sizeof(float));
    for (int i = 0; i < width * length; i++)
    {
        in[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        // Get random neagtive number
        if (i % 2 == 0)
            in[i] = -in[i];
    }

    reLu(in, outHost, length, width);

    dim3 blockSize(512); // Default
    if (argc == 2)
    	blockSize.x = atoi(argv[1]); 

	reLu(in, outDevice1, length, width, 1, blockSize);
    float err = checkCorrectFloat(outHost, outDevice1, width * length);
	printf("Error: %f\n\n", err);

    reLu(in, outDevice2, length, width, 2, blockSize);
    err = checkCorrectFloat(outHost, outDevice2, width * length);
	printf("Error: %f\n\n", err);

    reLu(in, outDevice3, length, width, 3, blockSize);
    err = checkCorrectFloat(outHost, outDevice3, width * length);
	printf("Error: %f\n\n", err);

    free(in);
    free(outHost);
    free(outDevice1);
    free(outDevice3);
    free(outDevice2);

    printf("******************************************************\n");
}
// ===================== ReLu =====================

int main(int argc, char ** argv)
{
	printDeviceInfo();
    checkReLu(argc, argv);
    return 0;
}
