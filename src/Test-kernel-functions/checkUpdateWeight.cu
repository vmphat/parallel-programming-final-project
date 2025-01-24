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
__global__ void updateWeightV1(float *in1, float *in2, float *out, int numRows, int numCols,
                                    float learningRate)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows && col < numCols)
    {
        int index = row * numCols + col;
        out[index] = in1[index] - learningRate * in2[index];
    }
}
__global__ void updateWeightV2(float *in1, float *in2, float *out, int size,
                                    float learningRate)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        out[index] = in1[index] - learningRate * in2[index];
    }
}
void updateWeight(float * in1, float * in2, float * out, int length, int width, int verDevice, float lr, dim3 blockSize = dim3(1)){
    GpuTimer timer;
    timer.Start();
	if (verDevice == 0){
		for (int row = 0; row < length; ++row){
        for (int col = 0; col < width; ++col){
            int index = row * width + col;
            out[index] = in1[index] - lr * in2[index];
        }
    }
	} else if (verDevice == 1){
		// Allocate device memories
		float * d_in1, * d_out, * d_in2;
		dim3 gridSize((width - 1) / blockSize.x + 1, 
					(length - 1) / blockSize.y + 1); // TODO: Compute gridSize from n and blockSize
		
		// TODO: Allocate device memories
        CHECK(cudaMalloc(&d_in1, width * length * sizeof(float)));
        CHECK(cudaMalloc(&d_in2, width * length * sizeof(float)));
        CHECK(cudaMalloc(&d_out, width * length * sizeof(float)));
		// TODO: Copy data to device memories
        CHECK(cudaMemcpy(d_in1, in1, width * length * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_in2, in2, width * length * sizeof(float), cudaMemcpyHostToDevice));

		// Call kernel
		updateWeightV1<<<gridSize, blockSize>>>(d_in1, d_in2, d_out, length, width, lr);

		cudaDeviceSynchronize();

		CHECK(cudaGetLastError());
		
		// TODO: Copy result from device memories
        CHECK(cudaMemcpy(out, d_out, width * length * sizeof(float), cudaMemcpyDeviceToHost));
		// TODO: Free device memories
        CHECK(cudaFree(d_in1));
        CHECK(cudaFree(d_in2));
        CHECK(cudaFree(d_out));
		// Print info
		printf("Kernel version 1, Grid size: %d, block size: %d\n", gridSize.x, blockSize.x);
	}
    else{
        float * d_in1, * d_out, * d_in2;
        int numElementPerBlock = blockSize.x * blockSize.y;
		dim3 gridSize((width * length - 1) / numElementPerBlock + 1);
		// TODO: Allocate device memories
        CHECK(cudaMalloc(&d_in1, width * length * sizeof(float)));
        CHECK(cudaMalloc(&d_in2, width * length * sizeof(float)));
        CHECK(cudaMalloc(&d_out, width * length * sizeof(float)));
		// TODO: Copy data to device memories
        CHECK(cudaMemcpy(d_in1, in1, width * length * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_in2, in2, width * length * sizeof(float), cudaMemcpyHostToDevice));

		// Call kernel
		updateWeightV2<<<gridSize, numElementPerBlock>>>(d_in1, d_in2, d_out, length * width, lr);

		cudaDeviceSynchronize();

		CHECK(cudaGetLastError());
		
		// TODO: Copy result from device memories
        CHECK(cudaMemcpy(out, d_out, width * length * sizeof(float), cudaMemcpyDeviceToHost));
		// TODO: Free device memories
        CHECK(cudaFree(d_in1));
        CHECK(cudaFree(d_in2));
        CHECK(cudaFree(d_out));
		// Print info
		printf("Kernel version 2, Grid size: %d, block size: %d\n", gridSize.x, numElementPerBlock);
    }

    timer.Stop();
    float time = timer.Elapsed();
    if (verDevice == 0) {
        printf("Host time : %f ms\n\n", time);
    }
    else if (verDevice == 1) {
        printf("Kernel version 1 time : %f ms\n", time);
    }
    else{
        printf("Kernel version 2 time : %f ms\n", time);
    }
}
void checkUpdateWeight(int argc, char ** argv){
    printf("****************** Update Weight ******************\n");
    
    float* h_in1, * h_in2, *h_out1, * h_out2; // The A matrix
    float* correct_out; // The output C matrix

    int m;    // number of rows in the matrix A
    int n; // number of columns in the matrix A, number of rows in the matrix B

    m = (1 << 12);
    n = (1 << 12);

    // Set up input data
    h_in1 = (float*)malloc(m * n * sizeof(float));
    h_in2 = (float*)malloc(m * n * sizeof(float));
    h_out1 = (float*)malloc(m * n * sizeof(float));
    h_out2 = (float*)malloc(m * n * sizeof(float));
    correct_out = (float*)malloc(m * n * sizeof(float));

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n;j++)
            h_in1[i * n + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n;j++)
            h_in2[i * n + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

    updateWeight(h_in1, h_in2, correct_out, m, n, 0, 0.1);
	printf("\n");

	dim3 blockSize(32, 32); // Default
	if (argc == 3)
	{
		blockSize.x = atoi(argv[1]);
		blockSize.y = atoi(argv[2]);
	} 
    
    updateWeight(h_in1, h_in2, h_out1, m, n, 1, 0.1, blockSize);
	float err = checkCorrectFloat(h_out1, correct_out, m * n);
	printf("Error: %f\n\n", err);

    updateWeight(h_in1, h_in2, h_out2, m, n, 2, 0.1, blockSize);
	err = checkCorrectFloat(h_out2, correct_out, m * n);
	printf("Error: %f\n\n", err);
	
    
    free(h_in1);
    free(h_in2);
    free(h_out1);
    free(h_out2);
    free(correct_out);

    printf("***************************************************\n");
}
int main(int argc, char ** argv)
{
	printDeviceInfo();
    checkUpdateWeight(argc, argv);
    return 0;
}