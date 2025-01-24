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
__global__ void ComputeCrossEntropyLossV1(float *in1, float *in2, float *out, int numClasses, int numSamples)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < numSamples)
    {
        float sum = 0.0f;
        for (int row = 0; row < numClasses; ++row)
        {
            sum += in1[row * numSamples + sample] * logf(in2[row * numSamples + sample] + FLT_MIN);
        }
        atomicAdd(out, -sum);
    }
}
void ComputeCrossEntropyLoss(float * in1, float * in2, float * out, int numClasses, int numSamples, int verDevice, dim3 blockSize = dim3(1)){
    GpuTimer timer;
    timer.Start();
	if (verDevice == 0){
		float sum = 0.0f;
        for (int row = 0; row < numClasses; ++row)
        {
            for (int col = 0; col < numSamples; ++col)
            {
                sum += in1[row * numSamples + col] * logf(in2[row * numSamples + col] + FLT_MIN);
            }
        }
        // Normalize the loss
        *out = -sum / numSamples;
    } else if (verDevice == 1){
		// Allocate device memories
		float * d_in1, * d_out, * d_in2;
		int numThreadsPerBlock = blockSize.x * blockSize.y;
        dim3 numBlocksPerGrid = ((numSamples * numClasses - 1) / numThreadsPerBlock + 1);
        // Init loss to zero
        *out = 0.0f;
		
		// TODO: Allocate device memories
        CHECK(cudaMalloc(&d_in1, numSamples * numClasses * sizeof(float)));
        CHECK(cudaMalloc(&d_in2, numSamples * numClasses * sizeof(float)));
        CHECK(cudaMalloc(&d_out, 1 * sizeof(float)));
		// TODO: Copy data to device memories
        CHECK(cudaMemcpy(d_in1, in1, numSamples * numClasses * sizeof(float), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_in2, in2, numSamples * numClasses * sizeof(float), cudaMemcpyHostToDevice));

		// Call kernel
		ComputeCrossEntropyLossV1<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_in1, d_in2, d_out, numClasses, numSamples);

		cudaDeviceSynchronize();

		CHECK(cudaGetLastError());
		
		// TODO: Copy result from device memories
        CHECK(cudaMemcpy(out, d_out, 1 * sizeof(float), cudaMemcpyDeviceToHost));
		// TODO: Free device memories
        CHECK(cudaFree(d_in1));
        CHECK(cudaFree(d_in2));
        CHECK(cudaFree(d_out));
		// Print info
        *out /= numSamples;
		printf("Kernel version 1, Grid size: %d, block size: %d\n", numBlocksPerGrid.x, numThreadsPerBlock);
	}
    timer.Stop();
    float time = timer.Elapsed();
    if (verDevice == 0) {
        printf("Host time : %f ms\n\n", time);
    }
    else{
        printf("Kernel version 1 time : %f ms\n", time);
    }
}
void checkComputeCrossEntropyLoss(int argc, char ** argv){
    printf("************ Compute Cross-Entropy Loss ************\n");
    
    float* h_in1, * h_in2, *h_out1, * h_out2; // The A matrix
    float* correct_out; // The output C matrix

    int m;    // number of rows in the matrix A
    int n; // number of columns in the matrix A, number of rows in the matrix B

    m = 10;
    n = (1 << 15);

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

    ComputeCrossEntropyLoss(h_in1, h_in2, correct_out, m, n, 0);
	printf("\n");

	dim3 blockSize(32, 32); // Default
	if (argc == 3)
	{
		blockSize.x = atoi(argv[1]);
		blockSize.y = atoi(argv[2]);
	} 
    
    ComputeCrossEntropyLoss(h_in1, h_in2, h_out1, m, n, 1, blockSize);
	float err = checkCorrectFloat(h_out1, correct_out, 1);
	printf("Error: %f\n\n", err);

    // ComputeCrossEntropyLoss(h_in1, h_in2, h_out2, m, n, 2, blockSize);
	// err = checkCorrectFloat(h_out2, correct_out, m * n);
	// printf("Error: %f\n\n", err);
	
    
    free(h_in1);
    free(h_in2);
    free(h_out1);
    free(h_out2);
    free(correct_out);

    printf("****************************************************\n");    
}
int main(int argc, char ** argv)
{
	printDeviceInfo();
    checkComputeCrossEntropyLoss(argc, argv);
    return 0;
}
