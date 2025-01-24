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
__global__ void OneHotEncodingV1(int *in, float *out, int numClasses, int numSamples)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < numSamples)
    {
        int label = in[sample];
        for (int row = 0; row < numClasses; ++row)
        {
            out[row * numSamples + sample] = (row == label) ? 1.0f : 0.0f;
        }
    }
}
__global__ void oneHotEncodingV2(int *labels, float *Y, int numClasses, int numSamples, int len, int start)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x; 
    if (sample < len)
    {
        int label = labels[sample + start];
        for (int row = 0; row < numClasses; ++row)
        {
            Y[row * numSamples + (sample + start)] = (row == label) ? 1.0f : 0.0f;
        }
    }
}
void OneHotEncoding(int * in, float * out, int numClasses, int numSamples, int verDevice, dim3 blockSize = dim3(1)){
    GpuTimer timer;
    timer.Start();
	if (verDevice == 0){
		for (int sample = 0; sample < numSamples; ++sample){
        int label = in[sample];
        for (int row = 0; row < numClasses; ++row){
            out[row * numSamples + sample] = (row == label) ? 1.0f : 0.0f;
        }
    }
    } else if (verDevice == 1){
		// Allocate device memories
		float *d_out;
        int *d_in;
		int numThreadsPerBlock = blockSize.x * blockSize.y;
        dim3 numBlocksPerGrid = ((numSamples * numClasses - 1) / numThreadsPerBlock + 1);
		
		// TODO: Allocate device memories
        CHECK(cudaMalloc(&d_in, numSamples * sizeof(int)));
        CHECK(cudaMalloc(&d_out, numSamples * numClasses  * sizeof(float)));
		// TODO: Copy data to device memories
        CHECK(cudaMemcpy(d_in, in, numSamples  * sizeof(int), cudaMemcpyHostToDevice));

		// Call kernel
		OneHotEncodingV1<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_in, d_out, numClasses, numSamples);

		cudaDeviceSynchronize();

		CHECK(cudaGetLastError());
		
		// TODO: Copy result from device memories
        CHECK(cudaMemcpy(out, d_out, numSamples * numClasses  * sizeof(float), cudaMemcpyDeviceToHost));
		// TODO: Free device memories
        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));
		// Print info
		printf("Kernel version 1, Grid size: %d, block size: %d\n", numBlocksPerGrid.x, numThreadsPerBlock);
	} else{
        cudaStream_t streams[3];
        for (int i = 0; i < 3; i++){
            cudaStreamCreate(&streams[i]);
        }

        float *d_out;
        int *d_in;
		int numThreadsPerBlock = blockSize.x * blockSize.y;
        dim3 numBlocksPerGrid = ((numSamples * numClasses - 1) / numThreadsPerBlock + 1);
        CHECK(cudaMalloc(&d_in, numSamples * sizeof(int)));
        CHECK(cudaMalloc(&d_out, numSamples * numClasses * sizeof(float)));
        // printf("numsample: %d \n", numSamples);
        int start, len;
        for (int  i = 0; i < 2; i++){
            start = i * numSamples / 3;
            len = numSamples / 3;
            // printf("start: %d - %d \n", start, len);
            CHECK(cudaMemcpyAsync(d_in, in, numSamples * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
            oneHotEncodingV2<<<numBlocksPerGrid, numThreadsPerBlock, 0, streams[i]>>>(d_in, d_out, numClasses, numSamples, len, start);
        }
        start = 2 * numSamples / 3;
        len = numSamples - start;
        // printf("start: %d - %d", start, len);
        CHECK(cudaMemcpyAsync(d_in, in, numSamples * sizeof(float), cudaMemcpyHostToDevice, streams[2]));
        oneHotEncodingV2<<<numBlocksPerGrid, numThreadsPerBlock, 0, streams[2]>>>(d_in, d_out, numClasses, numSamples, len, start);

        for (int i = 0; i < 3; i++){
            CHECK(cudaStreamSynchronize(streams[i]));
            cudaStreamDestroy(streams[i]);
        }
        cudaDeviceSynchronize();

		CHECK(cudaGetLastError());
		
		// TODO: Copy result from device memories
        CHECK(cudaMemcpy(out, d_out, numSamples * numClasses  * sizeof(float), cudaMemcpyDeviceToHost));
		// TODO: Free device memories
        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));
		// Print info
		printf("Kernel version 2, Grid size: %d, block size: %d\n", numBlocksPerGrid.x, numThreadsPerBlock);
    }
    timer.Stop();
    float time = timer.Elapsed();
    if (verDevice == 0) {
        printf("Host time : %f ms\n\n", time);
    }
    else if (verDevice == 1){
        printf("Kernel version 1 time : %f ms\n", time);
    }else{
        printf("Kernel version 2 time : %f ms\n", time);
    }
}
void checkOneHotEncoding(int argc, char ** argv){
    printf("***************** One-Hot Encoding *****************\n");
    
    int* h_in1;
    float * h_in2, *h_out1, * h_out2; // The A matrix
    float* correct_out; // The output C matrix

    int m;    // number of rows in the matrix A
    int n; // number of columns in the matrix A, number of rows in the matrix B

    m = 10;
    n = (1 << 15);

    // Set up input data
    h_in1 = (int*)malloc(n * sizeof(int));
    h_in2 = (float*)malloc(m * n * sizeof(float));
    h_out1 = (float*)malloc(m * n * sizeof(float));
    h_out2 = (float*)malloc(m * n * sizeof(float));
    correct_out = (float*)malloc(m * n * sizeof(float));

    for (int j = 0; j < n;j++)
        h_in1[j] = static_cast<int>(static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 9);
    

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n;j++)
            h_in2[i * n + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

    OneHotEncoding(h_in1, correct_out, m, n, 0);
	printf("\n");

	dim3 blockSize(32, 32); // Default
	if (argc == 3)
	{
		blockSize.x = atoi(argv[1]);
		blockSize.y = atoi(argv[2]);
	} 
    
    OneHotEncoding(h_in1, h_out1, m, n, 1, blockSize);
	float err = checkCorrectFloat(h_out1, correct_out, m * n);
	printf("Error: %f\n\n", err); 

    OneHotEncoding(h_in1, h_out2, m, n, 2, blockSize);
	err = checkCorrectFloat(h_out2, correct_out, m * n);
	printf("Error: %f\n\n", err);
	
    
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
    checkOneHotEncoding(argc, argv);
    return 0;
}