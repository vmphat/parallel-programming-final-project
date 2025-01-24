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
// ===================== Matrix Multiplication =====================
__global__ void matMulKernelV1(float *A, float *B, float *C,
                                     int m, int n, int k)
{
    //TODO
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < k) {
        float result = 0;
        for (int pos = 0; pos < n; pos++) {
            result += A[i * n + pos] * B[pos * k + j]; 
        }
        C[i * k + j] = result;
    }
}

__global__ void matMulKernelV2(float* A, float* B, float* C, int m, int n, int k)
{
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;


	float sum = 0.0f;
	for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
	{
		// Load one tile of matrix A and B from global memory
		// into shared memory
		if (row < m && (t * TILE_WIDTH + threadIdx.x) < n)
		{
			s_A[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_WIDTH + threadIdx.x];
		}
		else
		{
			s_A[threadIdx.y][threadIdx.x] = 0.0f;
		}

		if ((t * TILE_WIDTH + threadIdx.y) < n && col < k)
		{
			s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * k + col];
		}
		else
		{
			s_B[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads();

		for (int i = 0; i < TILE_WIDTH; ++i)
		{
			sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
		}
		__syncthreads();
	}
    
	if (row < m && col < k)
	{
		C[row * k + col] = sum;
	}
}

void matMulHost(float* A, float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float result = 0;
            for (int pos = 0; pos < n; pos++) {
                result += A[i * n + pos] * B[pos * k + j]; 
            }
            C[i * k + j] = result;
        }
    }
}

void matMul(float* A, float* B, float* C, int m, int n, int k,
    int verDevice = 0, dim3 blockSize = dim3(1))
{
    GpuTimer timer;
    timer.Start();
    if (verDevice == 0)
    {
        matMulHost(A, B, C, m, n, k);
    } else if (verDevice == 2) // Use device
    {
        // TODO: Allocate device memories
        float* d_A, * d_B, * d_C;
        size_t aBytes = m * n * sizeof(float);
        size_t bBytes = n * k * sizeof(float);
        size_t cBytes = m * k * sizeof(float);

        CHECK(cudaMalloc(&d_A, aBytes));
        CHECK(cudaMalloc(&d_B, bBytes));
        CHECK(cudaMalloc(&d_C, cBytes));
        // TODO: Copy data to device memories
        
        CHECK(cudaMemcpy(d_A, A, aBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, B, bBytes, cudaMemcpyHostToDevice));

        dim3 gridSize((m - 1) / blockSize.x + 1, 
					(k - 1) / blockSize.y + 1); // TODO: Compute gridSize
		matMulKernelV2<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);
        
        // TODO: Copy result from device memory
        CHECK(cudaMemcpy(C, d_C, cBytes, cudaMemcpyDeviceToHost));
        // TODO: Free device memories
		CHECK(cudaFree(d_A));
        CHECK(cudaFree(d_B));
        CHECK(cudaFree(d_C));

		printf("Kernel verion 2, Grid size: %d * %d, block size: %d * %d\n", 
			gridSize.x, gridSize.y, blockSize.x, blockSize.y);

    }
    else {
        // TODO: Allocate device memories
        float* d_A, * d_B, * d_C;
        size_t aBytes = m * n * sizeof(float);
        size_t bBytes = n * k * sizeof(float);
        size_t cBytes = m * k * sizeof(float);

        CHECK(cudaMalloc(&d_A, aBytes));
        CHECK(cudaMalloc(&d_B, bBytes));
        CHECK(cudaMalloc(&d_C, cBytes));
        // TODO: Copy data to device memories
        
        CHECK(cudaMemcpy(d_A, A, aBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, B, bBytes, cudaMemcpyHostToDevice));

        dim3 gridSize((m - 1) / blockSize.x + 1, 
					(k - 1) / blockSize.y + 1); // TODO: Compute gridSize
		matMulKernelV1<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);
        
        // TODO: Copy result from device memory
        CHECK(cudaMemcpy(C, d_C, cBytes, cudaMemcpyDeviceToHost));
        // TODO: Free device memories
		CHECK(cudaFree(d_A));
        CHECK(cudaFree(d_B));
        CHECK(cudaFree(d_C));

		printf("Kernel verion 1, Grid size: %d * %d, block size: %d * %d\n", 
			gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    }
    timer.Stop();
    float time = timer.Elapsed();
    if (verDevice == 0) {
        printf("Host time : %f ms\n\n", time);
    }
    else if (verDevice == 1) {
        printf("Kernel version 1 time : %f ms\n", time);
    }
    else {
        printf("Kernel version 2 time : %f ms\n", time);
    }
}

void checkMatMul(int argc, char ** argv){
    printf("*************** Matrix Multiplication ***************\n");
    
    //Declare variables
    float* h_A; // The A matrix
    float* h_B; // The B matrix
    float* h_C1, *h_C2; // The output C matrix
    float* correct_C; // The output C matrix

    int m; // number of rows in the matrix A
    int n; // number of columns in the matrix A, number of rows in the matrix B
    int k; // number of columns in the matrix B

    m = (1 << 10);
    n = (1 << 9);
    k = (1 << 10);

    // Set up input data
    h_A = (float*)malloc(m * n * sizeof(float));
    h_B = (float*)malloc(n * k * sizeof(float));
    h_C1 = (float*)malloc(m * k * sizeof(float));
    h_C2 = (float*)malloc(m * k * sizeof(float));
    correct_C = (float*)malloc(m * k * sizeof(float));

    for (int i = 0; i < m; i++) {
        for (int j = 0;j < n;j++) {
            h_A[i*n+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            // Get random negative value
            if (rand() % 2 == 0)
                h_A[i*n+j] = -h_A[i*n+j];
        }
    }
 
    for (int i = 0; i < n; i++) {
        for (int j = 0;j < k;j++) {
            h_B[i*k+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            // Get random negative value
            if (rand() % 2 == 0)
                h_B[i*k+j] = -h_B[i*k+j];
        }
    }

    matMul(h_A, h_B, correct_C, m, n, k);
	printf("\n");

	dim3 blockSize(32, 32); // Default
	if (argc == 3)
	{
		blockSize.x = atoi(argv[1]);
		blockSize.y = atoi(argv[2]);
	} 
    
    matMul(h_A, h_B, h_C1, m, n, k, 1, blockSize);
	float err = checkCorrectFloat(h_C1, correct_C, m*k);
	printf("Error: %f\n\n", err);

    matMul(h_A, h_B, h_C2, m, n, k, 2, blockSize);
	err = checkCorrectFloat(h_C2, correct_C, m*k);
	printf("Error: %f\n\n", err);
	
    
    free(h_A);
    free(h_B);
    free(h_C1);
    free(h_C2);
    free(correct_C);

    printf("*****************************************************\n");
}
// ===================== Matrix Multiplication =====================
int main(int argc, char ** argv)
{
	printDeviceInfo();
    checkMatMul(argc, argv);
    return 0;
}