#include <stdio.h>
#include <fstream>
#include <string>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <random>

#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
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

// ==================== MODEL AND DATASET PARAMETERS =====================
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define NUM_CLASSES 10
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define LEARNING_RATE 0.1f
#define NUM_EPOCHS 10
#define BATCH_SIZE 64
#define TILE_WIDTH 32
#define TILE_WIDTH_OUTPUT 16
#define DEFAULT_BLOCK_SIZE 32
#define NUM_LAYERS 3
#define DEFAULT_GPU_DEVICE_ID 0
#define NUM_STREAMS_LOAD_MINI_BATCH_DATA 4 // 16 streams = 75.470 ms / 8 - 45.645 ms / 4 - 47.023 ms / 2 - 58.936 ms
// =======================================================================

// ============ TIMING THE TRAINING PROCESS OF THE NETWORK ===============
// Timer for the entire neural network
GpuTimer forwardTimer;
GpuTimer backwardTimer;
GpuTimer updateWeightsTimer;
GpuTimer trainModelTimer;
GpuTimer softmaxTimer;
GpuTimer MY_TIMER;
GpuTimer evaluateTimer;
GpuTimer copyDataTimer;
// Total time taken by forward pass entire training set
float totalTimeForwardTrain[NUM_LAYERS] = {0.0f};
// Total time taken by forward pass entire test set
float totalTimeForwardTest[NUM_LAYERS] = {0.0f};
// Total time taken by forward pass mini-batch training set
float totalTimeForwardMiniBatch[NUM_LAYERS] = {0.0f};
// Total time taken by backward pass mini-batch training set
float totalTimeBackwardMiniBatch[NUM_LAYERS] = {0.0f};
// Total time to update the weights of the neural network
float totalTimeUpdateWeights = 0.0f;
// Total time taken by softmax activation function
float totalTimeSoftmax = 0.0f;
// Total time taken by matmul operation
float totalTimeMatMul = 0.0f;
// Total time taken by transpose operation
float totalTimeTranspose = 0.0f;
// Total time to evaluate the model on full training set
float totalTimeEvaluateTrain = 0.0f;
// Total time to copy mini-batch data
float totalTimeCopyData = 0.0f;
// =======================================================================

// ============== STREAMS FOR CONTROLING PROGRAM EXECUTION ===============
// Stream for one-hot encoding labels for mini-batch data
cudaStream_t streamOneHotMiniBatch;
// Streams for loading mini-batch data
cudaStream_t streamLoadMiniBatch[NUM_STREAMS_LOAD_MINI_BATCH_DATA];
// =======================================================================

// ================ WEIGHT MATRICES OF THE NEURAL NETWORK ================
// Dimensions of weight matrices
const int W1_ROWS = HIDDEN_SIZE;    // #rows in W1
const int W1_COLS = INPUT_SIZE + 1; // #cols in W1
const int W2_ROWS = HIDDEN_SIZE;    // #rows in W2
const int W2_COLS = W1_ROWS + 1;    // #cols in W2
const int W3_ROWS = NUM_CLASSES;    // #rows in W3
const int W3_COLS = W2_ROWS + 1;    // #cols in W3
// Pointers to weight matrices
float *W1 = nullptr;
float *W2 = nullptr;
float *W3 = nullptr;
// =======================================================================

// ====== NUMBER OF ROWS IN INPUT AND OUTPUT MATRICES AT EACH LAYER ======
const int X1_ROWS = W1_COLS; // #rows in X1
const int A1_ROWS = W1_ROWS; // #rows in A1
const int X2_ROWS = W2_COLS; // #rows in X2
const int A2_ROWS = W2_ROWS; // #rows in A2
const int X3_ROWS = W3_COLS; // #rows in X3
const int A3_ROWS = W3_ROWS; // #rows in A3
// =======================================================================

// ========= FORWARD PASS WHEN TRAINING MODEL (USING MINI-BATCH) =========
float *mb_Y = nullptr;  // One-hot encoded ground truth labels
float *mb_X1 = nullptr; // Input matrix at layer 1
float *mb_X2 = nullptr; // Input matrix at layer 2
float *mb_X3 = nullptr; // Input matrix at layer 3
float *mb_A1 = nullptr; // Output matrix at layer 1: mb_A1 = mb_X2[1:,:]
float *mb_A2 = nullptr; // Output matrix at layer 2: mb_A2 = mb_X3[1:,:]
float *mb_A3 = nullptr; // Output matrix at layer 3
// =======================================================================

// ======== BACKWARD PASS WHEN TRAINING MODEL (USING MINI-BATCH) =========
// For layer 3
float *mb_dLdZ3 = nullptr; // mb_dLdZ3 = mb_A3 - mb_Y
float *mb_X3T = nullptr;   // Transpose of mb_X3
float *mb_dLdW3 = nullptr; // mb_dLdW3 = mb_dLdZ3 * mb_X3T
float *mb_W3T = nullptr;   // Transpose of mb_W3
float *mb_dLdX3 = nullptr; // mb_dLdX3 = mb_W3T * mb_dLdZ3
// For layer 2
float *mb_dLdA2 = nullptr; // mb_dLdA2 = mb_dLdX3[1:,:]
float *mb_dLdZ2 = nullptr; // mb_dLdZ2 = mb_dLdA2 * relu'(mb_A2)
float *mb_X2T = nullptr;   // Transpose of mb_X2
float *mb_dLdW2 = nullptr; // mb_dLdW2 = mb_dLdZ2 * mb_X2T
float *mb_W2T = nullptr;   // Transpose of mb_W2
float *mb_dLdX2 = nullptr; // mb_dLdX2 = mb_W2T * mb_dLdZ2
// For layer 1
float *mb_dLdA1 = nullptr; // mb_dLdA1 = mb_dLdX2[1:,:]
float *mb_dLdZ1 = nullptr; // mb_dLdZ1 = mb_dLdA1 * relu'(mb_A1)
float *mb_X1T = nullptr;   // Transpose of mb_X1
float *mb_dLdW1 = nullptr; // mb_dLdW1 = mb_dLdZ1 * mb_X1T
// =======================================================================

// =================== ORIGINAL FASHION-MNIST DATASET ====================
float *trainData = nullptr;  // Training data
int *trainLabels = nullptr;  // Training labels
int *trainPred = nullptr;    // Predicted labels on training set
float *trainLoss = nullptr;  // Loss on training set
int *trainIndices = nullptr; // Shuffled indices for mini-batch training
float *testData = nullptr;   // Test data
int *testLabels = nullptr;   // Test labels
int *testPred = nullptr;     // Predicted labels on test set
float *testLoss = nullptr;   // Loss on test set
// =======================================================================

// ======== FORWARD PASS WHEN COMPUTING LOSS (FULL TRAINING SET) =========
float *train_Y = nullptr;  // One-hot encoded ground truth labels
float *train_X1 = nullptr; // Input matrix at layer 1
float *train_X2 = nullptr; // Input matrix at layer 2
float *train_X3 = nullptr; // Input matrix at layer 3
float *train_A1 = nullptr; // Output matrix at layer 1: train_A1 = train_X2[1:,:]
float *train_A2 = nullptr; // Output matrix at layer 2: train_A2 = train_X3[1:,:]
float *train_A3 = nullptr; // Output matrix at layer 3
// =======================================================================

// ======== FORWARD PASS WHEN EVALUATING MODEL (FULL TEST SET) ===========
float *test_Y = nullptr;  // One-hot encoded ground truth labels
float *test_X1 = nullptr; // Input matrix at layer 1
float *test_X2 = nullptr; // Input matrix at layer 2
float *test_X3 = nullptr; // Input matrix at layer 3
float *test_A1 = nullptr; // Output matrix at layer 1: test_A1 = test_X2[1:,:]
float *test_A2 = nullptr; // Output matrix at layer 2: test_A2 = test_X3[1:,:]
float *test_A3 = nullptr; // Output matrix at layer 3
// =======================================================================

// =======================================================================
// [========             MEMORY MANAGEMENT FUNCTIONS             ========]
// -----------------------------------------------------------------------
/** @brief Allocate memory for all declared pointers/arrays/matrices */
void ALLOCATE_MEMORY()
{
    // ================ WEIGHT MATRICES OF THE NEURAL NETWORK ================
    CHECK(cudaMallocManaged(&W1, W1_ROWS * W1_COLS * sizeof(float)));
    CHECK(cudaMallocManaged(&W2, W2_ROWS * W2_COLS * sizeof(float)));
    CHECK(cudaMallocManaged(&W3, W3_ROWS * W3_COLS * sizeof(float)));
    // =======================================================================

    // ========= FORWARD PASS WHEN TRAINING MODEL (USING MINI-BATCH) =========
    CHECK(cudaMallocManaged(&mb_Y, NUM_CLASSES * BATCH_SIZE * sizeof(float)));
    CHECK(cudaMallocManaged(&mb_X1, X1_ROWS * BATCH_SIZE * sizeof(float)));
    CHECK(cudaMallocManaged(&mb_X2, X2_ROWS * BATCH_SIZE * sizeof(float)));
    CHECK(cudaMallocManaged(&mb_X3, X3_ROWS * BATCH_SIZE * sizeof(float)));
    mb_A1 = mb_X2 + BATCH_SIZE; // mb_A1 = mb_X2[1:,:]
    mb_A2 = mb_X3 + BATCH_SIZE; // mb_A2 = mb_X3[1:,:]
    CHECK(cudaMallocManaged(&mb_A3, A3_ROWS * BATCH_SIZE * sizeof(float)));
    // =======================================================================

    // ======== BACKWARD PASS WHEN TRAINING MODEL (USING MINI-BATCH) =========
    // For layer 3
    CHECK(cudaMallocManaged(&mb_dLdZ3, W3_ROWS * BATCH_SIZE * sizeof(float)));
    CHECK(cudaMallocManaged(&mb_X3T, BATCH_SIZE * X3_ROWS * sizeof(float)));
    CHECK(cudaMallocManaged(&mb_dLdW3, W3_ROWS * W3_COLS * sizeof(float)));
    CHECK(cudaMallocManaged(&mb_W3T, W3_COLS * W3_ROWS * sizeof(float)));
    CHECK(cudaMallocManaged(&mb_dLdX3, X3_ROWS * BATCH_SIZE * sizeof(float)));
    // For layer 2
    mb_dLdA2 = mb_dLdX3 + BATCH_SIZE; // mb_dLdA2 = mb_dLdX3[1:,:]
    CHECK(cudaMallocManaged(&mb_dLdZ2, W2_ROWS * BATCH_SIZE * sizeof(float)));
    CHECK(cudaMallocManaged(&mb_X2T, BATCH_SIZE * X2_ROWS * sizeof(float)));
    CHECK(cudaMallocManaged(&mb_dLdW2, W2_ROWS * W2_COLS * sizeof(float)));
    CHECK(cudaMallocManaged(&mb_W2T, W2_COLS * W2_ROWS * sizeof(float)));
    CHECK(cudaMallocManaged(&mb_dLdX2, X2_ROWS * BATCH_SIZE * sizeof(float)));
    // For layer 1
    mb_dLdA1 = mb_dLdX2 + BATCH_SIZE; // mb_dLdA1 = mb_dLdX2[1:,:]
    CHECK(cudaMallocManaged(&mb_dLdZ1, W1_ROWS * BATCH_SIZE * sizeof(float)));
    CHECK(cudaMallocManaged(&mb_X1T, BATCH_SIZE * X1_ROWS * sizeof(float)));
    CHECK(cudaMallocManaged(&mb_dLdW1, W1_ROWS * W1_COLS * sizeof(float)));
    // =======================================================================

    // =================== ORIGINAL FASHION-MNIST DATASET ====================
    CHECK(cudaMallocManaged(&trainData, NUM_TRAIN * INPUT_SIZE * sizeof(float)));
    CHECK(cudaMallocManaged(&trainLabels, NUM_TRAIN * sizeof(int)));
    CHECK(cudaMallocManaged(&trainPred, NUM_TRAIN * sizeof(int)));
    CHECK(cudaMallocManaged(&trainLoss, sizeof(float)));
    CHECK(cudaMallocManaged(&trainIndices, NUM_TRAIN * sizeof(int)));
    CHECK(cudaMallocManaged(&testData, NUM_TEST * INPUT_SIZE * sizeof(float)));
    CHECK(cudaMallocManaged(&testLabels, NUM_TEST * sizeof(int)));
    CHECK(cudaMallocManaged(&testPred, NUM_TEST * sizeof(int)));
    CHECK(cudaMallocManaged(&testLoss, sizeof(float)));
    // =======================================================================

    // ======== FORWARD PASS WHEN COMPUTING LOSS (FULL TRAINING SET) =========
    CHECK(cudaMallocManaged(&train_Y, NUM_CLASSES * NUM_TRAIN * sizeof(float)));
    CHECK(cudaMallocManaged(&train_X1, X1_ROWS * NUM_TRAIN * sizeof(float)));
    CHECK(cudaMallocManaged(&train_X2, X2_ROWS * NUM_TRAIN * sizeof(float)));
    CHECK(cudaMallocManaged(&train_X3, X3_ROWS * NUM_TRAIN * sizeof(float)));
    train_A1 = train_X2 + NUM_TRAIN; // train_A1 = train_X2[1:,:]
    train_A2 = train_X3 + NUM_TRAIN; // train_A2 = train_X3[1:,:]
    CHECK(cudaMallocManaged(&train_A3, A3_ROWS * NUM_TRAIN * sizeof(float)));
    // =======================================================================

    // ======== FORWARD PASS WHEN EVALUATING MODEL (FULL TEST SET) ===========
    CHECK(cudaMallocManaged(&test_Y, NUM_TEST * NUM_CLASSES * sizeof(float)));
    CHECK(cudaMallocManaged(&test_X1, X1_ROWS * NUM_TEST * sizeof(float)));
    CHECK(cudaMallocManaged(&test_X2, X2_ROWS * NUM_TEST * sizeof(float)));
    CHECK(cudaMallocManaged(&test_X3, X3_ROWS * NUM_TEST * sizeof(float)));
    test_A1 = test_X2 + NUM_TEST; // test_A1 = test_X2[1:,:]
    test_A2 = test_X3 + NUM_TEST; // test_A2 = test_X3[1:,:]
    CHECK(cudaMallocManaged(&test_A3, A3_ROWS * NUM_TEST * sizeof(float)));
    // =======================================================================
}
/** @brief Free memory for all declared pointers/arrays/matrices */
void FREE_MEMORY()
{
    // ================ WEIGHT MATRICES OF THE NEURAL NETWORK ================
    CHECK(cudaFree(W1));
    CHECK(cudaFree(W2));
    CHECK(cudaFree(W3));
    // =======================================================================

    // ========= FORWARD PASS WHEN TRAINING MODEL (USING MINI-BATCH) =========
    CHECK(cudaFree(mb_Y));
    CHECK(cudaFree(mb_X1));
    CHECK(cudaFree(mb_X2));
    CHECK(cudaFree(mb_X3));
    mb_A1 = nullptr; // mb_A1 = mb_X2[1:,:]
    mb_A2 = nullptr; // mb_A2 = mb_X3[1:,:]
    CHECK(cudaFree(mb_A3));
    // =======================================================================

    // ======== BACKWARD PASS WHEN TRAINING MODEL (USING MINI-BATCH) =========
    // For layer 3
    CHECK(cudaFree(mb_dLdZ3));
    CHECK(cudaFree(mb_X3T));
    CHECK(cudaFree(mb_dLdW3));
    CHECK(cudaFree(mb_W3T));
    CHECK(cudaFree(mb_dLdX3));
    // For layer 2
    mb_dLdA2 = nullptr; // mb_dLdA2 = mb_dLdX3[1:,:]
    CHECK(cudaFree(mb_dLdZ2));
    CHECK(cudaFree(mb_X2T));
    CHECK(cudaFree(mb_dLdW2));
    CHECK(cudaFree(mb_W2T));
    CHECK(cudaFree(mb_dLdX2));
    // For layer 1
    mb_dLdA1 = nullptr; // mb_dLdA1 = mb_dLdX2[1:,:]
    CHECK(cudaFree(mb_dLdZ1));
    CHECK(cudaFree(mb_X1T));
    CHECK(cudaFree(mb_dLdW1));
    // =======================================================================

    // =================== ORIGINAL FASHION-MNIST DATASET ====================
    CHECK(cudaFree(trainData));
    CHECK(cudaFree(trainLabels));
    CHECK(cudaFree(trainPred));
    CHECK(cudaFree(trainLoss));
    CHECK(cudaFree(trainIndices));
    CHECK(cudaFree(testData));
    CHECK(cudaFree(testLabels));
    CHECK(cudaFree(testPred));
    CHECK(cudaFree(testLoss));
    // =======================================================================

    // ======== FORWARD PASS WHEN COMPUTING LOSS (FULL TRAINING SET) =========
    CHECK(cudaFree(train_Y));
    CHECK(cudaFree(train_X1));
    CHECK(cudaFree(train_X2));
    CHECK(cudaFree(train_X3));
    train_A1 = nullptr; // train_A1 = train_X2[1:,:]
    train_A2 = nullptr; // train_A2 = train_X3[1:,:]
    CHECK(cudaFree(train_A3));
    // =======================================================================

    // ======== FORWARD PASS WHEN EVALUATING MODEL (FULL TEST SET) ===========
    CHECK(cudaFree(test_Y));
    CHECK(cudaFree(test_X1));
    CHECK(cudaFree(test_X2));
    CHECK(cudaFree(test_X3));
    test_A1 = nullptr; // test_A1 = test_X2[1:,:]
    test_A2 = nullptr; // test_A2 = test_X3[1:,:]
    CHECK(cudaFree(test_A3));
    // =======================================================================
}

// =======================================================================
// [========            STREAMS MANAGEMENT FUNCTIONS             ========]
// -----------------------------------------------------------------------
/** @brief Create all streams */
void CREATE_STREAMS()
{
    // Stream for one-hot encoding labels for mini-batch data
    CHECK(cudaStreamCreate(&streamOneHotMiniBatch));
    // Streams for loading mini-batch data
    for (int i = 0; i < NUM_STREAMS_LOAD_MINI_BATCH_DATA; ++i)
    {
        CHECK(cudaStreamCreate(&streamLoadMiniBatch[i]));
    }
}
/** @brief Destroy all streams */
void DESTROY_STREAMS()
{
    // Stream for one-hot encoding labels for mini-batch data
    CHECK(cudaStreamSynchronize(streamOneHotMiniBatch));
    CHECK(cudaStreamDestroy(streamOneHotMiniBatch));
    // Streams for loading mini-batch data
    for (int i = 0; i < NUM_STREAMS_LOAD_MINI_BATCH_DATA; ++i)
    {
        // Sync before destroying stream
        CHECK(cudaStreamSynchronize(streamLoadMiniBatch[i]));
        CHECK(cudaStreamDestroy(streamLoadMiniBatch[i]));
    }
}

// =======================================================================
// [========       LOADING FASHION-MNIST DATASET FUNCTIONS       ========]
// -----------------------------------------------------------------------
/** @brief Load Fashion-MNIST image data */
bool loadImageData(const std::string &filePath, float *data, int numRows, int numCols)
{
    // Open image data file in binary mode
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open())
    {
        printf("Failed to open file: %s\n", filePath.c_str());
        return false;
    }
    // Skip the header (16 bytes for images)
    file.ignore(16);

    // Read pixel values and normalize to [0, 1]
    for (int r = 0; r < numRows; ++r)
    {
        for (int c = 0; c < numCols; ++c)
        {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char *>(&pixel), sizeof(pixel));
            data[r * numCols + c] = static_cast<float>(pixel) / 255.0f;
        }
    }

    // Close the file
    file.close();
    return true;
}
/** @brief Load labels for Fashion-MNIST dataset */
bool loadLabels(const std::string &filePath, int *labels, int numSamples)
{
    // Open label file in binary mode
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open())
    {
        printf("Failed to open file: %s\n", filePath.c_str());
        return false;
    }
    // Skip the header (8 bytes for labels)
    file.ignore(8);

    // Read each label and store it in the labels array
    for (int i = 0; i < numSamples; ++i)
    {
        unsigned char label = 0;
        file.read(reinterpret_cast<char *>(&label), sizeof(label));
        labels[i] = static_cast<int>(label);
    }

    // Close the file
    file.close();
    return true;
}

// =======================================================================
// [========    PREPARING INPUT FOR EACH LAYER OF THE NETWORK    ========]
// -----------------------------------------------------------------------
/** @brief Kernel function to add bias unit to the first row of the input matrix */
__global__ void addBiasUnitKernel(float *input, int numCols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < numCols)
    {
        input[col] = 1.0f;
    }
}
/** @brief Add bias unit to the first row of the input matrix */
void addBiasUnit(float *input, int numCols, dim3 blockSize)
{
    // Calculate grid size to cover the entire input matrix
    int numThreadsPerBlock = blockSize.x * blockSize.y;
    if (numCols < 2048)
    {
        numThreadsPerBlock = 32;
    }
    int numBlocksPerGrid = (numCols - 1) / numThreadsPerBlock + 1;

    // Call kernel function
    addBiasUnitKernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(input, numCols);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors
}

// =======================================================================
// [========  RELU ACTIVATION FUNCTION WHEN BACKWARD PROPAGATING  =======]
// -----------------------------------------------------------------------
/** @brief Kernel function to compute the derivative of ReLU activation function */
__global__ void reluBackwardKernel(const float *A, float *dLdA, float *dLdZ, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // Khoảng cách giữa các luồng

#pragma unroll
    for (int i = 0; i < 4; ++i) // Unrolling thủ công với 4 phần tử mỗi luồng
    {
        int currentIdx = idx + i * stride;
        if (currentIdx < size)
        {
            dLdZ[currentIdx] = (A[currentIdx] > 0.0f) ? dLdA[currentIdx] : 0.0f;
        }
    }
}
/** @brief Backward pass for ReLU activation function */
void reluBackward(float *A, float *dLdA, float *dLdZ, int numRows, int numCols,
                  dim3 blockSize)
{
    // Calculate grid size to cover the entire input matrix
    int size = numRows * numCols;
    int numThreadsPerBlock = blockSize.x * blockSize.y;
    int numElementsPerBlock = 4 * numThreadsPerBlock;
    dim3 gridSize((size - 1) / numElementsPerBlock + 1);

    // Call kernel function
    reluBackwardKernel<<<gridSize, numThreadsPerBlock>>>(A, dLdA, dLdZ, size);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors
}

// =======================================================================
// [========  SUBTRACTING TWO MATRICES (ELEMENT-WISE) FUNCTION   ========]
// -----------------------------------------------------------------------
/** @brief Kernel function to subtract two matrices (element-wise): C = A - B */
__global__ void subtractMatricesKernel(float *A, float *B, float *C, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        C[index] = A[index] - B[index];
    }
}
/** @brief Subtract two matrices (element-wise): C = A - B */
void subtractMatrices(float *A, float *B, float *C, int numRows, int numCols,
                      dim3 blockSize)
{
    // Calculate grid size to cover the entire input matrix
    int size = numRows * numCols;
    int numThreadsPerBlock = blockSize.x * blockSize.y;
    if (size < 2048)
    {
        numThreadsPerBlock = 32;
    }
    dim3 gridSize((size - 1) / numThreadsPerBlock + 1);
    // Call kernel function
    subtractMatricesKernel<<<gridSize, numThreadsPerBlock>>>(A, B, C, size);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors
}

// =======================================================================
// [========                MATRIX MULTIPLICATION                ========]
// -----------------------------------------------------------------------
/** @brief Kernel function to compute matrix multiplication: C = A * B */
__global__ void matrixMultiplyKernel(float *A, float *B, float *C,
                                     int m, int n, int k)
{
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Register blocking for partial sums
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
    {
        // Collaborative loading of tiles into shared memory
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

// Compute partial sums with 4x loop unrolling
#pragma unroll 4
        for (int i = 0; i < TILE_WIDTH; i += 4)
        {
            sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
            sum += s_A[threadIdx.y][i + 1] * s_B[i + 1][threadIdx.x];
            sum += s_A[threadIdx.y][i + 2] * s_B[i + 2][threadIdx.x];
            sum += s_A[threadIdx.y][i + 3] * s_B[i + 3][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < m && col < k)
    {
        C[row * k + col] = sum;
    }
}
/** @brief Optimized kernel function for matrix multiplication with ReLU: C = ReLU(A * B) */
__global__ void matrixMultiplyReluKernel(float *A, float *B, float *C, int m, int n, int k)
{
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Register blocking for partial sums
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
    {
        // Collaborative loading of tiles into shared memory
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

// Compute partial sums with 4x loop unrolling
#pragma unroll(4)
        for (int i = 0; i < TILE_WIDTH; i += 4)
        {
            sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
            sum += s_A[threadIdx.y][i + 1] * s_B[i + 1][threadIdx.x];
            sum += s_A[threadIdx.y][i + 2] * s_B[i + 2][threadIdx.x];
            sum += s_A[threadIdx.y][i + 3] * s_B[i + 3][threadIdx.x];
        }

        __syncthreads();
    }

    // Apply ReLU and write result
    if (row < m && col < k)
    {
        C[row * k + col] = fmaxf(0.0f, sum);
    }
}
/** @brief Optimized kernel function for matrix multiplication with Softmax: C = Softmax(A * B) */
__global__ void matrixMultiplySoftmaxKernel(float *A, float *B, float *C, int m, int n, int k)
{
    __shared__ float s_A[TILE_WIDTH_OUTPUT][TILE_WIDTH_OUTPUT];
    __shared__ float s_B[TILE_WIDTH_OUTPUT][TILE_WIDTH_OUTPUT];

    int row = blockIdx.y * TILE_WIDTH_OUTPUT + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH_OUTPUT + threadIdx.x;

    // Register blocking for partial sums
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (n + TILE_WIDTH_OUTPUT - 1) / TILE_WIDTH_OUTPUT; ++t)
    {
        // Collaborative loading of tiles into shared memory
        if (row < m && (t * TILE_WIDTH_OUTPUT + threadIdx.x) < n)
        {
            s_A[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_WIDTH_OUTPUT + threadIdx.x];
        }
        else
        {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((t * TILE_WIDTH_OUTPUT + threadIdx.y) < n && col < k)
        {
            s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH_OUTPUT + threadIdx.y) * k + col];
        }
        else
        {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

// Compute partial sums with 4x loop unrolling
#pragma unroll(4)
        for (int i = 0; i < TILE_WIDTH_OUTPUT; i += 4)
        {
            sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
            sum += s_A[threadIdx.y][i + 1] * s_B[i + 1][threadIdx.x];
            sum += s_A[threadIdx.y][i + 2] * s_B[i + 2][threadIdx.x];
            sum += s_A[threadIdx.y][i + 3] * s_B[i + 3][threadIdx.x];
        }

        __syncthreads();
    }

    // Step 1: Write intermediate result (Z) into shared memory (for Softmax)
    __shared__ float sharedZ[TILE_WIDTH_OUTPUT][TILE_WIDTH_OUTPUT];
    if (row < m && col < k)
    {
        sharedZ[threadIdx.y][threadIdx.x] = sum;
    }
    else
    {
        sharedZ[threadIdx.y][threadIdx.x] = 0.0f; // Out-of-bound values do not contribute to sum
    }
    __syncthreads();

    // Step 2: Compute max value in the column for numerical stability
    float maxVal = -FLT_MAX;
    for (int i = 0; i < TILE_WIDTH_OUTPUT; ++i)
    {
        maxVal = fmaxf(maxVal, sharedZ[i][threadIdx.x]);
    }
    __syncthreads();

    // Step 3: Compute exp(Z - maxVal) and store it back in shared memory
    float expVal = 0.0f;
    if (row < m && col < k)
    {
        expVal = expf(sharedZ[threadIdx.y][threadIdx.x] - maxVal);
        sharedZ[threadIdx.y][threadIdx.x] = expVal;
    }
    else
    {
        sharedZ[threadIdx.y][threadIdx.x] = 0.0f; // Out-of-bound values do not contribute to sum
    }
    __syncthreads();

    // Step 4: Compute sum of exp values in the column
    float sumExp = 0.0f;
    for (int i = 0; i < TILE_WIDTH_OUTPUT; ++i)
    {
        sumExp += sharedZ[i][threadIdx.x];
    }
    __syncthreads();

    // Step 5: Normalize values to obtain Softmax result
    if (row < m && col < k)
    {
        C[row * k + col] = sharedZ[threadIdx.y][threadIdx.x] / sumExp;
    }
}
/**
 * @brief Kernel function to compute matrix multiplication: C = A * BT
 * @param A Matrix A of size (m x n)
 * @param BT Matrix B transpose of size (n x k)
 */
__global__ void matrixMultiplyWithTransposeKernel_ABT(float *A, float *BT, float *C, int m, int n, int k)
{
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_BT[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
    {
        if (row < m && (t * TILE_WIDTH + threadIdx.x) < n)
        {
            s_A[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_WIDTH + threadIdx.x];
        }
        else
        {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < k && (t * TILE_WIDTH + threadIdx.y) < n)
        {
            s_BT[threadIdx.y][threadIdx.x] = BT[col * n + t * TILE_WIDTH + threadIdx.y];
        }
        else
        {
            s_BT[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

// Compute partial sums with 4x loop unrolling
#pragma unroll(4)
        for (int i = 0; i < TILE_WIDTH; i += 4)
        {
            sum += s_A[threadIdx.y][i] * s_BT[i][threadIdx.x];
            sum += s_A[threadIdx.y][i + 1] * s_BT[i + 1][threadIdx.x];
            sum += s_A[threadIdx.y][i + 2] * s_BT[i + 2][threadIdx.x];
            sum += s_A[threadIdx.y][i + 3] * s_BT[i + 3][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < k)
    {
        C[row * k + col] = sum;
    }
}
/**
 * @brief Kernel function to compute matrix multiplication: C = A * B
 * @param AT Transpose of matrix A. A has size (m x n)
 * @param B Matrix B of size (n x k)
 * @param C Output matrix C = A * B of size (m x k)
 */
__global__ void matrixMultiplyWithTransposeKernel_ATB(float *AT, float *B, float *C, int m, int n, int k)
{
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    // Calculate the row and column index of the element
    // that this thread is responsible for
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Accumulate the result for C[row, col] in a register
    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
    {
        // Load one tile of matrix A and B
        if (row < m && (t * TILE_WIDTH + threadIdx.x) < n)
        {
            s_A[threadIdx.y][threadIdx.x] = AT[(t * TILE_WIDTH + threadIdx.x) * m + row];
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

        // Synchronize
        __syncthreads();

// Compute partial sums with 4x loop unrolling
#pragma unroll(4)
        for (int i = 0; i < TILE_WIDTH; i += 4)
        {
            sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
            sum += s_A[threadIdx.y][i + 1] * s_B[i + 1][threadIdx.x];
            sum += s_A[threadIdx.y][i + 2] * s_B[i + 2][threadIdx.x];
            sum += s_A[threadIdx.y][i + 3] * s_B[i + 3][threadIdx.x];
        }

        __syncthreads();
    }

    // Check if the thread is within the matrix bounds
    if (row < m && col < k)
    {
        C[row * k + col] = sum;
    }
}
/** @brief Optimized kernel function for matrix multiplication with ReLU backward: C = ReLU'(A * B) */
__global__ void matrixMultiplyWithTransposeReluBackwardKernel(float *AT, float *B, float *C, float *Z, int m, int n, int k)
{
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    // Calculate the row and column index of the element
    // that this thread is responsible for
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Accumulate the result for C[row, col] in a register
    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
    {
        // Load one tile of matrix A and B
        if (row < m && (t * TILE_WIDTH + threadIdx.x) < n)
        {
            s_A[threadIdx.y][threadIdx.x] = AT[(t * TILE_WIDTH + threadIdx.x) * m + row];
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

        // Synchronize
        __syncthreads();

// Compute partial sums with 4x loop unrolling
#pragma unroll(4)
        for (int i = 0; i < TILE_WIDTH; i += 4)
        {
            sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
            sum += s_A[threadIdx.y][i + 1] * s_B[i + 1][threadIdx.x];
            sum += s_A[threadIdx.y][i + 2] * s_B[i + 2][threadIdx.x];
            sum += s_A[threadIdx.y][i + 3] * s_B[i + 3][threadIdx.x];
        }

        __syncthreads();
    }

    // Check if the thread is within the matrix bounds
    if (row > 0 && row < m && col < k)
    {
        // C[row * k + col] = sum;
        C[(row - 1) * k + col] = (Z[(row - 1) * k + col] > 0.0f) ? sum : 0.0f;
    }
}

/** @brief Compute matrix multiplication: C = A * B */
void matrixMultiply(float *A, float *B, float *C,
                    int numRowsA, int numColsA, int numColsB,
                    dim3 blockSize)
{
    MY_TIMER.Start(); // Start timing the matrix multiplication operation

    // Calculate grid size to cover the entire output matrix
    dim3 myBlockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((numColsB - 1) / TILE_WIDTH + 1,
                  (numRowsA - 1) / TILE_WIDTH + 1);
    // Call kernel function
    matrixMultiplyKernel<<<gridSize, myBlockSize>>>(A, B, C, numRowsA, numColsA, numColsB);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors

    MY_TIMER.Stop(); // Stop timing the matrix multiplication operation
    totalTimeMatMul += MY_TIMER.Elapsed();
}

// =======================================================================
// [========                TRANSPOSE OF A MATRIX                ========]
// -----------------------------------------------------------------------
/** @brief Kernel function to compute the transpose of a matrix */
__global__ void transposeMatrixKernel(float *in, int numRows, int numCols, float *out)
{
    __shared__ float s_blkData[DEFAULT_BLOCK_SIZE][DEFAULT_BLOCK_SIZE + 1];

    int iR = blockIdx.y * blockDim.y + threadIdx.y;
    int iC = blockIdx.x * blockDim.x + threadIdx.x;
    if (iR < numRows && iC < numCols)
        s_blkData[threadIdx.x][threadIdx.y] = in[iR * numCols + iC];
    __syncthreads();

    int oC = blockIdx.y * blockDim.y + threadIdx.x;
    int oR = blockIdx.x * blockDim.x + threadIdx.y;
    if (oR < numCols && oC < numRows)
        out[oR * numRows + oC] = s_blkData[threadIdx.y][threadIdx.x];
}
/** @brief Compute the transpose of a matrix */
void transposeMatrix(float *input, int numRows, int numCols, float *output,
                     dim3 blockSize)
{
    MY_TIMER.Start(); // Start timing the transpose operation

    // Calculate grid size to cover the entire input matrix
    dim3 gridSize((numCols - 1) / blockSize.x + 1,
                  (numRows - 1) / blockSize.y + 1);
    // Call kernel function
    transposeMatrixKernel<<<gridSize, blockSize>>>(input, numRows, numCols, output);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors

    MY_TIMER.Stop(); // Stop timing the transpose operation
    totalTimeTranspose += MY_TIMER.Elapsed();
}

// =======================================================================
// [========              ONE-HOT ENCODING FUNCTION              ========]
// -----------------------------------------------------------------------
/** @brief Kernel function to one-hot encode the labels (column-wise) */
__global__ void oneHotEncodingKernel(int *labels, float *Y, int numClasses, int numSamples)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < numSamples)
    {
        int label = labels[sample];
        for (int row = 0; row < numClasses; ++row)
        {
            Y[row * numSamples + sample] = (row == label) ? 1.0f : 0.0f;
        }
    }
}
/** @brief One-hot encode the labels (column-wise) */
void oneHotEncoding(int *labels, float *Y, int numClasses, int numSamples,
                    dim3 blockSize)
{
    // Compute grid size to cover the entire output matrix
    int numThreadsPerBlock = blockSize.x * blockSize.y;
    int numBlocksPerGrid = (numSamples - 1) / numThreadsPerBlock + 1;
    // Call kernel function
    oneHotEncodingKernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(labels, Y, numClasses, numSamples);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors
}

// =======================================================================
// [========             COMPUTE CROSS-ENTROPY LOSS              ========]
// -----------------------------------------------------------------------
/** @brief Kernel function to compute cross-entropy loss */
__global__ void crossEntropyLossKernel(float *Y, float *P, float *loss,
                                       int numClasses, int numSamples)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < numSamples)
    {
        float sum = 0.0f;
        for (int row = 0; row < numClasses; ++row)
        {
            sum += Y[row * numSamples + sample] * logf(P[row * numSamples + sample] + FLT_MIN);
        }
        atomicAdd(loss, -sum);
    }
}
/** @brief Compute cross-entropy loss */
void computeCrossEntropyLoss(float *Y, float *P, float *loss, int numClasses, int numSamples,
                             dim3 blockSize)
{
    // Compute grid size to cover the entire output matrix
    int numThreadsPerBlock = blockSize.x * blockSize.y;
    int numBlocksPerGrid = (numSamples - 1) / numThreadsPerBlock + 1;
    // Init loss to zero
    *loss = 0.0f;
    // Call kernel function
    crossEntropyLossKernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(Y, P, loss, numClasses, numSamples);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors
    // Normalize the loss
    *loss /= numSamples;
}

// =======================================================================
// [========          PREDICT LABELS FROM PROBABILITIES          ========]
// -----------------------------------------------------------------------
/** @brief Kernel function to predict labels from probabilities */
__global__ void predictLabelsKernel(float *P, int *labels, int numClasses, int numSamples)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < numSamples)
    {
        float maxProb = P[sample];
        int predLabel = 0;
        for (int row = 1; row < numClasses; ++row)
        {
            float currentProb = P[row * numSamples + sample];
            if (currentProb > maxProb)
            {
                maxProb = currentProb;
                predLabel = row;
            }
        }
        labels[sample] = predLabel;
    }
}
/** @brief Predict labels from probabilities */
void predictLabels(float *P, int *labels, int numClasses, int numSamples,
                   dim3 blockSize)
{
    // Compute grid size to cover the entire output matrix
    int numThreadsPerBlock = blockSize.x * blockSize.y;
    int numBlocksPerGrid = (numSamples - 1) / numThreadsPerBlock + 1;
    // Call kernel function
    predictLabelsKernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(P, labels, numClasses, numSamples);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors
}

// =======================================================================
// [========  FORWARD PASS INPUT FOR EACH LAYER OF THE NETWORK   ========]
// -----------------------------------------------------------------------
/**
 * @brief Forward pass the input through neural network
 * @note Timing the forward pass of each layer when training the model
 */
void forward(float *X1, float *X2, float *X3, float *A1, float *A2, float *A3,
             int numSamples, float *totalTimeForward, dim3 blockSize)
{
    // ********** Forward pass for layer 1 **********
    forwardTimer.Start();

    // A1 = ReLU(W1 * X1)
    dim3 myBlockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize_L1((numSamples - 1) / TILE_WIDTH + 1,
                     (W1_ROWS - 1) / TILE_WIDTH + 1);
    // Call kernel function
    matrixMultiplyReluKernel<<<gridSize_L1, myBlockSize>>>(W1, X1, A1, W1_ROWS, W1_COLS, numSamples);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors

    forwardTimer.Stop();
    totalTimeForward[0] += forwardTimer.Elapsed();

    // ********** Forward pass for layer 2 **********
    forwardTimer.Start();

    // A2 = ReLU(W2 * X2)
    dim3 gridSize_L2((numSamples - 1) / TILE_WIDTH + 1,
                     (W2_ROWS - 1) / TILE_WIDTH + 1);
    // Call kernel function
    matrixMultiplyReluKernel<<<gridSize_L2, myBlockSize>>>(W2, X2, A2, W2_ROWS, W2_COLS, numSamples);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors

    forwardTimer.Stop();
    totalTimeForward[1] += forwardTimer.Elapsed();

    // ********** Forward pass for layer 3 **********
    forwardTimer.Start();

    // A3 = Softmax(W3 * X3)
    dim3 myBlockSize_Output(TILE_WIDTH_OUTPUT, TILE_WIDTH_OUTPUT);
    dim3 gridSize_Output((numSamples - 1) / TILE_WIDTH_OUTPUT + 1,
                         (W3_ROWS - 1) / TILE_WIDTH_OUTPUT + 1);
    // Call kernel function
    matrixMultiplySoftmaxKernel<<<gridSize_Output, myBlockSize_Output>>>(W3, X3, A3, W3_ROWS, W3_COLS, numSamples);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors

    forwardTimer.Stop();
    totalTimeForward[2] += forwardTimer.Elapsed();
}

// =======================================================================
// [========          BACKWARD PASS WHEN TRAINING MODEL          ========]
// -----------------------------------------------------------------------
/** @brief Backward pass the gradients through neural network */
void backward(dim3 blockSize)
{
    // ********** Backward pass for layer 3 **********
    backwardTimer.Start(); // Timing the backward pass for layer 3

    // dLdZ3 = A3 - Y
    subtractMatrices(mb_A3, mb_Y, mb_dLdZ3, A3_ROWS, BATCH_SIZE, blockSize);

    // dLdW3 = dLdZ3 * X3T
    dim3 myBlockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize_dLdW3((X3_ROWS - 1) / TILE_WIDTH + 1,
                        (W3_ROWS - 1) / TILE_WIDTH + 1);
    // Call kernel function
    matrixMultiplyWithTransposeKernel_ABT<<<gridSize_dLdW3, myBlockSize>>>(mb_dLdZ3, mb_X3, mb_dLdW3, W3_ROWS, BATCH_SIZE, X3_ROWS);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());

    // Kernel fusion: - dLdX3 = W3T * dLdZ3
    //                - mb_dLdA2 = mb_dLdX3[1:,:]
    //                - mb_dLdZ2 = mb_dLdA2 * relu'(mb_A2)
    dim3 gridSize_dLdX3((BATCH_SIZE - 1) / TILE_WIDTH + 1,
                        (W3_COLS - 1) / TILE_WIDTH + 1);
    // Call kernel function
    matrixMultiplyWithTransposeReluBackwardKernel<<<gridSize_dLdX3, myBlockSize>>>(W3, mb_dLdZ3, mb_dLdZ2, mb_A2, W3_COLS, W3_ROWS, BATCH_SIZE);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());

    backwardTimer.Stop(); // Stop timing the backward pass for layer 3
    totalTimeBackwardMiniBatch[2] += backwardTimer.Elapsed();

    // ********** Backward pass for layer 2 **********
    backwardTimer.Start(); // Timing the backward pass for layer 2

    // mb_dLdW2 = mb_dLdZ2 * mb_X2T
    dim3 gridSize_dLdW2((X2_ROWS - 1) / TILE_WIDTH + 1,
                        (W2_ROWS - 1) / TILE_WIDTH + 1);
    // Call kernel function
    matrixMultiplyWithTransposeKernel_ABT<<<gridSize_dLdW2, myBlockSize>>>(mb_dLdZ2, mb_X2, mb_dLdW2, W2_ROWS, BATCH_SIZE, X2_ROWS);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());

    // Kernel fusion: - mb_dLdX2 = mb_W2T * mb_dLdZ2
    //                - mb_dLdA1 = mb_dLdX2[1:,:]
    //                - mb_dLdZ1 = mb_dLdA1 * relu'(mb_A1)
    dim3 gridSize_dLdX2((BATCH_SIZE - 1) / TILE_WIDTH + 1,
                        (W2_COLS - 1) / TILE_WIDTH + 1);
    // Call kernel function
    matrixMultiplyWithTransposeReluBackwardKernel<<<gridSize_dLdX2, myBlockSize>>>(W2, mb_dLdZ2, mb_dLdZ1, mb_A1, W2_COLS, W2_ROWS, BATCH_SIZE);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());

    backwardTimer.Stop(); // Stop timing the backward pass for layer 2
    totalTimeBackwardMiniBatch[1] += backwardTimer.Elapsed();

    // ********** Backward pass for layer 1 **********
    backwardTimer.Start(); // Timing the backward pass for layer 1

    // mb_dLdW1 = mb_dLdZ1 * mb_X1T
    dim3 gridSize_dLdW1((X1_ROWS - 1) / TILE_WIDTH + 1,
                        (W1_ROWS - 1) / TILE_WIDTH + 1);
    // Call kernel function
    matrixMultiplyWithTransposeKernel_ABT<<<gridSize_dLdW1, myBlockSize>>>(mb_dLdZ1, mb_X1, mb_dLdW1, W1_ROWS, BATCH_SIZE, X1_ROWS);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());

    backwardTimer.Stop(); // Stop timing the backward pass for layer 1
    totalTimeBackwardMiniBatch[0] += backwardTimer.Elapsed();
}

// =======================================================================
// [========        UPDATE WEIGHTS OF THE NEURAL NETWORK         ========]
// -----------------------------------------------------------------------
/** @brief Kernel function to update weights of the neural network */
__global__ void updateWeightsKernel(float *dLdW, float *W, int size,
                                    float learningRate, int batchSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        W[index] -= learningRate * dLdW[index] / batchSize;
    }
}
/** @brief Update weights of the neural network */
void updateWeights(float learningRate, int batchSize, dim3 blockSize)
{
    updateWeightsTimer.Start(); // Timing the update weights process

    // Flatten the weight matrices, and compute the number of elements per block
    int numElementsPerBlock = blockSize.x * blockSize.y / 2;

    // ********** Update weights for layer 1 **********
    dim3 gridSize_W1((W1_ROWS * W1_COLS - 1) / numElementsPerBlock + 1);
    updateWeightsKernel<<<gridSize_W1, numElementsPerBlock>>>(mb_dLdW1, W1, W1_ROWS * W1_COLS,
                                                              learningRate, batchSize);
    // ********** Update weights for layer 2 **********
    dim3 gridSize_W2((W2_ROWS * W2_COLS - 1) / numElementsPerBlock + 1);
    updateWeightsKernel<<<gridSize_W2, numElementsPerBlock>>>(mb_dLdW2, W2, W2_ROWS * W2_COLS,
                                                              learningRate, batchSize);
    // ********** Update weights for layer 3 **********
    dim3 gridSize_W3((W3_ROWS * W3_COLS - 1) / numElementsPerBlock + 1);
    updateWeightsKernel<<<gridSize_W3, numElementsPerBlock>>>(mb_dLdW3, W3, W3_ROWS * W3_COLS,
                                                              learningRate, batchSize);

    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors

    updateWeightsTimer.Stop(); // Stop timing the update weights process
    totalTimeUpdateWeights += updateWeightsTimer.Elapsed();
}

// =======================================================================
// [========      TRAINING THE NEURAL NETWORK ON MINI-BATCH      ========]
// -----------------------------------------------------------------------
/**
 * @brief Initialize weights of the neural network using normal distribution
 * @note He initialization is used for ReLU activation function
 */
__host__ void initializeWeights(float *W, int fan_in, int fan_out, std::mt19937 &gen)
{
    float std = sqrt(2.0f / fan_in); // He initialization
    std::normal_distribution<float> normalDist(0.0f, std);

    int total_weights = fan_in * fan_out;
    for (int i = 0; i < total_weights; ++i)
    {
        W[i] = normalDist(gen);
    }
}
/** @brief Kernel function to one-hot encode the labels for mini-batch */
__global__ void oneHotEncodingMiniBatchKernel(int *trainLabels, int *mb_Indices, float *mb_Y, int batchSize)
{
    int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sampleIdx < batchSize)
    {
        // One-hot encode the ground truth labels
        for (int row = 0; row < NUM_CLASSES; ++row)
        {
            mb_Y[row * batchSize + sampleIdx] = (row == trainLabels[mb_Indices[sampleIdx]]) ? 1.0f : 0.0f;
        }
    }
}
/** @brief Kernel function to copy data to mini-batch */
__global__ void copyDataToMiniBatchKernel(float *trainData, int *mb_Indices, float *mb_X1, int batchSize, int XRows, int startXRow)
{
    int sampleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sampleIdx < batchSize)
    {
        // Copy 1 row from trainData to remaining rows of each column in mb_X1
        for (int row = 0; row < XRows; ++row)
        {
            mb_X1[(row + 1 + startXRow) * batchSize + sampleIdx] = trainData[mb_Indices[sampleIdx] * INPUT_SIZE + row + startXRow];
        }
    }
}
/** @brief Launch the function to prepare mini-batch data using streams */
void prepareMiniBatchData(float *trainData, int *trainLabels, int *mb_Indices, float *mb_X1, float *mb_Y, int batchSize, dim3 blockSize)
{
    // Number of threads per block
    int numThreadsPerBlock = 32;

    // Use one stream to one-hot encode the labels
    dim3 gridSize_OneHot((batchSize - 1) / numThreadsPerBlock + 1);
    oneHotEncodingMiniBatchKernel<<<gridSize_OneHot, numThreadsPerBlock>>>(trainLabels, mb_Indices, mb_Y, batchSize);
    CHECK(cudaGetLastError()); // Check if there is any error

    // Use multiple streams to load data to mini-batch
    for (int i = 0; i < NUM_STREAMS_LOAD_MINI_BATCH_DATA; ++i)
    {
        // Calculate the number of blocks needed for each stream
        int numBlocksPerGrid = (batchSize - 1) / numThreadsPerBlock + 1;

        // Calculate the number of rows for each stream
        int XRows = (INPUT_SIZE - 1) / NUM_STREAMS_LOAD_MINI_BATCH_DATA + 1;
        int startXRow = i * XRows;
        if (startXRow + XRows > INPUT_SIZE)
        {
            XRows = INPUT_SIZE - startXRow;
        }

        // Call kernel function
        copyDataToMiniBatchKernel<<<numBlocksPerGrid, numThreadsPerBlock, 0, streamLoadMiniBatch[i]>>>(
            trainData, mb_Indices, mb_X1, batchSize, XRows, startXRow);
        CHECK(cudaGetLastError()); // Check if there is any error
    }

    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
}
/** @brief Train the neural network using mini-batch gradient descent */
void trainModel(float learningRate, int batchSize, bool initWeights,
                dim3 blockSize)
{
    trainModelTimer.Start(); // Start timing the training process

    // Set random seed for reproducibility
    std::mt19937 gen(0);

    // Initialize weights of the neural network
    if (initWeights)
    {
        MY_TIMER.Start(); // Start timing the weight initialization process

        // Initialize weights of the neural network using He initialization
        initializeWeights(W1, W1_COLS, W1_ROWS, gen);
        initializeWeights(W2, W2_COLS, W2_ROWS, gen);
        initializeWeights(W3, W3_COLS, W3_ROWS, gen);

        MY_TIMER.Stop(); // Stop timing the weight initialization process
        printf("[>] Weight initialization time: %.6f ms\n", MY_TIMER.Elapsed());
    }

    // Set forward pass hints
    /*
    CHECK(cudaMemAdvise(trainData, NUM_TRAIN * INPUT_SIZE * sizeof(float), cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
    CHECK(cudaMemPrefetchAsync(trainData, NUM_TRAIN * INPUT_SIZE * sizeof(float), cudaCpuDeviceId));
    CHECK(cudaMemAdvise(trainLabels, NUM_TRAIN * sizeof(int), cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
    CHECK(cudaMemPrefetchAsync(trainLabels, NUM_TRAIN * sizeof(int), cudaCpuDeviceId));
    */
    CHECK(cudaMemAdvise(mb_Y, NUM_CLASSES * batchSize * sizeof(float), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));
    CHECK(cudaMemAdvise(trainIndices, NUM_TRAIN * sizeof(int), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));

    // Train the model using mini-batch gradient descent
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch)
    {
        printf("[>] Epoch %3d/%d, ", epoch + 1, NUM_EPOCHS);

        // Shuffle the training data indices
        std::shuffle(trainIndices, trainIndices + NUM_TRAIN, gen);

        // Loop over mini-batches
        for (int mb_StartIdx = 0; mb_StartIdx < NUM_TRAIN; mb_StartIdx += batchSize)
        {
            // Break if the last mini-batch is smaller than batchSize
            if (mb_StartIdx + batchSize > NUM_TRAIN)
            {
                break;
            }

            // Get the current mini-batch indices
            int *mb_Indices = trainIndices + mb_StartIdx;

            // ********** Load mini-batch data and labels **********
            copyDataTimer.Start(); // Start timing the data copy process
            // Copy data to mini-batch and one-hot encode labels
            prepareMiniBatchData(trainData, trainLabels, mb_Indices, mb_X1, mb_Y, batchSize, blockSize);
            copyDataTimer.Stop(); // Stop timing the data copy process
            totalTimeCopyData += copyDataTimer.Elapsed();

            // Prefetch input matrices to GPU memory
            // CHECK(cudaMemPrefetchAsync(mb_X1, X1_ROWS * batchSize * sizeof(float), DEFAULT_GPU_DEVICE_ID));
            // Forward pass through the neural network
            forward(mb_X1, mb_X2, mb_X3, mb_A1, mb_A2, mb_A3,
                    batchSize, totalTimeForwardMiniBatch, blockSize);

            // Backward pass through the neural network
            backward(blockSize);

            // Update weights using mini-batch gradient descent
            updateWeights(learningRate, batchSize, blockSize);
        }

        // ********** Compute loss and accuracy on the full training set **********
        evaluateTimer.Start(); // Start timing the evaluation process

        // Compute Cross-Entropy loss on the full training set
        forward(train_X1, train_X2, train_X3, train_A1, train_A2, train_A3,
                NUM_TRAIN, totalTimeForwardTrain, blockSize);
        computeCrossEntropyLoss(train_Y, train_A3, trainLoss, NUM_CLASSES, NUM_TRAIN, blockSize);
        printf("Train loss: %.10f, ", *trainLoss); // Print with 10 decimal places

        // Compute accuracy on the full training set
        predictLabels(train_A3, trainPred, NUM_CLASSES, NUM_TRAIN, blockSize);
        int correct = 0;
        for (int i = 0; i < NUM_TRAIN; ++i)
        {
            correct += (trainPred[i] == trainLabels[i]);
        }
        float accuracy = static_cast<float>(correct) / NUM_TRAIN;
        printf("Train accuracy: %.6f\n", accuracy); // Print with 6 decimal places

        evaluateTimer.Stop(); // Stop timing the evaluation process
        totalTimeEvaluateTrain += evaluateTimer.Elapsed();
    }

    trainModelTimer.Stop(); // Stop timing the training process
}

// =======================================================================
// [========     PREPARING INPUT DATA FOR THE NEURAL NETWORK     ========]
// -----------------------------------------------------------------------
/** @brief Prepare the necessary data for the neural network */
void prepareData(dim3 blockSize)
{
    MY_TIMER.Start(); // Start timing the data preparation process

    // ********** One-hot encode ground truth labels **********
    // For train dataset
    CHECK(cudaMemAdvise(train_Y, NUM_CLASSES * NUM_TRAIN * sizeof(float), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));
    CHECK(cudaMemAdvise(trainLabels, NUM_TRAIN * sizeof(int), cudaMemAdviseSetReadMostly, DEFAULT_GPU_DEVICE_ID));
    CHECK(cudaMemPrefetchAsync(trainLabels, NUM_TRAIN * sizeof(int), DEFAULT_GPU_DEVICE_ID));
    oneHotEncoding(trainLabels, train_Y, NUM_CLASSES, NUM_TRAIN, blockSize);
    // For test dataset
    CHECK(cudaMemAdvise(test_Y, NUM_CLASSES * NUM_TEST * sizeof(float), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));
    CHECK(cudaMemAdvise(testLabels, NUM_TEST * sizeof(int), cudaMemAdviseSetReadMostly, DEFAULT_GPU_DEVICE_ID));
    CHECK(cudaMemPrefetchAsync(testLabels, NUM_TEST * sizeof(int), DEFAULT_GPU_DEVICE_ID));
    oneHotEncoding(testLabels, test_Y, NUM_CLASSES, NUM_TEST, blockSize);

    // ********** Add bias unit to the input matrices for each layer **********
    // Mini-batch
    CHECK(cudaMemAdvise(mb_X1, BATCH_SIZE * sizeof(float), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));
    CHECK(cudaMemAdvise(mb_X2, BATCH_SIZE * sizeof(float), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));
    CHECK(cudaMemAdvise(mb_X3, BATCH_SIZE * sizeof(float), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));
    addBiasUnit(mb_X1, BATCH_SIZE, blockSize);
    addBiasUnit(mb_X2, BATCH_SIZE, blockSize);
    addBiasUnit(mb_X3, BATCH_SIZE, blockSize);
    // Full train dataset
    CHECK(cudaMemAdvise(train_X1, NUM_TRAIN * sizeof(float), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));
    CHECK(cudaMemAdvise(train_X2, NUM_TRAIN * sizeof(float), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));
    CHECK(cudaMemAdvise(train_X3, NUM_TRAIN * sizeof(float), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));
    addBiasUnit(train_X1, NUM_TRAIN, blockSize);
    addBiasUnit(train_X2, NUM_TRAIN, blockSize);
    addBiasUnit(train_X3, NUM_TRAIN, blockSize);
    // Full test dataset
    CHECK(cudaMemAdvise(test_X1, NUM_TEST * sizeof(float), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));
    CHECK(cudaMemAdvise(test_X2, NUM_TEST * sizeof(float), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));
    CHECK(cudaMemAdvise(test_X3, NUM_TEST * sizeof(float), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));
    addBiasUnit(test_X1, NUM_TEST, blockSize);
    addBiasUnit(test_X2, NUM_TEST, blockSize);
    addBiasUnit(test_X3, NUM_TEST, blockSize);

    // ********** Prepare the input for the first layer **********
    // For train dataset
    CHECK(cudaMemAdvise(train_X1 + NUM_TRAIN, NUM_TRAIN * INPUT_SIZE * sizeof(float), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));
    CHECK(cudaMemAdvise(trainData, NUM_TRAIN * INPUT_SIZE * sizeof(float), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));
    CHECK(cudaMemPrefetchAsync(trainData, NUM_TRAIN * INPUT_SIZE * sizeof(float), DEFAULT_GPU_DEVICE_ID));
    transposeMatrix(trainData, NUM_TRAIN, INPUT_SIZE, train_X1 + NUM_TRAIN, blockSize);
    // For test dataset
    CHECK(cudaMemAdvise(test_X1 + NUM_TEST, NUM_TEST * INPUT_SIZE * sizeof(float), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));
    CHECK(cudaMemAdvise(testData, NUM_TEST * INPUT_SIZE * sizeof(float), cudaMemAdviseSetPreferredLocation, DEFAULT_GPU_DEVICE_ID));
    CHECK(cudaMemPrefetchAsync(testData, NUM_TEST * INPUT_SIZE * sizeof(float), DEFAULT_GPU_DEVICE_ID));
    transposeMatrix(testData, NUM_TEST, INPUT_SIZE, test_X1 + NUM_TEST, blockSize);

    // ********** Fill the array with indices from 0 to NUM_TRAIN **********
    // Reference: https://en.cppreference.com/w/cpp/algorithm/iota
    CHECK(cudaMemAdvise(trainIndices, NUM_TRAIN * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    std::iota(trainIndices, trainIndices + NUM_TRAIN, 0);

    MY_TIMER.Stop(); // Stop timing the data preparation process
    printf("[>] Data prepared in %.3f ms\n", MY_TIMER.Elapsed());
}
/** @brief Load the dataset from files */
void loadDataset()
{
    MY_TIMER.Start(); // Start timing the loading process

    // Set some hints about the data and do some prefetching
    CHECK(cudaMemAdvise(trainData, NUM_TRAIN * INPUT_SIZE * sizeof(float), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CHECK(cudaMemAdvise(testData, NUM_TEST * INPUT_SIZE * sizeof(float), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CHECK(cudaMemAdvise(trainLabels, NUM_TRAIN * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CHECK(cudaMemAdvise(testLabels, NUM_TEST * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));

    const std::string base_dir = "./";
    if (!loadImageData(base_dir + "train-images-idx3-ubyte", trainData, NUM_TRAIN, INPUT_SIZE))
    {
        throw std::runtime_error("Failed to load training data");
    }
    if (!loadImageData(base_dir + "t10k-images-idx3-ubyte", testData, NUM_TEST, INPUT_SIZE))
    {
        throw std::runtime_error("Failed to load test data");
    }
    if (!loadLabels(base_dir + "train-labels-idx1-ubyte", trainLabels, NUM_TRAIN))
    {
        throw std::runtime_error("Failed to load training labels");
    }
    if (!loadLabels(base_dir + "t10k-labels-idx1-ubyte", testLabels, NUM_TEST))
    {
        throw std::runtime_error("Failed to load test labels");
    }

    MY_TIMER.Stop(); // Stop timing the loading process
    printf("[>] Dataset loaded in %.3f ms\n", MY_TIMER.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU Info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);
    printf("****************************\n\n");
}
void initTimer()
{
    for (int i = 0; i < NUM_LAYERS; ++i)
    {
        totalTimeForwardTrain[i] = 0.0f;
        totalTimeForwardTest[i] = 0.0f;
        totalTimeForwardMiniBatch[i] = 0.0f;
        totalTimeBackwardMiniBatch[i] = 0.0f;
    }
}
int main(int argc, char *argv[])
{
    printDeviceInfo();
    dim3 blockSize(DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE);
    if (argc > 1)
    {
        blockSize.x = atoi(argv[1]);
        blockSize.y = atoi(argv[2]);
    }

    // Allocate memory for all declared pointers/arrays/matrices
    ALLOCATE_MEMORY();
    // Create all streams
    CREATE_STREAMS();

    // Load Fashion-MNIST dataset
    printf("Loading Fashion-MNIST dataset...\n");
    loadDataset();

    // Prepare the necessary data for the neural network
    prepareData(blockSize);
    // Initialize the timer
    initTimer();

    // Train the neural network using mini-batch gradient descent
    printf("Training the Neural Network using mini-batch gradient descent...\n");
    printf("==================+=================+========================\n");
    printf("[ NUM_EPOCHS = %2d | BATCH_SIZE = %d | LEARNING_RATE = %5.3f ]\n", NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE);
    printf("------------------+-----------------+------------------------\n");
    trainModel(LEARNING_RATE, BATCH_SIZE, true, blockSize);

    // Evaluate the model on the full test set, calculate accuracy
    printf("\nEvaluating the model on the full test set...\n");
    forward(test_X1, test_X2, test_X3, test_A1, test_A2, test_A3,
            NUM_TEST, totalTimeForwardTest, blockSize);
    computeCrossEntropyLoss(test_Y, test_A3, testLoss, NUM_CLASSES, NUM_TEST, blockSize);
    printf("[>] Test loss: %.10f\n", *testLoss); // Print with 10 decimal places
    predictLabels(test_A3, testPred, NUM_CLASSES, NUM_TEST, blockSize);
    int correct = 0;
    for (int i = 0; i < NUM_TEST; ++i)
    {
        correct += (testPred[i] == testLabels[i]);
    }
    float accuracy = static_cast<float>(correct) / NUM_TEST;
    printf("[>] Test accuracy: %.6f\n", accuracy); // Print with 6 decimal places

    // MODEL RUN TIME REPORT
    printf("\n[========== MODEL RUN TIME REPORT ==========]\n");
    printf("(*) Average time for forward and backward pass of each layer in 1 epoch\n");
    printf("+----------+--------------+---------------+\n");
    printf("| Layer    | Forward (ms) | Backward (ms) |\n");
    printf("|==========+==============+===============|\n");
    printf("| Hidden 1 | %12.3f | %13.3f |\n", totalTimeForwardMiniBatch[0] / NUM_EPOCHS, totalTimeBackwardMiniBatch[0] / NUM_EPOCHS);
    printf("| Hidden 2 | %12.3f | %13.3f |\n", totalTimeForwardMiniBatch[1] / NUM_EPOCHS, totalTimeBackwardMiniBatch[1] / NUM_EPOCHS);
    printf("| Output   | %12.3f | %13.3f |\n", totalTimeForwardMiniBatch[2] / NUM_EPOCHS, totalTimeBackwardMiniBatch[2] / NUM_EPOCHS);
    printf("*----------*--------------*---------------*\n\n");
    printf("(*) Average time to update weights in 1 epoch: %.3f ms\n", totalTimeUpdateWeights / NUM_EPOCHS);
    printf("(*) Average time for training the model in 1 epoch: %.3f ms\n", (trainModelTimer.Elapsed() - totalTimeUpdateWeights) / NUM_EPOCHS);
    printf("(*) Average time for forward pass when training model in 1 epoch: %.3f ms\n", (totalTimeForwardMiniBatch[0] + totalTimeForwardMiniBatch[1] + totalTimeForwardMiniBatch[2]) / NUM_EPOCHS);
    printf("(*) Average time for backward pass when training model in 1 epoch: %.3f ms\n", (totalTimeBackwardMiniBatch[0] + totalTimeBackwardMiniBatch[1] + totalTimeBackwardMiniBatch[2]) / NUM_EPOCHS);
    printf("(*) Average time for evaluating the model on the full training set: %.3f ms\n", totalTimeEvaluateTrain / NUM_EPOCHS);
    printf("(*) Average time for preparing mini-batch data: %.3f ms\n", totalTimeCopyData / NUM_EPOCHS);
    printf("(*) Average time for forward pass entire training set: %.3f ms\n", (totalTimeForwardTrain[0] + totalTimeForwardTrain[1] + totalTimeForwardTrain[2]) / NUM_EPOCHS);
    printf("(*) Average time for forward pass entire test set: %.3f ms\n", totalTimeForwardTest[0] + totalTimeForwardTest[1] + totalTimeForwardTest[2]);
    printf("(*) Total time for softmax activation function: %.3f ms\n", totalTimeSoftmax);
    printf("(*) Total time for matrix multiplication: %.3f ms\n", totalTimeMatMul);
    printf("(*) Total time for matrix transpose: %.3f ms\n", totalTimeTranspose);
    printf("[===========================================]\n\n");

    // Destroy all streams
    DESTROY_STREAMS();
    // Free memory for all declared pointers/arrays/matrices
    FREE_MEMORY();
    
    return 0;
}
