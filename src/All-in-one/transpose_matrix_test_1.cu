/**
 * @author ...
 * @date ...
 * @note ...
 */
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
// =======================================================================

// ============ TIMING THE TRAINING PROCESS OF THE NETWORK ===============
GpuTimer MY_TIMER; // Timer for the entire neural network
// Total time taken by forward pass at each layer
float totalTimeForwardLayer1 = 0.0f;
float totalTimeForwardLayer2 = 0.0f;
float totalTimeForwardLayer3 = 0.0f;
// Total time taken by backward pass at each layer
float totalTimeBackwardLayer3 = 0.0f;
float totalTimeBackwardLayer2 = 0.0f;
float totalTimeBackwardLayer1 = 0.0f;
// Total time to update the weights of the neural network
float totalTimeUpdateWeights = 0.0f;
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
    W1 = (float *)malloc(W1_ROWS * W1_COLS * sizeof(float));
    W2 = (float *)malloc(W2_ROWS * W2_COLS * sizeof(float));
    W3 = (float *)malloc(W3_ROWS * W3_COLS * sizeof(float));
    // =======================================================================

    // ========= FORWARD PASS WHEN TRAINING MODEL (USING MINI-BATCH) =========
    mb_Y = (float *)malloc(NUM_CLASSES * BATCH_SIZE * sizeof(float));
    mb_X1 = (float *)malloc(X1_ROWS * BATCH_SIZE * sizeof(float));
    mb_X2 = (float *)malloc(X2_ROWS * BATCH_SIZE * sizeof(float));
    mb_X3 = (float *)malloc(X3_ROWS * BATCH_SIZE * sizeof(float));
    mb_A1 = mb_X2 + BATCH_SIZE; // mb_A1 = mb_X2[1:,:]
    mb_A2 = mb_X3 + BATCH_SIZE; // mb_A2 = mb_X3[1:,:]
    mb_A3 = (float *)malloc(A3_ROWS * BATCH_SIZE * sizeof(float));
    // =======================================================================

    // ======== BACKWARD PASS WHEN TRAINING MODEL (USING MINI-BATCH) =========
    // For layer 3
    mb_dLdZ3 = (float *)malloc(W3_ROWS * BATCH_SIZE * sizeof(float));
    mb_X3T = (float *)malloc(BATCH_SIZE * X3_ROWS * sizeof(float));
    mb_dLdW3 = (float *)malloc(W3_ROWS * W3_COLS * sizeof(float));
    mb_W3T = (float *)malloc(W3_COLS * W3_ROWS * sizeof(float));
    mb_dLdX3 = (float *)malloc(X3_ROWS * BATCH_SIZE * sizeof(float));
    // For layer 2
    mb_dLdA2 = mb_dLdX3 + BATCH_SIZE; // mb_dLdA2 = mb_dLdX3[1:,:]
    mb_dLdZ2 = (float *)malloc(W2_ROWS * BATCH_SIZE * sizeof(float));
    mb_X2T = (float *)malloc(BATCH_SIZE * X2_ROWS * sizeof(float));
    mb_dLdW2 = (float *)malloc(W2_ROWS * W2_COLS * sizeof(float));
    mb_W2T = (float *)malloc(W2_COLS * W2_ROWS * sizeof(float));
    mb_dLdX2 = (float *)malloc(X2_ROWS * BATCH_SIZE * sizeof(float));
    // For layer 1
    mb_dLdA1 = mb_dLdX2 + BATCH_SIZE; // mb_dLdA1 = mb_dLdX2[1:,:]
    mb_dLdZ1 = (float *)malloc(W1_ROWS * BATCH_SIZE * sizeof(float));
    mb_X1T = (float *)malloc(BATCH_SIZE * X1_ROWS * sizeof(float));
    mb_dLdW1 = (float *)malloc(W1_ROWS * W1_COLS * sizeof(float));
    // =======================================================================

    // =================== ORIGINAL FASHION-MNIST DATASET ====================
    trainData = (float *)malloc(NUM_TRAIN * INPUT_SIZE * sizeof(float));
    trainLabels = (int *)malloc(NUM_TRAIN * sizeof(int));
    trainPred = (int *)malloc(NUM_TRAIN * sizeof(int));
    trainLoss = (float *)malloc(sizeof(float));
    trainIndices = (int *)malloc(NUM_TRAIN * sizeof(int));
    testData = (float *)malloc(NUM_TEST * INPUT_SIZE * sizeof(float));
    testLabels = (int *)malloc(NUM_TEST * sizeof(int));
    testPred = (int *)malloc(NUM_TEST * sizeof(int));
    testLoss = (float *)malloc(sizeof(float));
    // =======================================================================

    // ======== FORWARD PASS WHEN COMPUTING LOSS (FULL TRAINING SET) =========
    train_Y = (float *)malloc(NUM_CLASSES * NUM_TRAIN * sizeof(float));
    train_X1 = (float *)malloc(X1_ROWS * NUM_TRAIN * sizeof(float));
    train_X2 = (float *)malloc(X2_ROWS * NUM_TRAIN * sizeof(float));
    train_X3 = (float *)malloc(X3_ROWS * NUM_TRAIN * sizeof(float));
    train_A1 = train_X2 + NUM_TRAIN; // train_A1 = train_X2[1:,:]
    train_A2 = train_X3 + NUM_TRAIN; // train_A2 = train_X3[1:,:]
    train_A3 = (float *)malloc(A3_ROWS * NUM_TRAIN * sizeof(float));
    // =======================================================================

    // ======== FORWARD PASS WHEN EVALUATING MODEL (FULL TEST SET) ===========
    test_Y = (float *)malloc(NUM_TEST * NUM_CLASSES * sizeof(float));
    test_X1 = (float *)malloc(X1_ROWS * NUM_TEST * sizeof(float));
    test_X2 = (float *)malloc(X2_ROWS * NUM_TEST * sizeof(float));
    test_X3 = (float *)malloc(X3_ROWS * NUM_TEST * sizeof(float));
    test_A1 = test_X2 + NUM_TEST; // test_A1 = test_X2[1:,:]
    test_A2 = test_X3 + NUM_TEST; // test_A2 = test_X3[1:,:]
    test_A3 = (float *)malloc(A3_ROWS * NUM_TEST * sizeof(float));
    // =======================================================================
}
/** @brief Free memory for all declared pointers/arrays/matrices */
void FREE_MEMORY()
{
    // ================ WEIGHT MATRICES OF THE NEURAL NETWORK ================
    free(W1);
    free(W2);
    free(W3);
    // =======================================================================

    // ========= FORWARD PASS WHEN TRAINING MODEL (USING MINI-BATCH) =========
    free(mb_Y);
    free(mb_X1);
    free(mb_X2);
    free(mb_X3);
    mb_A1 = nullptr; // mb_A1 = mb_X2[1:,:]
    mb_A2 = nullptr; // mb_A2 = mb_X3[1:,:]
    free(mb_A3);
    // =======================================================================

    // ======== BACKWARD PASS WHEN TRAINING MODEL (USING MINI-BATCH) =========
    // For layer 3
    free(mb_dLdZ3);
    free(mb_X3T);
    free(mb_dLdW3);
    free(mb_W3T);
    free(mb_dLdX3);
    // For layer 2
    mb_dLdA2 = nullptr; // mb_dLdA2 = mb_dLdX3[1:,:]
    free(mb_dLdZ2);
    free(mb_X2T);
    free(mb_dLdW2);
    free(mb_W2T);
    free(mb_dLdX2);
    // For layer 1
    mb_dLdA1 = nullptr; // mb_dLdA1 = mb_dLdX2[1:,:]
    free(mb_dLdZ1);
    free(mb_X1T);
    free(mb_dLdW1);
    // =======================================================================

    // =================== ORIGINAL FASHION-MNIST DATASET ====================
    free(trainData);
    free(trainLabels);
    free(trainPred);
    free(trainLoss);
    free(trainIndices);
    free(testData);
    free(testLabels);
    free(testPred);
    free(testLoss);
    // =======================================================================

    // ======== FORWARD PASS WHEN COMPUTING LOSS (FULL TRAINING SET) =========
    free(train_Y);
    free(train_X1);
    free(train_X2);
    free(train_X3);
    train_A1 = nullptr; // train_A1 = train_X2[1:,:]
    train_A2 = nullptr; // train_A2 = train_X3[1:,:]
    free(train_A3);
    // =======================================================================

    // ======== FORWARD PASS WHEN EVALUATING MODEL (FULL TEST SET) ===========
    free(test_Y);
    free(test_X1);
    free(test_X2);
    free(test_X3);
    test_A1 = nullptr; // test_A1 = test_X2[1:,:]
    test_A2 = nullptr; // test_A2 = test_X3[1:,:]
    free(test_A3);
    // =======================================================================
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
void addBiasUnit(float *input, int numCols, dim3 blockSize = dim3(1, 1))
{
    //Allocate device memory
    float * d_in;
    CHECK(cudaMalloc(&d_in, numCols * sizeof(float)));
    CHECK(cudaMemcpy(d_in, input, numCols * sizeof(float), cudaMemcpyHostToDevice));
    // Calculate grid size to cover the entire input matrix
    int numThreadsPerBlock = blockSize.x * blockSize.y;
    int numBlocksPerGrid = (numCols - 1) / numThreadsPerBlock + 1;
    // Call kernel function
    addBiasUnitKernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_in, numCols);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors
    // Free device memory
    CHECK(cudaMemcpy(input, d_in, numCols * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_in));
}

// =======================================================================
// [========  RELU ACTIVATION FUNCTION WHEN FORWARD PROPAGATING  ========]
// -----------------------------------------------------------------------
/** @brief Kernel function to apply ReLU activation function to input matrix */
__global__ void applyReluKernel(float *input, int numRows, int numCols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows && col < numCols)
    {
        int index = row * numCols + col;
        input[index] = fmaxf(0.0f, input[index]);
    }
}
/** @brief Apply ReLU activation function to input matrix */
void applyRelu(float *input, int numRows, int numCols, dim3 blockSize = dim3(1, 1))
{
    //Allocate device memory
    float * d_in;
    CHECK(cudaMalloc(&d_in, numRows * numCols * sizeof(float)));
    CHECK(cudaMemcpy(d_in, input, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice));
    // Calculate grid size to cover the entire input matrix
    dim3 gridSize((numCols - 1) / blockSize.x + 1,
                  (numRows - 1) / blockSize.y + 1);
    // Call kernel function
    applyReluKernel<<<gridSize, blockSize>>>(d_in, numRows, numCols);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors
    // Free device memory
    CHECK(cudaMemcpy(input, d_in, numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_in));
}

// =======================================================================
// [========  RELU ACTIVATION FUNCTION WHEN BACKWARD PROPAGATING  =======]
// -----------------------------------------------------------------------
/** @brief Kernel function to compute the derivative of ReLU activation function */
__global__ void reluBackwardKernel(float *A, float *dLdA, float *dLdZ, int numRows, int numCols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows && col < numCols)
    {
        int index = row * numCols + col;
        dLdZ[index] = (A[index] > 0.0f) ? dLdA[index] : 0.0f;
    }
}
/** @brief Backward pass for ReLU activation function */
void reluBackward(float *A, float *dLdA, float *dLdZ, int numRows, int numCols,
                  dim3 blockSize = dim3(1, 1))
{
    //Allocate device memory
    float *d_A, *d_dLdA, *d_dLdZ;
    CHECK(cudaMalloc(&d_A, numRows * numCols * sizeof(float)));
    CHECK(cudaMalloc(&d_dLdA, numRows * numCols * sizeof(float)));
    CHECK(cudaMalloc(&d_dLdZ, numRows * numCols * sizeof(float)));
    CHECK(cudaMemcpy(d_A, A, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dLdA, dLdA, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice));
    // Calculate grid size to cover the entire input matrix
    dim3 gridSize((numCols - 1) / blockSize.x + 1,
                  (numRows - 1) / blockSize.y + 1);
    // Call kernel function
    reluBackwardKernel<<<gridSize, blockSize>>>(d_A, d_dLdA, d_dLdZ, numRows, numCols);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors
    // Free device memory
    CHECK(cudaMemcpy(dLdZ, d_dLdZ, numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_dLdA));
    CHECK(cudaFree(d_dLdZ));
}

// =======================================================================
// [========  SOFTMAX ACTIVATION FUNCTION WHEN FORWARD PROPAGATING  =====]
// -----------------------------------------------------------------------
/** @brief Kernel function to apply softmax activation function to input matrix (column-wise) */
__global__ void applySoftmaxKernel(float *input, int numRows, int numCols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < numCols)
    {
        // Find maximum value in the column
        float maxVal = -FLT_MAX;
        for (int row = 0; row < numRows; ++row)
        {
            maxVal = fmaxf(maxVal, input[row * numCols + col]);
        }

        // Compute sum of exponentials of input values
        float expSum = 0.0f;
        for (int row = 0; row < numRows; ++row)
        {
            input[row * numCols + col] = expf(input[row * numCols + col] - maxVal);
            expSum += input[row * numCols + col];
        }

        // Normalize the column
        for (int row = 0; row < numRows; ++row)
        {
            input[row * numCols + col] /= expSum;
        }
    }
}
/** @brief Apply softmax activation function to input matrix (column-wise) */
void applySoftmax(float *input, int numRows, int numCols, dim3 blockSize = dim3(1, 1))
{
    //Allocate device memory
    float * d_in;
    CHECK(cudaMalloc(&d_in, numRows * numCols * sizeof(float)));
    CHECK(cudaMemcpy(d_in, input, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice));
    // Calculate grid size to cover the entire input matrix
    int numThreadsPerBlock = blockSize.x * blockSize.y;
    int numBlocksPerGrid = (numCols - 1) / numThreadsPerBlock + 1;
    // Call kernel function
    applySoftmaxKernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_in, numRows, numCols);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors
    // Free device memory
    CHECK(cudaMemcpy(input, d_in, numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_in));
    
}

// =======================================================================
// [========  SUBTRACTING TWO MATRICES (ELEMENT-WISE) FUNCTION   ========]
// -----------------------------------------------------------------------
/** @brief Kernel function to subtract two matrices (element-wise): C = A - B */
__global__ void subtractMatricesKernel(float *A, float *B, float *C, int numRows, int numCols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows && col < numCols)
    {
        int index = row * numCols + col;
        C[index] = A[index] - B[index];
    }
}
/** @brief Subtract two matrices (element-wise): C = A - B */
void subtractMatrices(float *A, float *B, float *C, int numRows, int numCols,
                      dim3 blockSize = dim3(1, 1))
{
    //Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, numRows * numCols * sizeof(float)));
    CHECK(cudaMalloc(&d_B, numRows * numCols * sizeof(float)));
    CHECK(cudaMalloc(&d_C, numRows * numCols * sizeof(float)));
    CHECK(cudaMemcpy(d_A, A, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice));
    // Calculate grid size to cover the entire input matrix
    dim3 gridSize((numCols - 1) / blockSize.x + 1,
                  (numRows - 1) / blockSize.y + 1);
    // Call kernel function
    subtractMatricesKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, numRows, numCols);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors
    // Free device memory
    CHECK(cudaMemcpy(C, d_C, numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
}

// =======================================================================
// [========                MATRIX MULTIPLICATION                ========]
// -----------------------------------------------------------------------
/** @brief Kernel function to compute matrix multiplication: C = A * B */
__global__ void matrixMultiplyKernel(float *A, float *B, float *C,
                                     int numRowsA, int numColsA, int numColsB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRowsA && col < numColsB)
    {
        float sum = 0.0f;
        for (int i = 0; i < numColsA; ++i)
        {
            sum += A[row * numColsA + i] * B[i * numColsB + col];
        }
        C[row * numColsB + col] = sum;
    }
}
/** @brief Compute matrix multiplication: C = A * B */
void matrixMultiply(float *A, float *B, float *C,
                    int numRowsA, int numColsA, int numColsB,
                    dim3 blockSize = dim3(1, 1))
{
    //Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, numRowsA * numColsA * sizeof(float)));
    CHECK(cudaMalloc(&d_B, numColsA * numColsB * sizeof(float)));
    CHECK(cudaMalloc(&d_C, numRowsA * numColsB * sizeof(float)));
    CHECK(cudaMemcpy(d_A, A, numRowsA * numColsA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, numColsA * numColsB * sizeof(float), cudaMemcpyHostToDevice));
    // Calculate grid size to cover the entire output matrix
    dim3 gridSize((numColsB - 1) / blockSize.x + 1,
                  (numRowsA - 1) / blockSize.y + 1);
    // Call kernel function
    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, numRowsA, numColsA, numColsB);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors
    // Free device memory
    CHECK(cudaMemcpy(C, d_C, numRowsA * numColsB * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
}

// =======================================================================
// [========                TRANSPOSE OF A MATRIX                ========]
// -----------------------------------------------------------------------
/** @brief Kernel function to compute the transpose of a matrix */
__global__ void transposeMatrixKernel(float *input, int numRows, int numCols, float *output)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows && col < numCols)
    {
        output[col * numRows + row] = input[row * numCols + col];
    }
}

__global__ void transposeMatrixKernelV2(float *in, int numRows, int numCols, float *out) {
    __shared__ float s_blkData[32][33];

    int iR = blockIdx.y * blockDim.y + threadIdx.y;
    int iC = blockIdx.x * blockDim.x + threadIdx.x;
    if (iR < numRows && iC < numCols) s_blkData[threadIdx.x][threadIdx.y] = in[iR * numCols + iC];
    __syncthreads();

    int oC = blockIdx.y * blockDim.y + threadIdx.x;
    int oR = blockIdx.x * blockDim.x + threadIdx.y;
    if (oR < numCols && oC < numRows) out[oR * numRows + oC] = s_blkData[threadIdx.y][threadIdx.x];
}

/** @brief Compute the transpose of a matrix */
void transposeMatrix(float *input, int numRows, int numCols, float *output,
                     dim3 blockSize = dim3(1, 1))
{
    //Allocate device memory
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, numRows * numCols * sizeof(float)));
    CHECK(cudaMalloc(&d_out, numRows * numCols * sizeof(float)));
    CHECK(cudaMemcpy(d_in, input, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice));
    // Calculate grid size to cover the entire input matrix
    dim3 gridSize((numCols - 1) / blockSize.x + 1,
                  (numRows - 1) / blockSize.y + 1);
    // Call kernel function
    transposeMatrixKernelV2<<<gridSize, blockSize>>>(d_in, numRows, numCols, d_out);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors
    // Free device memory
    CHECK(cudaMemcpy(output, d_out, numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
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
                    dim3 blockSize = dim3(1, 1))
{
    //Allocate device memory
    float *d_Y;
    int *d_labels;
    CHECK(cudaMalloc(&d_labels, numSamples * sizeof(int)));
    CHECK(cudaMalloc(&d_Y, numSamples * numClasses * sizeof(float)));
    CHECK(cudaMemcpy(d_labels, labels, numSamples * sizeof(int), cudaMemcpyHostToDevice));
    // Compute grid size to cover the entire output matrix
    int numThreadsPerBlock = blockSize.x * blockSize.y;
    int numBlocksPerGrid = (numSamples - 1) / numThreadsPerBlock + 1;
    // Call kernel function
    oneHotEncodingKernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_labels, d_Y, numClasses, numSamples);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors
    // Free device memory
    CHECK(cudaMemcpy(Y, d_Y, numSamples * numClasses * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_labels));
    CHECK(cudaFree(d_Y));
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
                             dim3 blockSize = dim3(1, 1))
{
    //Allocate device memory
    float *d_P, *d_Y, *d_loss;
    CHECK(cudaMalloc(&d_P, numSamples * numClasses * sizeof(float)));
    CHECK(cudaMalloc(&d_Y, numSamples * numClasses * sizeof(float)));
    CHECK(cudaMalloc(&d_loss, 1 * sizeof(float)));
    CHECK(cudaMemcpy(d_P, P, numSamples * numClasses * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Y, Y, numSamples * numClasses * sizeof(float), cudaMemcpyHostToDevice));
    // Compute grid size to cover the entire output matrix
    int numThreadsPerBlock = blockSize.x * blockSize.y;
    int numBlocksPerGrid = (numSamples - 1) / numThreadsPerBlock + 1;
    // Init loss to zero
    *loss = 0.0f;
    //notice......................
    // Call kernel function
    crossEntropyLossKernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_Y, d_P, d_loss, numClasses, numSamples);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors
    // Free device memory
    CHECK(cudaMemcpy(loss, d_loss, 1 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_loss));
    CHECK(cudaFree(d_Y));
    CHECK(cudaFree(d_P));
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
        float maxProb = -FLT_MAX;
        int predLabel = -1;
        for (int row = 0; row < numClasses; ++row)
        {
            if (P[row * numSamples + sample] > maxProb)
            {
                maxProb = P[row * numSamples + sample];
                predLabel = row;
            }
        }
        labels[sample] = predLabel;
    }
}
/** @brief Predict labels from probabilities */
void predictLabels(float *P, int *labels, int numClasses, int numSamples,
                   dim3 blockSize = dim3(1, 1))
{
    //Allocate device memory
    int *d_labels;
    float *d_P;
    CHECK(cudaMalloc(&d_P, numSamples * numClasses * sizeof(float)));
    CHECK(cudaMalloc(&d_labels, numSamples * sizeof(int)));
    CHECK(cudaMemcpy(d_P, P, numSamples * numClasses * sizeof(float), cudaMemcpyHostToDevice));
    // Compute grid size to cover the entire output matrix
    int numThreadsPerBlock = blockSize.x * blockSize.y;
    int numBlocksPerGrid = (numSamples - 1) / numThreadsPerBlock + 1;
    // Call kernel function
    predictLabelsKernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_P, d_labels, numClasses, numSamples);
    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors
    // Free device memory
    CHECK(cudaMemcpy(labels, d_labels, numSamples * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_labels));
    CHECK(cudaFree(d_P));
}

// =======================================================================
// [========  FORWARD PASS INPUT FOR EACH LAYER OF THE NETWORK   ========]
// -----------------------------------------------------------------------
/**
 * @brief Forward pass the input through neural network
 * @note Timing the forward pass of each layer when training the model
 */
void forward(float *X1, float *X2, float *X3, float *A1, float *A2, float *A3,
             int numSamples, bool isTraining, dim3 blockSize = dim3(1, 1))
{
    // ********** Forward pass for layer 1 **********
    if (isTraining)
    {
        MY_TIMER.Start();
    }

    // Z1 = W1 * X1
    matrixMultiply(W1, X1, A1, W1_ROWS, W1_COLS, numSamples, blockSize);
    // A1 = ReLU(Z1)
    applyRelu(A1, A1_ROWS, numSamples, blockSize);

    if (isTraining)
    {
        MY_TIMER.Stop();
        totalTimeForwardLayer1 += MY_TIMER.Elapsed();
    }

    // ********** Forward pass for layer 2 **********
    if (isTraining)
    {
        MY_TIMER.Start();
    }

    // Z2 = W2 * X2
    matrixMultiply(W2, X2, A2, W2_ROWS, W2_COLS, numSamples, blockSize);
    // A2 = ReLU(Z2)
    applyRelu(A2, A2_ROWS, numSamples, blockSize);

    if (isTraining)
    {
        MY_TIMER.Stop();
        totalTimeForwardLayer2 += MY_TIMER.Elapsed();
    }

    // ********** Forward pass for layer 3 **********
    if (isTraining)
    {
        MY_TIMER.Start();
    }

    // Z3 = W3 * X3
    matrixMultiply(W3, X3, A3, W3_ROWS, W3_COLS, numSamples, blockSize);
    // A3 = softmax(Z3)
    applySoftmax(A3, A3_ROWS, numSamples, blockSize);

    if (isTraining)
    {
        MY_TIMER.Stop();
        totalTimeForwardLayer3 += MY_TIMER.Elapsed();
    }
}

// =======================================================================
// [========          BACKWARD PASS WHEN TRAINING MODEL          ========]
// -----------------------------------------------------------------------
/** @brief Backward pass the gradients through neural network */
void backward(dim3 blockSize = dim3(1, 1))
{
    // ********** Backward pass for layer 3 **********
    MY_TIMER.Start(); // Timing the backward pass for layer 3

    // dLdZ3 = A3 - Y
    subtractMatrices(mb_A3, mb_Y, mb_dLdZ3, A3_ROWS, BATCH_SIZE, blockSize);
    // dLdW3 = dLdZ3 * X3T
    transposeMatrix(mb_X3, X3_ROWS, BATCH_SIZE, mb_X3T, blockSize);
    matrixMultiply(mb_dLdZ3, mb_X3T, mb_dLdW3, W3_ROWS, BATCH_SIZE, X3_ROWS, blockSize);
    // dLdX3 = W3T * dLdZ3
    transposeMatrix(W3, W3_ROWS, W3_COLS, mb_W3T, blockSize);
    matrixMultiply(mb_W3T, mb_dLdZ3, mb_dLdX3, W3_COLS, W3_ROWS, BATCH_SIZE, blockSize);

    MY_TIMER.Stop(); // Stop timing the backward pass for layer 3
    totalTimeBackwardLayer3 += MY_TIMER.Elapsed();

    // ********** Backward pass for layer 2 **********
    MY_TIMER.Start(); // Timing the backward pass for layer 2

    // mb_dLdA2 = mb_dLdX3[1:,:]
    // mb_dLdZ2 = mb_dLdA2 * relu'(mb_A2)
    reluBackward(mb_A2, mb_dLdA2, mb_dLdZ2, A2_ROWS, BATCH_SIZE, blockSize);
    // mb_dLdW2 = mb_dLdZ2 * mb_X2T
    transposeMatrix(mb_X2, X2_ROWS, BATCH_SIZE, mb_X2T, blockSize);
    matrixMultiply(mb_dLdZ2, mb_X2T, mb_dLdW2, W2_ROWS, BATCH_SIZE, X2_ROWS, blockSize);
    // mb_dLdX2 = mb_W2T * mb_dLdZ2
    transposeMatrix(W2, W2_ROWS, W2_COLS, mb_W2T, blockSize);
    matrixMultiply(mb_W2T, mb_dLdZ2, mb_dLdX2, W2_COLS, W2_ROWS, BATCH_SIZE, blockSize);

    MY_TIMER.Stop(); // Stop timing the backward pass for layer 2
    totalTimeBackwardLayer2 += MY_TIMER.Elapsed();

    // ********** Backward pass for layer 1 **********
    MY_TIMER.Start(); // Timing the backward pass for layer 1

    // mb_dLdA1 = mb_dLdX2[1:,:]
    // mb_dLdZ1 = mb_dLdA1 * relu'(mb_A1)
    reluBackward(mb_A1, mb_dLdA1, mb_dLdZ1, A1_ROWS, BATCH_SIZE, blockSize);
    // mb_dLdW1 = mb_dLdZ1 * mb_X1T
    transposeMatrix(mb_X1, X1_ROWS, BATCH_SIZE, mb_X1T, blockSize);
    matrixMultiply(mb_dLdZ1, mb_X1T, mb_dLdW1, W1_ROWS, BATCH_SIZE, X1_ROWS, blockSize);

    MY_TIMER.Stop(); // Stop timing the backward pass for layer 1
    totalTimeBackwardLayer1 += MY_TIMER.Elapsed();
}

// =======================================================================
// [========        UPDATE WEIGHTS OF THE NEURAL NETWORK         ========]
// -----------------------------------------------------------------------
/** @brief Kernel function to update weights of the neural network */
__global__ void updateWeightsKernel(float *dLdW, float *W, int numRows, int numCols,
                                    float learningRate, int batchSize)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows && col < numCols)
    {
        int index = row * numCols + col;
        W[index] -= learningRate * dLdW[index] / batchSize;
    }
}
/** @brief Update weights of the neural network */
void updateWeights(float learningRate, int batchSize, dim3 blockSize = dim3(1, 1))
{
    MY_TIMER.Start(); // Timing the update weights process

    //Allocate device memory
    float *d_mb_dLdW1, *d_mb_dLdW2, *d_mb_dLdW3, *d_W1, *d_W2, *d_W3;
    CHECK(cudaMalloc(&d_mb_dLdW1, W1_ROWS * W1_COLS * sizeof(float)));
    CHECK(cudaMalloc(&d_mb_dLdW2, W2_ROWS * W2_COLS * sizeof(float)));
    CHECK(cudaMalloc(&d_mb_dLdW3, W3_ROWS * W3_COLS * sizeof(float)));
    CHECK(cudaMalloc(&d_W1, W1_ROWS * W1_COLS * sizeof(float)));
    CHECK(cudaMalloc(&d_W2, W2_ROWS * W2_COLS * sizeof(float)));
    CHECK(cudaMalloc(&d_W3, W3_ROWS * W3_COLS * sizeof(float)));

    //Copy to device
    CHECK(cudaMemcpy(d_mb_dLdW1, mb_dLdW1, W1_ROWS * W1_COLS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mb_dLdW2, mb_dLdW2, W2_ROWS * W2_COLS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mb_dLdW3, mb_dLdW3, W3_ROWS * W3_COLS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_W1, W1, W1_ROWS * W1_COLS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_W2, W2, W2_ROWS * W2_COLS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_W3, W3, W3_ROWS * W3_COLS * sizeof(float), cudaMemcpyHostToDevice));

    // ******** Update weights for layer 1 ********
    dim3 gridSize_W1((W1_COLS - 1) / blockSize.x + 1,
                     (W1_ROWS - 1) / blockSize.y + 1);                 
    updateWeightsKernel<<<gridSize_W1, blockSize>>>(d_mb_dLdW1, d_W1, W1_ROWS, W1_COLS,
                                                    learningRate, batchSize);

    // ******** Update weights for layer 2 ********
    dim3 gridSize_W2((W2_COLS - 1) / blockSize.x + 1,
                     (W2_ROWS - 1) / blockSize.y + 1);
    updateWeightsKernel<<<gridSize_W2, blockSize>>>(d_mb_dLdW2, d_W2, W2_ROWS, W2_COLS,
                                                    learningRate, batchSize);

    // ******** Update weights for layer 3 ********
    dim3 gridSize_W3((W3_COLS - 1) / blockSize.x + 1,
                     (W3_ROWS - 1) / blockSize.y + 1);
    updateWeightsKernel<<<gridSize_W3, blockSize>>>(d_mb_dLdW3, d_W3, W3_ROWS, W3_COLS,
                                                    learningRate, batchSize);

    CHECK(cudaDeviceSynchronize()); // Synchronize to wait for kernel to finish
    CHECK(cudaGetLastError());      // Check for kernel launch errors

    MY_TIMER.Stop(); // Stop timing the update weights process
    totalTimeUpdateWeights += MY_TIMER.Elapsed();

    //Copy device to host
    CHECK(cudaMemcpy(W1, d_W1, W1_ROWS * W1_COLS * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(W2, d_W2, W2_ROWS * W2_COLS * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(W3, d_W3, W3_ROWS * W3_COLS * sizeof(float), cudaMemcpyDeviceToHost));

    //Free device memory
    CHECK(cudaFree(d_mb_dLdW1));
    CHECK(cudaFree(d_mb_dLdW2));
    CHECK(cudaFree(d_mb_dLdW3));
    CHECK(cudaFree(d_W1));
    CHECK(cudaFree(d_W2));
    CHECK(cudaFree(d_W3));
}

// =======================================================================
// [========      TRAINING THE NEURAL NETWORK ON MINI-BATCH      ========]
// -----------------------------------------------------------------------
/** @brief Train the neural network using mini-batch gradient descent */
void trainModel(float learningRate, int batchSize, bool initializeWeights = true,
                dim3 blockSize = dim3(1, 1))
{
    // Set random seed for reproducibility
    std::mt19937 gen(0);

    // Initialize weights of the neural network
    if (initializeWeights)
    {
        // Initialize weights of the neural network using normal distribution
        std::normal_distribution<float> normalDist(0.0f, 0.1f);
        for (int i = 0; i < W1_ROWS * W1_COLS; ++i)
        {
            W1[i] = normalDist(gen);
        }
        for (int i = 0; i < W2_ROWS * W2_COLS; ++i)
        {
            W2[i] = normalDist(gen);
        }
        for (int i = 0; i < W3_ROWS * W3_COLS; ++i)
        {
            W3[i] = normalDist(gen);
        }
    }

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

            // Load mini-batch data and labels
            for (int sampleIdx = 0; sampleIdx < batchSize; ++sampleIdx)
            {
                // Bias unit is at the first row of each column
                // Copy 1 row from trainData to remaining rows of each column in mb_X1
                for (int row = 0; row < INPUT_SIZE; ++row)
                {
                    mb_X1[(row + 1) * batchSize + sampleIdx] = trainData[mb_Indices[sampleIdx] * INPUT_SIZE + row];
                }
                // One-hot encode the ground truth labels
                // Copy from 1 column from train_Y to mb_Y
                for (int row = 0; row < NUM_CLASSES; ++row)
                {
                    mb_Y[row * batchSize + sampleIdx] = train_Y[row * NUM_TRAIN + mb_Indices[sampleIdx]];
                }
            }

            // Forward pass through the neural network
            forward(mb_X1, mb_X2, mb_X3, mb_A1, mb_A2, mb_A3,
                    batchSize, true, blockSize);

            // Backward pass through the neural network
            backward(blockSize);

            // Update weights using mini-batch gradient descent
            updateWeights(learningRate, batchSize, blockSize);
        }

        // Compute Cross-Entropy loss on the full training set
        forward(train_X1, train_X2, train_X3, train_A1, train_A2, train_A3,
                NUM_TRAIN, false, blockSize);
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
    }
}

// =======================================================================
// [========     PREPARING INPUT DATA FOR THE NEURAL NETWORK     ========]
// -----------------------------------------------------------------------
/** @brief Prepare the necessary data for the neural network */
void prepareData(dim3 blockSize)
{
    // Fill the array with indices from 0 to NUM_TRAIN
    // Reference: https://en.cppreference.com/w/cpp/algorithm/iota
    std::iota(trainIndices, trainIndices + NUM_TRAIN, 0);

    // Prepare the input for the first layer
    transposeMatrix(trainData, NUM_TRAIN, INPUT_SIZE, train_X1 + NUM_TRAIN, blockSize);
    transposeMatrix(testData, NUM_TEST, INPUT_SIZE, test_X1 + NUM_TEST, blockSize);
    // One-hot encode ground truth labels
    oneHotEncoding(trainLabels, train_Y, NUM_CLASSES, NUM_TRAIN, blockSize);
    oneHotEncoding(testLabels, test_Y, NUM_CLASSES, NUM_TEST, blockSize);

    // Add bias unit to the input matrices for each layer
    // Mini-batch training
    addBiasUnit(mb_X1, BATCH_SIZE, blockSize);
    addBiasUnit(mb_X2, BATCH_SIZE, blockSize);
    addBiasUnit(mb_X3, BATCH_SIZE, blockSize);
    // Full training set
    addBiasUnit(train_X1, NUM_TRAIN, blockSize);
    addBiasUnit(train_X2, NUM_TRAIN, blockSize);
    addBiasUnit(train_X3, NUM_TRAIN, blockSize);
    // Full test set
    addBiasUnit(test_X1, NUM_TEST, blockSize);
    addBiasUnit(test_X2, NUM_TEST, blockSize);
    addBiasUnit(test_X3, NUM_TEST, blockSize);
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
int main(int argc, char *argv[])
{
    printDeviceInfo();
    dim3 blockSize(32, 32);
    if (argc > 1)
    {
        blockSize.x = atoi(argv[1]);
        blockSize.y = atoi(argv[2]);
    }

    // Allocate memory for all declared pointers/arrays/matrices
    ALLOCATE_MEMORY();

    // Load Fashion-MNIST dataset
    printf("Loading Fashion-MNIST dataset...\n");
    const std::string base_dir = "./";
    if (!loadImageData(base_dir + "train-images-idx3-ubyte", trainData, NUM_TRAIN, INPUT_SIZE))
    {
        return -1;
    }
    if (!loadLabels(base_dir + "train-labels-idx1-ubyte", trainLabels, NUM_TRAIN))
    {
        return -1;
    }
    if (!loadImageData(base_dir + "t10k-images-idx3-ubyte", testData, NUM_TEST, INPUT_SIZE))
    {
        return -1;
    }
    if (!loadLabels(base_dir + "t10k-labels-idx1-ubyte", testLabels, NUM_TEST))
    {
        return -1;
    }

    // Prepare the necessary data for the neural network
    prepareData(blockSize);

    // Train the neural network using mini-batch gradient descent
    printf("Training the Neural Network using mini-batch gradient descent...\n");
    printf("==================+=================+========================\n");
    printf("[ NUM_EPOCHS = %d | BATCH_SIZE = %d | LEARNING_RATE = %5.3f ]\n", NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE);
    printf("------------------+-----------------+------------------------\n");
    trainModel(LEARNING_RATE, BATCH_SIZE, true, blockSize);

    // Evaluate the model on the full test set, calculate accuracy
    printf("\nEvaluating the model on the full test set...\n");
    forward(test_X1, test_X2, test_X3, test_A1, test_A2, test_A3,
            NUM_TEST, false, blockSize);
    computeCrossEntropyLoss(test_Y, test_A3, testLoss, NUM_CLASSES, NUM_TEST, blockSize);
    printf("[>] Test loss: %.10f\n", *testLoss); // Print with 20 decimal places
    predictLabels(test_A3, testPred, NUM_CLASSES, NUM_TEST, blockSize);
    int correct = 0;
    for (int i = 0; i < NUM_TEST; ++i)
    {
        correct += (testPred[i] == testLabels[i]);
    }
    float accuracy = static_cast<float>(correct) / NUM_TEST;
    printf("[>] Test accuracy: %.6f\n", accuracy); // Print with 6 decimal places

    // Free memory for all declared pointers/arrays/matrices
    FREE_MEMORY();

    // Print the summary of training time analysis
    printf("\n[========== Training Time Analysis ==========]\n");
    printf("(*) Average time for forward and backward pass of each layer in 1 epoch\n");
    printf("+----------+--------------+---------------+\n");
    printf("| Layer    | Forward (ms) | Backward (ms) |\n");
    printf("|==========+==============+===============|\n");
    printf("| Hidden 1 | %12.3f | %13.3f |\n", totalTimeForwardLayer1 / NUM_EPOCHS, totalTimeBackwardLayer1 / NUM_EPOCHS);
    printf("| Hidden 2 | %12.3f | %13.3f |\n", totalTimeForwardLayer2 / NUM_EPOCHS, totalTimeBackwardLayer2 / NUM_EPOCHS);
    printf("| Output   | %12.3f | %13.3f |\n", totalTimeForwardLayer3 / NUM_EPOCHS, totalTimeBackwardLayer3 / NUM_EPOCHS);
    printf("*----------*--------------*---------------*\n\n");
    printf("(*) Average time to update weights in 1 epoch: %.3f ms\n", totalTimeUpdateWeights / NUM_EPOCHS);
    printf("[============================================]\n\n");

    return 0;
}
