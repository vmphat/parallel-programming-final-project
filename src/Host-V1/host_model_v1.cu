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
#define NUM_EPOCHS 5
#define BATCH_SIZE 64
#define NUM_LAYERS 3
// =======================================================================

// ============ TIMING THE TRAINING PROCESS OF THE NETWORK ===============
GpuTimer forwardTimer;
GpuTimer backwardTimer;
GpuTimer updateWeightsTimer;
GpuTimer trainModelTimer;
GpuTimer softmaxTimer;
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
// =======================================================================

// ================ WEIGHT MATRICES OF THE NEURAL NETWORK ================
// Dimensions of weight matrices
const int W1_ROWS = INPUT_SIZE + 1; // #rows in W1
const int W1_COLS = HIDDEN_SIZE;    // #cols in W1
const int W2_ROWS = W1_COLS + 1;    // #rows in W2
const int W2_COLS = HIDDEN_SIZE;    // #cols in W2
const int W3_ROWS = W2_COLS + 1;    // #rows in W3
const int W3_COLS = NUM_CLASSES;    // #cols in W3
// Pointers to weight matrices
float *W1 = nullptr;
float *W2 = nullptr;
float *W3 = nullptr;
// =======================================================================

// ==== NUMBER OF COLUMNS IN INPUT AND OUTPUT MATRICES AT EACH LAYER =====
const int X1_COLS = INPUT_SIZE + 1;  // #cols in X1
const int A1_COLS = HIDDEN_SIZE;     // #cols in A1
const int X2_COLS = HIDDEN_SIZE + 1; // #cols in X2
const int A2_COLS = HIDDEN_SIZE;     // #cols in A2
const int X3_COLS = HIDDEN_SIZE + 1; // #cols in X3
const int A3_COLS = NUM_CLASSES;     // #cols in A3
// =======================================================================

// ========= FORWARD PASS WHEN TRAINING MODEL (USING MINI-BATCH) =========
float *mb_Y = nullptr;  // One-hot encoded ground truth labels
float *mb_X1 = nullptr; // Input matrix at layer 1
float *mb_A1 = nullptr; // Output matrix at layer 1
float *mb_X2 = nullptr; // Input matrix at layer 2
float *mb_A2 = nullptr; // Output matrix at layer 2
float *mb_X3 = nullptr; // Input matrix at layer 3
float *mb_A3 = nullptr; // Output matrix at layer 3
// =======================================================================

// ======== BACKWARD PASS WHEN TRAINING MODEL (USING MINI-BATCH) =========
// For layer 3
float *mb_dLdZ3 = nullptr; // mb_dLdZ3 = mb_A3 - mb_Y
float *mb_X3T = nullptr;   // Transpose of mb_X3
float *mb_dLdW3 = nullptr; // mb_dLdW3 = mb_X3T * mb_dLdZ3
float *mb_W3T = nullptr;   // Transpose of mb_W3
float *mb_dLdX3 = nullptr; // mb_dLdX3 = mb_dLdZ3 * mb_W3T
// For layer 2
float *mb_dLdA2 = nullptr; // mb_dLdA2 = mb_dLdX3[:, 1:]
float *mb_dLdZ2 = nullptr; // mb_dLdZ2 = mb_dLdA2 * relu'(mb_A2)
float *mb_X2T = nullptr;   // Transpose of mb_X2
float *mb_dLdW2 = nullptr; // mb_dLdW2 = mb_X2T * mb_dLdZ2
float *mb_W2T = nullptr;   // Transpose of mb_W2
float *mb_dLdX2 = nullptr; // mb_dLdX2 = mb_dLdZ2 * mb_W2T
// For layer 1
float *mb_dLdA1 = nullptr; // mb_dLdA1 = mb_dLdX2[:, 1:]
float *mb_dLdZ1 = nullptr; // mb_dLdZ1 = mb_dLdA1 * relu'(mb_A1)
float *mb_X1T = nullptr;   // Transpose of mb_X1
float *mb_dLdW1 = nullptr; // mb_dLdW1 = mb_X1T * mb_dLdZ1
// =======================================================================

// =================== ORIGINAL FASHION-MNIST DATASET ====================
float *trainData = nullptr;  // Training data
int *trainLabels = nullptr;  // Training labels
int *trainIndices = nullptr; // Shuffled indices for mini-batch training
float *testData = nullptr;   // Test data
int *testLabels = nullptr;   // Test labels
// =======================================================================

// ======== FORWARD PASS WHEN COMPUTING LOSS (FULL TRAINING SET) =========
float *train_Y = nullptr;  // One-hot encoded ground truth labels
float *train_X1 = nullptr; // Input matrix at layer 1
float *train_A1 = nullptr; // Output matrix at layer 1
float *train_X2 = nullptr; // Input matrix at layer 2
float *train_A2 = nullptr; // Output matrix at layer 2
float *train_X3 = nullptr; // Input matrix at layer 3
float *train_A3 = nullptr; // Output matrix at layer 3
// =======================================================================

// ======== FORWARD PASS WHEN EVALUATING MODEL (FULL TEST SET) ===========
float *test_Y = nullptr;  // One-hot encoded ground truth labels
float *test_X1 = nullptr; // Input matrix at layer 1
float *test_A1 = nullptr; // Output matrix at layer 1
float *test_X2 = nullptr; // Input matrix at layer 2
float *test_A2 = nullptr; // Output matrix at layer 2
float *test_X3 = nullptr; // Input matrix at layer 3
float *test_A3 = nullptr; // Output matrix at layer 3
// =======================================================================

// =================== FUNCTIONS FOR MEMORY MANAGEMENT ===================
/**
 * @brief Allocate memory for all declared pointers/arrays/matrices
 */
void ALLOCATE_MEMORY()
{
    // ================ WEIGHT MATRICES OF THE NEURAL NETWORK ================
    W1 = (float *)malloc(W1_ROWS * W1_COLS * sizeof(float));
    W2 = (float *)malloc(W2_ROWS * W2_COLS * sizeof(float));
    W3 = (float *)malloc(W3_ROWS * W3_COLS * sizeof(float));
    // =======================================================================

    // ========= FORWARD PASS WHEN TRAINING MODEL (USING MINI-BATCH) =========
    mb_Y = (float *)malloc(BATCH_SIZE * NUM_CLASSES * sizeof(float));
    mb_X1 = (float *)malloc(BATCH_SIZE * X1_COLS * sizeof(float));
    mb_A1 = (float *)malloc(BATCH_SIZE * A1_COLS * sizeof(float));
    mb_X2 = (float *)malloc(BATCH_SIZE * X2_COLS * sizeof(float));
    mb_A2 = (float *)malloc(BATCH_SIZE * A2_COLS * sizeof(float));
    mb_X3 = (float *)malloc(BATCH_SIZE * X3_COLS * sizeof(float));
    mb_A3 = (float *)malloc(BATCH_SIZE * A3_COLS * sizeof(float));
    // =======================================================================

    // ======== BACKWARD PASS WHEN TRAINING MODEL (USING MINI-BATCH) =========
    // For layer 3
    mb_dLdZ3 = (float *)malloc(BATCH_SIZE * W3_COLS * sizeof(float));
    mb_X3T = (float *)malloc(X3_COLS * BATCH_SIZE * sizeof(float));
    mb_dLdW3 = (float *)malloc(W3_ROWS * W3_COLS * sizeof(float));
    mb_W3T = (float *)malloc(W3_COLS * W3_ROWS * sizeof(float));
    mb_dLdX3 = (float *)malloc(BATCH_SIZE * X3_COLS * sizeof(float));
    // For layer 2
    mb_dLdA2 = (float *)malloc(BATCH_SIZE * W2_COLS * sizeof(float));
    mb_dLdZ2 = (float *)malloc(BATCH_SIZE * W2_COLS * sizeof(float));
    mb_X2T = (float *)malloc(X2_COLS * BATCH_SIZE * sizeof(float));
    mb_dLdW2 = (float *)malloc(W2_ROWS * W2_COLS * sizeof(float));
    mb_W2T = (float *)malloc(W2_COLS * W2_ROWS * sizeof(float));
    mb_dLdX2 = (float *)malloc(BATCH_SIZE * X2_COLS * sizeof(float));
    // For layer 1
    mb_dLdA1 = (float *)malloc(BATCH_SIZE * W1_COLS * sizeof(float));
    mb_dLdZ1 = (float *)malloc(BATCH_SIZE * W1_COLS * sizeof(float));
    mb_X1T = (float *)malloc(X1_COLS * BATCH_SIZE * sizeof(float));
    mb_dLdW1 = (float *)malloc(W1_ROWS * W1_COLS * sizeof(float));
    // =======================================================================

    // =================== ORIGINAL FASHION-MNIST DATASET ====================
    trainData = (float *)malloc(NUM_TRAIN * INPUT_SIZE * sizeof(float));
    trainLabels = (int *)malloc(NUM_TRAIN * sizeof(int));
    trainIndices = (int *)malloc(NUM_TRAIN * sizeof(int));
    testData = (float *)malloc(NUM_TEST * INPUT_SIZE * sizeof(float));
    testLabels = (int *)malloc(NUM_TEST * sizeof(int));
    // =======================================================================

    // ======== FORWARD PASS WHEN COMPUTING LOSS (FULL TRAINING SET) =========
    train_Y = (float *)malloc(NUM_TRAIN * NUM_CLASSES * sizeof(float));
    train_X1 = (float *)malloc(NUM_TRAIN * X1_COLS * sizeof(float));
    train_A1 = (float *)malloc(NUM_TRAIN * A1_COLS * sizeof(float));
    train_X2 = (float *)malloc(NUM_TRAIN * X2_COLS * sizeof(float));
    train_A2 = (float *)malloc(NUM_TRAIN * A2_COLS * sizeof(float));
    train_X3 = (float *)malloc(NUM_TRAIN * X3_COLS * sizeof(float));
    train_A3 = (float *)malloc(NUM_TRAIN * A3_COLS * sizeof(float));
    // =======================================================================

    // ======== FORWARD PASS WHEN EVALUATING MODEL (FULL TEST SET) ===========
    test_Y = (float *)malloc(NUM_TEST * NUM_CLASSES * sizeof(float));
    test_X1 = (float *)malloc(NUM_TEST * X1_COLS * sizeof(float));
    test_A1 = (float *)malloc(NUM_TEST * A1_COLS * sizeof(float));
    test_X2 = (float *)malloc(NUM_TEST * X2_COLS * sizeof(float));
    test_A2 = (float *)malloc(NUM_TEST * A2_COLS * sizeof(float));
    test_X3 = (float *)malloc(NUM_TEST * X3_COLS * sizeof(float));
    test_A3 = (float *)malloc(NUM_TEST * A3_COLS * sizeof(float));
    // =======================================================================
}
/**
 * @brief Free memory for all declared pointers/arrays/matrices
 */
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
    free(mb_A1);
    free(mb_X2);
    free(mb_A2);
    free(mb_X3);
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
    free(mb_dLdA2);
    free(mb_dLdZ2);
    free(mb_X2T);
    free(mb_dLdW2);
    free(mb_W2T);
    free(mb_dLdX2);
    // For layer 1
    free(mb_dLdA1);
    free(mb_dLdZ1);
    free(mb_X1T);
    free(mb_dLdW1);
    // =======================================================================

    // =================== ORIGINAL FASHION-MNIST DATASET ====================
    free(trainData);
    free(trainLabels);
    free(trainIndices);
    free(testData);
    free(testLabels);
    // =======================================================================

    // ======== FORWARD PASS WHEN COMPUTING LOSS (FULL TRAINING SET) =========
    free(train_Y);
    free(train_X1);
    free(train_A1);
    free(train_X2);
    free(train_A2);
    free(train_X3);
    free(train_A3);
    // =======================================================================

    // ======== FORWARD PASS WHEN EVALUATING MODEL (FULL TEST SET) ===========
    free(test_Y);
    free(test_X1);
    free(test_A1);
    free(test_X2);
    free(test_A2);
    free(test_X3);
    free(test_A3);
    // =======================================================================
}
// =======================================================================

// Load Fashion-MNIST image data
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
// Load labels for Fashion-MNIST dataset
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

// Add bias unit to the input matrix
void addBiasUnit(float *input, int numRows, int numCols)
{
    for (int r = 0; r < numRows; ++r)
    {
        input[r * numCols] = 1.0f;
    }
}

// ReLU activation function
void relu(float *input, float *output, int size)
{
    for (int i = 0; i < size; ++i)
    {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

// Softmax activation function
void softmax(float *input, float *output, int size)
{
    // Find max value in input array
    float maxVal = *std::max_element(input, input + size);

    // Compute softmax values
    float sumExp = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        output[i] = exp(input[i] - maxVal);
        sumExp += output[i];
    }
    for (int i = 0; i < size; ++i)
    {
        output[i] /= sumExp;
    }
}

/**
 * @brief Multiply two matrices: C = A * B
 *
 * @param A input matrix A
 * @param ARows number of rows in matrix A
 * @param ACols number of columns in matrix A
 * @param B input matrix B
 * @param BRows number of rows in matrix B
 * @param BCols number of columns in matrix B
 * @param C output matrix C
 * @param useDevice flag to use device implementation
 * @return void
 */
void multiplyMatrices(const float *A, int ARows, int ACols,
                      const float *B, int BRows, int BCols,
                      float *C)
{
    // Check matrix dimensions
    if (ACols != BRows)
    {
        printf("Matrix dimensions are not compatible for multiplication: (%d, %d) * (%d, %d)\n", ARows, ACols, BRows, BCols);
        return;
    }

    for (int i = 0; i < ARows; ++i)
    {
        for (int j = 0; j < BCols; ++j)
        {
            float sumVal = 0.0f;
            for (int k = 0; k < ACols; ++k)
            {
                sumVal += A[i * ACols + k] * B[k * BCols + j];
            }
            C[i * BCols + j] = sumVal;
        }
    }
}

// Transpose matrix
void transposeMatrix(const float *A, int rows, int cols, float *AT)
{
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            AT[c * rows + r] = A[r * cols + c];
        }
    }
}

/**
 * @brief Subtract two matrices element-wise: C = A - B
 */
void subtractMatrices(const float *A, const float *B, int rows, int cols, float *C)
{
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            C[r * cols + c] = A[r * cols + c] - B[r * cols + c];
        }
    }
}

// Prepare the input for the next layer: X_{i+1} = [1, A_i]
void prepareLayerInput(const float *A, int ARows, int ACols,
                       float *X, int XRows, int XCols)
{
    // Check input matrix dimensions
    if (ARows != XRows || ACols + 1 != XCols)
    {
        printf("Matrix dimensions are not compatible for creating layer input: (%d, %d) -> (%d, %d)\n", ARows, ACols, XRows, XCols);
        return;
    }

    for (int r = 0; r < XRows; ++r)
    {
        // Not need to copy the bias unit
        for (int c = 1; c < XCols; ++c)
        {
            X[r * XCols + c] = A[r * ACols + c - 1];
        }
    }
}

/**
 * @brief Remove bias unit when backpropagating through the network: dLdA_{i} = dLdX_{i+1}[:, 1:]
 */
void removeBiasUnit(float *dLdA, int ACols, const float *dLdX, int XCols,
                    int numRows)
{
    for (int r = 0; r < numRows; ++r)
    {
        for (int c = 0; c < ACols; ++c)
        {
            dLdA[r * ACols + c] = dLdX[r * XCols + c + 1];
        }
    }
}

// Compute dLdZ = dLdA * relu'(A)
void reluBackward(const float *dLdA, const float *A, int size, float *dLdZ)
{
    for (int i = 0; i < size; ++i)
    {
        dLdZ[i] = (A[i] > 0.0f) ? dLdA[i] : 0.0f;
    }
}

// Forward pass through the neural network
void forward(const float *X1, float *X2, float *X3, float *A1, float *A2, float *A3,
             int numSamples, float *totalTimeForward)
{
    // ******************** Forward pass at layer 1 ********************
    forwardTimer.Start();
    // Calculate weighted sum: Z1 = X1 * W1
    multiplyMatrices(X1, numSamples, X1_COLS, W1, W1_ROWS, W1_COLS, A1);
    // Apply ReLU activation function: A1 = ReLU(Z1)
    relu(A1, A1, numSamples * W1_COLS);
    forwardTimer.Stop();
    totalTimeForward[0] += forwardTimer.Elapsed();

    // ******************** Forward pass at layer 2 ********************
    forwardTimer.Start();
    // Prepare input matrix: X2 = [1, A1]
    prepareLayerInput(A1, numSamples, W1_COLS, X2, numSamples, X2_COLS);
    // Calculate weighted sum: Z2 = X2 * W2
    multiplyMatrices(X2, numSamples, X2_COLS, W2, W2_ROWS, W2_COLS, A2);
    // Apply ReLU activation function: A2 = ReLU(Z2)
    relu(A2, A2, numSamples * W2_COLS);
    forwardTimer.Stop();
    totalTimeForward[1] += forwardTimer.Elapsed();

    // ******************** Forward pass at layer 3 ******************
    forwardTimer.Start();
    // Prepare input matrix: X3 = [1, A2]
    prepareLayerInput(A2, numSamples, W2_COLS, X3, numSamples, X3_COLS);
    // Calculate weighted sum: Z3 = X3 * W3
    multiplyMatrices(X3, numSamples, X3_COLS, W3, W3_ROWS, W3_COLS, A3);
    // Apply softmax activation function: A3[i, :] = softmax(Z3[i, :])
    softmaxTimer.Start();
    for (int i = 0; i < numSamples; ++i)
    {
        softmax(A3 + i * W3_COLS, A3 + i * W3_COLS, W3_COLS);
    }
    softmaxTimer.Stop();
    totalTimeSoftmax += softmaxTimer.Elapsed();
    forwardTimer.Stop();
    totalTimeForward[2] += forwardTimer.Elapsed();
}

/**
 * @brief Compute cross-entropy loss (negative log-likelihood)
 *
 * @param Y one-hot encoded ground truth labels
 * @param P predicted probabilities
 * @param numSamples number of samples
 * @param lossVal computed loss scalar value
 * @param useDevice flag to use device implementation
 * @return void
 */
void computeCrossEntropyLoss(const float *Y, const float *P, int numSamples,
                             float *lossVal)
{
    // Calculate the sum of negative log-likelihood
    float sumLoss = 0.0f;
    for (int i = 0; i < numSamples; ++i)
    {
        for (int j = 0; j < NUM_CLASSES; ++j)
        {
            sumLoss -= Y[i * NUM_CLASSES + j] * logf(P[i * NUM_CLASSES + j] + FLT_MIN);
        }
    }

    // Compute the average loss and store it in lossVal
    *lossVal = sumLoss / numSamples;
}

/**
 * @brief Backward pass through the neural network
 *
 * @param useDevice flag to use device implementation
 * @return void
 */
void backward()
{
    // ******************** Backward pass at layer 3 ********************
    backwardTimer.Start(); // Timing the backward pass for layer 3

    // Compute dLdZ3 = A3 - Y
    subtractMatrices(mb_A3, mb_Y, BATCH_SIZE, NUM_CLASSES, mb_dLdZ3);
    // Compute dLdW3 = X3^T * dLdZ3
    transposeMatrix(mb_X3, BATCH_SIZE, X3_COLS, mb_X3T);
    multiplyMatrices(mb_X3T, X3_COLS, BATCH_SIZE, mb_dLdZ3, BATCH_SIZE, W3_COLS,
                     mb_dLdW3);
    // Compute dLdX3 = dLdZ3 * W3^T
    transposeMatrix(W3, W3_ROWS, W3_COLS, mb_W3T);
    multiplyMatrices(mb_dLdZ3, BATCH_SIZE, W3_COLS, mb_W3T, W3_COLS, W3_ROWS,
                     mb_dLdX3);

    backwardTimer.Stop(); // Stop timing the backward pass for layer 3
    totalTimeBackwardMiniBatch[2] += backwardTimer.Elapsed();

    // ******************** Backward pass at layer 2 ********************
    backwardTimer.Start(); // Timing the backward pass for layer 2

    // Compute dLdA2 = dLdX3[:, 1:]
    removeBiasUnit(mb_dLdA2, A2_COLS, mb_dLdX3, X3_COLS, BATCH_SIZE);
    // Compute dLdZ2 = dLdA2 * relu'(A2)
    reluBackward(mb_dLdA2, mb_A2, BATCH_SIZE * W2_COLS, mb_dLdZ2);
    // Compute dLdW2 = X2^T * dLdZ2
    transposeMatrix(mb_X2, BATCH_SIZE, X2_COLS, mb_X2T);
    multiplyMatrices(mb_X2T, X2_COLS, BATCH_SIZE, mb_dLdZ2, BATCH_SIZE, W2_COLS,
                     mb_dLdW2);
    // Compute dLdX2 = dLdZ2 * W2^T
    transposeMatrix(W2, W2_ROWS, W2_COLS, mb_W2T);
    multiplyMatrices(mb_dLdZ2, BATCH_SIZE, W2_COLS, mb_W2T, W2_COLS, W2_ROWS,
                     mb_dLdX2);

    backwardTimer.Stop(); // Stop timing the backward pass for layer 2
    totalTimeBackwardMiniBatch[1] += backwardTimer.Elapsed();

    // ******************** Backward pass at layer 1 ********************
    backwardTimer.Start(); // Timing the backward pass for layer 1

    // Compute dLdA1 = dLdX2[:, 1:]
    removeBiasUnit(mb_dLdA1, A1_COLS, mb_dLdX2, X2_COLS, BATCH_SIZE);
    // Compute dLdZ1 = dLdA1 * relu'(A1)
    reluBackward(mb_dLdA1, mb_A1, BATCH_SIZE * W1_COLS, mb_dLdZ1);
    // Compute dLdW1 = X1^T * dLdZ1
    transposeMatrix(mb_X1, BATCH_SIZE, X1_COLS, mb_X1T);
    multiplyMatrices(mb_X1T, X1_COLS, BATCH_SIZE, mb_dLdZ1, BATCH_SIZE, W1_COLS,
                     mb_dLdW1);

    backwardTimer.Stop(); // Stop timing the backward pass for layer 1
    totalTimeBackwardMiniBatch[0] += backwardTimer.Elapsed();
}

// Update weights using gradient descent
void updateWeights()
{
    updateWeightsTimer.Start(); // Timing the update weights process

    // ********** Update weights for layer 1 **********
    for (int row = 0; row < W1_ROWS; ++row)
    {
        for (int col = 0; col < W1_COLS; ++col)
        {
            int index = row * W1_COLS + col;
            W1[index] -= LEARNING_RATE * mb_dLdW1[index] / BATCH_SIZE;
        }
    }

    // ********** Update weights for layer 2 **********
    for (int row = 0; row < W2_ROWS; ++row)
    {
        for (int col = 0; col < W2_COLS; ++col)
        {
            int index = row * W2_COLS + col;
            W2[index] -= LEARNING_RATE * mb_dLdW2[index] / BATCH_SIZE;
        }
    }

    // ********** Update weights for layer 3 **********
    for (int row = 0; row < W3_ROWS; ++row)
    {
        for (int col = 0; col < W3_COLS; ++col)
        {
            int index = row * W3_COLS + col;
            W3[index] -= LEARNING_RATE * mb_dLdW3[index] / BATCH_SIZE;
        }
    }

    updateWeightsTimer.Stop(); // Stop timing the update weights process
    totalTimeUpdateWeights += updateWeightsTimer.Elapsed();
}

// Train the neural network using mini-batch gradient descent
void trainModel(bool initializeWeights = true)
{
    // Set random seed for reproducibility
    std::mt19937 gen(0);

    // Initialize weights
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

    trainModelTimer.Start(); // Start timing the training process

    // Train the model using mini-batch gradient descent
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch)
    {
        printf("[>] Epoch %d/%d, ", epoch + 1, NUM_EPOCHS);

        // Shuffle the training data indices
        std::shuffle(trainIndices, trainIndices + NUM_TRAIN, gen);

        // Loop over mini-batches
        for (int mb_StartIdx = 0; mb_StartIdx < NUM_TRAIN; mb_StartIdx += BATCH_SIZE)
        {
            // Break if the last mini-batch is smaller than BATCH_SIZE
            if (mb_StartIdx + BATCH_SIZE > NUM_TRAIN)
            {
                break;
            }

            // Get the current mini-batch indices
            int *mb_Indices = trainIndices + mb_StartIdx;

            // Load mini-batch data and labels
            for (int i = 0; i < BATCH_SIZE; ++i)
            {
                // Load mini-batch data
                prepareLayerInput(trainData + mb_Indices[i] * INPUT_SIZE, 1, INPUT_SIZE,
                                  mb_X1 + i * X1_COLS, 1, X1_COLS);
                // Load mini-batch labels
                for (int j = 0; j < NUM_CLASSES; ++j)
                {
                    mb_Y[i * NUM_CLASSES + j] = (j == trainLabels[mb_Indices[i]]) ? 1.0f : 0.0f;
                }
            }

            // Forward pass through the neural network
            forward(mb_X1, mb_X2, mb_X3, mb_A1, mb_A2, mb_A3,
                    BATCH_SIZE, totalTimeForwardMiniBatch);

            // Backward pass through the neural network
            backward();

            // Update weights using gradient descent
            updateWeights();
        }

        // Compute loss on the full training set
        forward(train_X1, train_X2, train_X3, train_A1, train_A2, train_A3,
                NUM_TRAIN, totalTimeForwardTrain);
        float lossVal = 0.0f;
        computeCrossEntropyLoss(train_Y, train_A3, NUM_TRAIN, &lossVal);
        printf("Train loss: %.10f, ", lossVal); // Print with 10 decimal places

        // Calculate the accuracy of the model
        int numCorrect = 0;
        for (int i = 0; i < NUM_TRAIN; ++i)
        {
            int predLabel = std::distance(train_A3 + i * NUM_CLASSES, std::max_element(train_A3 + i * NUM_CLASSES, train_A3 + (i + 1) * NUM_CLASSES));
            if (predLabel == trainLabels[i])
            {
                numCorrect++;
            }
        }
        float accuracy = static_cast<float>(numCorrect) / NUM_TRAIN;
        printf("Train accuracy: %.6f\n", accuracy); // Print with 6 decimal places
    }

    trainModelTimer.Stop(); // Stop timing the training process
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
    totalTimeUpdateWeights = 0.0f;
    totalTimeSoftmax = 0.0f;
}
int main()
{
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

    // One-hot encode ground truth labels
    for (int i = 0; i < NUM_TRAIN; ++i)
    {
        for (int j = 0; j < NUM_CLASSES; ++j)
        {
            train_Y[i * NUM_CLASSES + j] = (j == trainLabels[i]) ? 1.0f : 0.0f;
        }
    }
    for (int i = 0; i < NUM_TEST; ++i)
    {
        for (int j = 0; j < NUM_CLASSES; ++j)
        {
            test_Y[i * NUM_CLASSES + j] = (j == testLabels[i]) ? 1.0f : 0.0f;
        }
    }

    // Fill the array with indices from 0 to NUM_TRAIN
    // Reference: https://en.cppreference.com/w/cpp/algorithm/iota
    std::iota(trainIndices, trainIndices + NUM_TRAIN, 0);

    // Add bias unit to the input matrices for each layer
    // Mini-batch training
    addBiasUnit(mb_X1, BATCH_SIZE, X1_COLS);
    addBiasUnit(mb_X2, BATCH_SIZE, X2_COLS);
    addBiasUnit(mb_X3, BATCH_SIZE, X3_COLS);
    // Full training set
    addBiasUnit(train_X1, NUM_TRAIN, X1_COLS);
    addBiasUnit(train_X2, NUM_TRAIN, X2_COLS);
    addBiasUnit(train_X3, NUM_TRAIN, X3_COLS);
    // Full test set
    addBiasUnit(test_X1, NUM_TEST, X1_COLS);
    addBiasUnit(test_X2, NUM_TEST, X2_COLS);
    addBiasUnit(test_X3, NUM_TEST, X3_COLS);

    // Prepare the input for the first layer
    prepareLayerInput(trainData, NUM_TRAIN, INPUT_SIZE, train_X1, NUM_TRAIN, X1_COLS);
    prepareLayerInput(testData, NUM_TEST, INPUT_SIZE, test_X1, NUM_TEST, X1_COLS);

    // Initialize the timer
    initTimer();
    // Train the neural network using mini-batch gradient descent
    printf("Training the Neural Network using mini-batch gradient descent...\n");
    printf("==================+=================+========================\n");
    printf("[ NUM_EPOCHS = %2d | BATCH_SIZE = %d | LEARNING_RATE = %5.3f ]\n", NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE);
    printf("------------------+-----------------+------------------------\n");
    trainModel();

    // Evaluate the model on the full test set
    printf("\nEvaluating the model on the full test set...\n");
    forward(test_X1, test_X2, test_X3, test_A1, test_A2, test_A3,
            NUM_TEST, totalTimeForwardTest);
    float testLoss = 0.0f;
    computeCrossEntropyLoss(test_Y, test_A3, NUM_TEST, &testLoss);
    printf("[>] Test loss: %.10f\n", testLoss); // Print with 10 decimal places

    // Calculate the accuracy of the model
    int numCorrect = 0;
    for (int i = 0; i < NUM_TEST; ++i)
    {
        int predLabel = std::distance(test_A3 + i * NUM_CLASSES, std::max_element(test_A3 + i * NUM_CLASSES, test_A3 + (i + 1) * NUM_CLASSES));
        if (predLabel == testLabels[i])
        {
            numCorrect++;
        }
    }
    float accuracy = static_cast<float>(numCorrect) / NUM_TEST;
    printf("[>] Test accuracy: %.6f\n", accuracy); // Print with 6 decimal places

    // Free memory for all declared pointers/arrays/matrices
    FREE_MEMORY();

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
    printf("(*) Average time for training the model in 1 epoch: %.3f ms\n", trainModelTimer.Elapsed() / NUM_EPOCHS);
    printf("(*) Average time for forward pass entire training set: %.3f ms\n", (totalTimeForwardTrain[0] + totalTimeForwardTrain[1] + totalTimeForwardTrain[2]) / NUM_EPOCHS);
    printf("(*) Average time for forward pass entire test set: %.3f ms\n", totalTimeForwardTest[0] + totalTimeForwardTest[1] + totalTimeForwardTest[2]);
    printf("(*) Total time for softmax activation function: %.3f ms\n", totalTimeSoftmax);
    printf("[============================================]\n\n");

    return 0;
}
