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

// ============ TIMING EACH PHASE OF THE NEURAL NETWORK MODEL ============
// Timer for the entire neural network
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
    mb_A3 = (float *)malloc(A3_ROWS * BATCH_SIZE * sizeof(float));
    mb_A1 = mb_X2 + BATCH_SIZE; // mb_A1 = mb_X2[1:,:]
    mb_A2 = mb_X3 + BATCH_SIZE; // mb_A2 = mb_X3[1:,:]
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
    trainLoss = (float *)malloc(sizeof(float));
    testData = (float *)malloc(NUM_TEST * INPUT_SIZE * sizeof(float));
    testLoss = (float *)malloc(sizeof(float));

    trainLabels = (int *)malloc(NUM_TRAIN * sizeof(int));
    trainPred = (int *)malloc(NUM_TRAIN * sizeof(int));
    trainIndices = (int *)malloc(NUM_TRAIN * sizeof(int));
    testLabels = (int *)malloc(NUM_TEST * sizeof(int));
    testPred = (int *)malloc(NUM_TEST * sizeof(int));
    // =======================================================================

    // ======== FORWARD PASS WHEN COMPUTING LOSS (FULL TRAINING SET) =========
    train_Y = (float *)malloc(NUM_CLASSES * NUM_TRAIN * sizeof(float));
    train_X1 = (float *)malloc(X1_ROWS * NUM_TRAIN * sizeof(float));
    train_X2 = (float *)malloc(X2_ROWS * NUM_TRAIN * sizeof(float));
    train_X3 = (float *)malloc(X3_ROWS * NUM_TRAIN * sizeof(float));
    train_A3 = (float *)malloc(A3_ROWS * NUM_TRAIN * sizeof(float));
    train_A1 = train_X2 + NUM_TRAIN; // train_A1 = train_X2[1:,:]
    train_A2 = train_X3 + NUM_TRAIN; // train_A2 = train_X3[1:,:]
    // =======================================================================

    // ======== FORWARD PASS WHEN EVALUATING MODEL (FULL TEST SET) ===========
    test_Y = (float *)malloc(NUM_TEST * NUM_CLASSES * sizeof(float));
    test_X1 = (float *)malloc(X1_ROWS * NUM_TEST * sizeof(float));
    test_X2 = (float *)malloc(X2_ROWS * NUM_TEST * sizeof(float));
    test_X3 = (float *)malloc(X3_ROWS * NUM_TEST * sizeof(float));
    test_A3 = (float *)malloc(A3_ROWS * NUM_TEST * sizeof(float));
    test_A1 = test_X2 + NUM_TEST; // test_A1 = test_X2[1:,:]
    test_A2 = test_X3 + NUM_TEST; // test_A2 = test_X3[1:,:]
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
/** @brief Add bias unit to the first row of the input matrix */
void addBiasUnit(float *input, int numCols)
{
    for (int col = 0; col < numCols; ++col)
    {
        input[col] = 1.0f;
    }
}

// =======================================================================
// [========  RELU ACTIVATION FUNCTION WHEN FORWARD PROPAGATING  ========]
// -----------------------------------------------------------------------
/** @brief Apply ReLU activation function to input matrix */
void applyRelu(float *input, int numRows, int numCols)
{
    for (int row = 0; row < numRows; ++row)
    {
        for (int col = 0; col < numCols; ++col)
        {
            input[row * numCols + col] = fmaxf(0.0f, input[row * numCols + col]);
        }
    }
}

// =======================================================================
// [========  RELU ACTIVATION FUNCTION WHEN BACKWARD PROPAGATING  =======]
// -----------------------------------------------------------------------
/** @brief Backward pass for ReLU activation function */
void reluBackward(float *A, float *dLdA, float *dLdZ, int numRows, int numCols)
{
    for (int row = 0; row < numRows; ++row)
    {
        for (int col = 0; col < numCols; ++col)
        {
            int index = row * numCols + col;
            dLdZ[index] = (A[index] > 0.0f) ? dLdA[index] : 0.0f;
        }
    }
}

// =======================================================================
// [========  SOFTMAX ACTIVATION FUNCTION WHEN FORWARD PROPAGATING  =====]
// -----------------------------------------------------------------------
/** @brief Apply softmax activation function to input matrix (column-wise) */
void applySoftmax(float *input, int numRows, int numCols, dim3 blockSize = dim3(1, 1))
{
    softmaxTimer.Start(); // Start timing the softmax activation function

    for (int col = 0; col < numCols; ++col)
    {
        // Temporal array to store column vector
        float colVec[numRows];
        // Find maximum value in the column
        float maxVal = -FLT_MAX;
        for (int row = 0; row < numRows; ++row)
        {
            colVec[row] = input[row * numCols + col];
            maxVal = fmaxf(maxVal, colVec[row]);
        }
        // Compute sum of exponentials of input values
        float expSum = 0.0f;
        for (int row = 0; row < numRows; ++row)
        {
            colVec[row] = expf(colVec[row] - maxVal);
            expSum += colVec[row];
        }
        // Normalize the column
        for (int row = 0; row < numRows; ++row)
        {
            input[row * numCols + col] = colVec[row] / expSum;
        }
    }

    softmaxTimer.Stop(); // Stop timing the softmax activation function
    totalTimeSoftmax += softmaxTimer.Elapsed();
}

// =======================================================================
// [========  SUBTRACTING TWO MATRICES (ELEMENT-WISE) FUNCTION   ========]
// -----------------------------------------------------------------------
/** @brief Subtract two matrices (element-wise): C = A - B */
void subtractMatrices(float *A, float *B, float *C, int numRows, int numCols)
{
    for (int row = 0; row < numRows; ++row)
    {
        for (int col = 0; col < numCols; ++col)
        {
            int index = row * numCols + col;
            C[index] = A[index] - B[index];
        }
    }
}

// =======================================================================
// [========                MATRIX MULTIPLICATION                ========]
// -----------------------------------------------------------------------
/** @brief Compute matrix multiplication: C = A * B */
void matrixMultiply(float *A, float *B, float *C,
                    int numRowsA, int numColsA, int numColsB)
{
    for (int row = 0; row < numRowsA; ++row)
    {
        for (int col = 0; col < numColsB; ++col)
        {
            float sum = 0.0f;
            for (int i = 0; i < numColsA; ++i)
            {
                sum += A[row * numColsA + i] * B[i * numColsB + col];
            }
            C[row * numColsB + col] = sum;
        }
    }
}

// =======================================================================
// [========                TRANSPOSE OF A MATRIX                ========]
// -----------------------------------------------------------------------
/** @brief Compute the transpose of a matrix */
void transposeMatrix(float *input, int numRows, int numCols, float *output)
{
    for (int row = 0; row < numRows; ++row)
    {
        for (int col = 0; col < numCols; ++col)
        {
            output[col * numRows + row] = input[row * numCols + col];
        }
    }
}

// =======================================================================
// [========              ONE-HOT ENCODING FUNCTION              ========]
// -----------------------------------------------------------------------
/** @brief One-hot encode the labels (column-wise) */
void oneHotEncoding(int *labels, float *Y, int numClasses, int numSamples)
{
    for (int sample = 0; sample < numSamples; ++sample)
    {
        int label = labels[sample];
        for (int row = 0; row < numClasses; ++row)
        {
            Y[row * numSamples + sample] = (row == label) ? 1.0f : 0.0f;
        }
    }
}

// =======================================================================
// [========             COMPUTE CROSS-ENTROPY LOSS              ========]
// -----------------------------------------------------------------------
/** @brief Compute cross-entropy loss */
void computeCrossEntropyLoss(float *Y, float *P, float *loss, int numClasses, int numSamples)
{
    // Init sum to zero
    float sum = 0.0f;
    for (int row = 0; row < numClasses; ++row)
    {
        for (int col = 0; col < numSamples; ++col)
        {
            sum += Y[row * numSamples + col] * logf(P[row * numSamples + col] + FLT_MIN);
        }
    }
    // Normalize the loss
    *loss = -sum / numSamples;
}

// =======================================================================
// [========          PREDICT LABELS FROM PROBABILITIES          ========]
// -----------------------------------------------------------------------
/** @brief Predict labels from probabilities */
void predictLabels(float *P, int *labels, int numClasses, int numSamples)
{
    for (int sample = 0; sample < numSamples; ++sample)
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

// =======================================================================
// [========  FORWARD PASS INPUT FOR EACH LAYER OF THE NETWORK   ========]
// -----------------------------------------------------------------------
void hiddenLayerForward(float *W, float *X, float *A, int m, int n, int k)
{
    // Matrix multiplication: A = W * X
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < k; ++col)
        {
            float sum = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                sum += W[row * n + i] * X[i * k + col];
            }
            // Apply ReLU activation function: A = ReLU(A)
            A[row * k + col] = fmaxf(0.0f, sum);
        }
    }
}
/**
 * @brief Forward pass the input through neural network
 * @note Timing the forward pass of each layer when training the model
 */
void forward(float *X1, float *X2, float *X3, float *A1, float *A2, float *A3,
             int numSamples, float *totalTimeForward)
{
    // ********** Forward pass for layer 1 **********
    forwardTimer.Start();
    hiddenLayerForward(W1, X1, A1, W1_ROWS, W1_COLS, numSamples);
    forwardTimer.Stop();
    totalTimeForward[0] += forwardTimer.Elapsed();

    // ********** Forward pass for layer 2 **********
    forwardTimer.Start();
    hiddenLayerForward(W2, X2, A2, W2_ROWS, W2_COLS, numSamples);
    forwardTimer.Stop();
    totalTimeForward[1] += forwardTimer.Elapsed();

    // ********** Forward pass for layer 3 **********
    forwardTimer.Start();
    // Z3 = W3 * X3
    matrixMultiply(W3, X3, A3, W3_ROWS, W3_COLS, numSamples);
    // A3 = softmax(Z3)
    applySoftmax(A3, A3_ROWS, numSamples);
    forwardTimer.Stop();
    totalTimeForward[2] += forwardTimer.Elapsed();
}

// =======================================================================
// [========          BACKWARD PASS WHEN TRAINING MODEL          ========]
// -----------------------------------------------------------------------
/** @brief Backward pass the gradients through neural network */
void backward()
{
    // ********** Backward pass for layer 3 **********
    backwardTimer.Start(); // Timing the backward pass for layer 3

    // dLdZ3 = A3 - Y
    subtractMatrices(mb_A3, mb_Y, mb_dLdZ3, A3_ROWS, BATCH_SIZE);
    // dLdW3 = dLdZ3 * X3T
    transposeMatrix(mb_X3, X3_ROWS, BATCH_SIZE, mb_X3T);
    matrixMultiply(mb_dLdZ3, mb_X3T, mb_dLdW3, W3_ROWS, BATCH_SIZE, X3_ROWS);
    // dLdX3 = W3T * dLdZ3
    transposeMatrix(W3, W3_ROWS, W3_COLS, mb_W3T);
    matrixMultiply(mb_W3T, mb_dLdZ3, mb_dLdX3, W3_COLS, W3_ROWS, BATCH_SIZE);

    backwardTimer.Stop(); // Stop timing the backward pass for layer 3
    totalTimeBackwardMiniBatch[2] += backwardTimer.Elapsed();

    // ********** Backward pass for layer 2 **********
    backwardTimer.Start(); // Timing the backward pass for layer 2

    // mb_dLdA2 = mb_dLdX3[1:,:]
    // mb_dLdZ2 = mb_dLdA2 * relu'(mb_A2)
    reluBackward(mb_A2, mb_dLdA2, mb_dLdZ2, A2_ROWS, BATCH_SIZE);
    // mb_dLdW2 = mb_dLdZ2 * mb_X2T
    transposeMatrix(mb_X2, X2_ROWS, BATCH_SIZE, mb_X2T);
    matrixMultiply(mb_dLdZ2, mb_X2T, mb_dLdW2, W2_ROWS, BATCH_SIZE, X2_ROWS);
    // mb_dLdX2 = mb_W2T * mb_dLdZ2
    transposeMatrix(W2, W2_ROWS, W2_COLS, mb_W2T);
    matrixMultiply(mb_W2T, mb_dLdZ2, mb_dLdX2, W2_COLS, W2_ROWS, BATCH_SIZE);

    backwardTimer.Stop(); // Stop timing the backward pass for layer 2
    totalTimeBackwardMiniBatch[1] += backwardTimer.Elapsed();

    // ********** Backward pass for layer 1 **********
    backwardTimer.Start(); // Timing the backward pass for layer 1

    // mb_dLdA1 = mb_dLdX2[1:,:]
    // mb_dLdZ1 = mb_dLdA1 * relu'(mb_A1)
    reluBackward(mb_A1, mb_dLdA1, mb_dLdZ1, A1_ROWS, BATCH_SIZE);
    // mb_dLdW1 = mb_dLdZ1 * mb_X1T
    transposeMatrix(mb_X1, X1_ROWS, BATCH_SIZE, mb_X1T);
    matrixMultiply(mb_dLdZ1, mb_X1T, mb_dLdW1, W1_ROWS, BATCH_SIZE, X1_ROWS);

    backwardTimer.Stop(); // Stop timing the backward pass for layer 1
    totalTimeBackwardMiniBatch[0] += backwardTimer.Elapsed();
}

// =======================================================================
// [========        UPDATE WEIGHTS OF THE NEURAL NETWORK         ========]
// -----------------------------------------------------------------------
/** @brief Update weights of the neural network */
void updateWeights(float learningRate, int batchSize)
{
    updateWeightsTimer.Start(); // Timing the update weights process

    // ********** Update weights for layer 1 **********
    for (int row = 0; row < W1_ROWS; ++row)
    {
        for (int col = 0; col < W1_COLS; ++col)
        {
            int index = row * W1_COLS + col;
            W1[index] -= learningRate * mb_dLdW1[index] / batchSize;
        }
    }

    // ********** Update weights for layer 2 **********
    for (int row = 0; row < W2_ROWS; ++row)
    {
        for (int col = 0; col < W2_COLS; ++col)
        {
            int index = row * W2_COLS + col;
            W2[index] -= learningRate * mb_dLdW2[index] / batchSize;
        }
    }

    // ********** Update weights for layer 3 **********
    for (int row = 0; row < W3_ROWS; ++row)
    {
        for (int col = 0; col < W3_COLS; ++col)
        {
            int index = row * W3_COLS + col;
            W3[index] -= learningRate * mb_dLdW3[index] / batchSize;
        }
    }

    updateWeightsTimer.Stop(); // Stop timing the update weights process
    totalTimeUpdateWeights += updateWeightsTimer.Elapsed();
}

// =======================================================================
// [========      TRAINING THE NEURAL NETWORK ON MINI-BATCH      ========]
// -----------------------------------------------------------------------
/** @brief Train the neural network using mini-batch gradient descent */
void trainModel(float learningRate, int batchSize, bool initializeWeights = true)
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

    trainModelTimer.Start(); // Start timing the training process

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

            // // Load mini-batch data and labels
            // for (int sampleIdx = 0; sampleIdx < batchSize; ++sampleIdx)
            // {
            //     // Bias unit is at the first row of each column
            //     // Copy 1 row from trainData to remaining rows of each column in mb_X1
            //     for (int row = 0; row < INPUT_SIZE; ++row)
            //     {
            //         mb_X1[(row + 1) * batchSize + sampleIdx] = trainData[mb_Indices[sampleIdx] * INPUT_SIZE + row];
            //     }
            //     // One-hot encode the ground truth labels
            //     // Copy from 1 column from train_Y to mb_Y
            //     for (int row = 0; row < NUM_CLASSES; ++row)
            //     {
            //         mb_Y[row * batchSize + sampleIdx] = train_Y[row * NUM_TRAIN + mb_Indices[sampleIdx]];
            //     }
            // }

            // Load mini-batch data
            for (int row = 0; row < INPUT_SIZE; ++row)
            {
                for (int col = 0; col < batchSize; ++col)
                {
                    mb_X1[(row + 1) * batchSize + col] = trainData[mb_Indices[col] * INPUT_SIZE + row];
                }
            }
            // Load mini-batch labels
            for (int row = 0; row < NUM_CLASSES; ++row)
            {
                for (int col = 0; col < batchSize; ++col)
                {
                    mb_Y[row * batchSize + col] = train_Y[row * NUM_TRAIN + mb_Indices[col]];
                }
            }

            // Forward pass through the neural network
            forward(mb_X1, mb_X2, mb_X3, mb_A1, mb_A2, mb_A3,
                    batchSize, totalTimeForwardMiniBatch);

            // Backward pass through the neural network
            backward();

            // Update weights using mini-batch gradient descent
            updateWeights(learningRate, batchSize);
        }

        // Compute Cross-Entropy loss on the full training set
        forward(train_X1, train_X2, train_X3, train_A1, train_A2, train_A3,
                NUM_TRAIN, totalTimeForwardTrain);
        computeCrossEntropyLoss(train_Y, train_A3, trainLoss, NUM_CLASSES, NUM_TRAIN);
        printf("Train loss: %.10f, ", *trainLoss); // Print with 10 decimal places

        // Compute accuracy on the full training set
        predictLabels(train_A3, trainPred, NUM_CLASSES, NUM_TRAIN);
        int correct = 0;
        for (int i = 0; i < NUM_TRAIN; ++i)
        {
            correct += (trainPred[i] == trainLabels[i]);
        }
        float accuracy = static_cast<float>(correct) / NUM_TRAIN;
        printf("Train accuracy: %.6f\n", accuracy); // Print with 6 decimal places
    }

    trainModelTimer.Stop(); // Stop timing the training process
}

// =======================================================================
// [========     PREPARING INPUT DATA FOR THE NEURAL NETWORK     ========]
// -----------------------------------------------------------------------
/** @brief Prepare the necessary data for the neural network */
void prepareData()
{
    // Fill the array with indices from 0 to NUM_TRAIN
    // Reference: https://en.cppreference.com/w/cpp/algorithm/iota
    std::iota(trainIndices, trainIndices + NUM_TRAIN, 0);

    // Prepare the input for the first layer
    transposeMatrix(trainData, NUM_TRAIN, INPUT_SIZE, train_X1 + NUM_TRAIN);
    transposeMatrix(testData, NUM_TEST, INPUT_SIZE, test_X1 + NUM_TEST);
    // One-hot encode ground truth labels
    oneHotEncoding(trainLabels, train_Y, NUM_CLASSES, NUM_TRAIN);
    oneHotEncoding(testLabels, test_Y, NUM_CLASSES, NUM_TEST);

    // Add bias unit to the input matrices for each layer
    // Mini-batch training
    addBiasUnit(mb_X1, BATCH_SIZE);
    addBiasUnit(mb_X2, BATCH_SIZE);
    addBiasUnit(mb_X3, BATCH_SIZE);
    // Full training set
    addBiasUnit(train_X1, NUM_TRAIN);
    addBiasUnit(train_X2, NUM_TRAIN);
    addBiasUnit(train_X3, NUM_TRAIN);
    // Full test set
    addBiasUnit(test_X1, NUM_TEST);
    addBiasUnit(test_X2, NUM_TEST);
    addBiasUnit(test_X3, NUM_TEST);
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
int main(int argc, char *argv[])
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

    // Prepare the necessary data for the neural network
    prepareData();
    // Initialize the timer
    initTimer();

    // Train the neural network using mini-batch gradient descent
    printf("Training the Neural Network using mini-batch gradient descent...\n");
    printf("==================+=================+========================\n");
    printf("[ NUM_EPOCHS = %2d | BATCH_SIZE = %d | LEARNING_RATE = %5.3f ]\n", NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE);
    printf("------------------+-----------------+------------------------\n");
    trainModel(LEARNING_RATE, BATCH_SIZE, true);

    // Evaluate the model on the full test set, calculate accuracy
    printf("\nEvaluating the model on the full test set...\n");
    forward(test_X1, test_X2, test_X3, test_A1, test_A2, test_A3,
            NUM_TEST, totalTimeForwardTest);
    computeCrossEntropyLoss(test_Y, test_A3, testLoss, NUM_CLASSES, NUM_TEST);
    printf("[>] Test loss: %.10f\n", *testLoss); // Print with 10 decimal places
    predictLabels(test_A3, testPred, NUM_CLASSES, NUM_TEST);
    int correct = 0;
    for (int i = 0; i < NUM_TEST; ++i)
    {
        correct += (testPred[i] == testLabels[i]);
    }
    float accuracy = static_cast<float>(correct) / NUM_TEST;
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
