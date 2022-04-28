#include <iostream>
#include <vector>
#include <sstream>
#include "read_mnist.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>

#define blockSize 128

/**
 * Computes ceil(x / y) for integral nonnegative values.
 */
static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
    return (nominator + denominator - 1) / denominator;
}

#define cudaCheckErrors(ans)                  \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define cublasCheckErrors(fn)                               \
    do                                                      \
    {                                                       \
        cublasStatus_t __err = fn;                          \
        if (__err != CUBLAS_STATUS_SUCCESS)                 \
        {                                                   \
            fprintf(stderr, "Fatal error: %s (at %s:%d)\n", \
                    _cudaGetErrorEnum(__err),               \
                    __FILE__, __LINE__);                    \
            fprintf(stderr, "*** FAILED - ABORTING\n");     \
            exit(1);                                        \
        }                                                   \
    } while (0)

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

#define cudnnCheckErrors(f)                                    \
    {                                                          \
        cudnnStatus_t err = (f);                               \
        if (err != CUDNN_STATUS_SUCCESS)                       \
        {                                                      \
            std::cout                                          \
                << "    Error occurred: " << err << std::endl; \
            std::exit(1);                                      \
        }                                                      \
    }

__global__ void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size)
        return;

    const int label_value = static_cast<int>(label[idx]);

    // For each item in the batch, decrease the result of the label's value by 1
    diff[idx * num_labels + label_value] -= 1.0f;
}

struct FullyConnectedLayer
{
    int inputs, outputs;
    std::vector<float> pneurons, pbias;

    FullyConnectedLayer(int inputs_, int outputs_) : outputs(outputs_), inputs(inputs_),
                                                     pneurons(inputs_ * outputs_), pbias(outputs_) {}

    bool FromFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";

        // Read weights file
        FILE *fp = fopen(ssf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            return false;
        }
        fread(&pneurons[0], sizeof(float), inputs * outputs, fp);
        fclose(fp);

        // Read bias file
        fp = fopen(ssbf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            return false;
        }
        fread(&pbias[0], sizeof(float), outputs, fp);
        fclose(fp);
        return true;
    }

    void ToFile(const char *fileprefix)
    {
        std::stringstream ssf, ssbf;
        ssf << fileprefix << ".bin";
        ssbf << fileprefix << ".bias.bin";

        // Write weights file
        FILE *fp = fopen(ssf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
            exit(2);
        }
        fwrite(&pneurons[0], sizeof(float), inputs * outputs, fp);
        fclose(fp);

        // Write bias file
        fp = fopen(ssbf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            exit(2);
        }
        fwrite(&pbias[0], sizeof(float), outputs, fp);
        fclose(fp);
    }
};

class Network
{
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;

    cudnnTensorDescriptor_t dataTensor, fc1Tensor, fc2Tensor;
    cudnnActivationDescriptor_t fc1Activation;

    int m_gpuid;
    int m_batchSize;
    size_t m_workspaceSize;

    FullyConnectedLayer &ref_fc1, &ref_fc2;

    Network &operator=(const Network &) = delete;
    Network(const Network &) = delete;

public:
public:
public:
    Network(int gpuid, int batch_size, FullyConnectedLayer &fc1, FullyConnectedLayer &fc2) : ref_fc1(fc1), ref_fc2(fc2), m_gpuid(gpuid)
    {
        m_batchSize = batch_size;

        // Create CUBLAS and CUDNN handles
        cudaCheckErrors(cudaSetDevice(gpuid));
        cublasCheckErrors(cublasCreate(&cublasHandle));
        cudnnCheckErrors(cudnnCreate(&cudnnHandle));

        // Create tensor descriptors
        cudnnCheckErrors(cudnnCreateTensorDescriptor(&dataTensor));
        cudnnCheckErrors(cudnnCreateTensorDescriptor(&fc1Tensor));
        cudnnCheckErrors(cudnnCreateTensorDescriptor(&fc2Tensor));
        cudnnCheckErrors(cudnnCreateActivationDescriptor(&fc1Activation));

        cudnnCheckErrors(cudnnSetTensor4dDescriptor(fc1Tensor,
                                                    CUDNN_TENSOR_NCHW,
                                                    CUDNN_DATA_FLOAT,
                                                    batch_size, fc1.outputs, 1, 1));

        cudnnCheckErrors(cudnnSetTensor4dDescriptor(fc2Tensor,
                                                    CUDNN_TENSOR_NCHW,
                                                    CUDNN_DATA_FLOAT,
                                                    batch_size, fc2.outputs, 1, 1));

        cudnnCheckErrors(cudnnSetActivationDescriptor(fc1Activation, CUDNN_ACTIVATION_RELU,
                                                      CUDNN_PROPAGATE_NAN, 0.0));
    }

    ~Network()
    {
        cudaCheckErrors(cudaSetDevice(m_gpuid));

        cublasCheckErrors(cublasDestroy(cublasHandle));
        cudnnCheckErrors(cudnnDestroy(cudnnHandle));
        cudnnCheckErrors(cudnnDestroyTensorDescriptor(dataTensor));
        cudnnCheckErrors(cudnnDestroyTensorDescriptor(fc1Tensor));
        cudnnCheckErrors(cudnnDestroyTensorDescriptor(fc2Tensor));
        cudnnCheckErrors(cudnnDestroyActivationDescriptor(fc1Activation));
    }

    void forward(float *data, float *fc1, float *fc1relu, float *fc2, float *result, float *pfc1, float *pfc1bias,
                 float *pfc2, float *pfc2bias, void *workspace, float *onevec)
    {
        float alpha = 1.0f, beta = 0.0f;
        // FC1 layer
        // Forward propagate neurons using weights (fc1 = pfc1'*data)
        cublasCheckErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                      ref_fc1.outputs, m_batchSize, ref_fc1.inputs,
                                      &alpha,
                                      pfc1, ref_fc1.inputs,
                                      data, ref_fc1.inputs,
                                      &beta,
                                      fc1, ref_fc1.outputs));
        // Add bias using GEMM's "beta" (fc1 += pfc1bias*1_vec')
        cublasCheckErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                      ref_fc1.outputs, m_batchSize, 1,
                                      &alpha,
                                      pfc1bias, ref_fc1.outputs,
                                      onevec, 1,
                                      &alpha,
                                      fc1, ref_fc1.outputs));

        // ReLU activation
        cudnnCheckErrors(cudnnActivationForward(cudnnHandle, fc1Activation, &alpha,
                                                fc1Tensor, fc1, &beta, fc1Tensor, fc1relu));

        // FC2 layer
        // Forward propagate neurons using weights (fc2 = pfc2'*fc1relu)
        cublasCheckErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                      ref_fc2.outputs, m_batchSize, ref_fc2.inputs,
                                      &alpha,
                                      pfc2, ref_fc2.inputs,
                                      fc1relu, ref_fc2.inputs,
                                      &beta,
                                      fc2, ref_fc2.outputs));
        // Add bias using GEMM's "beta" (fc2 += pfc2bias*1_vec')
        cublasCheckErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                      ref_fc2.outputs, m_batchSize, 1,
                                      &alpha,
                                      pfc2bias, ref_fc2.outputs,
                                      onevec, 1,
                                      &alpha,
                                      fc2, ref_fc2.outputs));

        // Softmax loss
        cudnnCheckErrors(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                             &alpha, fc2Tensor, fc2, &beta, fc2Tensor, result));
    }

    void backward(float *data, float *labels, float *fc1, float *fc1relu,
                  float *fc2, float *fc2smax, float *dloss_data,
                  float *pfc1, float *pfc1bias,
                  float *pfc2, float *pfc2bias,
                  float *gfc1, float *gfc1bias, float *dfc1, float *dfc1relu,
                  float *gfc2, float *gfc2bias, float *dfc2,
                  void *workspace, float *onevec)
    {
        float alpha = 1.0f, beta = 0.0f;

        float scalVal = 1.0f / static_cast<float>(m_batchSize);

        cudaCheckErrors(cudaSetDevice(m_gpuid));

        // Initialization (using the training error function)
        cudaCheckErrors(cudaMemcpyAsync(dloss_data, fc2smax, sizeof(float) * m_batchSize * ref_fc2.outputs, cudaMemcpyDeviceToDevice));

        // Softmax layer
        SoftmaxLossBackprop<<<RoundUp(m_batchSize, blockSize), blockSize>>>(labels, ref_fc2.outputs, m_batchSize, dloss_data);

        // Accounting for batch size in SGD
        cublasCheckErrors(cublasSscal(cublasHandle, ref_fc2.outputs * m_batchSize, &scalVal, dloss_data, 1));

        // FC2 layer
        // Compute derivative with respect to weights: gfc2 = (fc1relu * dfc2smax')
        cublasCheckErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc2.inputs, ref_fc2.outputs, m_batchSize,
                                      &alpha, fc1relu, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, gfc2, ref_fc2.inputs));
        // Compute derivative with respect to bias: gfc2bias = dfc2smax * 1_vec
        cublasCheckErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc2.outputs, m_batchSize,
                                      &alpha, dloss_data, ref_fc2.outputs, onevec, 1, &beta, gfc2bias, 1));
        // Compute derivative with respect to data (for previous layer): pfc2*dfc2smax (500x10*10xN)
        cublasCheckErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc2.inputs, m_batchSize, ref_fc2.outputs,
                                      &alpha, pfc2, ref_fc2.inputs, dloss_data, ref_fc2.outputs, &beta, dfc2, ref_fc2.inputs));

        // ReLU activation
        cudnnCheckErrors(cudnnActivationBackward(cudnnHandle, fc1Activation, &alpha,
                                                 fc1Tensor, fc1relu, fc1Tensor, dfc2,
                                                 fc1Tensor, fc1, &beta, fc1Tensor, dfc1relu));

        // FC1 layer
        // Compute derivative with respect to weights: gfc1 = (pool2 * dfc1relu')
        cublasCheckErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc1.inputs, ref_fc1.outputs, m_batchSize,
                                      &alpha, data, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, gfc1, ref_fc1.inputs));
        // Compute derivative with respect to bias: gfc1bias = dfc1relu * 1_vec
        cublasCheckErrors(cublasSgemv(cublasHandle, CUBLAS_OP_N, ref_fc1.outputs, m_batchSize,
                                      &alpha, dfc1relu, ref_fc1.outputs, onevec, 1, &beta, gfc1bias, 1));
        // Compute derivative with respect to data (for previous layer): pfc1*dfc1relu (800x500*500xN)
        cublasCheckErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc1.inputs, m_batchSize, ref_fc1.outputs,
                                      &alpha, pfc1, ref_fc1.inputs, dfc1relu, ref_fc1.outputs, &beta, dfc1, ref_fc1.inputs));
    }

    void update(float learning_rate,
                float *pfc1, float *pfc1bias,
                float *pfc2, float *pfc2bias,
                float *gfc1, float *gfc1bias,
                float *gfc2, float *gfc2bias)
    {
        float alpha = -learning_rate;

        cudaCheckErrors(cudaSetDevice(m_gpuid));

        // Fully connected 1
        cublasCheckErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pneurons.size()),
                                      &alpha, gfc1, 1, pfc1, 1));
        cublasCheckErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc1.pbias.size()),
                                      &alpha, gfc1bias, 1, pfc1bias, 1));

        // Fully connected 2
        cublasCheckErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pneurons.size()),
                                      &alpha, gfc2, 1, pfc2, 1));
        cublasCheckErrors(cublasSaxpy(cublasHandle, static_cast<int>(ref_fc2.pbias.size()),
                                      &alpha, gfc2bias, 1, pfc2bias, 1));
    }
};

int main(void)
{

    unsigned char **train_images, **test_images, *train_labels, *test_labels;
    std::cout << "MNIST data directory: " << MNIST_DATA_DIR << std::endl;

    mnist::MnistDataset dataset = mnist::read_dataset(MNIST_DATA_DIR, 0, 0);
    train_images = dataset.training_images;
    train_labels = dataset.training_labels;
    test_images = dataset.test_images;
    test_labels = dataset.training_labels;

    for (int i = 0; i < 100; i++)
    {
        std::cout << +train_images[i][i] << " ";
    }
    std::cout << "\n";

    return 0;
}
