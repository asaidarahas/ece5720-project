#include <iostream>
#include <vector>
#include <sstream>
#include <random>

#include <curand.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "read_mnist.hpp"

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

#define cudnnCheckErrors(fn)                                \
    do                                                      \
    {                                                       \
        cudnnStatus_t __err = fn;                           \
        if (__err != CUDNN_STATUS_SUCCESS)                  \
        {                                                   \
            fprintf(stderr, "Fatal error: %s (at %s:%d)\n", \
                    _cudnnGetErrorEnum(__err),              \
                    __FILE__, __LINE__);                    \
            fprintf(stderr, "*** FAILED - ABORTING\n");     \
            exit(1);                                        \
        }                                                   \
    } while (0)

static const char *_cudnnGetErrorEnum(cudnnStatus_t error)
{
    switch (error)
    {
    case CUDNN_STATUS_SUCCESS:
        return "CUDNN_STATUS_SUCCESS";

    case CUDNN_STATUS_NOT_INITIALIZED:
        return "CUDNN_STATUS_NOT_INITIALIZED";

    case CUDNN_STATUS_ALLOC_FAILED:
        return "CUDNN_STATUS_ALLOC_FAILED";

    case CUDNN_STATUS_INVALID_VALUE:
        return "CUDNN_STATUS_INVALID_VALUE";

    case CUDNN_STATUS_ARCH_MISMATCH:
        return "CUDNN_STATUS_ARCH_MISMATCH";

    case CUDNN_STATUS_MAPPING_ERROR:
        return "CUDNN_STATUS_MAPPING_ERROR";

    case CUDNN_STATUS_EXECUTION_FAILED:
        return "CUDNN_STATUS_EXECUTION_FAILED";

    case CUDNN_STATUS_INTERNAL_ERROR:
        return "CUDNN_STATUS_INTERNAL_ERROR";

    case CUDNN_STATUS_BAD_PARAM:
        return "CUDNN_STATUS_BAD_PARAM";

    case CUDNN_STATUS_NOT_SUPPORTED:
        return "CUDNN_STATUS_BAD_PARAM";

    case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
        return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";

    case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
        return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";

    case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
        return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
    }

    return "<unknown>";
}

/**
 * Computes ceil(x / y) for integral nonnegative values.
 */
static inline unsigned int RoundUp(unsigned int numer, unsigned int denom)
{
    return (numer + denom - 1) / denom;
}

__global__ void CrossEntropyLoss(double *dloss, double *probs, const double *label, int num_labels, int batchSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize)
        return;
    double loss = 0.0;
    for (int c = 0; c < num_labels; c++)
    {
        loss += label[idx * num_labels + c] * logf(probs[idx * num_labels + c]);
        dloss[idx * num_labels + c] = -(label[idx * num_labels + c] * logf(probs[idx * num_labels + c]));
    }
}

__global__ void FillOnes(double *vec, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    vec[idx] = 1.0;
}

struct Dense
{
    int inputs, outputs;
    std::vector<double> weights, bias;

    Dense(int inputs_, int outputs_) : outputs(outputs_), inputs(inputs_),
                                       weights(inputs_ * outputs_), bias(outputs_) {}

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
        fread(&weights[0], sizeof(double), inputs * outputs, fp);
        fclose(fp);

        // Read bias file
        fp = fopen(ssbf.str().c_str(), "rb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            return false;
        }
        fread(&bias[0], sizeof(double), outputs, fp);
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
        fwrite(&weights[0], sizeof(double), inputs * outputs, fp);
        fclose(fp);

        // Write bias file
        fp = fopen(ssbf.str().c_str(), "wb");
        if (!fp)
        {
            printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
            exit(2);
        }
        fwrite(&bias[0], sizeof(double), outputs, fp);
        fclose(fp);
    }
};

class Network
{
    // create handles
    cudnnHandle_t cudnnH;
    cublasHandle_t cublasH;

    // create tensors
    cudnnTensorDescriptor_t dataTensor, fc1Tensor, fc2Tensor, fc3Tensor, fc4Tensor, fc5Tensor;
    cudnnActivationDescriptor_t fc1Act, fc2Act, fc3Act, fc4Act;

    Dense &ref_fc1, &ref_fc2, &ref_fc3, &ref_fc4, &ref_fc5;

    Network &operator=(const Network &) = delete;
    Network(const Network &) = delete;

public:
    int batchSize;
    Network(int batchSize, Dense &fc1, Dense &fc2, Dense &fc3,
            Dense &fc4, Dense &fc5) : ref_fc1(fc1), ref_fc2(fc2), ref_fc3(fc3), ref_fc4(fc4),
                                      ref_fc5(fc5), batchSize(batchSize)
    {

        // Create CUBLAS and CUDNN handles
        cublasCheckErrors(cublasCreate(&cublasH));
        cudnnCheckErrors(cudnnCreate(&cudnnH));

        // Create tensor descriptors
        cudnnCheckErrors(cudnnCreateTensorDescriptor(&dataTensor));
        cudnnCheckErrors(cudnnCreateTensorDescriptor(&fc1Tensor));
        cudnnCheckErrors(cudnnCreateTensorDescriptor(&fc2Tensor));
        cudnnCheckErrors(cudnnCreateTensorDescriptor(&fc3Tensor));
        cudnnCheckErrors(cudnnCreateTensorDescriptor(&fc4Tensor));
        cudnnCheckErrors(cudnnCreateTensorDescriptor(&fc5Tensor));

        cudnnCheckErrors(cudnnCreateActivationDescriptor(&fc1Act));
        cudnnCheckErrors(cudnnCreateActivationDescriptor(&fc2Act));
        cudnnCheckErrors(cudnnCreateActivationDescriptor(&fc3Act));
        cudnnCheckErrors(cudnnCreateActivationDescriptor(&fc4Act));

        cudnnCheckErrors(cudnnSetTensor4dDescriptor(fc1Tensor,
                                                    CUDNN_TENSOR_NCHW,
                                                    CUDNN_DATA_DOUBLE,
                                                    batchSize, fc1.outputs, 1, 1));

        cudnnCheckErrors(cudnnSetTensor4dDescriptor(fc2Tensor,
                                                    CUDNN_TENSOR_NCHW,
                                                    CUDNN_DATA_DOUBLE,
                                                    batchSize, fc2.outputs, 1, 1));

        cudnnCheckErrors(cudnnSetTensor4dDescriptor(fc3Tensor,
                                                    CUDNN_TENSOR_NCHW,
                                                    CUDNN_DATA_DOUBLE,
                                                    batchSize, fc3.outputs, 1, 1));

        cudnnCheckErrors(cudnnSetTensor4dDescriptor(fc4Tensor,
                                                    CUDNN_TENSOR_NCHW,
                                                    CUDNN_DATA_DOUBLE,
                                                    batchSize, fc4.outputs, 1, 1));

        cudnnCheckErrors(cudnnSetTensor4dDescriptor(fc5Tensor,
                                                    CUDNN_TENSOR_NCHW,
                                                    CUDNN_DATA_DOUBLE,
                                                    batchSize, fc5.outputs, 1, 1));

        cudnnCheckErrors(cudnnSetActivationDescriptor(fc1Act, CUDNN_ACTIVATION_RELU,
                                                      CUDNN_PROPAGATE_NAN, 0.0));

        cudnnCheckErrors(cudnnSetActivationDescriptor(fc2Act, CUDNN_ACTIVATION_RELU,
                                                      CUDNN_PROPAGATE_NAN, 0.0));

        cudnnCheckErrors(cudnnSetActivationDescriptor(fc3Act, CUDNN_ACTIVATION_RELU,
                                                      CUDNN_PROPAGATE_NAN, 0.0));

        cudnnCheckErrors(cudnnSetActivationDescriptor(fc4Act, CUDNN_ACTIVATION_RELU,
                                                      CUDNN_PROPAGATE_NAN, 0.0));
    }

    ~Network()
    {
        cublasCheckErrors(cublasDestroy(cublasH));
        cudnnCheckErrors(cudnnDestroyTensorDescriptor(dataTensor));
        cudnnCheckErrors(cudnnDestroyTensorDescriptor(fc1Tensor));
        cudnnCheckErrors(cudnnDestroyTensorDescriptor(fc2Tensor));
        cudnnCheckErrors(cudnnDestroyTensorDescriptor(fc3Tensor));
        cudnnCheckErrors(cudnnDestroyTensorDescriptor(fc4Tensor));
        cudnnCheckErrors(cudnnDestroyTensorDescriptor(fc5Tensor));
        cudnnCheckErrors(cudnnDestroyActivationDescriptor(fc1Act));
        cudnnCheckErrors(cudnnDestroyActivationDescriptor(fc2Act));
        cudnnCheckErrors(cudnnDestroyActivationDescriptor(fc3Act));
        cudnnCheckErrors(cudnnDestroyActivationDescriptor(fc4Act));
    }

    void forward(double *data, double *fc1, double *fc1act, double *fc2, double *fc2act, double *fc3, double *fc3act,
                 double *fc4, double *fc4act, double *fc5, double *out, double *wfc1, double *fc1bias, double *wfc2,
                 double *fc2bias, double *wfc3, double *fc3bias, double *wfc4, double *fc4bias, double *wfc5, double *fc5bias,
                 double *onevec)
    {
        double alpha = 1.0, beta = 0.0;
        // FC1 layer
        // Forward propagate neurons using weights (fc1 = wfc1'*data)
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                                      ref_fc1.outputs, batchSize, ref_fc1.inputs,
                                      &alpha,
                                      wfc1, ref_fc1.inputs,
                                      data, ref_fc1.inputs,
                                      &beta,
                                      fc1, ref_fc1.outputs));
        // Add bias using GEMM's "beta" (fc1 += fc1bias*1_vec')
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                      ref_fc1.outputs, batchSize, 1,
                                      &alpha,
                                      fc1bias, ref_fc1.outputs,
                                      onevec, 1,
                                      &alpha,
                                      fc1, ref_fc1.outputs));

        // ReLU activation
        cudnnCheckErrors(cudnnActivationForward(cudnnH, fc1Act, &alpha,
                                                fc1Tensor, fc1, &beta, fc1Tensor, fc1act));

        // FC2 layer
        // Forward propagate neurons using weights (fc2 = wfc2'*fc1act)
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                                      ref_fc2.outputs, batchSize, ref_fc2.inputs,
                                      &alpha,
                                      wfc2, ref_fc2.inputs,
                                      fc1act, ref_fc2.inputs,
                                      &beta,
                                      fc2, ref_fc2.outputs));
        // Add bias using GEMM's "beta" (fc2 += wfc2bias*1_vec')
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                      ref_fc2.outputs, batchSize, 1,
                                      &alpha,
                                      fc2bias, ref_fc2.outputs,
                                      onevec, 1,
                                      &alpha,
                                      fc2, ref_fc2.outputs));

        // ReLU activation
        cudnnCheckErrors(cudnnActivationForward(cudnnH, fc2Act, &alpha,
                                                fc2Tensor, fc2, &beta, fc2Tensor, fc2act));

        // FC3 Layer
        // Forward propagate neurons using weights (fc3 = wfc3'*fc2act)
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                                      ref_fc3.outputs, batchSize, ref_fc3.inputs,
                                      &alpha,
                                      wfc3, ref_fc3.inputs,
                                      fc2act, ref_fc3.inputs,
                                      &beta,
                                      fc3, ref_fc3.outputs));
        // Add bias using GEMM's "beta" (fc3 += fc3bias*1_vec')
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                      ref_fc3.outputs, batchSize, 1,
                                      &alpha,
                                      fc3bias, ref_fc3.outputs,
                                      onevec, 1,
                                      &alpha,
                                      fc3, ref_fc3.outputs));

        // ReLU activation
        cudnnCheckErrors(cudnnActivationForward(cudnnH, fc3Act, &alpha,
                                                fc3Tensor, fc3, &beta, fc3Tensor, fc3act));

        // FC4 Layer
        // Forward propagate neurons using weights (fc4 = wfc4'*fc3act)
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                                      ref_fc4.outputs, batchSize, ref_fc4.inputs,
                                      &alpha,
                                      wfc4, ref_fc4.inputs,
                                      fc3act, ref_fc4.inputs,
                                      &beta,
                                      fc4, ref_fc4.outputs));
        // Add bias using GEMM's "beta" (fc4 += fc4bias*1_vec')
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                      ref_fc4.outputs, batchSize, 1,
                                      &alpha,
                                      fc4bias, ref_fc4.outputs,
                                      onevec, 1,
                                      &alpha,
                                      fc4, ref_fc4.outputs));

        // ReLU activation
        cudnnCheckErrors(cudnnActivationForward(cudnnH, fc4Act, &alpha,
                                                fc4Tensor, fc4, &beta, fc4Tensor, fc4act));

        // FC5 Layer
        // Forward propagate neurons using weights (fc5 = wfc5'*fc4act)
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                                      ref_fc5.outputs, batchSize, ref_fc5.inputs,
                                      &alpha,
                                      wfc5, ref_fc5.inputs,
                                      fc4act, ref_fc5.inputs,
                                      &beta,
                                      fc5, ref_fc5.outputs));

        // Add bias using GEMM's "beta" (fc5 += wfc5bias*1_vec')
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                      ref_fc5.outputs, batchSize, 1,
                                      &alpha,
                                      fc5bias, ref_fc5.outputs,
                                      onevec, 1,
                                      &alpha,
                                      fc5, ref_fc5.outputs));

        // Softmax loss
        cudnnCheckErrors(cudnnSoftmaxForward(cudnnH, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                             &alpha, fc5Tensor, fc5, &beta, fc5Tensor, out));
    }

    void backward(double *data, double *labels, double *fc1, double *fc1act, double *fc2, double *fc2act, double *fc3, double *fc3act,
                  double *fc4, double *fc4act, double *fc5, double *dout, double *wfc1, double *fc1bias,
                  double *wfc2, double *fc2bias, double *wfc3, double *fc3bias, double *wfc4, double *fc4bias, double *wfc5, double *fc5bias,
                  double *gfc1, double *gfc1bias, double *gfc2, double *gfc2bias, double *gfc3, double *gfc3bias, double *gfc4, double *gfc4bias,
                  double *gfc5, double *gfc5bias, double *dfc1, double *dfc1act, double *dfc2, double *dfc2act, double *dfc3, double *dfc3act,
                  double *dfc4, double *dfc4act, double *dfc5, double *onevec)
    {
        double alpha = 1.0, beta = 0.0, gamma = -1.0;

        double scale = 1.0 / static_cast<double>(batchSize);

        cublasCheckErrors(cublasDaxpy(cublasH, ref_fc5.outputs * batchSize, &gamma, labels, 1, dout, 1));

        cublasCheckErrors(cublasDscal(cublasH, ref_fc5.outputs * batchSize, &scale, dout, 1));

        // FC5 layer
        // Compute derivative with respect to weights: gfc5 = (fc4act * dout')
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc5.inputs, ref_fc5.outputs, batchSize,
                                      &alpha, fc4act, ref_fc5.inputs, dout, ref_fc5.outputs, &beta, gfc5, ref_fc5.inputs));
        // Compute derivative with respect to bias: gfc5bias = dout * 1_vec -> TODO: cublasDcopy
        cublasCheckErrors(cublasDgemv(cublasH, CUBLAS_OP_N, ref_fc5.outputs, batchSize,
                                      &alpha, dout, ref_fc5.outputs, onevec, 1, &beta, gfc5bias, 1));
        // Compute derivative with respect to data (for previous layer): wfc5*dout
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc5.inputs, batchSize, ref_fc5.outputs,
                                      &alpha, wfc5, ref_fc5.inputs, dout, ref_fc5.outputs, &beta, dfc5, ref_fc5.inputs));
        // act activation
        cudnnCheckErrors(cudnnActivationBackward(cudnnH, fc4Act, &alpha,
                                                 fc4Tensor, fc4act, fc4Tensor, dfc5,
                                                 fc4Tensor, fc4, &beta, fc4Tensor, dfc4act));

        // FC4 layer
        // Compute derivative with respect to weights: gfc4 = (fc3act * dfc4act')
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc4.inputs, ref_fc4.outputs, batchSize,
                                      &alpha, fc3act, ref_fc4.inputs, dfc4act, ref_fc4.outputs, &beta, gfc4, ref_fc4.inputs));
        // Compute derivative with respect to bias: gfc4bias = dfc4act * 1_vec
        cublasCheckErrors(cublasDgemv(cublasH, CUBLAS_OP_N, ref_fc4.outputs, batchSize,
                                      &alpha, dfc4act, ref_fc4.outputs, onevec, 1, &beta, gfc4bias, 1));
        // Compute derivative with respect to data (for previous layer): wfc4*dfc4act (800x500*500xN)
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc4.inputs, batchSize, ref_fc4.outputs,
                                      &alpha, wfc4, ref_fc4.inputs, dfc4act, ref_fc4.outputs, &beta, dfc4, ref_fc4.inputs));
        // act activation
        cudnnCheckErrors(cudnnActivationBackward(cudnnH, fc3Act, &alpha, fc3Tensor, fc3act, fc3Tensor, dfc4,
                                                 fc3Tensor, fc3, &beta, fc3Tensor, dfc3act));

        // FC3 layer
        // Compute derivative with respect to weights: gfc3 = (fc2act * dfc3act')
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc3.inputs, ref_fc3.outputs, batchSize,
                                      &alpha, fc2act, ref_fc3.inputs, dfc3act, ref_fc3.outputs, &beta, gfc3, ref_fc3.inputs));
        // Compute derivative with respect to bias: gfc3bias = dfc3act * 1_vec
        cublasCheckErrors(cublasDgemv(cublasH, CUBLAS_OP_N, ref_fc3.outputs, batchSize,
                                      &alpha, dfc3act, ref_fc3.outputs, onevec, 1, &beta, gfc3bias, 1));
        // Compute derivative with respect to data (for previous layer): wfc3*dfc3act (800x500*500xN)
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc3.inputs, batchSize, ref_fc3.outputs,
                                      &alpha, wfc3, ref_fc3.inputs, dfc3act, ref_fc3.outputs, &beta, dfc3, ref_fc3.inputs));
        // act activation
        cudnnCheckErrors(cudnnActivationBackward(cudnnH, fc2Act, &alpha, fc2Tensor, fc2act, fc2Tensor, dfc3,
                                                 fc2Tensor, fc2, &beta, fc2Tensor, dfc2act));

        // FC2 layer
        // Compute derivative with respect to weights: gfc2 = (fc1act * dfc2act')
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc2.inputs, ref_fc2.outputs, batchSize,
                                      &alpha, fc1act, ref_fc2.inputs, dfc2act, ref_fc2.outputs, &beta, gfc2, ref_fc2.inputs));
        // Compute derivative with respect to bias: gfc2bias = dfc2act * 1_vec
        cublasCheckErrors(cublasDgemv(cublasH, CUBLAS_OP_N, ref_fc2.outputs, batchSize,
                                      &alpha, dfc2act, ref_fc2.outputs, onevec, 1, &beta, gfc2bias, 1));
        // Compute derivative with respect to data (for previous layer): wfc2*dfc2act (800x500*500xN)
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc2.inputs, batchSize, ref_fc2.outputs,
                                      &alpha, wfc2, ref_fc2.inputs, dfc2act, ref_fc2.outputs, &beta, dfc2, ref_fc2.inputs));
        // act activation
        cudnnCheckErrors(cudnnActivationBackward(cudnnH, fc1Act, &alpha, fc1Tensor, fc1act, fc1Tensor, dfc2,
                                                 fc1Tensor, fc1, &beta, fc1Tensor, dfc1act));

        // FC1 layer
        // Compute derivative with respect to weights: gfc1 = (data * dfc1act')
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, ref_fc1.inputs, ref_fc1.outputs, batchSize,
                                      &alpha, data, ref_fc1.inputs, dfc1act, ref_fc1.outputs, &beta, gfc1, ref_fc1.inputs));
        // Compute derivative with respect to bias: gfc1bias = dfc1act * 1_vec
        cublasCheckErrors(cublasDgemv(cublasH, CUBLAS_OP_N, ref_fc1.outputs, batchSize,
                                      &alpha, dfc1act, ref_fc1.outputs, onevec, 1, &beta, gfc1bias, 1));
        // Compute derivative with respect to data (for previous layer): wfc1*dfc1act (800x500*500xN)
        cublasCheckErrors(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, ref_fc1.inputs, batchSize, ref_fc1.outputs,
                                      &alpha, wfc1, ref_fc1.inputs, dfc1act, ref_fc1.outputs, &beta, dfc1, ref_fc1.inputs));
    }

    void update(double learning_rate,
                double *wfc1, double *fc1bias,
                double *wfc2, double *fc2bias,
                double *wfc3, double *fc3bias,
                double *wfc4, double *fc4bias,
                double *wfc5, double *fc5bias,
                double *gfc1, double *gfc1bias,
                double *gfc2, double *gfc2bias,
                double *gfc3, double *gfc3bias,
                double *gfc4, double *gfc4bias,
                double *gfc5, double *gfc5bias)
    {
        double alpha = -learning_rate;

        // Fully connected 1
        cublasCheckErrors(cublasDaxpy(cublasH, static_cast<int>(ref_fc1.weights.size()),
                                      &alpha, gfc1, 1, wfc1, 1));
        cublasCheckErrors(cublasDaxpy(cublasH, static_cast<int>(ref_fc1.bias.size()),
                                      &alpha, gfc1bias, 1, fc1bias, 1));

        // Fully connected 2
        cublasCheckErrors(cublasDaxpy(cublasH, static_cast<int>(ref_fc2.weights.size()),
                                      &alpha, gfc2, 1, wfc2, 1));
        cublasCheckErrors(cublasDaxpy(cublasH, static_cast<int>(ref_fc2.bias.size()),
                                      &alpha, gfc2bias, 1, fc2bias, 1));

        // Fully connected 3
        cublasCheckErrors(cublasDaxpy(cublasH, static_cast<int>(ref_fc3.weights.size()),
                                      &alpha, gfc3, 1, wfc3, 1));
        cublasCheckErrors(cublasDaxpy(cublasH, static_cast<int>(ref_fc3.bias.size()),
                                      &alpha, gfc3bias, 1, fc3bias, 1));

        // Fully connected 4
        cublasCheckErrors(cublasDaxpy(cublasH, static_cast<int>(ref_fc4.weights.size()),
                                      &alpha, gfc4, 1, wfc4, 1));
        cublasCheckErrors(cublasDaxpy(cublasH, static_cast<int>(ref_fc4.bias.size()),
                                      &alpha, gfc4bias, 1, fc4bias, 1));

        // Fully connected 5
        cublasCheckErrors(cublasDaxpy(cublasH, static_cast<int>(ref_fc5.weights.size()),
                                      &alpha, gfc5, 1, wfc5, 1));
        cublasCheckErrors(cublasDaxpy(cublasH, static_cast<int>(ref_fc5.bias.size()),
                                      &alpha, gfc5bias, 1, fc5bias, 1));
    }
};

int main(int argc, char **argv)
{
    std::vector<uint8_t> train_images, test_images;
    std::vector<int> train_labels, test_labels;
    int batchSize, epochs;
    size_t width, height;
    float elapsedTime;
    double lr, lr_decay = 0.00005;
    int train_size, test_size;
    int blockSize;   // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the
                     // maximum occupancy for a full device launch
    int gridSize;    // The actual grid size needed, based on input size
    int num_classes = 10;

    if (argc == 1)
    {
        batchSize = 8;
        epochs = 100;
        lr = 0.02;
    }
    else
    {
        batchSize = atoi(argv[2]);
        epochs = atoi(argv[4]);
        lr = atof(argv[6]);
    }

    cudaCheckErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, FillOnes, 0, 0));

    std::cout << "MNIST data directory: " << MNIST_DATA_DIR << std::endl;

    mnist::get_image_size(MNIST_DATA_DIR, width, height);

    mnist::MnistDataset dataset = mnist::read_dataset(MNIST_DATA_DIR);
    train_images = dataset.training_images;
    train_labels = dataset.training_labels;
    test_images = dataset.test_images;
    test_labels = dataset.test_labels;

    train_size = train_labels.size() / num_classes;
    test_size = test_labels.size() / num_classes;

    std::cout << "Number of training images = " << train_size << std::endl;
    std::cout << "Number of test images = " << test_size << std::endl;
    std::cout << "Image size = " << width << "x" << height << std::endl;
    std::cout << "Batch Size = " << batchSize << std::endl;

    // Defining architecture
    Dense fc1(width * height, 400);
    Dense fc2(fc1.outputs, 300);
    Dense fc3(fc2.outputs, 100);
    Dense fc4(fc3.outputs, 50);
    Dense fc5(fc4.outputs, num_classes);

    // He weight initialization (double precision)
    std::random_device rd;
    std::mt19937 gen(32 < 0 ? rd() : static_cast<unsigned int>(32));

    double wfc1 = sqrt(6.0 / (fc1.inputs)); // mu + 3*std
    std::uniform_real_distribution<> dfc1(-wfc1, wfc1);
    double wfc2 = sqrt(6.0 / (fc2.inputs));
    std::uniform_real_distribution<> dfc2(-wfc2, wfc2);
    double wfc3 = sqrt(6.0 / (fc3.inputs));
    std::uniform_real_distribution<> dfc3(-wfc3, wfc3);
    double wfc4 = sqrt(6.0 / (fc4.inputs));
    std::uniform_real_distribution<> dfc4(-wfc4, wfc4);
    double wfc5 = sqrt(6.0 / (fc5.inputs));
    std::uniform_real_distribution<> dfc5(-wfc5, wfc5);

    for (auto &&iter : fc1.weights)
        iter = static_cast<double>(dfc1(gen));
    for (auto &&iter : fc1.bias)
        iter = static_cast<double>(dfc1(gen));
    for (auto &&iter : fc2.weights)
        iter = static_cast<double>(dfc2(gen));
    for (auto &&iter : fc2.bias)
        iter = static_cast<double>(dfc2(gen));
    for (auto &&iter : fc3.weights)
        iter = static_cast<double>(dfc3(gen));
    for (auto &&iter : fc3.bias)
        iter = static_cast<double>(dfc3(gen));
    for (auto &&iter : fc4.weights)
        iter = static_cast<double>(dfc4(gen));
    for (auto &&iter : fc4.bias)
        iter = static_cast<double>(dfc4(gen));
    for (auto &&iter : fc5.weights)
        iter = static_cast<double>(dfc5(gen));
    for (auto &&iter : fc5.bias)
        iter = static_cast<double>(dfc5(gen));

    // timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialize network
    Network model(batchSize, fc1, fc2, fc3, fc4, fc5);

    // start timing
    cudaEventRecord(start, 0);

    // Allocate Memory on device
    double *dev_data, *dev_labels, *dev_fc1, *dev_fc1Act, *dev_fc2, *dev_fc2Act, *dev_fc3, *dev_fc3Act, *dev_fc4, *dev_fc4Act, *dev_fc5;
    double *dev_wfc1, *dev_fc1bias, *dev_wfc2, *dev_fc2bias, *dev_wfc3, *dev_fc3bias, *dev_wfc4, *dev_fc4bias, *dev_wfc5, *dev_fc5bias;
    double *dev_out, *dev_onevec;

    cudaCheckErrors(cudaMalloc(&dev_data, sizeof(double) * batchSize * width * height));
    cudaCheckErrors(cudaMalloc(&dev_labels, sizeof(double) * num_classes * batchSize));
    cudaCheckErrors(cudaMalloc(&dev_fc1, sizeof(double) * batchSize * fc1.outputs));
    cudaCheckErrors(cudaMalloc(&dev_fc2, sizeof(double) * batchSize * fc2.outputs));
    cudaCheckErrors(cudaMalloc(&dev_fc3, sizeof(double) * batchSize * fc3.outputs));
    cudaCheckErrors(cudaMalloc(&dev_fc4, sizeof(double) * batchSize * fc4.outputs));
    cudaCheckErrors(cudaMalloc(&dev_fc5, sizeof(double) * batchSize * fc5.outputs));
    cudaCheckErrors(cudaMalloc(&dev_fc1Act, sizeof(double) * batchSize * fc1.outputs));
    cudaCheckErrors(cudaMalloc(&dev_fc2Act, sizeof(double) * batchSize * fc2.outputs));
    cudaCheckErrors(cudaMalloc(&dev_fc3Act, sizeof(double) * batchSize * fc3.outputs));
    cudaCheckErrors(cudaMalloc(&dev_fc4Act, sizeof(double) * batchSize * fc4.outputs));
    cudaCheckErrors(cudaMalloc(&dev_out, sizeof(double) * batchSize * fc5.outputs));
    cudaCheckErrors(cudaMalloc(&dev_onevec, sizeof(double) * model.batchSize));

    // Weights and Biases
    cudaCheckErrors(cudaMalloc(&dev_wfc1, sizeof(double) * fc1.weights.size()));
    cudaCheckErrors(cudaMalloc(&dev_wfc2, sizeof(double) * fc2.weights.size()));
    cudaCheckErrors(cudaMalloc(&dev_wfc3, sizeof(double) * fc3.weights.size()));
    cudaCheckErrors(cudaMalloc(&dev_wfc4, sizeof(double) * fc4.weights.size()));
    cudaCheckErrors(cudaMalloc(&dev_wfc5, sizeof(double) * fc5.weights.size()));

    cudaCheckErrors(cudaMalloc(&dev_fc1bias, sizeof(double) * fc1.bias.size()));
    cudaCheckErrors(cudaMalloc(&dev_fc2bias, sizeof(double) * fc2.bias.size()));
    cudaCheckErrors(cudaMalloc(&dev_fc3bias, sizeof(double) * fc3.bias.size()));
    cudaCheckErrors(cudaMalloc(&dev_fc4bias, sizeof(double) * fc4.bias.size()));
    cudaCheckErrors(cudaMalloc(&dev_fc5bias, sizeof(double) * fc5.bias.size()));

    // Gradients
    double *dev_gfc1, *dev_gfc1bias, *dev_gfc2, *dev_gfc2bias, *dev_gfc3, *dev_gfc3bias, *dev_gfc4, *dev_gfc4bias, *dev_gfc5, *dev_gfc5bias;
    cudaCheckErrors(cudaMalloc(&dev_gfc1, sizeof(double) * fc1.weights.size()));
    cudaCheckErrors(cudaMalloc(&dev_gfc2, sizeof(double) * fc2.weights.size()));
    cudaCheckErrors(cudaMalloc(&dev_gfc3, sizeof(double) * fc3.weights.size()));
    cudaCheckErrors(cudaMalloc(&dev_gfc4, sizeof(double) * fc4.weights.size()));
    cudaCheckErrors(cudaMalloc(&dev_gfc5, sizeof(double) * fc5.weights.size()));

    cudaCheckErrors(cudaMalloc(&dev_gfc1bias, sizeof(double) * fc1.bias.size()));
    cudaCheckErrors(cudaMalloc(&dev_gfc2bias, sizeof(double) * fc2.bias.size()));
    cudaCheckErrors(cudaMalloc(&dev_gfc3bias, sizeof(double) * fc3.bias.size()));
    cudaCheckErrors(cudaMalloc(&dev_gfc4bias, sizeof(double) * fc4.bias.size()));
    cudaCheckErrors(cudaMalloc(&dev_gfc5bias, sizeof(double) * fc5.bias.size()));

    // Diff wrt data
    double *dev_dfc1, *dev_dfc1act, *dev_dfc2, *dev_dfc2act, *dev_dfc3, *dev_dfc3act, *dev_dfc4, *dev_dfc4act, *dev_dfc5, *dev_dout, *dev_dloss;
    cudaCheckErrors(cudaMalloc(&dev_dfc1, sizeof(double) * model.batchSize * fc1.inputs));
    cudaCheckErrors(cudaMalloc(&dev_dfc2, sizeof(double) * model.batchSize * fc2.inputs));
    cudaCheckErrors(cudaMalloc(&dev_dfc3, sizeof(double) * model.batchSize * fc3.inputs));
    cudaCheckErrors(cudaMalloc(&dev_dfc4, sizeof(double) * model.batchSize * fc4.inputs));
    cudaCheckErrors(cudaMalloc(&dev_dfc5, sizeof(double) * model.batchSize * fc5.inputs));

    cudaCheckErrors(cudaMalloc(&dev_dfc1act, sizeof(double) * model.batchSize * fc1.outputs));
    cudaCheckErrors(cudaMalloc(&dev_dfc2act, sizeof(double) * model.batchSize * fc2.outputs));
    cudaCheckErrors(cudaMalloc(&dev_dfc3act, sizeof(double) * model.batchSize * fc3.outputs));
    cudaCheckErrors(cudaMalloc(&dev_dfc4act, sizeof(double) * model.batchSize * fc4.outputs));
    cudaCheckErrors(cudaMalloc(&dev_dout, sizeof(double) * model.batchSize * fc5.outputs));
    cudaCheckErrors(cudaMalloc(&dev_dloss, sizeof(double) * model.batchSize * fc5.outputs));

    // Transfer Host to Device
    cudaCheckErrors(cudaMemcpyAsync(dev_wfc1, &fc1.weights[0], sizeof(double) * fc1.weights.size(), cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpyAsync(dev_wfc2, &fc2.weights[0], sizeof(double) * fc2.weights.size(), cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpyAsync(dev_wfc3, &fc3.weights[0], sizeof(double) * fc3.weights.size(), cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpyAsync(dev_wfc4, &fc4.weights[0], sizeof(double) * fc4.weights.size(), cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpyAsync(dev_wfc5, &fc5.weights[0], sizeof(double) * fc5.weights.size(), cudaMemcpyHostToDevice));

    cudaCheckErrors(cudaMemcpyAsync(dev_fc1bias, &fc1.bias[0], sizeof(double) * fc1.bias.size(), cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpyAsync(dev_fc2bias, &fc2.bias[0], sizeof(double) * fc2.bias.size(), cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpyAsync(dev_fc3bias, &fc3.bias[0], sizeof(double) * fc3.bias.size(), cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpyAsync(dev_fc4bias, &fc4.bias[0], sizeof(double) * fc4.bias.size(), cudaMemcpyHostToDevice));
    cudaCheckErrors(cudaMemcpyAsync(dev_fc5bias, &fc5.bias[0], sizeof(double) * fc5.bias.size(), cudaMemcpyHostToDevice));

    // Round up according to array size
    gridSize = RoundUp(model.batchSize, blockSize);

    FillOnes<<<gridSize, blockSize>>>(dev_onevec, model.batchSize);

    // Normalize images
    std::vector<double> train_images_double(train_images.size()), train_labels_double(num_classes * train_size);
    for (size_t i = 0; i < train_size * width * height; ++i)
        train_images_double[i] = (double)train_images[i] / 255.0;

    for (size_t i = 0; i < num_classes * train_size; ++i)
    {
        train_labels_double[i] = (double)train_labels[i];
    }
    cudaCheckErrors(cudaDeviceSynchronize());

    /* Training */
    printf("\nTraining...\n");
    double *loss, total_loss;
    loss = (double *)malloc(batchSize * fc5.outputs * sizeof(double));
    int num_steps = (train_size) / batchSize;
    for (int iter = 0; iter < epochs; ++iter)
    {
        for (int step = 0; step < num_steps; step++)
        {
            int imageid = step;

            // Prepare current batch on device
            cudaCheckErrors(cudaMemcpyAsync(dev_data, &train_images_double[imageid * model.batchSize * width * height],
                                            sizeof(double) * model.batchSize * width * height, cudaMemcpyHostToDevice));
            cudaCheckErrors(cudaMemcpyAsync(dev_labels, &train_labels_double[imageid * model.batchSize * num_classes],
                                            sizeof(double) * model.batchSize * num_classes, cudaMemcpyHostToDevice));

            // Forward propagation
            model.forward(dev_data, dev_fc1, dev_fc1Act, dev_fc2, dev_fc2Act, dev_fc3, dev_fc3Act, dev_fc4, dev_fc4Act,
                          dev_fc5, dev_out, dev_wfc1, dev_fc1bias, dev_wfc2, dev_fc2bias, dev_wfc3, dev_fc3bias,
                          dev_wfc4, dev_fc4bias, dev_wfc5, dev_fc5bias, dev_onevec);

            // compute loss
            if ((step + 1) % 2000 == 0)
            {
                CrossEntropyLoss<<<RoundUp(batchSize, 128), 128>>>(dev_dloss, dev_out, dev_labels, fc5.outputs, batchSize);

                cudaCheckErrors(cudaMemcpy(loss, dev_dloss, fc5.outputs * batchSize * sizeof(double), cudaMemcpyDeviceToHost));
                total_loss = 0;
                for (int i = 0; i < fc5.outputs * batchSize; i++)
                {
                    total_loss += loss[i];
                }
                std::cout << "Epoch " << iter + 1 << "/" << epochs << ", Step = " << step + 1 << "/" << num_steps << ", Loss = " << total_loss / batchSize << std::endl;
            }

            // Backpropagation
            model.backward(dev_data, dev_labels, dev_fc1, dev_fc1Act, dev_fc2, dev_fc2Act, dev_fc3, dev_fc3Act,
                           dev_fc4, dev_fc4Act, dev_fc5, dev_out, dev_wfc1, dev_fc1bias, dev_wfc2, dev_fc2bias,
                           dev_wfc3, dev_fc3bias, dev_wfc4, dev_fc4bias, dev_wfc5, dev_fc5bias, dev_gfc1, dev_gfc1bias,
                           dev_gfc2, dev_gfc2bias, dev_gfc3, dev_gfc3bias, dev_gfc4, dev_gfc4bias, dev_gfc5, dev_gfc5bias,
                           dev_dfc1, dev_dfc1act, dev_dfc2, dev_dfc2act, dev_dfc3, dev_dfc3act, dev_dfc4, dev_dfc4act,
                           dev_dfc5, dev_onevec);

            // compute learning rate
            lr *= 1.f / (1.f + lr_decay * step * iter);

            // Update weights
            model.update(lr, dev_wfc1, dev_fc1bias, dev_wfc2, dev_fc2bias, dev_wfc3, dev_fc3bias, dev_wfc4, dev_fc4bias,
                         dev_wfc5, dev_fc5bias, dev_gfc1, dev_gfc1bias, dev_gfc2, dev_gfc2bias, dev_gfc3, dev_gfc3bias,
                         dev_gfc4, dev_gfc4bias, dev_gfc5, dev_gfc5bias);
        }
    }
    cudaCheckErrors(cudaDeviceSynchronize());

    // stop timing
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    // resolution is 0.5 micro second
    printf("%d images, epochs = %d, batch size = %d, time = %10.3e ms\n", train_size, epochs, batchSize, elapsedTime);

    /* Testing */
    printf("\nTesting...\n");

    // start timing
    cudaEventRecord(start, 0);

    // Initialize
    Network test(1, fc1, fc2, fc3, fc4, fc5);
    double accuracy;
    int count = 0;

    for (int i = 0; i < test_size; ++i)
    {
        std::vector<double> data(width * height);

        // Normalize image to be in [0,1]
        for (int j = 0; j < width * height; ++j)
            data[j] = (double)test_images[i * width * height + j] / 255.0;

        cudaCheckErrors(cudaMemcpyAsync(dev_data, &data[0], sizeof(double) * width * height, cudaMemcpyHostToDevice));

        // Forward propagate test image
        test.forward(dev_data, dev_fc1, dev_fc1Act, dev_fc2, dev_fc2Act, dev_fc3, dev_fc3Act, dev_fc4, dev_fc4Act,
                     dev_fc5, dev_out, dev_wfc1, dev_fc1bias, dev_wfc2, dev_fc2bias, dev_wfc3, dev_fc3bias,
                     dev_wfc4, dev_fc4bias, dev_wfc5, dev_fc5bias, dev_onevec);

        // Transfer output to host
        std::vector<double> probs(num_classes);

        cudaCheckErrors(cudaMemcpy(&probs[0], dev_out, sizeof(double) * num_classes, cudaMemcpyDeviceToHost));

        // Get label with maximum probability
        int max_label = 0;
        for (int id = 0; id < num_classes; ++id)
        {
            if (probs[max_label] < probs[id])
                max_label = id;
        }

        if (test_labels[i * num_classes + max_label] == 1)
            ++count;
    }
    accuracy = (double)count / (double)test_size;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Accuracy = %.2f%% \n", accuracy * 100.0);
    printf("%d images, batch size = %d, time = %10.3e ms\n", test_size, 1, elapsedTime);

    // free host memory
    free(loss);

    // free device memory
    cudaCheckErrors(cudaFree(dev_data));
    cudaCheckErrors(cudaFree(dev_labels));
    cudaCheckErrors(cudaFree(dev_fc1));
    cudaCheckErrors(cudaFree(dev_fc1Act));
    cudaCheckErrors(cudaFree(dev_fc2));
    cudaCheckErrors(cudaFree(dev_fc2Act));
    cudaCheckErrors(cudaFree(dev_fc3));
    cudaCheckErrors(cudaFree(dev_fc3Act));
    cudaCheckErrors(cudaFree(dev_fc4));
    cudaCheckErrors(cudaFree(dev_fc4Act));
    cudaCheckErrors(cudaFree(dev_fc5));
    cudaCheckErrors(cudaFree(dev_fc1bias));
    cudaCheckErrors(cudaFree(dev_fc2bias));
    cudaCheckErrors(cudaFree(dev_fc3bias));
    cudaCheckErrors(cudaFree(dev_fc4bias));
    cudaCheckErrors(cudaFree(dev_fc5bias));
    cudaCheckErrors(cudaFree(dev_onevec));
    cudaCheckErrors(cudaFree(dev_out));
    cudaCheckErrors(cudaFree(dev_wfc1));
    cudaCheckErrors(cudaFree(dev_wfc2));
    cudaCheckErrors(cudaFree(dev_wfc3));
    cudaCheckErrors(cudaFree(dev_wfc4));
    cudaCheckErrors(cudaFree(dev_wfc5));
    cudaCheckErrors(cudaFree(dev_gfc1));
    cudaCheckErrors(cudaFree(dev_gfc2));
    cudaCheckErrors(cudaFree(dev_gfc3));
    cudaCheckErrors(cudaFree(dev_gfc4));
    cudaCheckErrors(cudaFree(dev_gfc5));
    cudaCheckErrors(cudaFree(dev_gfc1bias));
    cudaCheckErrors(cudaFree(dev_gfc2bias));
    cudaCheckErrors(cudaFree(dev_gfc3bias));
    cudaCheckErrors(cudaFree(dev_gfc4bias));
    cudaCheckErrors(cudaFree(dev_gfc5bias));
    cudaCheckErrors(cudaFree(dev_dfc1));
    cudaCheckErrors(cudaFree(dev_dfc2));
    cudaCheckErrors(cudaFree(dev_dfc3));
    cudaCheckErrors(cudaFree(dev_dfc4));
    cudaCheckErrors(cudaFree(dev_dfc5));
    cudaCheckErrors(cudaFree(dev_dfc1act));
    cudaCheckErrors(cudaFree(dev_dfc2act));
    cudaCheckErrors(cudaFree(dev_dfc3act));
    cudaCheckErrors(cudaFree(dev_dfc4act));
    cudaCheckErrors(cudaFree(dev_dloss));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
