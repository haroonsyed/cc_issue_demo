#include <unordered_map>

#include "./cuda_kernels.cuh"

bool init_pool = false;
size_t mat_generated_count(0);
std::unordered_map<size_t, float*> mat_map;

// Error checking macro: https://stackoverflow.com/a/14038590
#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = false) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void cuda_synchronize() {
    cudaDeviceSynchronize();
}

/////////////////////
/// Matrix Setup API
/////////////////////
void init_min_pool_size() {
    int device;
    cudaGetDevice(&device);
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, device);
    size_t threshold = sizeof(float) * 2048 * 2048;  // Around 68 Mb reserved
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
    init_pool = false;
}
size_t register_matrix_buffer(float* gpu_buffer) {
    if (init_pool) {
        init_min_pool_size();
    }

    // Register with the map for retrieval later
    mat_map[mat_generated_count] = gpu_buffer;
    return mat_generated_count++;  // Fine if this overflows
}

size_t register_matrix(size_t rows, size_t cols) {
    // Upload the data
    float* gpu_buffer;
    gpuErrchk(cudaMallocAsync(&gpu_buffer, sizeof(float) * rows * cols, 0));

    return register_matrix_buffer(gpu_buffer);
}

size_t register_matrix(float* data, size_t rows, size_t cols) {
    // Upload the data
    float* gpu_buffer;
    gpuErrchk(cudaMallocAsync(&gpu_buffer, sizeof(float) * rows * cols, 0));
    gpuErrchk(cudaMemcpy(gpu_buffer, data, sizeof(float) * rows * cols, cudaMemcpyHostToDevice));
    // Potentially nasty bug by acting like you copied data when you havent finished if using cudaMemCpyAsync...
    return register_matrix_buffer(gpu_buffer);
}

void unregister_matrix(size_t mat_id) {
    gpuErrchk(cudaFreeAsync(mat_map[mat_id], 0));
    mat_map.erase(mat_id);
}

void get_matrix_data(size_t mat_id, int rows, int cols, float* data_buffer) {
    float* gpu_buffer = mat_map[mat_id];
    gpuErrchk(cudaMemcpy(data_buffer, gpu_buffer, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost));
}

//////////////////////////
/// Matrix Operations API
//////////////////////////
__global__ void matrix_multiply_kernel(float* mat1_buffer, int mat1_rows, int mat1_cols, float* mat2_buffer, int mat2_rows, int mat2_cols, float* out_buffer, int out_rows, int out_cols) {
    // Go by col row instead of row col. Enabled memory coalescing
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < out_rows && col < out_cols) {
        // O[i][j] = mat1[i][:] weighted sum mat2[:][j]
        // Where common dimension : is mat1col/mat2row

        float weighted_sum = 0.0;
        for (int common = 0; common < mat1_cols; common++) {
            // mat1[i][common]
            int mat1_index = mat1_cols * row + common;
            // mat1[common][j]
            int mat2_index = mat2_cols * common + col;

            weighted_sum += mat1_buffer[mat1_index] * mat2_buffer[mat2_index];
        }

        int output_index = row * out_cols + col;
        out_buffer[output_index] = weighted_sum;
    }
}

size_t cuda_matrix_multiply(size_t mat1_id, size_t mat1_rows, size_t mat1_cols, size_t mat2_id, size_t mat2_rows, size_t mat2_cols) {
    // Create output buffer
    int out_rows = mat1_rows;
    int out_cols = mat2_cols;
    size_t out_mat_id = register_matrix(out_rows, out_cols);

    // Get the gpu buffers to operate on
    float* gpu_mat1_buffer = mat_map[mat1_id];
    float* gpu_mat2_buffer = mat_map[mat2_id];
    float* gpu_out_buffer = mat_map[out_mat_id];

    // Kernel launch parameters
    const int THREADS_PER_BLOCK_X = 16;
    const int THREADS_PER_BLOCK_Y = 16;

    dim3 block_dim(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
    dim3 grid_dim((out_cols / block_dim.x) + 1, (out_rows / block_dim.y) + 1, 1);

    // Run the kernels
    matrix_multiply_kernel<<<grid_dim, block_dim>>>(gpu_mat1_buffer, mat1_rows, mat1_cols, gpu_mat2_buffer, mat2_rows, mat2_cols, gpu_out_buffer, out_rows, out_cols);

    gpuErrchk(cudaPeekAtLastError());

    // Return result matrix id
    return out_mat_id;
}