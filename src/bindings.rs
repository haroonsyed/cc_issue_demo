use std::ffi::c_float;

extern "C" {
    pub fn cuda_synchronize();
    pub fn register_matrix(data: *const c_float, rows: usize, cols: usize) -> usize;
    pub fn unregister_matrix(mat_id: usize) -> usize;
    pub fn cuda_matrix_multiply(
        mat1_id: usize,
        mat1_rows: usize,
        mat1_cols: usize,
        mat2_buffer: usize,
        mat2_rows: usize,
        mat2_cols: usize,
    ) -> usize;
}
