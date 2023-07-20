mod bindings;

#[cfg(test)]
mod tests {
    use crate::bindings::*;
    use std::time::Instant;

    #[test]
    fn mat_mult_benchmark() {
        // Random numbers for generation
        let mat_dim = 4096;

        let id_1;
        let id_2;
        unsafe {
            id_1 = register_matrix(vec![0.0; mat_dim * mat_dim].as_ptr(), mat_dim, mat_dim);
            id_2 = register_matrix(vec![0.0; mat_dim * mat_dim].as_ptr(), mat_dim, mat_dim);
        }

        let num_iterations = 100;
        let start = Instant::now();

        let mut result_id = 0;
        for _ in 0..num_iterations {
            unsafe {
                result_id = cuda_matrix_multiply(id_1, mat_dim, mat_dim, id_2, mat_dim, mat_dim);
                cuda_synchronize();
                unregister_matrix(result_id);
            }
        }
        unsafe { cuda_synchronize() }
        let elapsed = start.elapsed();
        println!(
    "\n=================================\nTime per iteration: {} ms\n=================================",
    elapsed.as_millis() as f64 / num_iterations as f64
  );

        print!("{}", result_id);

        assert!(false);
    }
}
