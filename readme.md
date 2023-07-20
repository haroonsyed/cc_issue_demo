# ANSWER: CUDA DEBUGGING IS KEPT ON UNLESS EXPLICITY RUNNING RELEASE MODE
# https://github.com/rust-lang/cc-rs/issues/830

1. First build the kernels (will be used to demonstrate without CC performance) `chmod +x build_kernels.sh && ./build_kernels.sh`
2. Configure build.rs to run with direct linking to lib file. Or have cc compile the cuda and link it itself.<br>
   Do this by commenting everything above xor below 
```
    ////////////////////////////////////////
    //  BUILDING WITH CC IS SLOWER BELOW
    ///////////////////////////////////////
```
3. Run cargo test. Test will fail with printout of time taken to complete a 4096x4096 matrix multiplication 100 times (reduce if taking too long).
