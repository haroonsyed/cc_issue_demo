To see the performance difference first build the cuda kernel for the version without CC

1. `chmod +x build_kernels.sh && ./build_kernels.sh`
2. Configure build.rs to run with direct linking to lib file. Or have cc compile the cuda and link it itself.<br>
   Do this by commenting everything above xor below 
```
    ////////////////////////////////////////
    //  BUILDING WITH CC IS SLOWER BELOW
    ///////////////////////////////////////
```
3. Run cargo test. Test will fail with printout of time taken to complete a 4096x4096 matrix multiplication 100 times (reduce if taking too long).
