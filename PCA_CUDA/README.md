# Parallelized_PCA_CUDA
CUDA implementation of PCA

# Compilation
nvcc -lm main_cuda.cu lab3_cuda.cu lab3_io.cu -o pca -lgomp
