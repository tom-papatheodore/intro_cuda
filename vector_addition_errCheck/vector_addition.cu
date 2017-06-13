#include <stdio.h>

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
    cudaError_t cuErr = call;                                                             \
    if(cudaSuccess != cuErr){                                                             \
      printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
      exit(0);                                                                            \
    }                                                                                     \
}while(0)

// Size of array
#define N 10240

// Kernel
__global__ void add_vectors(int *a, int *b, int *c, int n)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < n) c[id] = a[id] + b[id];
}

// Main program
int main()
{
	// Number of bytes to allocate for N integers
	size_t bytes = N*sizeof(int);

	// Allocate memory for arrays A, B, and C on host
	int *A = (int*)malloc(bytes);
	int *B = (int*)malloc(bytes);
	int *C = (int*)malloc(bytes);

	// Allocate memory for arrays d_A, d_B, and d_C on device
	int *d_A, *d_B, *d_C;
	cudaErrorCheck( cudaMalloc(&d_A, bytes) );	
	cudaErrorCheck( cudaMalloc(&d_B, bytes) );
	cudaErrorCheck( cudaMalloc(&d_C, bytes) );

	// Fill host arrays A and B
	for(int i=0; i<N; i++)
	{
		A[i] = 1;
		B[i] = 2;
		C[i] = 0;
	}

	// Copy data from host arrays A and B to device arrays d_A and d_B
	cudaErrorCheck( cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) );
	cudaErrorCheck( cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice) );

	// Set execution configuration parameters
	//		thr_per_blk: number of CUDA threads per grid block
	//		blk_in_grid: number of blocks in grid
	int thr_per_blk = 128;
	int blk_in_grid = ceil( float(N) / thr_per_blk );

	// Launch kernel
	add_vectors<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C, N);

  // Check for errors in kernel launch (e.g. invalid execution configuration paramters)
	cudaError_t cuErrSync  = cudaGetLastError();

  // Check for errors on the GPU after control is returned to CPU
	cudaError_t cuErrAsync = cudaDeviceSynchronize();

	if (cuErrSync != cudaSuccess) { printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErrSync)); exit(0); }
	if (cuErrAsync != cudaSuccess) { printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErrAsync)); exit(0); }

	// Copy data from device array d_C to host array C
	cudaErrorCheck( cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost) );

	// Verify results
	for(int i=0; i<N; i++)
	{
		if(C[i] != 3)
		{ 
			printf("Error: value of C[%d] = %d instead of 3\n", i, C[i]);
			exit(-1);
		}
	}	

	// Free CPU memory
	free(A);
	free(B);
	free(C);

	// Free GPU memory
	cudaErrorCheck( cudaFree(d_A) );
	cudaErrorCheck( cudaFree(d_B) );
	cudaErrorCheck( cudaFree(d_C) );

	printf("__SUCCESS__\n");

	return 0;
}
