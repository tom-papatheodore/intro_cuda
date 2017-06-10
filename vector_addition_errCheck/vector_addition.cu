#include <stdio.h>

#define cudaErrorCheck(call)																															\
do{																																												\
		cudaError_t cuErr = call;																															\
		if(cudaSuccess != cuErr){																															\
			printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
			exit(0);																																						\
		}																																											\
}while(0)

#define N 10240


__global__ void add_vectors(int *a, int *b, int *c, int n)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < n) c[id] = a[id] + b[id];
}


int main()
{
	// Number of bytes to allocate for N integers
	size_t bytes = N*sizeof(int);

  // Allocate memory for arrays A, B, and C on host
  int *A = (int*)malloc(bytes);
  int *B = (int*)malloc(bytes);
  int *C = (int*)malloc(bytes);

	// Fill host arrays A and B
	for(int i=0; i<N; i++)
	{
		A[i] = 1;
		B[i] = 2;
	}

	// Allocate memory for arrays d_A, d_B, and d_C on device
	int *d_A, *d_B, *d_C;

	cudaErrorCheck( cudaMalloc(&d_A, bytes) );
	cudaErrorCheck( cudaMalloc(&d_B, bytes) );
	cudaErrorCheck( cudaMalloc(&d_C, bytes) );

	// Copy data from host arrays A and B to device arrays d_A and d_B
	cudaErrorCheck( cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) );
	cudaErrorCheck( cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice) );

	// Set execution configuration parameters and launch kernel
	int threads_in_block = 4096;
	int blocks_in_grid = ceil(N/threads_in_block);

	printf("blocks in grid: %d \n", blocks_in_grid);
	add_vectors<<< blocks_in_grid, threads_in_block >>>(d_A, d_B, d_C, N);

	cudaError_t cuErrSync  = cudaGetLastError();
	cudaError_t cuErrAsync = cudaDeviceSynchronize();

	// This is needed to find errors in kernel launch (e.g. invalid execution configuration parameters)
  if (cuErrSync != cudaSuccess) { printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErrSync)); exit(0); }	 

	// This is needed to find errors on the GPU after control is returned to CPU
  if (cuErrAsync != cudaSuccess) { printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErrAsync)); exit(0); }

// Should actually be this line to reset the runtime error state
//  if (cuErrAsync != cudaSuccess) { printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cudaGetLastError())); exit(0); }

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

	printf("__SUCCESS__\n");

	free(A);
	free(B);
	free(C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}
