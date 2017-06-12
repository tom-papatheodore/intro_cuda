#include <stdio.h>

#define N 1024

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

	// Allocate memory for arrays d_A, d_B, and d_C on device
	int *d_A, *d_B, *d_C;

	cudaMalloc(&d_A, bytes);	
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);

  // Fill host arrays A and B
  for(int i=0; i<N; i++)
  {
    A[i] = 1;
    B[i] = 2;
  }

	// Copy data from host arrays A and B to device arrays d_A and d_B
	cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

	// Set execution configuration parameters and launch kernel
	//		thr_per_blk: number of CUDA threads per grid block
	//		blk_in_grid: number of blocks in grid
	int thr_per_blk = 128;
	int blk_in_grid = ceil( float(N) / thr_per_blk );

  add_vectors<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C, N);

	// Copy data from device array d_C to host array C
	cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

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
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	printf("__SUCCESS__\n");

	return 0;
}
