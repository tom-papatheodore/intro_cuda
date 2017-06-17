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

// Values for MxN matrix
#define M 100
#define N	200

// Kernel
__global__ void add_matrices(int *a, int *b, int *c)
{
	int column = blockDim.x * blockIdx.x + threadIdx.x;
	int row    = blockDim.y * blockIdx.y + threadIdx.y;

	if (row < M && column < N)
	{
		int thread_id = row * N + column;
		c[thread_id] = a[thread_id] + b[thread_id];
	}
}

// Main program
int main()
{
	// Number of bytes to allocate for MxN matrix
	size_t bytes = M*N*sizeof(int);

	// Allocate memory for arrays A, B, and C on host
	int A[M][N];
	int B[M][N];
	int C[M][N];

	// Allocate memory for arrays d_A, d_B, and d_C on device
	int *d_A, *d_B, *d_C;
	cudaErrorCheck( cudaMalloc(&d_A, bytes) );
	cudaErrorCheck( cudaMalloc(&d_B, bytes) );
	cudaErrorCheck( cudaMalloc(&d_C, bytes) );

	// Initialize host arrays A and B
	for(int i=0; i<M; i++)
	{
		for(int j=0; j<N; j++)
		{
			A[i][j] = 1;
			B[i][j] = 2;
		}
	}

	// Copy data from host arrays A and B to device arrays d_A and d_B
	cudaErrorCheck( cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) );
	cudaErrorCheck( cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice) );

	// Set execution configuration parameters
	// 		threads_per_block: number of CUDA threads per grid block
	//		blocks_in_grid   : number of blocks in grid
	//		(These are c structs with 3 member variables x, y, x)
	dim3 threads_per_block( 16, 16, 1 );
	dim3 blocks_in_grid( ceil( float(N) / threads_per_block.x ), ceil( float(M) / threads_per_block.y ), 1 );

	// Launch kernel
	add_matrices<<< blocks_in_grid, threads_per_block >>>(d_A, d_B, d_C);

	// Check for errors in kernel launch (e.g. invalid execution configuration paramters)
  cudaError_t cuErrSync  = cudaGetLastError();

	// Check for errors on the GPU after control is returned to CPU
  cudaError_t cuErrAsync = cudaDeviceSynchronize();

  if (cuErrSync != cudaSuccess) 
	{ printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErrSync)); exit(0); }

  if (cuErrAsync != cudaSuccess) 
	{ printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErrAsync)); exit(0); }

	// Copy data from device array d_C to host array C
	cudaErrorCheck( cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost) );

	// Verify results
	for(int i=0; i<M; i++)
	{
		for(int j=0; j<N; j++)
		{
			if (C[i][j] != 3)
			{
				printf("C[%d][%d] = %d instread of 3\n", i, j, C[i][j]);
			}
		}
	}

	// Free GPU memory
	cudaErrorCheck( cudaFree(d_A) );
	cudaErrorCheck( cudaFree(d_B) );
	cudaErrorCheck( cudaFree(d_C) );

  printf("\n--------------------------------\n");
  printf("__SUCCESS__\n");
  printf("--------------------------------\n");
  printf("M                         = %d\n", M);
	printf("N                         = %d\n", N);
  printf("Threads Per Block (x-dim) = %d\n", threads_per_block.x);
  printf("Threads Per Block (y-dim) = %d\n", threads_per_block.y);
  printf("Blocks In Grid (x-dim)    = %d\n", blocks_in_grid.x);
	printf("Blocks In Grid (y-dim)    = %d\n", blocks_in_grid.y);
  printf("--------------------------------\n\n");

	return 0;
}
