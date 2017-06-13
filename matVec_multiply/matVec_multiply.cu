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
#define N 200

// Kernel
__global__ void multiply_mat_vec(int *a, int *x, int *y, int m, int n)
{
  int row    = blockDim.y * blockIdx.y + threadIdx.y;

	if(row < m)
	{
		for(int i=0; i<n; i++)
		{
			y[row] = y[row] + a[row*n + i]*x[i];
		}
	}
}

// Main program
int main()
{
  // Number of bytes to allocate for MxN matrix
  size_t bytes = M*N*sizeof(int);

  // Allocate memory for matrix A on host
  int A[M][N];

	// Allocate memory for arrays x and y on host
	int x[N], y[M];

  // Allocate memory for matrix d_A on device
  int *d_A;
  cudaErrorCheck( cudaMalloc(&d_A, bytes) );

	// Allocate memory for arrays d_x and d_y on device
	int *d_x, *d_y;
	cudaErrorCheck( cudaMalloc(&d_x, N*sizeof(int)) );
	cudaErrorCheck( cudaMalloc(&d_y, M*sizeof(int)) );

  // Initialize host matrix A
  for(int i=0; i<M; i++)
  {
    for(int j=0; j<N; j++)
    {
      A[i][j] = 1;
    }
  }

	// Initialize host array x
	for(int i=0; i<N; i++)
	{
		x[i] = 1;
	}

	// Initialize host array y
	for(int i=0; i<M; i++)
	{
		y[i] = 0;
	}

	// Copy data from host matrix A to device matrix d_A
	cudaErrorCheck( cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) );
	
	// Copy data from host arrays x and y to device arrays d_x and d_y
	cudaErrorCheck( cudaMemcpy(d_x, x, N*sizeof(int), cudaMemcpyHostToDevice) );

  // Set execution configuration parameters
  //    threads_per_block: number of CUDA threads per grid block
  //    blocks_in_grid   : number of blocks in grid
  //    (These are c structs with 3 member variables x, y, x)
  dim3 threads_per_block( 1, 128, 1 );
  dim3 blocks_in_grid( 1, ceil( float(M) / threads_per_block.y ), 1 );

	multiply_mat_vec<<< blocks_in_grid, threads_per_block >>>(d_A, d_x, d_y, M, N);

  // Check for errors in kernel launch (e.g. invalid execution configuration paramters)
  cudaError_t cuErrSync  = cudaGetLastError();

  // Check for errors on the GPU after control is returned to CPU
  cudaError_t cuErrAsync = cudaDeviceSynchronize();

  if (cuErrSync != cudaSuccess)
  { printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErrSync)); exit(0); }

  if (cuErrAsync != cudaSuccess)
  { printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErrAsync)); exit(0); }

	// Copy the data from device array d_y to host array y
	cudaErrorCheck( cudaMemcpy(y, d_y, M*sizeof(int), cudaMemcpyDeviceToHost) );

	for(int i=0; i<M; i++)
	{
		if(y[i] != N)
		{
			printf("Error - y[%d] = %d instead of %d\n", i, y[i], N);
			exit(0);
		}
	}

	printf("__SUCCESS__\n");

	return 0;
}
