#include <stdio.h>

#define M 100
#define N	200

__global__ void add_matrices(int *a, int *b, int *c, int m, int n)
{
	int column = blockDim.x * blockIdx.x + threadIdx.x;
	int row    = blockDim.y * blockIdx.y + threadIdx.y;

	if (row < m && column < n)
	{
		int thread_id = row * n + column;
		c[thread_id] = a[thread_id] + b[thread_id];
	}

}

int main()
{

	int A[M][N];
	int B[M][N];
	int C[M][N];

	for(int i=0; i<M; i++)
	{
		for(int j=0; j<N; j++)
		{
			A[i][j] = 1;
			B[i][j] = 2;
			C[i][j] = 0;
		}
	}

	int *d_A, *d_B, *d_C;

	size_t bytes = M*N*sizeof(int);

	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);

	cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

	int num_threads_x = 16;
	int num_threads_y = 16;

	int num_blocks_x  = ceil(float(N)/num_threads_x);
	int num_blocks_y  = ceil(float(M)/num_threads_y);

	printf("num_threads_x: %d\n", num_threads_x);
	printf("num_threads_y: %d\n", num_threads_y);
	printf("num_blocks_x: %d\n", num_blocks_x);
	printf("num_blocks_y: %d\n", num_blocks_y);

	dim3 blocks_in_grid( num_blocks_x, num_blocks_y, 1);
	dim3 threads_in_block( num_threads_x, num_threads_y, 1 );

	add_matrices<<< blocks_in_grid, threads_in_block >>>(d_A, d_B, d_C, M, N);

	cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

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

	printf("__SUCCESS__\n");

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

	return 0;
}
