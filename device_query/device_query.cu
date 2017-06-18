#include <stdio.h>

int main()
{
	int num_devices;

	cudaGetDeviceCount(&num_devices);
	printf("\n---------------------------------\n");
	printf("Number of CUDA-capable devices: %d\n", num_devices);
	printf("---------------------------------\n\n");

	for(int i=0; i<num_devices; i++)
	{
		cudaDeviceProp dev_property;
		cudaGetDeviceProperties(&dev_property, i);

		printf("Device %d: %s\n", i, dev_property.name);

		printf("  Compute capability: %d.%d\n\n", dev_property.major, dev_property.minor);

		for(int i=0; i<3; i++)
		{
			printf("  Max size of grid (blocks in dim %d): %d\n", i, dev_property.maxGridSize[i]);
		}
		printf("\n");

		printf("  Max number of (total) threads per block : %d\n", dev_property.maxThreadsPerBlock);
		
		for(int i=0; i<3; i++)
		{
			printf("  Max number of threads per block in dim %d: %d\n", i, dev_property.maxThreadsDim[i]);
		}
		printf("\n");

		printf("  Number of multiprocessors: %d\n\n", dev_property.multiProcessorCount);

    printf("  Total global memory (B)    : %zu\n", dev_property.totalGlobalMem);
		printf("  Shared memory per block (B): %zu\n\n", dev_property.sharedMemPerBlock);

		// Add in user-defined theoretical memory bandwidth (in GB/s) here.

	  printf("\n---------------------------------\n\n");

	}
}
