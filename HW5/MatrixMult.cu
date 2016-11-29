#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TILE 2;

__global__ void gpu_mult_kernel(int* A, int* B, int* C, const int n)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	for (int k = 0; k < n; k++)
		C[i * n + j] += A[i * n + k] * B[k * n + j];
}

__global__ void sgpu_mult_kernel(int* A, int* B, int* C, const int n)
{
	__shared__ int sharedA[TILE][TILE];
	__shared__ int sharedB[TILE][TILE];

	int i = blockIdx.y * TILE + threadIdx.y;
	int j = blockIdx.x * TILE + threadIdx.x;

	// for each tiled section
	for (int m = 0; m < n / TILE; m++)
	{
		// copy data to for this section (each thread works to acheive this goal)
		sharedA[threadIdx.y][threadIdx.x] = A[i * n + (m * TILE + threadIdx.x)];
		sharedB[threadIdx.y][threadIdx.x] = B[(m * TILE + threadIdx.y) * n + j];

		// synchronize
		__syncthreads();

		// calculate the values for the tile section
		for (int k = 0; k < TILE_WIDTH; k++)
			C[i * n + j] += sharedA[threadIdx.x][k] * sharedB[k][threadIdx.y];

		// synchronize
		__syncthreads();
	}
}

void cpu_mult(int n, int* A, int* B, int* C);
int* allocate_matrix(int n);
void randomize_matrix(int n, int* A);
void print_matrix(int n, int* A);

int main(int argc, char * argv[])
{
	if (argc != 2)
	{
		printf("Error: Missing argument");
		return 0;
	}

	srand(time(NULL));

	int n = atoi(argv[1]);
	int m = n * n * sizeof(int);
	if (n <= 2)
	{
		printf("Error: Invalid Matrix Size");
		return 0;
	}

	int *X, *Y, *Zcpu, *Zgpu, *Zsgpu;
	int *X_d, *Y_d, *Zgpu_d, *Zsgpu_d;
	allocate_matrix(n);
	X = allocate_matrix(n);
	Y = allocate_matrix(n);
	Zcpu = allocate_matrix(n);
	Zgpu = allocate_matrix(n);
	Zsgpu = allocate_matrix(n);
	randomize_matrix(n, X);
	randomize_matrix(n, Y);

	// print X
	print_matrix(n, X);
	print_matrix(n, Y);
	cpu_mult(n, X, Y, Zcpu);
	print_matrix(n, Zcpu);

	// allocate memory on gpu
	cudaMalloc((void **)&X_d, m);
	cudaMalloc((void **)&Y_d, m);
	cudaMalloc((void **)&Zgpu_d, m);

	// copy host data to gpu
	cudaMemcpy(X_d, X, m, cudaMemcpyHostToDevice);
	cudaMemcpy(Y_d, Y, m, cudaMemcpyHostToDevice);
	cudaMemcpy(Zgpu_d, Zgpu, m, cudaMemcpyHostToDevice);
	cudaMemcpy(Zsgpu_d, Zsgpu, m, cudaMemcpyHostToDevice);

	// kernel parameters
	dim3 dimGrid(n / TILE, n / TILE, 1);
	dim3 dimBlock(TILE, TILE, 1);

	// run kernels
	gpu_mult_kernel << <dimGrid, dimBlock >> > (X_d, Y_d, Zgpu_d, n);
	sgpu_mult_kernel << <dimGrid, dimBlock >> > (X_d, Y_d, Zgpu_d, n);

	// copy result back
	cudaMemcpy(Zgpu, Zgpu_d, m, cudaMemcpyDeviceToHost);
	cudaMemcpy(Zsgpu, Zsgpu_d, m, cudaMemcpyDeviceToHost);
	print_matrix(n, Zgpu);
	print_matrix(n, Zsgpu);
}

void cpu_mult(int n, int* A, int* B, int* C)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			for (int k = 0; k < n; k++)
				C[i * n + j] += A[i * n + k] * B[k * n + j];
}

int* allocate_matrix(int n)
{
	int* A = (int *)malloc(n * n * sizeof(int*));
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			A[i * n + j] = 0;
	return A;
}

void randomize_matrix(int n, int * A)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			A[i * n + j] = rand() % 10;
}

void print_matrix(int n, int* A)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			printf("%4d ", A[i * n + j]);
		printf("\n");
	}
	printf("\n");
}










