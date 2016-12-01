#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

const int MAXTILE = 32;

__global__ void gpu_mult_kernel(int* A, int* B, int* C, const int n)
{
	// determine access location based on block ids and threadids
	int i = blockIdx.y * blockDim.y + threadIdx.y; // row
	int j = blockIdx.x * blockDim.x + threadIdx.x; // col

	for (int k = 0; k < n; k++)
	{
		// each thread responsible for one cell in C
		C[i * n + j] += A[i * n + k] * B[k * n + j];
	}
};

__global__ void sgpu_mult_kernel(int* A, int* B, int* C, const int n, const int b)
{		
	__shared__ int sharedA[MAXTILE][MAXTILE];
	__shared__ int sharedB[MAXTILE][MAXTILE];

	int i = (blockIdx.y * b) + threadIdx.y; // row
	int j = (blockIdx.x * b) + threadIdx.x; // col

	// for each tiled section
	int ntile = n / b;
	for (int t = 0; t < ntile; t++)
	{
		// copy data to for this shared tiles (each thread works to acheive this goal)
		sharedA[threadIdx.y][threadIdx.x] = A[i * n + (t * b + threadIdx.x)];
		sharedB[threadIdx.y][threadIdx.x] = B[(t * b + threadIdx.y) * n + j];

		// synchronize to ensure shared tiles are updated by all threads
		__syncthreads();

		// calculate the values for the tile section
		for (int k = 0; k < b; k++)
		{
			// each thread responsible for one cell in C
			C[i * n + j] += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x]; 
		}

		// synchronize before moving to next tile (prevents premature changes to shared memory)
		__syncthreads();
	}
};

void cpu_mult(int n, int* A, int* B, int* C);
int* allocate_matrix(int n);
void randomize_matrix(int n, int* A);
void print_matrix(int n, int* A);
bool equal_matrix(int n, int* A, int* B);
double rtclock();

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
	if (n < 1)
	{
		printf("Error: Invalid Matrix Size");
		return 0;
	}
	
	int b = 1;
	if (n <= MAXTILE)
	{
		b = n;
	}
	else
	{
		b = MAXTILE;
		while (b > 1 && n % b != 0) // while b is not a factor of n
		{
			b--;
		}
	}
	int *X, *Y, *Zcpu, *Zgpu, *Zsgpu;
	int *X_d, *Y_d, *Zgpu_d, *Zsgpu_d;
	double start, end;
	allocate_matrix(n);
	X = allocate_matrix(n);
	Y = allocate_matrix(n);
	Zcpu = allocate_matrix(n);
	Zgpu = allocate_matrix(n);
	Zsgpu = allocate_matrix(n);
	randomize_matrix(n, X);
	randomize_matrix(n, Y);

	// print X
	//print_matrix(n, X);
	//print_matrix(n, Y);
	start = rtclock();
	cpu_mult(n, X, Y, Zcpu);
	end = rtclock();
	printf("CPU time:\t%f\n", end - start);
	//print_matrix(n, Zcpu);

	// allocate memory on gpu
	cudaMalloc((void **)&X_d, m);
	cudaMalloc((void **)&Y_d, m);

	// copy host data to gpu
	cudaMemcpy(X_d, X, m, cudaMemcpyHostToDevice);
	cudaMemcpy(Y_d, Y, m, cudaMemcpyHostToDevice);

	// kernel parameters
	dim3 dimGrid(n / b, n / b, 1);
	dim3 dimBlock(b, b, 1);

	// run kernels
	start = rtclock();
	cudaMalloc((void **)&Zgpu_d, m);
	cudaMemcpy(Zgpu_d, Zgpu, m, cudaMemcpyHostToDevice);
	gpu_mult_kernel <<<dimGrid, dimBlock>>> (X_d, Y_d, Zgpu_d, n);
	cudaMemcpy(Zgpu, Zgpu_d, m, cudaMemcpyDeviceToHost);
	end = rtclock();
	printf("GPU time:\t%f\n", end - start);
	
	start = rtclock();
	cudaMalloc((void **)&Zsgpu_d, m);
	cudaMemcpy(Zsgpu_d, Zsgpu, m, cudaMemcpyHostToDevice);
	sgpu_mult_kernel <<<dimGrid, dimBlock>>> (X_d, Y_d, Zsgpu_d, n, b);
	cudaMemcpy(Zsgpu, Zsgpu_d, m, cudaMemcpyDeviceToHost);
	end = rtclock();
	printf("sGPU time:\t%f\n", end - start);

	//print_matrix(n, Zgpu);
	//print_matrix(n, Zsgpu);
	printf("Zcpu == Zgpu? %s\n", equal_matrix(n, Zcpu, Zgpu) ? "true" : "false");
	printf("Zcpu == Zsgpu? %s\n", equal_matrix(n, Zcpu, Zsgpu) ? "true" : "false");
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

double rtclock()
{
	struct timezone Tzp;
	struct timeval Tp;
	int stat;
	stat = gettimeofday(&Tp, &Tzp);
	if (stat != 0) 
	{
		printf("Error return from gettimeofday: %d\n", stat);
	}
	return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}


bool equal_matrix(int n, int* A, int* B)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
		{
			int index = i * n + j;
			if (A[index] != B[index])
				return false;
		}
	return true;
}







