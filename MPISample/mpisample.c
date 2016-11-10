/*
mpicc -std=c99 mpisample.c -o cpi
mpirun -H borg,cauchy,fermat,godel,granville,lamarr,mckusick,naur,perlman -npernode 2 ./cpi
*/

#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#define PI 3.141592653589793238462643

double f(double);
double f(double a)
{
	return 4.0 / (1.0 + a * a);
}

int main(int argc, char* argv[])
{
	bool done = false;
	int n, myid, numprocs;
	double mypi, pi, h, sum, x;
	double startwtime = 0.0, endwtime;
	int namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Get_processor_name(processor_name, &namelen);
	
	fprintf(stderr, "Process %d on %s\n", myid, processor_name);
	n = 0;
	while (!done)
	{
		if (myid == 0)
		{
			printf("Enter the number of intervals: (0 quits)\n");
			scanf("%d", &n);
			
			if (n < 0)
				n = 100;
		
			startwtime = MPI_Wtime();
		}
		MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (n == 0)
			done = 1;
		else
		{
			h = 1.0 / (double) n;
			sum = 0.0;
			for (int i = myid + 1; i <= n; i += numprocs)
			{
				x = h * ((double) i - 0.5);
				sum += f(x);
			}
			mypi = h * sum;
		
			MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		
			if (myid == 0)
			{
				printf("pi is approximately %.16f, Error is %.16f\n",
					pi, fabs(pi - PI));
				endwtime = MPI_Wtime();
				printf("wall clock time = %f\n", endwtime - startwtime);
			}
		}
	}
	MPI_Finalize();
	return 0;
}





















