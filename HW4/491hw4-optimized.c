/*
mpicc -std=c99 -g 491hw4-optimized.c -o jacobi
mpirun -H borg,cauchy,fermat,godel,granville,lamarr,mckusick,naur,perlman -npernode 4 ./jacobi
*/

#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>

#define N 101 
#define M 103
#define maxiter 1000000
#define epsilon 0.0000001


int main(int argc, char * argv[])
{

	double xold[M][M];
	double xnew[M][M];
	double rtclock();
	double clkbegin, clkend;
	int iter = 0;

	// MPI Variables
	int myid, numprocs;

	// Generate the Data
	for (int i = 0; i < N + 2; i++) xold[i][0] = i*50.0 / (N + 1);
	for (int i = 0; i < N + 2; i++) xold[i][N + 1] = (i + N + 1)*50.0 / (N + 1);
	for (int j = 0; j < N + 2; j++) xold[0][j] = j*50.0 / (N + 1);
	for (int j = 0; j < N + 2; j++) xold[N + 1][j] = (j + N + 1)*50.0 / (N + 1);
	for (int i = 1; i < N + 1; i++)
		for (int j = 1; j < N + 1; j++)
			xold[i][j] = 0;

	// intialize nodes
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	// root will keep track of time
	if (myid == 0)
		clkbegin = rtclock();

	// divide up data
	int centerid = numprocs / 2;
	int centerTag = 1000 * centerid;
	int start = (N / numprocs) * myid + 1;
	int end = (N / numprocs) * (myid + 1) + 1;
	if (myid == numprocs - 1) end = N + 1;

	do
	{
		MPI_Status status;
		MPI_Request requests[4];
		double diff = 0;
		double maxdiff = 0;
		double center = 0;
		for (int i = start; i < end; i++)
		{
			for (int j = 1; j < N + 1; j++)
			{
				double temp = 0.25*(xold[i - 1][j] + xold[i + 1][j] + xold[i][j - 1] + xold[i][j + 1]);
				double thisdiff;
				xnew[i][j] = temp;
				if ((thisdiff = fabs(temp - xold[i][j])) > diff)
					diff = thisdiff;
				if (i == j && j == (N + 1) / 2)
					center = temp;
			}

			// begin sending shared data as it becomes available
			if (i == start && myid > 0) // first row is ready, begin sending
				MPI_Isend(xnew[start], M, MPI_DOUBLE, myid - 1, start, MPI_COMM_WORLD, &requests[1]);
			else if (i == end - 1 && myid < numprocs - 1) // last row is ready, begin sending
				MPI_Isend(xnew[end - 1], M, MPI_DOUBLE, myid + 1, end - 1, MPI_COMM_WORLD, &requests[0]);
		}

		// Begin receiving data
		if (myid < numprocs - 1)
			MPI_Irecv(xold[end], M, MPI_DOUBLE, myid + 1, end, MPI_COMM_WORLD, &requests[2]);
		if (myid > 0)
			MPI_Irecv(xold[start - 1], M, MPI_DOUBLE, myid - 1, start - 1, MPI_COMM_WORLD, &requests[3]);

		// calculate the maxdiff
		MPI_Reduce(&diff, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		// update xold
		for (int i = start; i < end; i++)
			for (int j = 1; j < N + 1; j++)
				xold[i][j] = xnew[i][j];

		if (myid == centerid)
			MPI_Send(&center, 1, MPI_DOUBLE, 0, centerTag, MPI_COMM_WORLD);
		else if (myid == 0)
		{
			MPI_Recv(&center, 1, MPI_DOUBLE, centerid, centerTag, MPI_COMM_WORLD, &status);
			if (maxdiff < epsilon) {
				clkend = rtclock();
				double t = clkend - clkbegin;
				printf("Solution converged in  %d iterations\n", iter + 1);
				printf("Solution at center of grid : %f\n", center);
				printf("Base-Jacobi: %.1f MFLOPS; Time = %.3f sec; \n", 4.0*N*N*(iter + 1) / t / 1000000, t);
				iter = maxiter; // this will end the loop
			}
			iter++;
		}

		// tell all nodes if we are done and update iteration
		MPI_Bcast(&iter, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// wait for new data
		if (myid > 0)
		{
			MPI_Wait(&requests[3], &status);
			MPI_Wait(&requests[1], &status);
		}
		if (myid < numprocs - 1)
		{
			MPI_Wait(&requests[0], &status);
			MPI_Wait(&requests[2], &status);
		}
	} while (iter < maxiter);
	MPI_Finalize();
}


double rtclock()
{
	//struct timezone Tzp;
	struct timeval Tp;
	int stat;
	stat = gettimeofday(&Tp, NULL);//&Tzp);
	if (stat != 0)
		printf("Error return from gettimeofday: %d", stat);
	return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

