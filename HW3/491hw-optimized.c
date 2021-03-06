#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>

#define N 101 
#define maxiter 1000000
#define epsilon 0.0000001

double xold[(N + 2)][(N + 2)];
double xnew[(N + 2)][(N + 2)];

int main(int argc, char * argv[])
{
	double rtclock();
	double clkbegin, clkend, t;

	double thisdiff, maxdiff;
	int i, j, iter;

	/*  N is size of physical grid over which the heat equation is solved
		epsilon is the threshold for the convergence criterion
		xnew and xold hold the N*N new and old iterates for the temperature over grid
	*/


	// Initialization

	for (i = 0; i < N + 2; i++) xold[i][0] = i*50.0 / (N + 1);
	for (i = 0; i < N + 2; i++) xold[i][N + 1] = (i + N + 1)*50.0 / (N + 1);
	for (j = 0; j < N + 2; j++) xold[0][j] = j*50.0 / (N + 1);
	for (j = 0; j < N + 2; j++) xold[N + 1][j] = (j + N + 1)*50.0 / (N + 1);
	for (i = 1; i < N + 1; i++)
		for (j = 1; j < N + 1; j++) xold[i][j] = 0;

	clkbegin = rtclock();

	int done = 0;
	omp_set_dynamic( 0 );
	omp_set_num_threads(4);
	int num_threads = 0;
	iter = 0;
	maxdiff = 0;
	#pragma omp parallel private(i,j,thisdiff)
	{	

		#pragma omp single
		num_threads = omp_get_num_threads();
		#pragma omp barrier

		int id = omp_get_thread_num();
		int start = (N / num_threads) * id + 1; 
		int end = (N / num_threads) * (id + 1) + 1;
		if (id == num_threads - 1) end = N + 1;
		
		do
		{
			double diff = 0;
			for (i = start; i < end; i++)
				for (j = 1; j < N + 1; j++)
				{	
					double temp =  0.25*(xold[i - 1][j] + xold[i + 1][j] + xold[i][j - 1] + xold[i][j + 1]);
					xnew[i][j] = temp;
					if ((thisdiff = fabs(temp - xold[i][j])) > diff) 
						diff = thisdiff;
				}
		
			#pragma omp critical
			if (diff > maxdiff)
				maxdiff = diff;
			#pragma omp barrier
		
			#pragma omp single
			if (maxdiff < epsilon) {
				clkend = rtclock();
				done = 1;
				printf("Solution converged in  %d iterations\n", iter + 1);
				printf("Solution at center of grid : %f\n", xnew[(N + 1) / 2][(N + 1) / 2]);
				t = clkend - clkbegin;
				printf("Base-Jacobi: %.1f MFLOPS; Time = %.3f sec; \n", 4.0*N*N*(iter + 1) / t / 1000000, t);
			}
			else
			{
				maxdiff = 0;
				iter++;
			}

			for (i = start; i < end; i++)
				for (j = 1; j < N + 1; j++)
					xold[i][j] = xnew[i][j];		
			#pragma omp barrier
		} while (!done && iter < maxiter);
	} // end parallel region
}


double rtclock()
{
	struct timezone Tzp;
	struct timeval Tp;
	int stat;
	stat = gettimeofday(&Tp, &Tzp);
	if (stat != 0) printf("Error return from gettimeofday: %d", stat);
	return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

