/*
mpicc -std=c99 mpihello.c -o hello
mpirun -H borg,cauchy,fermat,godel,granville,lamarr,mckusick,naur,perlman -npernode 2 ./hello
*/
#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[])
{
	int rank, size, rest, result;
	char namech[MPI_MAX_PROCESSOR_NAME+1];
	char* name = namech;
	int* res;
	res = &result;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(name,res);
	printf("Hello world, I am %s %d of %d\n", name, rank, size);
	MPI_Finalize();
	
	return 0;
}
