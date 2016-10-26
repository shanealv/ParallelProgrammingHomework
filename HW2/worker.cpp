#include <cstdlib>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <ios>
#include <iostream>
#include <limits>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#define EIGHT_KB 8192
using namespace std;

pthread_t * workers;
pthread_mutex_t worker_mutex;
pthread_mutex_t record_id_mutex;
char * file;
long filesize;
int num_workers;
int fd;
long record_id = 0;

void * DoWork (void * arg);
long GetFileSize(int fp);
void WriteRecord(const char filename[], int address, char buffer[]);
char * CreateRecord (long tid, long address);
long GetNextID ();

int main (int argc, char * argv[])
{
	if (argc != 3)
	{
		cerr << "Worker: Invalid Number of Arguments" << endl;
		return 0;
	}
	
	file = argv[1];
	num_workers = atoi(argv[2]);
	
	
	if (num_workers < 2 || num_workers > 100)
	{
		cerr << "Unsupported number of workers" << endl;
		return 0;
	}
	
	cout << "Opening file: " << file << endl;
	fd = open(file, O_RDWR, 0);
	if (fd < 0)
	{
		cerr << "Could not open: " << file << endl;
		return 1;
	}
	filesize = GetFileSize(fd);
	cout << "File is of size: " << filesize << endl;
	cout << "Number of workers: " << num_workers << endl;
	
	
	workers = new pthread_t[num_workers];
	if (pthread_mutex_init(&worker_mutex, NULL) != 0)
	{
		cerr << "Error: mutex init failed" << endl;
		return 1;
	}
	
	if (pthread_mutex_init(&record_id_mutex, NULL) != 0)
	{
		cerr << "Error: mutex init failed" << endl;
		return 1;
	}
	
	srand(time(NULL));
	
	for (long i = 0; i < num_workers; i++)
	{
		int returnCode = pthread_create(workers + (i-1), NULL, DoWork, (void *) (i + 1));
		if (returnCode)
		{
			cerr << "Error: unable to create thread, return code: " << returnCode << endl;
			return 1;
		}
	}
	
	for (long i = 0; i < num_workers; i++)
	{
		void * out;
		int returnCode = pthread_join(*(workers + i), &out);
		if (returnCode)
		{
			cerr << "Error: unable to join thread, return code: " << returnCode << endl;
			return 1;
		}
	}
	
	close(fd);
	pthread_mutex_destroy(&record_id_mutex);
}

void * DoWork (void * threadid)
{
	long id = (long) threadid;
	int num_records = filesize / EIGHT_KB;
	while (true)
	{
		int rn = rand() % (num_records + 1);
		long address = rn * EIGHT_KB;
		char * record = CreateRecord(id, address);
		
		pthread_mutex_lock(&worker_mutex);
		WriteRecord(file, address, record);
		
#ifdef DEBUG
		cout << "Thread ID:  " << id  << "\tAddress:  " << address << "  \tID:  " << id << endl;
#endif
		pthread_mutex_unlock(&worker_mutex);
	}
	return NULL;
}

long GetFileSize(int fp)
{
	long size = (long) lseek(fp, 0L, SEEK_END);
	return size;
}

void WriteRecord(const char filename[], int address, char buffer[])
{
#ifdef DEBUG
	long data = 0;
	memcpy(&data, buffer + 0 * sizeof(long), sizeof(long));
	cout << "tid: " << data << "\t";
	memcpy(&data, buffer + 1 * sizeof(long), sizeof(long));
	cout << "add: " << data << "\t";
	memcpy(&data, buffer + 2 * sizeof(long), sizeof(long));
	cout << "rid: " << data << "\t";
	memcpy(&data, buffer + 3 * sizeof(long), sizeof(long));
	cout << "chk: " << data << endl;
#endif
	
	int partitions = 64;
	for (int i = 0; i < partitions; i++)
	{
		int size = EIGHT_KB / partitions;
		int offset = address + i * size;
		pwrite(fd, buffer + i * size, size, offset);
	}
}

char * CreateRecord (long tid, long address)
{
	char * record = new char[EIGHT_KB];
	long rid = GetNextID();
	memcpy(record + 0 * sizeof(long), &tid, sizeof(long));
	memcpy(record + 1 * sizeof(long), &address, sizeof(long));
	memcpy(record + 2 * sizeof(long), &rid, sizeof(long));
	
	long checksum = 0;
	for (int i = 4 * sizeof(long); i < EIGHT_KB; i += sizeof(int))
	{
		int randomNumber = rand();
		checksum += randomNumber;
		memcpy(record + i, &randomNumber, sizeof(int));
	}
	memcpy(record + 3 * sizeof(long), &checksum, sizeof(long));

	return record;
}

long GetNextID ()
{
	long id;
	pthread_mutex_lock(&record_id_mutex);
	id = record_id++;
	pthread_mutex_unlock(&record_id_mutex);
	return id;
	
}















