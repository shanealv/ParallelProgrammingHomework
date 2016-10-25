#include <cstdlib>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <ios>
#include <iostream>
#include <limits>
#include <pthread.h>
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
long record_id = 0;

void * DoWork (void * arg);
bool IsFileValid (const char fileName[]);
long GetFileSize(const char filename[]);
void WriteRecord(const char filename[], int address, char buffer[]);
char * CreateRecord (long tid, long address);
long GetNextID ();

int main (int argc, char * argv[])
{
	if (argc != 3)
	{
		cerr << "Invalid Number of Arguments" << endl;
		return 0;
	}
	
	file = argv[1];
	num_workers = atoi(argv[2]);
	
	if (!IsFileValid(file))
	{
		cerr << "Invalid File" << endl;
		return 0;
	}
	
	if (num_workers < 2 || num_workers > 100)
	{
		cerr << "Unsupported number of workers" << endl;
		return 0;
	}
	
	cout << "Trying to use file: " << file << endl;
	filesize = GetFileSize(file);
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
	
	for (long i = 1; i <= num_workers; i++)
	{
		int returnCode = pthread_create(workers + (i-1), NULL, DoWork, (void *) i);
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
	
	pthread_mutex_destroy(&record_id_mutex);
}

void * DoWork (void * threadid)
{
	long id = (long) threadid;
	while (true)
	{
		int rn = rand() % filesize;
		long address = EIGHT_KB * (rn / EIGHT_KB);
		char * record = CreateRecord(id, address);
		pthread_mutex_lock(&worker_mutex);
		WriteRecord(file, address, record);
		cout << "Thread ID:\t" << id  << "\tAddress:\t" << address << "  \tID:\t" << id << endl;
		pthread_mutex_unlock(&worker_mutex);
	}
	return NULL;
}

bool IsFileValid (const char fileName[])
{
	ifstream infile(fileName);
	return infile.good();
}

long GetFileSize(const char filename[])
{
	ifstream file;
	file.open(filename, ios::in | ios::binary);
	file.ignore(numeric_limits<streamsize>::max());
	streamsize length = file.gcount();
	file.clear();
	file.seekg(0, ios_base::beg);
	file.close();
	return (long)length;
}

void WriteRecord(const char filename[], int address, char buffer[])
{
	int fd = open(filename, O_WRONLY);
	if (fd != -1)
	{
		cerr << "Could not open " << filename << endl;
		return;
	}
	
	for (int i = 0; i < 8; i++)
	{
		int size = EIGHT_KB / 8;
		int offset = i * size;
		pwrite(fd, buffer + offset, size, offset);
	}
	close(fd);
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















