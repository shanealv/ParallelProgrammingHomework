#include <cstdlib>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <ios>
#include <iostream>
#include <limits>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#define EIGHT_KB 8192

using namespace std;

long GetFileSize(int fp);
int ReadRecords(int fp, long numRecords);
int ProcessRecord (char record[]);


typedef struct
{
	long rCount = 0;
	long lastRecordID = 0;
	bool lastRecordValid = true;
} RecordStat;

RecordStat theadStats[100];

int main (int argc, char * argv[])
{
	if (argc != 2)
	{
		cerr << "Checker: Invalid Number of Arguments" << endl;
		return 0;
	}

	char * file = argv[1];

	cout << "Opening file: " << file << endl;
	int fd = open(file, O_RDWR, 0);
	if (fd < 0)
	{
		cerr << "Could not open: " << file << endl;
		return 1;
	}
	long filesize = GetFileSize(fd);
	long numRecords = filesize / EIGHT_KB;
	cout << "File is of size: " << filesize << endl;
	cout << "num records " << numRecords << endl;

	long numThreads = ReadRecords(fd, numRecords);
	
	cout << "num threads " << numThreads << endl;
	for (int i = 0; i < numThreads; i++)
	{
		RecordStat stat = theadStats[i];
		cout << "Thread\t" << i + 1 << "\t";
		cout << "NUM Recs\t" << stat.rCount << "\t";
		cout << "Last RID\t" << stat.lastRecordID << "\t";
		cout << "Safe\t" << stat.lastRecordValid << endl;
	}
	close(fd);
}


long GetFileSize(int fd)
{
	long size = (long) lseek(fd, 0L, SEEK_END);
	return size;
}

int ReadRecords(int fd, long numRecords)
{
	char * buffer = new char[EIGHT_KB];
	long numThreads = 0;
	for (int i = 0; i < numRecords; i++)
	{
		pread(fd, buffer, EIGHT_KB, i * EIGHT_KB);
		int tid = ProcessRecord(buffer);
		if (tid > numThreads)
			numThreads = tid;
	}
	return numThreads;
}

int ProcessRecord (char record[])
{
	long tid;
	long address;
	long rid;
	long oldchecksum;

	memcpy(&tid, 		record + 0 * sizeof(long), sizeof(long));
	memcpy(&address, 	record + 1 * sizeof(long), sizeof(long));
	memcpy(&rid, 		record + 2 * sizeof(long), sizeof(long));
	memcpy(&oldchecksum,record + 3 * sizeof(long), sizeof(long));

	cout << "thread id: " << tid << "\t";
	cout << "address:   " << address << "\t";
	cout << "record id: " << rid << "\t";
	cout << "checksum:  " << oldchecksum << endl;

	if(tid == 0 || tid > 100) return 0;

	long newchecksum = 0;
	for (int i = 4 * sizeof(long); i < EIGHT_KB; i += sizeof(int))
	{
		int value;
		memcpy(&value, record + i, sizeof(int));
		newchecksum += value;
	}

	theadStats[tid-1].rCount++;
	if (theadStats[tid-1].lastRecordID < rid)
	{
		theadStats[tid-1].lastRecordID = rid;
		theadStats[tid-1].lastRecordValid = (newchecksum == oldchecksum);
	}
	return tid;
}











