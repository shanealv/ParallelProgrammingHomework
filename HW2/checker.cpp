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

bool IsFileValid (const char fileName[]);
long GetFileSize(const char filename[]);
int ReadRecords(char filename[], long numRecords);
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
		cerr << "Invalid Number of Arguments" << endl;
		return 0;
	}

	char * file = argv[1];

	if (!IsFileValid(file))
	{
		cerr << "Invalid File" << endl;
		return 0;
	}
	
	long filesize = GetFileSize(file);
	long numRecords = filesize / EIGHT_KB;
	cout << "file size   " << filesize << endl;
	cout << "num records " << numRecords << endl;

	long numThreads = ReadRecords(file, numRecords);
	
	cout << "num threads " << numThreads << endl;
	for (int i = 0; i < numThreads + 1; i++)
	{
		RecordStat stat = theadStats[i];
		cout << "Thread\t" << i << "\t";
		cout << "NUM Recs\t" << stat.rCount << "\t";
		cout << "Last RID\t" << stat.lastRecordID << "\t";
		cout << "Safe\t" << stat.lastRecordValid << endl;
	}
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

int ReadRecords(char filename[], long numRecords)
{
	ifstream inf(filename, ios::in | ios::binary);
	char * buffer = new char[EIGHT_KB];
	long numThreads = 0;
	for (int i = 0; i < numRecords; i++)
	{
		long address = i * EIGHT_KB;
		inf.seekg(address, ios::beg);
		inf.read(buffer, EIGHT_KB);
		int tid = ProcessRecord(buffer);
		if (tid > numThreads)
			numThreads = tid;
	}
	inf.close();
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
	memcpy(&oldchecksum, 	record + 3 * sizeof(long), sizeof(long));
	cout << tid << " " << address << " " << rid << " " << oldchecksum << "    ";
	if(tid == 0) return 0;

	long newchecksum = 0;
	for (int i = 4 * sizeof(long); i < EIGHT_KB; i += sizeof(int))
	{
		int value;
		memcpy(&value, record + i, sizeof(int));
		newchecksum += value;
	}

	theadStats[tid].rCount++;
	if (theadStats[tid].lastRecordID < rid)
	{
		theadStats[tid].lastRecordID = rid;
		theadStats[tid].lastRecordValid = (newchecksum == oldchecksum);
	}
	return tid;
}











