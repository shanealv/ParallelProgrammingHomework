/**
 * Simple shell interface program.
 *
 * Operating System Concepts - Ninth Edition
 * Copyright John Wiley & Sons - 2013
 */
#define _POSIX_SOURCE
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define MAX_LINE		80 /* 80 chars per line, per command */
#define H_COUNT			10 /* 10 lines of history */

typedef struct HistoryNode
{
	struct HistoryNode * next;
	char * command;
} HistoryNode;
typedef struct History
{
	HistoryNode * head;
	int size;
} History;

History history = 
{ 
	.head = NULL,
	.size = 0 
};

int GetCommand (char buffer[], char* argv[]);
void Execute (int argc, char * argv[]);
void AddToHistory(char* buffer);
int HistoryLength();
void PrintHistory();
char* GetFromHistory(int n);
void sigHandler(int signum);

bool ExitWait = false;

int main(void)
{
	int argc = 0;
	char* argv[MAX_LINE/2 + 1];	// command line (of 80) has max of 40 arguments
	char buffer[MAX_LINE + 1]; // stores individual commands
	int should_run = 1;
	
	for (int i = 0; i < MAX_LINE/2 + 1; i++)
		argv[i] = NULL;
		
	do {   
		printf("osh>");
		fflush(stdout);

		if ((argc = GetCommand(buffer, argv)) < 0)
			continue;

		if (strcmp(*argv, "exit") == 0)
			return 0;
		
		if (strcmp(*argv, "history") == 0)
		{
			PrintHistory();
			continue;
		}

		AddToHistory(buffer);
		
		Execute(argc, argv);
	} while (true);
    
	return 0;
}

int GetCommand (char buffer[], char* argv[])
{
	char command[MAX_LINE + 1];
	const char space[2] = " ";
	char * argNew;
	int argc = 0;

	fgets(command, MAX_LINE + 1, stdin);
	command[MAX_LINE] = 0;

	int lastIdx = strlen(command) - 1;
	
	if (lastIdx == 0)
		return -1;
	else if (command[lastIdx] == '\n')
		command[lastIdx] = 0; // remove newline

	// if a history indexer was used, replace command with it
	if (lastIdx >= 2 && command[0] == '!')
		if (command[1] == '!')
		{
			char * histCommand = GetFromHistory(history.size);
			if (histCommand == NULL)
				return -1;
			strcpy(command, histCommand);
			printf("%s\n", command);
		}
		else
		{
			argNew = strtok(command, space);
			int histIdx = atoi(argNew + 1);
			char * histCommand = GetFromHistory(histIdx);
			if (histCommand == NULL)
				return -1;
			strcpy(command, histCommand);
			printf("%s\n", command);
		}

	strcpy(buffer, command); // save the original command

	argNew = strtok(command, space);

	do
	{	
		argv[argc] = (char *) malloc(sizeof(char) * strlen(argNew) + 1);
		strcpy(argv[argc], argNew);
		argc++;
	} while ((argNew = strtok(NULL, space)) != NULL);
	return argc;
}

void Execute (int argc, char * argv[])
{
	pid_t pid;	
	int status;
	bool blocking = true;
	if (strcmp(argv[argc - 1], "&") == 0)
	{
		blocking = false;
		free(argv[argc - 1]);
		argv[argc - 1] = NULL;
		argc--;
		
	}

	// execute in new process
	if ((pid = fork()) < 0)
	{
		perror("Failed to create child process");
	}
	else if (pid == 0)
	{		
		if (execvp(*argv, argv) < 0)
		{
			perror("Execution failed");
		}
		exit(1);
	}
	else if (blocking)
	{
		// since we are blocking, override interrupt
		// so we can kill the child process
   		signal(SIGINT, sigHandler);
		while (wait(&status) != pid && !ExitWait);
		if (ExitWait)
		{
			kill(pid, SIGKILL); // kill the child process
			ExitWait = false;
			fflush(stdout);
			printf("\n");
		}
		// resotre signal handler
		signal(SIGINT, SIG_DFL);
	}
	
	// clear arguments
	for (int i = 0; i < argc; i++)
	{
		free(argv[i]);
		argv[i] = NULL;
	}
}

void AddToHistory(char* buffer)
{
	HistoryNode * newNode = (HistoryNode *) malloc(sizeof(HistoryNode));
	newNode->command = (char *) malloc(sizeof(char) * strlen(buffer) + 1);
	strcpy(newNode->command, buffer);
	newNode->next = history.head;
	history.head = newNode;
	history.size++;
}

void PrintHistory()
{
	int index = history.size;
	int i = 0;
	for (HistoryNode * curser = history.head; curser != NULL; curser = curser->next)
	{
		printf("%2i %s\n", index--, curser->command);
		if (++i >= 10)
			return;
	}
}

char* GetFromHistory(int n)
{
	if (n <= 0 || n > history.size)
	{
		printf("No such command in history.\n");
		return NULL;
	}
	int index = history.size;
	for (HistoryNode * curser = history.head; curser != NULL; curser = curser->next)
	{
		if (index-- == n)
			return curser->command;
	}
	return NULL;
}

void sigHandler(int signum)
{
	ExitWait = true;
}




















