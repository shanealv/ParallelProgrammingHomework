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


int GetCommand (char buffer[], char* argv[]);
void Execute (int argc, char * argv[]);
void AddToHistory(char* buffer);
int HistoryLength();
void PrintHistory();
char* GetFromHistory(int n);
void sigHandler(int signum);

char* history[H_COUNT];
int historyIdx = 0;
bool ExitWait = false;

int main(void)
{
	int argc = 0;
	char* argv[MAX_LINE/2 + 1];	/* command line (of 80) has max of 40 arguments */
	char buffer[MAX_LINE];
    	int should_run = 1;

	for (int i = 0; i < H_COUNT; i++)
		history[i] = NULL;
	
	for (int i = 0; i < MAX_LINE/2 + 1; i++)
		argv[i] = NULL;
		
	do {   
		printf("osh>");
		fflush(stdout);

		if ((argc = GetCommand(buffer, argv)) < 0)
			continue;

		if (strcmp(*argv, "exit") == 0)
			return 0;
		
		if (strcmp(*argv, "!!") == 0)
		{
			PrintHistory();
			continue;
		}

		AddToHistory(buffer);
		
		Execute(argc, argv);
	} while (should_run);
    
	return 0;
}

int GetCommand (char buffer[], char* argv[])
{
	char command[MAX_LINE];
	const char space[2] = " ";
	char * argNew;
	int argc = 0;

	fgets(command, MAX_LINE, stdin);

	int lastIdx = strlen(command) - 1;
	
	if (lastIdx == 0)
		return -1;
	else if (command[lastIdx] == '\n')
		command[lastIdx] = 0; // remove newline

	// if a history indexer was used, replace command with it
	if (lastIdx >= 2 && command[0] == '!' && command[1] != '!')
	{
		argNew = strtok(command, space);
		int histIdx = atoi(argNew + 1);
		char * histCommand = GetFromHistory(histIdx);
		if (histCommand == NULL)
			return -1;
		strcpy(command, histCommand);
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
		perror("Failed to create child process\n");
	}
	else if (pid == 0)
	{		
		if (execvp(*argv, argv) < 0)
		{
			perror("Error: execution failed\n");
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
	free(history[historyIdx]);
	history[historyIdx] = (char *) malloc(sizeof(char) * strlen(buffer) + 1);
	strcpy(history[historyIdx], buffer);
	historyIdx = (historyIdx + 1) % H_COUNT;
}

int HistoryLength()
{
	int total = 0;
	for (int i = 0; i < H_COUNT; i++)
	{
		if (history[i] != NULL)
			total++;
	}
	return total;
}

void PrintHistory()
{
	int id = HistoryLength();
	for (int i = historyIdx; i < historyIdx + H_COUNT; i++)
	{
		if (history[i % H_COUNT] == NULL) continue;
		printf("%2i %s\n", id--, history[i % H_COUNT]);
	}
}

char* GetFromHistory(int n)
{
	if (n <= 0 || n > HistoryLength())
	{
		printf("No such command in history.\n");
		return NULL;
	}
	char * command = history[(historyIdx - n) % H_COUNT];
	printf("%s\n", command);
	return command;
}

void sigHandler(int signum)
{
	ExitWait = true;
}




















