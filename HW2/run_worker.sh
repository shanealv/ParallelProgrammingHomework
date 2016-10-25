#!/bin/bash

if [ $# == 0 ]; then
	echo "Usage: ./run_worker.sh [timeout]"
	exit 
fi

timeout $1 ./worker.out device-file 50
