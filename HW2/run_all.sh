#!/bin/bash

rm -f device-file

make clean

make

./make_device_file.sh

./run_worker.sh 10

./run_checker.sh 
