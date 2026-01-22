#!/bin/bash

# extract the output of the lscpu command to get the values (Cores and Thread(s) per core)
# generate a YAML file to be read in by run-tests.py and
# compared with the expected values in the configuration file

file=$1

# get CPU(s)
cores=`grep "CPU(s):" $file | awk '(NF == 2) {print $2}'`
# get Threads per core
threadspercore=`grep "Thread(s) per core" $file | awk '{print $4}'`
echo "---"
echo "  output:"
echo "    Cores:"
echo "      value: $cores"
echo "    ThreadsPerCore:"
echo "      value: $threadspercore"

