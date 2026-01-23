#!/bin/bash

# extract the output of the lscpu command to get the values (Cores and Thread(s) per core)
# generate a YAML file to be read in by run-tests.py and
# compared with the expected values in the configuration file

file=$1

# get GFLOPs
gflops=`grep "Benchmark Time Summary::Total" $file | cut -d'=' -f2`
# get SpMV GFLOPs
SpMV=`grep "GFLOP/s Summary::Raw SpMV" $file | | cut -d'=' -f2`
# get MG
MG=`grep "GFLOP/s Summary::Raw MG" $file | | cut -d'=' -f2`
# get Memory Bandwidth
MG=`grep "GB/s Summary::Raw Total B/W" $file | | cut -d'=' -f2`

# get Memory Bandwidth
MG=`grep "GB/s Summary::Raw Total B/W" $file | | cut -d'=' -f2`

# MPI communication (MPI_Allreduce) imbalance is captured by Min/Max/Avg timing of DDOT.
# This depends on the placement of MPI procs on the NUMA nodes and or inter-node network.

MinAllreduce=`grep "Min DDOT MPI_Allreduce time" $file | | cut -d'=' -f2`
MaxAllreduce=`grep "Max DDOT MPI_Allreduce time" $file | | cut -d'=' -f2`
AvgAllreduce=`grep "Avg DDOT MPI_Allreduce time" $file | | cut -d'=' -f2`
imbalance=`echo "scale=4; ($MaxAllreduce - $MinAllreduce)/$AvgAllreduce" | bc`
 
echo "---"
echo "  output:"
echo "    GFLOPs:"
echo "      value: $gflops"
echo "    SpMV:"
echo "      value: $SpMV"
echo "    MG:"
echo "      value: $MG"
echo "    TotalBW:"
echo "      value: $TotalBW"
echo "    MPIAllreduceImbalance:"
echo "      value: $imbalance"


