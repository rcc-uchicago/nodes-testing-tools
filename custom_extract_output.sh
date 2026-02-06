#!/bin/bash

# extract the output of the python test_numpy.py command to get the values
# generate a YAML file to be read in by run-tests.py and
# compared with the expected values in the configuration file

file=$1

# get e-value1
evalue1=`grep -1 "Eigenvalues" $file | tail -n1 | awk '(NF == 2) {print $1}'`
# get e-vector1
evector1=`grep -1 "Eigenvectors" $file | tail -n1 | sed 's/]//g' | sed 's/\[//g'`
evector11=`echo $evector1 | awk '{print $1}'`
evector12=`echo $evector1 | awk '{print $2}'`
echo "---"
echo "  output:"
echo "    Eigenvalue1:"
echo "      value: $evalue1"
echo "    Eigenvector11:"
echo "      value: $evector11"
echo "    Eigenvector12:"
echo "      value: $evector12"

