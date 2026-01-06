#!/bin/bash

# extract the output of the LAMMPS run to get the values (TotEng and TPS)
# generate a YAML file to be read in by run-tests.py and
# compared with the expected values in the configuration file

file=$1
n=`grep -n Loop $1 | cut -d ':' -f1`
# get TotEng
TotEng=`awk -v n="$n" '{if (NR==n-1) {print $5}}' $file`
# get timesteps/s (TPS)
TPS=`grep "timesteps/s" $file | awk '{print $4}'`
echo "---"
echo "  output:"
echo "    TotEng:"
echo "      value: $TotEng"
echo "    TPS:"
echo "      value: $TPS"

