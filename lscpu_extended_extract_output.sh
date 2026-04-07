#!/bin/bash

# extract the output of the lscpu command to get the values (Cores and Thread(s) per core)
# generate a YAML file to be read in by run-tests.py and
# compared with the expected values in the configuration file

file=$1
# check if hyperthreading is enabled
ht=`awk 'BEGIN{ht = 0;} {if ($1 != $4) ht = 1;} END{ printf("%d\n", ht); }' $file`

echo "---"
echo "  output:"
echo "    Hyperthreading:"
echo "      value: $ht"


