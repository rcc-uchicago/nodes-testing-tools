#!/bin/bash

# This script is to extract certain values from the output of the mpgadget run and
# to generate a YAML file to be read in by run-tests.py which is then 
# compared against the expected values in the configuration file.
# Values are extracted from three files.


# get values from the file that corresponds to output to the terminal
filetmp=$1
n=$(grep -n ", Time: 1 (1c00000000000)" "$filetmp")
Redshift=$(echo "$n" | awk '{print substr($11, 1, length($11)-1)}')
Nf=$(echo "$n" | awk '{print substr($14, 1, length($14)-1)}')
Nf=$((10#$Nf)) 

# get timestep/s (TPS) from the cpu.txt file
filecpu="tmp/mp-gadget/tests/dm-only/output/cpu.txt"
TPS=$(grep Step "$filecpu" | tail -n1 | sed 's/,//g' |  awk '{print(($2/$10)) }')

# get values from the power-spectrum file
filepw="tmp/mp-gadget/tests/dm-only/output/powerspectrum-1.0000.txt"
Powspec=$(awk 'NR==4{print}' "$filepw")
Powspec1=$(echo "$Powspec" | awk '{print$1}')
Powspec2=$(echo "$Powspec" | awk '{print$2}')
Powspec3=$(echo "$Powspec" | awk '{print$3}')
Powspec4=$(echo "$Powspec" | awk '{print$4}')

echo "---"
echo "  output:"
echo "    PowerSpectra_1k:"
echo "      value: $Powspec1"
echo "    PowerSpectra_1P:"
echo "      value: $Powspec2"
echo "    PowerSpectra_1N:"
echo "      value: $Powspec3"
echo "    PowerSpectra_1P(z=0):"
echo "      value: $Powspec4"
echo "    RedShift:"
echo "      value: $Redshift"
echo "    Nf:"
echo "      value: $Nf"
echo "    TPS:"
echo "      value: $TPS"
