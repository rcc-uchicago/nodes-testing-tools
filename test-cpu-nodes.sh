#!/bin/sh

NODENAMES=''
if [ $# -ge 1 ]
then
  NODENAMES=$1
else
  echo "Need to specify a list of node names, e.g. midway3-[0383-0384], or midway3-0128,midway3-0231"
  exit
fi

RESERVATION="Test_CPP"
if [ $# -ge 2 ]
then
  RESERVATION=$2
fi

# get the number of CPU cores of the node
max_cores=`scontrol show node $NODENAMES | grep CPUTot | awk '{print $2}' | head -n1 | cut -d'=' -f2`
echo "Node has $max_cores CPU cores"

output="output-cpu-$NODENAMES.txt"

if [ -n "${NODENAMES}" ]
then
  echo "Testing $NODENAMES .."

  if [ -e $output ]
  then
    echo "Test output output-cpu-$NODENAMES.txt exists"
    read -p "Rerun the test? (y/n): " rerun
    if [ $rerun == 'y' ]
    then
      rm -rf output-cpu-$NODENAMES.txt
    fi
  fi

  cp queue-cpu-nodes.txt queue-cpu-$NODENAMES.txt
  sed -i "s/--reservation=Test_CPP/--reservation=$RESERVATION/g" queue-cpu-$NODENAMES.txt
  echo "Submitting job script queue-cpu-$NODENAMES.txt.."
  sbatch --nodelist=$NODENAMES --ntasks-per-node=$max_cores --output=output-cpu-$NODENAMES.txt queue-cpu-$NODENAMES.txt

else
  echo "No node names given: $NODENAMES"
  echo "Submitting to queue-cpu-nodes.txt as is"
  sbatch queue-cpu-nodes.txt
fi

echo "Waiting for $output"
echo "Checking job output log for details"


