#!/bin/sh

NODENAMES=''
if [ $# -ge 1 ]
then
  NODENAMES=$1
else
  echo "Need a node name: e.g. midway3-0279"
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

output="output-gpu-$NODENAMES.txt"

if [ -n "${NODENAMES}" ]
then
  echo "Testing $NODENAMES .."

  if [ -e $output ]
  then
    echo "Test output output-gpu-$NODENAMES.txt exists"
    read -p "Rerun the test? (y/n): " rerun
    if [ $rerun == 'y' ]
    then
      rm -rf output-gpu-$NODENAMES.txt
    fi
  fi

  cp queue-gpu-nodes.txt queue-gpu-$NODENAMES.txt
  sed -i "s/--reservation=Test_CPP/--reservation=$RESERVATION/g" queue-gpu-$NODENAMES.txt
  echo "Submitting job script queue-gpu-$NODENAMES.txt.."
  sbatch --nodelist=$NODENAMES --ntasks-per-node=$max_cores --output=output-gpu-$NODENAMES.txt queue-gpu-$NODENAMES.txt

else
  echo "No node names given: $NODENAMES"
  echo "Submitting to queue-cpu-nodes.txt as is"
  sbatch queue-gpu-nodes.txt
fi

echo "Waiting for $output"
echo "Checking job output log for details"


