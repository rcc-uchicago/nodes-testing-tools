#!/bin/sh
# This script is used to test CPP node(s) by submitting a batch job to the nodes and run selected applications
# Example uses:
#  ./test-nodes.sh midway3-0541
#  ./test-nodes.sh midway3-0426 Test_CPP

nodelist=''
if [ $# -ge 1 ]
then
  nodelist=$1
else
  echo "Need a node name: e.g. midway3-0279"
  exit
fi

reservation="Test_CPP"
if [ $# -ge 2 ]
then
  reservation=$2
fi

reservation_exist=`scontrol show res $reservation | grep ReservationName | sed 's/ReservationName=//g'`
echo "Reservation: $reservation_exist"

if [ -n "$reservation_exist" ]
then
    scontrol show res $reservation > res_info.txt
    nodelist=`grep Nodes res_info.txt | awk '{print $1}'| sed 's/Nodes=//g'`
    num_nodes=`grep NodeCnt res_info.txt | awk '{print $2}'| sed 's/NodeCnt=//g'`
    max_cores=`grep CoreCnt res_info.txt | awk '{print $3}'| sed 's/CoreCnt=//g'`


    scontrol show node $nodelist > node_info.txt
    real_mem=`grep RealMemory node_info.txt | awk '{print $1}'| sed 's/RealMemory=//g'`
    max_ppn=`grep CPUTot node_info.txt | awk '{print $2}'| sed 's/CPUTot=//g'`
    threads_per_node=`grep ThreadsPerCore node_info.txt | awk '{print $2}'| sed 's/ThreadsPerCore=//g'`
    has_gpu=`grep "gpu:" node_info.txt`

    if [ $max_cores -ne $max_ppn];
    then
        echo "Inconsistent core numbers: CoreCnt=$max_cores CPUTot=$max_ppn"
    fi


    gpu_nodes=0
    output="output-cpu-$nodelist.txt"
    queue_template="queue-cpu-nodes.txt"
    queue_file="queue-cpu-$nodelist.txt"
    gres=""
    if [ -n "$has_gpu" ]; then
        gpu_nodes=1
        gres=`grep "Gres=" node_info.txt | awk '{print $1}'| sed 's/Gres=//g'`
        output="output-gpu-$nodelist.txt"
        queue_template="queue-gpu-nodes.txt"
        queue_file="queue-gpu-$nodelist.txt"

        echo "Node has $max_ppn CPU cores; $real_mem memory and $gres GPUs"
    else
        echo "Node has $max_ppn CPU cores; $real_mem memory"
    fi

    echo "Testing $nodelist .."

    if [ -e $output ]
    then
        echo "Test output $output exists"
        read -p "Rerun the test? (y/n): " rerun
        if [ $rerun == 'y' ]
        then
            rm -rf $output
        fi
    fi


    cp $queue_template.txt $queue_file
    sed -i "s/--reservation=Test_CPP/--reservation=$reservation/g" $queue_file

    echo "Submitting job script $queue_file.."
    sbatch --nodelist=$nodelist --ntasks-per-node=$max_cores --output=$output $queue_file

    echo "Checking job output log $output for details"

fi
