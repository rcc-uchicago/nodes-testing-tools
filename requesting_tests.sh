#!/bin/bash

# Suppose that a reservation has been created for the test via
#   scontrol create reservation=Test_CPP starttime=Now duration=06:00:00 users=$USER flags=ignore_jobs partitionname=gpu Nodes=midway3-0278
#   scontrol create reservation=gpu_test starttime=Now duration=06:00:00 users=$USER flags=ignore_jobs partitionname=gpu Nodes=midway3-0278

# Run this script with nohup and disown
#   nohup ./requesting_tests.sh > out.log 2>&1 & disown
#
# check:     ps aux | grep requesting_tests.sh
# terminate: kill -9 [pid]

# reservation=Test_CPP
reservation="gpu_test"
if [ $# -ge 1 ]
then
  reservation=$1
fi

cwd=$PWD
pid=`ps aux | grep requesting_tests.sh | grep bash | awk '{print $2}'`

echo "Process ID: $pid"
echo "Check:      ps aux | grep $0"
echo "Terminate:  kill -9 $pid"

# enforce rerunning the test
enforce=0

delay=60

while true; do

    timestamp=`date`
    reservation_exist=`scontrol show res $reservation | grep ReservationName | sed 's/ReservationName=//g'`
    echo "Reservation: $reservation_exist"
    if [ -n "$reservation_exist" ]; then
 
        echo $timestamp
        scontrol show res $reservation > res_info.txt
        nodelist=`grep Nodes res_info.txt | awk '{print $1}'| sed 's/Nodes=//g'`
        num_nodes=`grep NodeCnt res_info.txt | awk '{print $2}'| sed 's/NodeCnt=//g'`
        max_cores=`grep CoreCnt res_info.txt | awk '{print $3}'| sed 's/CoreCnt=//g'`

        scontrol show node $nodelist > node_info.txt
        real_mem=`grep RealMemory node_info.txt | awk '{print $1}'| sed 's/RealMemory=//g'`
        max_ppn=`grep CPUTot node_info.txt | awk '{print $2}'| sed 's/CPUTot=//g'`
        threads_per_node=`grep ThreadsPerCore node_info.txt | awk '{print $2}'| sed 's/ThreadsPerCore=//g'`
        has_gpu=`grep "gpu:" node_info.txt`
        gpu_nodes=0
        gres=""
        if [ -n "$has_gpu" ]; then
            gpu_nodes=1
            gres=`grep "Gres=" node_info.txt | awk '{print $1}'| sed 's/Gres=//g'`
        fi

        output="output-$nodelist.txt"
        running_test=0
        # if the output test file exists
        if [ -f $output ]
        then
            # if the output file contains "PASSED" 
            echo "$output exists"
            passed=`grep PASSED $output`
            if [ -n "$passed" ]
            then
                running_test=0
            else
                echo "$output exists but does not contain PASSED. Need to rerun."
                running_test=1
            fi
        else
            echo "$output does not exist yet."
            running_test=1
        fi

        # if the job is actually running but not finished yet
        if [ -f $output ]
        then
            finished=`grep "Finished ALL tests" $output`
            if [ -n "$finished" ]
            then
                 running_test=0
            fi
        fi

        if [ $enforce -eq 1 ]
        then
            running_test=1
        fi


        # Submitting job to compute nodes

        if [ $running_test -eq 1 ]
        then

            echo "Testing $nodelist .."

            job_script="queue-$nodelist.txt"
            if [ $gpu_nodes -eq 1 ]
            then
                cp queue-gpu-nodes.txt $job_script
                echo "GPU: $gres"
                sed -i "s/--gres=gpu:4/--gres=$gres/g" $job_script
            else
                cp queue-cpu-nodes.txt $job_script
            fi

            # modify the job script
            sed -i "s/--reservation=Test_CPP/--reservation=$reservation/g" $job_script

            echo "Submitting job script $job_script.."

            #sbatch --nodelist=$nodelist --ntasks-per-node=$max_ppn $job_script $output
            sbatch $job_script $output
            echo "Check output $output for testing results"

        fi

    fi

    sleep $delay  # Check every 300 seconds
done

