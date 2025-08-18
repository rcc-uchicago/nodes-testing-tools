# This streamlit script allows users/testers to submit testing jobs to selected compute nodes
# Contact: ndtrung@uchicago.edu

from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import streamlit as st
from io import StringIO

import subprocess
from streamlit_autorefresh import st_autorefresh

class JobDescriptor:
    def __init__(self, params):
        self.sbatch_header = params['sbatch_header']
        self.job_name = params['job_name']
        self.job_type = params['job_type']
        self.partition = params['partition']
        self.account = params['account']
        self.reservation = params['reservation']
        self.nnodes = params['nnodes']
        self.nodelist = params['nodelist']
        self.exclusive = params['exclusive']
        self.ntasks_per_node = params['ntasks_per_node']
        self.cpus_per_task = params['cpus_per_task']
        self.gpus = params['gpus']
        self.mem = params['mem']
        self.walltime = params['walltime']
        self.constraint_list = params['constraint_list']
        self.execution = params['execution']
        self.work_dir = params['work_dir']
    
    def generate_job_script(self):
        if len(self.constraint_list) == 0:
            return

        job_script_name = "queue.txt"
        for constraint in self.constraint_list:
            
            if self.job_type == "CPU-only jobs":
                job_script_name = f"queue-cpu.txt"
            elif self.job_type == "High-memory CPU-only jobs":
                job_script_name = f"queue-hmem-cpu.txt"    
            elif self.job_type == "GPU jobs":
                job_script_name = f"queue-gpu-{constraint}.txt"
            elif self.job_type == "Custom":
                job_script_name = f"queue-custom.txt"

            with open(job_script_name, "w") as f:
                f.write(f"#!/bin/bash -l\n")
                f.write(f"{self.sbatch_header}")
                f.write(f"\n")
                # This is necessary because the Python environment of the job might be conflict with that of the streamlit app
                f.write(f"module purge\n")
                f.write(f"module load slurm/current rcc/default\n")
                f.write(f"cd {self.work_dir}\n")
                f.write(f"{self.execution}\n")
        return job_script_name

    def submitJob(self):
        job_script_name = self.generate_job_script()

        if self.execution == "":
            st.success('WARNING: Specific the command lines used for bencmark.')
            submitted = False
        else:
            for constraint in self.constraint_list:
                cmd = f"sbatch {job_script_name}"
                p = subprocess.run(cmd, shell=True, text=True, capture_output=True)

            st.success('Your jobs have been submitted!')
            submitted = True

        return submitted

def get_jobs():
    '''
    get the active (pending and running) jobs from the queue
    '''
    user_name = os.environ['USER']
    # job_id state job_name account nodelist   runningtime starttime comment user_name"
    cmd = f"export SQUEUE_FORMAT=\"%13i %.4t %24j %16a %N %M %V %k %u\"; squeue -u {user_name} -h"

    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
  
    # encode output as utf-8 from bytes, remove newline character
    m = out.decode('utf-8').strip()

    i = 0
    jobs = []
    for line in m.splitlines():

        node_info = line.split()

        jobid = node_info[0]
        state = node_info[1]
        job_name = node_info[2]
        
        instance_id = node_info[4]   # nodelist, can be used as public_host_ip
        running_time = node_info[5]
        #start_time = node_info[6]

        jobs.append([job_name,
                      state,
                      jobid,
                      instance_id,
                      running_time])
    return jobs

def extract_app_output_log(application_filter: str, output_str: str):
    info = {}
    info['passed'] = True
    info['notes'] = ""

    # specific for LAMMPS
    if application_filter == "LAMMPS":
        if "Loop" in output_str:
            info['passed']  = True
        lines = output_str.splitlines()
        for line in lines:
            if "tau/day" in line:
                # Performance: 5865.261 tau/day, 13.577 timesteps/s, 750.753 Matom-step/s
                fields = line.split()
                info['notes'] = "LAMMPS perf: " + " " + fields[3] + " timesteps/s "

    # specific for PyTorch NN training
    if application_filter == "NeuralNetwork":
        if "Epoch" in output_str and "Accuracy" in output_str:
            info['passed']  = True
        lines = output_str.splitlines()
        for line in lines:
            # get the last Accuracy line
            if "Accuracy" in line:
                # Accuracy: 70.4%, Avg loss: 0.791656
                fields = line.split()
                info['notes'] = "NN training accuracy: " + fields[1]
    return info           
        

def check_result(testing_output: str, reference=None):
    passed = True
    notes = ""
    if "PASSED" in testing_output and "Done" in testing_output:
        
        info = extract_app_output_log("LAMMPS", testing_output)
        passed = info['passed']
        notes = info['notes']

        info = extract_app_output_log("NeuralNetwork", testing_output)
        passed = passed & info['passed']
        notes = notes + info['notes']
    else:
        passed = False

    test_result = {
        'passed': passed,
        'notes': notes,
    }    
    return test_result

def get_output(work_dir):
    '''
    parse the info from the output files
    '''
    # job_id state job_name account nodelist   runningtime starttime comment user_name"
    cmd = f"ls {work_dir}/output-*.txt"

    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
  
    # encode output as utf-8 from bytes, remove newline character
    m = out.decode('utf-8').strip()
    
    results = []
    for line in m.splitlines():
        # for each output file
        outputfile = line.split()[0]
        with open(outputfile, "r") as f:
            file_content = f.read()

        jobid = ""
        nodelist = ""
        status = "PASSED"
        notes = ""

        # these are the indicators of PASSED in the output file (Custom jobs should generate these flags)
        test_result = check_result(testing_output=file_content)
        test_passed = test_result['passed']
        if test_passed == True:
            status = "PASSED"
        else:
            status = "FAILED"
        notes = test_result['notes']
        for line in file_content.splitlines():
            # for each line in the output file
            out = line.split()

            if "Job ID:" in line:
                jobid = out[2]
            if "Nodes" in line:
                nodelist = out[2]
            if "Job type:" in line:
                job_type = out[2]
        results.append([jobid, nodelist, job_type, status, notes])

    return results


if __name__ == "__main__":
    #nest_asyncio.apply()

    st.set_page_config(layout='wide')

    logo_file = os.path.join(os.path.dirname(__file__), 'logo.png')
    st.image(logo_file,width=500)

    # autorefresh every 30 seconds, maximum 200 times
    count = st_autorefresh(interval=30000, limit=200, key="autorefreshcounter")

    #st.markdown("### 📘 RCC User Guide Chatbot 🤖")
    st.markdown("## 📘 Node testing tool")
    st.markdown("""Created by the RCC team for the RCC team""")

    col1, col2 = st.columns((1,1))

    cmd = "groups"
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    groups = p.stdout.split(' ')
    i = 0
    for grp in enumerate(groups):
        if "rcc-staff" in grp:
            index = i
        i = i + 1
    accounts_avail = tuple(groups)

    execution = ""

    with col1:
        st.markdown("#### Requested resources")
        job_name = st.text_input(r"$\textsf{\large Job name}$", "node-test")
        
        job_type = st.selectbox(r"$\textsf{\large Job type}$", ('CPU-only jobs', 'High-memory CPU-only jobs', 'GPU jobs', 'Data transfers', 'Custom'))

        account = st.selectbox(r"$\textsf{\large Account}$", accounts_avail, help='Accounts accessible',index=index)
        partition = st.selectbox(r"$\textsf{\large Partition}$", ("test", "caslake", "gpu"), help='Partitions used for testing',index=0)
        reservation = st.selectbox(r"$\textsf{\large Reservation}$", ("Test_CPP", "None"), help='Reservation, usually Test_CPP already created by System or CS',index=0)

        nnodes = st.text_input(r"$\textsf{\large Number of nodes}$", "1")
        nodelist = st.text_input(r"$\textsf{\large List of nodes}$", "", help='Specify a list of target nodes in the reservation to test')

        exclusive = True
        exclusive = st.checkbox("Exclusive nodes")
        ntasks_per_node = 32
        cpus_per_task = 1
        if exclusive is False:
            ntasks_per_node = st.text_input(r"$\textsf{\large Number of tasks per node}$", "32")
            cpus_per_task = st.text_input(r"$\textsf{\large Number of CPUs per task}$", "1")
        mem = st.text_input(r"$\textsf{\large Memory per node (GB)}$", "0",help='Memory required, 0 for all the available on the node(s).')
        walltime = st.text_input(r"$\textsf{\large Walltime (HH:MM:SS)}$", "00:30:00")

        gpus = "0"
        gputype = "a100"
        if job_type == "GPU jobs":
            gpus = st.text_input(r"$\textsf{\large Number of GPUs per node}$", "4")
            gputype = st.text_input(r"$\textsf{\large GPU type}$", "a100", help='GPU type: a100, a40, L40S, H100',)

        current_dir = os.environ['PWD']
        work_dir = st.text_input(r"$\textsf{\large Output directory}$", f"{current_dir}",
                                 help='The folder where testing output files are located.',)

        job_script = ""
        sjobtype = ""
        if job_type == "Custom":
            uploaded_file = st.file_uploader(r"$\textsf{\large [Custom] Specify the batch job script}$",
                                         help='The script contains the body of the job script for custom jobs.',
                                         accept_multiple_files=False)
            if uploaded_file is not None:
                job_script = uploaded_file.name
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8")) 
                script_content = stringio.read()
            sjobtype = "custom"
        else:
            if job_type == "CPU-only jobs":
                job_script = "/project/rcc/shared/nodes-testing/node-testing-tool/queue-cpu-nodes.txt"
                sjobtype = "cpu"
            elif job_type == "High-memory CPU-only jobs":
                job_script = "/project/rcc/shared/nodes-testing/node-testing-tool/queue-hmem-cpu-nodes.txt"
                sjobtype = "hmem-cpu"
            elif job_type == "GPU jobs":
                job_script = "/project/rcc/shared/nodes-testing/node-testing-tool/queue-gpu-nodes.txt"
                sjobtype = "gpu"
            
            with open(job_script, "r") as f:
                script_content = f.read()

        if len(job_script) > 0:

            lines = script_content.splitlines()
            for line in lines:
                if ('#SBATCH' not in line and '#!' not in line) and len(line) > 0:
                    execution += line + "\n"

            # To read file as string:
            show_script_content = False
            show_script_content = st.checkbox("Show the job script content")

            res_box = st.empty()
            if show_script_content:
                st.markdown(f"###### Commands in the testing script {job_script} (excluding the #SBATCH lines) for {job_type}")
                res_box.code(execution, language='c++')

        else:
            module = ""
        
        #timing_compute_script = ""
        #timing_compute_file = st.file_uploader(r"$\textsf{\large Specify the post-processing script to compute the job timing}$",
        #                                 help='The script that measure timings of the job (optional).',
        #                                 accept_multiple_files=False)
        #if timing_compute_file is not None:
        #    timing_compute_script = timing_compute_file.name


    with col2:
      
      st.markdown("#### Generated batch job script header")
      job_script_header = st.empty()

      sbatch_header = f"""
#!/bin/bash -l
#SBATCH -J {job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --output=job-{sjobtype}-%J.out
#SBATCH --error=error-{sjobtype}-%J.err
#SBATCH --nodes={nnodes}
#SBATCH --mem={mem}G
#SBATCH --time={walltime}
"""

      if nodelist != "":
         sbatch_header += f"#SBATCH --nodelist={nodelist}"
      if exclusive:
         sbatch_header += f"#SBATCH --exclusive"
      else:
         sbatch_header += f"#SBATCH --ntasks-per-node={ntasks_per_node}\n"
         sbatch_header += f"#SBATCH --cpus-per-task={cpus_per_task}\n"
      if reservation != "None":
         sbatch_header += f"#SBATCH --reservation={reservation}"

      constraint_list = []
      constraint =  gputype
      constraint_list.append(constraint)
      
      if job_type == "GPU jobs":
          if int(gpus) > 0:
              sbatch_header += f"#SBATCH --gres=gpu:{gpus}\n"
              sbatch_header += f"#SBATCH --constraint={gputype}"


      job_script_header.code(sbatch_header, language='c++')

      params = {
        "sbatch_header": sbatch_header,
        "job_name": job_name,
        "job_type": job_type,
        "partition": partition,
        "account": account,
        "reservation": reservation,
        "nnodes": nnodes,
        "nodelist": nodelist,
        "exclusive": exclusive,
        "ntasks_per_node": ntasks_per_node,
        "cpus_per_task": cpus_per_task,
        "gpus": gpus,
        "mem": mem,
        "walltime": walltime,
        "execution": execution,
        "constraint_list": constraint_list,
        "work_dir": work_dir,
      }

      jobdescriptor = JobDescriptor(params)


      if st.button('Submit', type='primary', on_click=jobdescriptor.submitJob):
          st.markdown("#### Job status")
          jobs = st.empty()
          #jobs.write("Your job is pending..")

      st.markdown("#### Active jobs")
      headers=['Name', 'Status', 'JobID', 'Node', 'Elasped time' ]

      # listing all the running jobs
      jobs = get_jobs()
      #nodes = [["jobid", "Running", "jobID", "Node", "Running Time"]]
      df = pd.DataFrame(jobs, columns=headers)
      df.style.hide(axis="index")
      st.table(df)
      
      st.markdown("#### Testing results")
      st.markdown("Parsed from the files the output directory")
      headers=['JobID', 'Nodes', 'Job type', 'Status', 'Notes']
      output = get_output(work_dir)
      df2 = pd.DataFrame(output, columns=headers)
      df2.style.hide(axis="index")
      st.table(df2)
    
