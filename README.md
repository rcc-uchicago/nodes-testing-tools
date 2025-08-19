This project contains tools to test compute nodes before delivery and after maintenance.

Given a set of nodes in a partition, there are typical tests that need to be done

  * the CPUs in a node are functional (giving accurate results, performing as expected)
  * the memory in a node is accessible and functional (allocable memory as advertised)
  * the communication between the nodes (e.g. through MPI runs)
  * the GPUs are functional (as advertised)
  * the data transfers between the node(s) and storage spaces are at the expected rates


The tools serve the following 2 use cases:

  * Manual testing: the CS/System team members specify the node(s) to test.
    + the GUI app: `nodes-testing.py`
    + the CLI tools: `test-cpu-nodes.sh` and `test-gpu-nodes.sh`
  * Automated testing: the tool run in the background in the user space and automatically submits testing jobs
    + the shell script: `requesting_tests.sh`

The GUI tool `nodes-testing.py` is a Python script that runs through `streamlit` that
allows users to select the job type and an associated job script to submit the nodes.
The job script will run selected applications and print to an output log file. 

The underlying job scripts `queue-cpu-nodes.txt` and `queue-gpu-nodes.txt` should be responsible
to the tell if the applications and shell commands therein complete successfully and/or reports
the performance and accuracy of the run.

The shell script `requesting_tests.sh` is used for automated testing, which fills in the details
in the job scripts.

To-Do:

  * need to define a reference file for each test case: for example, with LAMMPS runs,
    the reference file lists the expected performance and numerical result of the last
    time step and the accepted tolerance.
  * more testing


=== Installation ====

```
git clone https://github.com/rcc-uchicago/nodes-testing-tools.git
cd nodes-testing-tools
```

On Midway3, you can run the GUI tool with
```
module load python/miniforge-25.3.0
source /project/rcc/shared/nodes-testing/testing-env/bin/activate
streamlit run nodes-testing.py
```

For development, you can create your own enviroment with

```
module load python/miniforge-25.3.0
python3 -m venv my-env
source my-env/bin/activate
pip install -r requirements.txt
```

Note: The enviroment contains torch and nvidia* in case you need to run a ML training script with GPU.
