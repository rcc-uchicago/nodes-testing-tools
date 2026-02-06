This project contains tools to test compute nodes before delivery of CPP nodes and new clusters and after maintenance. This is important for both CS and System teams to ensure the continuously high quality of service.

Given a set of nodes in a partition, there are typical tests that need to be done

  * the CPUs in a node are functional (giving accurate results, performing as expected)
  * the memory in a node is accessible and functional (allocable memory as advertised)
  * the communication between the nodes (e.g. through MPI runs)
  * the GPUs are functional (as advertised)
  * the data transfers between the node(s) and storage spaces are at the expected rates

The tools should serve the following 2 primary use cases:

  * Manual testing: the CS/System team members specify the node(s) to test.
    + the GUI app: `nodes-testing.py`
    + the CLI tools: `test-cpu-nodes.sh` and `test-gpu-nodes.sh`
  * Automated testing: the tool run in the background in the user space and automatically submits testing jobs
    + the shell script: `requesting_tests.sh`

The general idea for these testing tools is to submit jobs from the login node to the compute nodes of interest, within each job, run the selected applications (from the [RCC Benchmarking Suite](https://github.com/rcc-uchicago/RCC-benchmark-suite) repo) and compare the output with the expected values. If the run completed successfully and the differences are within the specified tolerances, then the nodes pass the tests.

The GUI tool `nodes-testing.py` is a Python script that runs through `streamlit` that
allows users to select the job type and an associated job script to submit the nodes.
The job script will run selected applications and print to an output log file. 

The underlying job scripts `queue-cpu-nodes.txt` and `queue-gpu-nodes.txt` should be responsible
to the tell if the applications and shell commands therein complete successfully and/or reports
the performance and accuracy of the run.

The shell script `requesting_tests.sh` is used for automated testing, which fills in the details
in the job scripts.

To-Do:

  - need to define a reference file for each test case: for example, with LAMMPS runs,
    the reference file lists the expected performance and numerical result of the last
    time step and the accepted tolerance (in progress, see `run-tests.py` and `lammps.yaml`).
  - more testing


## Usage

```
git clone https://github.com/rcc-uchicago/nodes-testing-tools.git
cd nodes-testing-tools
```

### GUI app

On Midway3, you can run the GUI app `nodes-testing.py` with
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

Note: The enviroment contains torch and nvidia in case you need to run a ML training script with GPU.

### Command-line tools

Currently the two shell scripts `test-cpu-nodes.sh` and `test-gpu-nodes.sh` submit jobs to the compute nodes of interest.
The GUI app similarly submits the job scripts `queue-cpu-nodes.sh` and `queue-gpu-nodes.sh` to the nodes, within which
the output from the runs is analyzed to judge if the runs produce expected output.

To eventually consolidate these approaches, we develop the Python script `run-tests.py` which executes a testing pipeline
defined by a configuration file:

```
python run-tests.py --config-file lammps.yaml
```

The configuration file `lammps.yaml` specifies the application executable to run and the expected output (e.g. numerical accuracy and performance and tolerances).
Similar configuration files can be created for other applications such as HPCG and HPCC.

The idea is to use this Python script inside the existing CLI tools and job scripts.

You can test a custom application by creating a `custom-app.yaml` file that describes the application and the expected output.

```
python3 run-tests.py --config-file custom-app.yaml 
2026-02-06 15:28:29,567 - INFO - Using the configuration file:
  /project/rcc/users/trung/nodes-testing-tools/custom-app.yaml
2026-02-06 15:28:29,568 - INFO - Execute:
2026-02-06 15:28:29,568 - INFO -   module load python/miniforge-25.3.0 && python3 test_numpy.py
2026-02-06 15:28:30,584 - INFO - Eigenvalue1: Actual = 5.0 Expected = 5.0 absdiff = 0.00000 abstol = 0.001
2026-02-06 15:28:30,584 - INFO - Eigenvector11: Actual = 0.89442719 Expected = 0.89442719 absdiff = 0.00000 abstol = 0.0001
2026-02-06 15:28:30,585 - INFO - Eigenvector12: Actual = -0.70710678 Expected = -0.70710678 absdiff = 0.00000 abstol = 0.0001
2026-02-06 15:28:30,585 - INFO - PASSED
```

### How to contribute

After cloning the repo, create a new branch, made changes to your branch, commit the changes to your branch and push the branch to the repo

```
git checkout -b my-contributions
# made changes to existing files, added new files
# ...
git add new_file
git commit -m "description of the changes in this commit"
git push --set-upstream origin my-contributions
```

Then go to the GitHub repo, create a pull request (PR) to merge the branch `my-contributions` to the `main` branch. Assign the reviewers to the PR.

You can keep working on the branch in your local copy (e.g. to address the suggestions from the reviewers), committing the changes to the branch.

When the PR is approved, the admins will merge the PR to the `main` branch.



