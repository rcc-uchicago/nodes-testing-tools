# Contact: ndtrung@uchicago.edu
# usage: on a compute node, activate the env, then run the script with a YAML file
#
#   module load python/miniforge-25.3.0
#   source /project/rcc/shared/nodes-testing/testing-env/bin/activate
#   
#   python3 node-tester-wizard.py

import sys
import subprocess
from PySide6.QtCore import QProcess, QTimer
from PySide6.QtWidgets import (
    QApplication, QWizard, QWizardPage,
    QLabel, QVBoxLayout, QHBoxLayout, QComboBox,
    QLineEdit, QMessageBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton
)

# ---------- Page 1: Welcome ----------
class WelcomePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Welcome")

        label = QLabel(
            "Welcome to the RCC Node Tester Wizard!\n"
            "Click Next to continue."
        )
        label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)


# ---------- Page 2: Node Test Selection ----------
class NodeTestPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Node Test Selection")

        # IMPORTANT: this is the last page
        #self.setFinalPage(True)

        self.test_combo = QComboBox()
        self.test_combo.addItems(["CPU-only Nodes", "GPU Nodes", "Data Transfers", "I/O Bandwidth"])

        self.project_name = QLineEdit()
        self.project_name.setPlaceholderText("test-cpp-ticket-1234")

        self.nodelist = QLineEdit()
        self.nodelist.setPlaceholderText("midway3-0001")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Project:"))
        layout.addWidget(self.project_name)
        layout.addWidget(QLabel("Node List:"))
        layout.addWidget(self.nodelist)
        layout.addWidget(QLabel("Select a Test Pipeline:"))
        layout.addWidget(self.test_combo)
        self.setLayout(layout)

        # Notify wizard when inputs change
        self.test_combo.currentIndexChanged.connect(self.completeChanged)
        self.project_name.textChanged.connect(self.completeChanged)
        self.nodelist.textChanged.connect(self.completeChanged)

        # Optional UX improvement
        #self.vendor_combo.currentIndexChanged.connect(self.update_ui)
        #self.update_ui()

        # Register fields
        self.registerField("test*", self.test_combo, "currentText")
        self.registerField("project_name", self.project_name)
        self.registerField("nodelist", self.nodelist)

    #def update_ui(self):
    #    is_cpu_only = self.test_combo.currentText() == "CPU-only Nodes"
    #    self.project_name.setEnabled(is_cpu_only)

    # Controls Next button enabled
    def isComplete(self):
        if self.project_name.text() != "":
            return bool(self.project_name.text().strip())
        return True
    
    # Validation when Finish is clicked
    def validatePage(self):
        if self.test_combo.currentText() == "GPU nodes":
            if not self.project_name.text().strip():
                QMessageBox.warning(
                    self,
                    "Input Required",
                    "GPUs need to be requested."
                )
                return False
        pname = self.project_name.text().strip()
        if not pname:
            placeholderText = self.project_name.placeholderText()
            self.project_name.setText(placeholderText)

        nlist = self.nodelist.text().strip()
        if not nlist:
            placeholderText = self.nodelist.placeholderText()
            self.nodelist.setText(placeholderText)

        return True

# ---------- Page 3: Node Test Configration ----------

class PipelineConfigurationPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Test Pipeline Configuration")

        # IMPORTANT: this is the last page
        #self.setFinalPage(True)

        self.run_lscpu = QCheckBox("lscpu (lscpu.yaml)")

        self.run_hpcc = QCheckBox("HPCC")
        self.hpcc_config = QLineEdit()
        self.hpcc_config.setPlaceholderText("hpcc-n8.yaml")

        self.run_hpcg = QCheckBox("HPCG")
        self.hpcg_config = QLineEdit()
        self.hpcg_config.setPlaceholderText("hpcg.yaml")
        
        self.run_lammps = QCheckBox("LAMMPS")
        self.lammps_config = QLineEdit()
        self.lammps_config.setPlaceholderText("lammps.yaml")

        self.run_mpgadget = QCheckBox("MP-Gadget (mpgadget.yaml)")
        self.run_nvidiasmi = QCheckBox("nvidia-smi (nvidiasmi.yaml)")

        self.job_script = QLineEdit()
        self.job_script.setPlaceholderText("job_script.txt")

        self.submit_btn = QPushButton("Submit the generated job script")
        self.submit_btn.clicked.connect(self.submit_jobs)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select the application(s) to run and specify the configuration files:"))
        layout.addWidget(self.run_lscpu)

        row1 = QHBoxLayout()
        row1.addWidget(self.run_hpcc)
        row1.addWidget(self.hpcc_config)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(self.run_hpcg)
        row2.addWidget(self.hpcg_config)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(self.run_lammps)
        row3.addWidget(self.lammps_config)
        layout.addLayout(row3)

        layout.addWidget(self.run_mpgadget)
        layout.addWidget(self.run_nvidiasmi)
        layout.addWidget(QLabel("Job script:"))
        layout.addWidget(self.job_script)

        layout.addSpacing(20)
        layout.addWidget(self.submit_btn)

        layout.addWidget(QLabel(f"Clicking Next will also generate the job script."))
        self.setLayout(layout)

        # Register fields
        self.registerField("run_lscpu", self.run_lscpu)
        self.registerField("run_hpcc", self.run_hpcc)
        self.registerField("hpcc_config", self.hpcc_config)
        self.registerField("run_hpcg", self.run_hpcg)
        self.registerField("hpcg_config", self.hpcg_config)
        self.registerField("run_lammps", self.run_lammps)
        self.registerField("lammps_config", self.lammps_config)
        self.registerField("run_mpgadget", self.run_mpgadget)
        self.registerField("run_nvidiasmi", self.run_nvidiasmi)
        self.registerField("job_script", self.job_script)

    def initializePage(self):        
        test = self.field("test")
        is_running_nvidiasmi = (test == "GPU Nodes")
        self.run_nvidiasmi.setVisible(is_running_nvidiasmi)

        if not is_running_nvidiasmi:
            self.run_nvidiasmi.setChecked(False)

    def submit_jobs(self):
        filename = self.field("job_script")
        self.generate_job_script(filename)

        self.proc = QProcess(self)
        self.proc.finished.connect(self.on_finished)
        self.proc.start("sbatch", [filename])

    def on_finished(self, exit_code, exit_status):
        print("Submitting job script finished")
        print(f"Exit code: {exit_code}")
        print(f"Exit status: {exit_status}")

    def validatePage(self):

        # retrieve the config file names, using the placeholder text if none is explicitly given
        config = self.hpcc_config.text().strip()
        if not config:
            placeholderText = self.hpcc_config.placeholderText()
            self.hpcc_config.setText(placeholderText)

        config = self.hpcg_config.text().strip()
        if not config:
            placeholderText = self.hpcg_config.placeholderText()
            self.hpcg_config.setText(placeholderText)

        config = self.lammps_config.text().strip()
        if not config:
            placeholderText = self.lammps_config.placeholderText()
            self.lammps_config.setText(placeholderText)         

        script_name = self.job_script.text().strip()
        if not script_name:
            placeholderText = self.job_script.placeholderText()
            self.job_script.setText(placeholderText)         

        # generate the job script for submission
        filename = self.job_script.text().strip()
        self.generate_job_script(filename)

        # invoke the wizard serialize() to store the settings and options being made at this point
        self.wizard().serialize()

        return True

    def generate_job_script(self, filename):
        # generate a job script file ready for submission but don't submit the job

        run_lscpu = self.field("run_lscpu")
        run_hpcc = self.field("run_hpcc")
        hpcc_config = self.field("hpcc_config")
        run_hpcg = self.field("run_hpcg")
        hpcg_config = self.field("hpcg_config")
        run_lammps = self.field("run_lammps")
        lammps_config = self.field("lammps_config")
        run_mpgadget = self.field("run_mpgadget")
        nodelist = self.field("nodelist")

        content  = f"#!/bin/bash\n"
        content += f"#SBATCH --account=rcc-staff\n"
        content += f"#SBATCH --partition=test\n"
        content += f"#SBATCH --nodelist={nodelist}\n"
        content += f"#SBATCH --reservation=TestCPP\n"
        content += f"#SBATCH --mem=0\n"
        content += f"#SBATCH --time=00:30:00\n"
        content += f"#SBATCH --exclusive\n"
        content += f"nodelist=$SLURM_NODELIST\n"
        content += f"OUTPUT=$1\n"
        content += f"echo \"Job ID: \$SLURM_JOB_ID\" > $OUTPUT\n"
        content += f"echo \"Nodes = \$nodelist\" >> $OUTPUT\n"
        content += f"echo \"Job type: CPU-only\" >> $OUTPUT\n"
        content += f"echo \"Date: `date`\" >> $OUTPUT\n"
        content += f"cd $SLURM_SUBMIT_DIR\n"
        content += f"CWD=`pwd`\n"
        content += "max_ppn=`scontrol show node $nodelist | grep CPUTot | awk '{print $2}'| sed 's/CPUTot=//g'`\n"

        content += "nodes=$SLURM_NNODES\n"
        content += "if [ -n \"$SLURM_NTASKS_PER_NODE\" ]\n"
        content += "then\n"
        content += "  ppn=$SLURM_NTASKS_PER_NODE\n"
        content += "else\n"
        content += "   ppn=$max_ppn\n"
        content += "fi\n"
        content += "n=$(( ppn * nodes ))\n"
        content += "ulimit -l unlimited\n"
        content += "ulimit -s unlimited\n"
        content += "module load load python/miniforge-25.3.0\n"
        content += "source /project/rcc/shared/nodes-testing/testing-env/bin/activate\n"
        if run_lscpu:
            content += "python3 run-tests.py --config-file lscpu.yaml\n"
        if run_hpcc:
            content += f"python3 run-tests.py --config-file {hpcc_config}\n"
        if run_hpcg:
            content += f"python3 run-tests.py --config-file {hpcg_config}\n"
        if run_lammps:
            content += f"python3 run-tests.py --config-file {lammps_config}\n"
        if run_mpgadget:
            content += "python3 run-tests.py --config-file mp-gadget.yaml\n"    

        with open(filename, "w") as f:
            f.write(content)


# ---------- Page 4: Submit job and waiting for results ----------

class JobMonitorPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Test Monitor")

        # IMPORTANT: this is the last page
        self.setFinalPage(True)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Project", "Job ID", "Job script", "Status", "Test Type"]
        )

        # UI polish
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Ongoing jobs:"))
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_job_list)
        self.timer.start(5000)


    def update_job_list(self):
        test = self.field("test")
        project_name = self.field("project_name")
        cmd_str = "squeue -u $USER -h --format \"%i %j %t \" "
        p = subprocess.run(cmd_str, shell=True, text=True, capture_output=True, timeout=60)
        status = { 
            'cmd_str': cmd_str,
            'stdout': p.stdout,
            'stderr': p.stderr,
            'returncode': p.returncode,
        }
        jobs = p.stdout.strip().splitlines()
        self.table.setRowCount(0)
        row = 0
        for job in jobs:
            self.table.insertRow(row)
            jobid, job_script, status = job.split()
            if status == "PD":
                status = "Pending"
            elif status == "R":
                status = "Running"

            self.table.setItem(row, 0, QTableWidgetItem(project_name))                
            self.table.setItem(row, 1, QTableWidgetItem(jobid))
            self.table.setItem(row, 2, QTableWidgetItem(job_script))
            self.table.setItem(row, 3, QTableWidgetItem(status))
            self.table.setItem(row, 4, QTableWidgetItem(test))

            row = row + 1


# ---------- Wizard ----------
class NodeTesterWizard(QWizard):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Node Tester Wizard")
        self.setWizardStyle(QWizard.ModernStyle)

        self.addPage(WelcomePage())
        self.addPage(NodeTestPage())
        self.addPage(PipelineConfigurationPage())
        self.addPage(JobMonitorPage())

    def serialize(self):
        test = self.field("test")
        project_name = self.field("project_name")
        run_lscpu = self.field("run_lscpu")
        run_hpcg = self.field("run_hpcg")
        run_lammps = self.field("run_lammps")
        run_mpgadget = self.field("run_mpgadget")
        run_nvidiasmi = self.field("run_nvidiasmi")
        job_script = self.field("job_script")

        outputfile = project_name + ".txt"
        with open(outputfile, "w") as f:
            f.write(f"test: \"{test}\"\n")
            f.write(f"project_name: {project_name}\n")
            f.write(f"applications:\n")
            if run_lscpu:
                f.write(f"  lspcu\n")
            if run_hpcg:
                f.write(f"  HPCG\n")
            if run_lammps:
                f.write(f"  LAMMPS\n")
            if run_mpgadget:
                f.write(f"  MP-Gadget\n")
            if run_nvidiasmi:
                f.write(f"  nvidia-smi\n")
            if test == "CPU-only Nodes":
                f.write(f"Job script: {job_script}\n")

    # Final commit
    def accept(self):
        self.serialize()
        super().accept()

# ---------- Main ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    wizard = NodeTesterWizard()
    wizard.show()
    sys.exit(app.exec())
