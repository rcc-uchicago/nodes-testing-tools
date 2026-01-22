# Contact: ndtrung@uchicago.edu
# usage: on a compute node, activate the env, then run the script with a YAML file
#
#   module load python/miniforge-25.3.0
#   source /project/rcc/shared/nodes-testing/testing-env/bin/activate
#   
#   python3 node-tester-wizard.py

import sys
from PySide6.QtWidgets import (
    QApplication, QWizard, QWizardPage,
    QLabel, QVBoxLayout, QComboBox,
    QLineEdit, QMessageBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)

# ---------- Page 1: Welcome ----------
class WelcomePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Welcome")

        label = QLabel(
            "Welcome to the RCC Node Tester Wizard.\n"
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
        self.project_name.setPlaceholderText("test-cpp-ticket-1234.txt")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Project:"))
        layout.addWidget(self.project_name)
        layout.addWidget(QLabel("Select a Test Pipeline:"))
        layout.addWidget(self.test_combo)
        self.setLayout(layout)

        # Notify wizard when inputs change
        self.test_combo.currentIndexChanged.connect(self.completeChanged)
        self.project_name.textChanged.connect(self.completeChanged)

        # Optional UX improvement
        #self.vendor_combo.currentIndexChanged.connect(self.update_ui)
        #self.update_ui()

        # Register fields
        self.registerField("test*", self.test_combo, "currentText")
        self.registerField("project_name", self.project_name)

    #def update_ui(self):
    #    is_vendor1 = self.vendor_combo.currentText() == "Vendor 1"
    #    self.project_id.setEnabled(is_vendor1)

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
        name = self.project_name.text().strip()
        if not name:
            placeholderText = self.project_name.placeholderText()
            self.project_name.setText(placeholderText)

        return True

# ---------- Page 3: Node Test Configration ----------

class PipelineConfigurationPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Test Pipeline Configuration")

        # IMPORTANT: this is the last page
        #self.setFinalPage(True)

        self.run_lscpu = QCheckBox("lscpu")
        self.run_hpcg = QCheckBox("HPCG")
        self.run_lammps = QCheckBox("LAMMPS")
        self.run_mpgadget = QCheckBox("MP-Gadget")
        
        self.run_nvidiasmi = QCheckBox("nvidia-smi")

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select the applications:"))
        layout.addWidget(self.run_lscpu)
        layout.addWidget(self.run_hpcg)
        layout.addWidget(self.run_lammps)
        layout.addWidget(self.run_mpgadget)
        layout.addWidget(self.run_nvidiasmi)
        self.setLayout(layout)

        # Register fields
        self.registerField("run_lscpu", self.run_lscpu)
        self.registerField("run_hpcg", self.run_hpcg)
        self.registerField("run_lammps", self.run_lammps)
        self.registerField("run_mpgadget", self.run_mpgadget)
        self.registerField("run_nvidiasmi", self.run_nvidiasmi)

    def initializePage(self):        
        test = self.field("test")
        is_running_nvidiasmi = (test == "GPU Nodes")
        self.run_nvidiasmi.setVisible(is_running_nvidiasmi)

        if not is_running_nvidiasmi:
            self.run_nvidiasmi.setChecked(False)

# ---------- Page 4: Submit job and waiting for results ----------

class JobMonitorPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Test Monitor")

        # IMPORTANT: this is the last page
        self.setFinalPage(True)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["Job ID", "Status", "Test Type"]
        )

        # UI polish
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Test job status:"))
        layout.addWidget(self.table)        
        
        self.setLayout(layout)

        # Register fields
        


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


    def generateJobScript(self):
        run_lscpu = self.field("run_lscpu")
        run_hpcg = self.field("run_hpcg")
        run_lammps = self.field("run_lammps")
        run_mpgadget = self.field("run_mpgadget")

        content  = f"#!/bin/bash\n"
        content += f"#SBATCH --account=rcc-staff\n"
        content += f"#SBATCH --partition=test\n"
        content += f"#SBATCH --nodelist=\n"

        content += f"#SBATCH --reservation=Test_CPP\n"
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
        if run_hpcg:
            content += "python3 run-tests.py --config-file hpcg.yaml\n"    
        if run_lammps:
            content += "python3 run-tests.py --config-file lammps.yaml\n"
        if run_mpgadget:
            content += "python3 run-tests.py --config-file mp-gadget.yaml\n"    

        with open("job_script.txt", "w") as f:
            f.write(content)

    # Final commit
    def accept(self):
        test = self.field("test")
        project_name = self.field("project_name")
        run_lscpu = self.field("run_lscpu")
        run_hpcg = self.field("run_hpcg")
        run_lammps = self.field("run_lammps")
        run_mpgadget = self.field("run_mpgadget")

        with open("test.txt", "w") as f:
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
            if test == "CPU-only Nodes":
                f.write(f"Script=queue-cpu-nodes.txt\n")
            elif test == "GPU Nodes":
                f.write(f"Script=queue-cpu-nodes.txt\n")

            self.generateJobScript()

        super().accept()

    
        


# ---------- Main ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    wizard = NodeTesterWizard()
    wizard.show()
    sys.exit(app.exec())
