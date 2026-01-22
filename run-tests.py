
# Contact: ndtrung@uchicago.edu
# usage: on a compute node, activate the env, then run the script with a YAML file
#
#   module load python/miniforge-25.3.0
#   source /project/rcc/shared/nodes-testing/testing-env/bin/activate
#   ulimit -l unlimited
#   python3 run-tests.py --config-file lammps.yaml

from argparse import ArgumentParser
from datetime import datetime
import logging
import numpy as np
import os
import subprocess
import yaml
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader


def execute(config):

    # need to load the modules in the same command to be executed
    # otherwise the modules are only loaded in the subprocess that is created
    cmd_str = ""
    if config['modules_needed']:
        cmd_str = f"module load {config['modules_needed']} && "

    # check if mpiexec/mpirun is used
    if 'mpiexec' in config:
        if config['mpiexec']:
            cmd_str += config['mpiexec'] + " " + config['mpiexec_numproc_flag'] + " " + config['nprocs'] + " "

    args = ""
    if 'input_dir' in config:
        args = config['args'].replace("$input_dir", config['input_dir'])

    cmd_str += config['app_binary'] + " " + args
    logging.info(f"Execute:")
    logging.info(f"  {cmd_str}")
    try:
        p = subprocess.run(cmd_str, shell=True, text=True, capture_output=True, timeout=60)
        status = { 
            'cmd_str': cmd_str,
            'stdout': p.stdout,
            'stderr': p.stderr,
            'returncode': p.returncode,
        }

        if config['run_completed_marker'] in status['stdout']:
            #print('Run completed')    
            # Write the output to a temporary file .tmp.txt
            with open(".tmp.txt", "w") as f:
                f.write(status['stdout'])
                f.close()
            # Run the script to extract output
            working_dir = "./"
            if 'working_dir' in config:
                working_dir = config['working_dir']
            extract_script = config['extract_output_script'].replace("$working_dir", working_dir)
            cmd_str = "source " + extract_script + " .tmp.txt > output.yaml"
            p = subprocess.run(cmd_str, shell=True, text=True, capture_output=True, timeout=60)
            output = p.stdout.split()

            output_results = None
            with open("output.yaml", 'r') as f:
                output_results = yaml.load(f, Loader=Loader)
                f.close()
            if output_results is not None:
                status['output_results'] = output_results['output']
        else:
            msg = f"The run did not completed successfully. Rerun {cmd_str} to troubleshoot."
            logging.error(msg)

        return status

    except subprocess.TimeoutExpired:
        msg = f"     Timeout for: {cmd_str}"
        logging.error(msg)

    
    status = { 
        'cmd_str': cmd_str,
        'stdout': "",
        'stderr': "",
        'returncode': -1,
    }
    return status

if __name__ == "__main__":

    configFileName = "config.yaml"

    parser = ArgumentParser()
    parser.add_argument("--config-file", dest="config_file", default="", help="Configuration YAML file")
    parser.add_argument('--log-file', type=str, dest="log_file", default=None,
                       help='Path to log file (default: log to console only)')
    parser.add_argument('--log-level', type=str, dest="log_level", default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    args = parser.parse_args()
    
    # Configure logging based on arguments
    handlers = [logging.StreamHandler()]  # Always log to console
    
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


    if len(args.config_file) > 0:
        configFileName = args.config_file
    
    # read in the configuration of the tests
    with open(configFileName, 'r') as f:
        config = yaml.load(f, Loader=Loader)
        absolute_path = os.path.abspath(configFileName)
        logging.info(f"Using the configuration file:\n  {absolute_path}")
        f.close()

    # Execute the pipeline in the configuration file
    status = execute(config)
    #logging.info(f"Ouput: {status['output_results']}")
    #logging.info(f"Expected: {config['expected']}")

    passed = True
    failed_quantities = []
    for quantity in config['expected']:
        if quantity not in status['output_results']:
            logging.info(f"{quantity} is missing in the output")
            logging.info("Failed")
            break
        
        expected_value = config['expected'][quantity]['value']
        actual_value = status['output_results'][quantity]['value']
        absdiff = np.abs(np.float64(expected_value) - np.float64(actual_value))
        reldiff = absdiff / np.abs(np.float64(expected_value)) * 100.0

        if 'abstol' in config['expected'][quantity]:
            abstol = np.float64(config['expected'][quantity]['abstol'])
            logging.info(f"{quantity}: Actual = {actual_value} Expected = {expected_value} absdiff = {absdiff:.5f} abstol = {abstol}")
            if absdiff > abstol:
                passed = False
                failed_quantities.append(quantity)

        if 'reltol' in config['expected'][quantity]:
            reltol = np.float64(config['expected'][quantity]['reltol'])
            logging.info(f"{quantity}: Actual = {actual_value} Expected = {expected_value} reldiff = {reldiff:.3f} reltol = {reltol}")
            if reldiff > reltol:
                passed = False
                failed_quantities.append(quantity)

    if passed == True:
        logging.info(f"PASSED")
    else:
        logging.info(f"FAILED for the following output: {failed_quantities}")
        



