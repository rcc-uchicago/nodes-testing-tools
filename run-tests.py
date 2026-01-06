
# Contact: ndtrung@uchicago.edu
# usage:
#   python3 run-tests.py --config-file=lammps.yaml

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
    if config['mpiexec']:
        cmd_str += config['mpiexec'] + " " + config['mpiexec_numproc_flag'] + " " + config['nprocs'] + " "

    cmd_str += config['app_binary'] + " " + config['args']
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

        if "Loop" in status['stdout']:
            #print('Run completed')    
            # Write the output to a temporary file .tmp.txt
            with open(".tmp.txt", "w") as f:
                f.write(status['stdout'])
                f.close()
            # Run the script to extract output
            cmd_str = "source " + config['extract_output_script'] + " .tmp.txt > output.yaml"
            p = subprocess.run(cmd_str, shell=True, text=True, capture_output=True, timeout=60)
            output = p.stdout.split()

            output_results = None
            with open("output.yaml", 'r') as f:
                output_results = yaml.load(f, Loader=Loader)
                f.close()
            if output_results is not None:
                status['output_results'] = output_results['output']
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
        logging.info(f"\nUsing the configuration file:\n  {absolute_path}")
        f.close()

    status = execute(config)
    logging.info(f"Ouput: {status['output_results']}")
    logging.info(f"Expected: {config['expected']}")

