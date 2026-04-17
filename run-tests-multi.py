# This Python script is part of the nodes testing tools.
# Contact: Trung Nguyen  (@ndtrung81)
#          Akhil Francis (@Akhil-Francis)
#          Bailey Howell (@bkhowell)
#
# Usage: on a compute node, activate the env, then run the script with a YAML file
#   each YAML file can have multiple tasks to run multiple tests
#
#   module load python/miniforge-25.3.0
#   source /project/rcc/shared/nodes-testing/testing-env/bin/activate
#   ulimit -l unlimited
#   python3 run-tests-multi.py --config-file lscpu-multi.yaml

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

def execute(config, verbose=False):
    '''
    Execute the pipeline (aka task) in the configuration file
    '''
    # need to load the modules in the same command to be executed
    # otherwise the modules are only loaded in the subprocess that is created
    cmd_str = ""
    if config['modules_needed']:
        cmd_str = f"module load {config['modules_needed']} && "
    
    if 'preprocess' in config:
        if isinstance(config['preprocess'], str):
            cmd_str += f"{config['preprocess']} && "
            if '$input_dir' in cmd_str:
                cmd_str = cmd_str.replace("$input_dir", config['input_dir'])

        else:
            for cmd_i in config['preprocess']:
                cmd_i = cmd_i.replace("$input_dir", config['input_dir'])
                cmd_i = cmd_i.replace("$working_dir", config['working_dir'])
                cmd_str += cmd_i + " && "

    # if multiple app files are used
    if isinstance(config['app_binary'], list):
        for i,appbin in enumerate(config['app_binary']):
            if 'time_command' in config:
                cmd_str += config['time_command'] +" "
            # check if mpiexec/mpirun is used
            if 'mpiexec' in config:
                cmd_str += config['mpiexec'] + " " + config['mpiexec_numproc_flag'] + " " + config['nprocs'] + " "
            if 'mpiexec_ppn_flag' in config:
                cmd_str += "-" + config['mpiexec_ppn_flag'] + " "
            if 'mpiexec_bind_flag' in config:
                cmd_str += config['mpiexec_bind_flag'] + " "

            appbin = appbin.replace("$working_dir",config['working_dir'])
            cmd_str += appbin + " "
            
            if 'args' in config:
                args = config['args'][i]
                args = args.replace("$working_dir", config['working_dir'])
                args = args.replace("$input_dir", config['input_dir'])
                cmd_str += args

            cmd_str += " && "
        cmd_str = cmd_str[:-4] # to remove extra " && "
    else:
        # check if mpiexec/mpirun is used
        if 'mpiexec' in config:
            cmd_str += config['mpiexec'] + " " + config['mpiexec_numproc_flag'] + " " + config['nprocs'] + " "
        
        appbin = config['app_binary']
        appbin = appbin.replace("$working_dir",config['working_dir'])
        cmd_str += appbin + " " 
        
        if 'args' in config:
            args = config['args']
            args = args.replace("$working_dir", config['working_dir'])
            args = args.replace("$input_dir", config['input_dir'])
            cmd_str += args

    # multiple runs may need larger timeout value
    if 'timeout_value' in config:
        timeout_value = config["timeout_value"]
    else:
        timeout_value = 60

    logging.info(f"Executing:")
    logging.info(f"  {cmd_str}")
    try:
        p = subprocess.run(cmd_str, shell=True, text=True, capture_output=True, timeout=timeout_value)
        status = { 
            'cmd_str': cmd_str,
            'stdout': p.stdout,
            'stderr': p.stderr,
            'returncode': p.returncode,
        } 
        if verbose:
            logging.info(f"stderr:")
            logging.info(f"  {status['stderr']}")

        logging.info(f'Run completed')
        # Write the output to a temporary file .tmp.txt
        with open(".tmp.txt", "w") as f:
            f.write(status['stdout'])
            f.close()
        # Run the script to extract output
        working_dir = "./"
        if 'working_dir' in config:
            working_dir = config['working_dir']
        extract_script = config['extract_output_script'].replace("$working_dir", working_dir)
        cmd_str = "sh " + extract_script + " .tmp.txt > output.yaml"
        p = subprocess.run(cmd_str, shell=True, text=True, timeout=60)

        output_results = None
        with open("output.yaml", 'r') as f:
            output_results = yaml.load(f, Loader=Loader)
            f.close()
        if output_results is not None:
            status['output_results'] = output_results['output']

        if config['run_completed_marker'] not in status['stdout']:                
            logging.info(msg = f"The run might not have completed successfully. Rerun {cmd_str} to troubleshoot.")

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

def check_output(output, config):
    '''
    Check the output results with the expected values in the configuration file.
    '''
    passed = True
    failed_quantities = []
    for quantity in config['expected']:
        if 'output_results' not in output:
            logging.info(f"output_results is missing in the output")
            logging.info("Failed")
            passed = False
            failed_quantities.append(quantity)
            break

        if quantity not in output['output_results']:
            logging.info(f"{quantity} is missing in the output")
            logging.info("Failed")
            passed = False
            failed_quantities.append(quantity)
            break
        
        expected_value = config['expected'][quantity]['value']
        actual_value = output['output_results'][quantity]['value']

        if isinstance(expected_value, str):
            logging.info(f"{quantity}: Actual = {actual_value} Expected = {expected_value}")
            if not isinstance(actual_value, str):
                logging.info("Failed")
                passed = False
                failed_quantities.append(quantity)    
            else:
                actual_value = actual_value.strip()
                if expected_value != actual_value:
                    logging.info("Failed")
                    passed = False
                    failed_quantities.append(quantity)
            
        else:
            # numeric values, check if the actual value is a number
            if not isinstance(actual_value, (int, float)):
                logging.info(f"{quantity} is not a number in the actual output")
                logging.info(f"{quantity}: Actual = {actual_value} Expected = {expected_value}")
                logging.info("Failed")
                passed = False
                failed_quantities.append(quantity)
                continue

            absdiff = np.abs(np.float64(expected_value) - np.float64(actual_value))

            if 'abstol' in config['expected'][quantity]:
                abstol = np.float64(config['expected'][quantity]['abstol'])
                logging.info(f"{quantity}: Actual = {actual_value} Expected = {expected_value} absdiff = {absdiff:.5f} abstol = {abstol}")
                if absdiff > abstol:
                    passed = False
                    failed_quantities.append(quantity)

            if 'reltol' in config['expected'][quantity]:
                reltol = np.float64(config['expected'][quantity]['reltol'])
                reldiff = absdiff / np.abs(np.float64(expected_value)) * 100.0
                logging.info(f"{quantity}: Actual = {actual_value} Expected = {expected_value} reldiff = {reldiff:.3f} reltol = {reltol}")
                if reldiff > reltol:
                    passed = False
                    failed_quantities.append(quantity)

    results = {
        'passed': passed,
        'failed_quantities': failed_quantities,
    }

    return results

if __name__ == "__main__":

    configFileName = "config.yaml"

    parser = ArgumentParser()
    parser.add_argument("--config-file", dest="config_file", default="", help="Configuration YAML file")
    parser.add_argument('--log-file', type=str, dest="log_file", default=None,
                       help='Path to log file (default: log to console only)')
    parser.add_argument('--log-level', type=str, dest="log_level", default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")
    args = parser.parse_args()
    verbose = args.verbose

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
        tasks = yaml.load(f, Loader=Loader)
        absolute_path = os.path.abspath(configFileName)
        logging.info(f"Using the configuration file:\n  {absolute_path}")
        f.close()

    for task in tasks:
        logging.info(f"Task: {task['task']}")

        skip = task.get('skip', False)
        if skip:
            logging.info(f"skip: True in {configFileName}")
            continue

        # Execute the task pipeline in the configuration file
        output = execute(task, verbose=verbose)

        # check the output results with the expected values in the configuration file
        results = check_output(output, task)

        if results['passed']:
            result_str = "PASSED"
        else:
            result_str = "FAILED"
        logging.info(f"Testing results:  {result_str}")
        if not results['passed']:
            logging.info(f"Failed quantities:  {results['failed_quantities']}")

    



