import sys 
import os 
sys.path.append(os.getcwd())

import json 
from run import convert2debm_and_ucl, run_debm, run_ucl
from pysaebm import run_ebm
from typing import Dict, List, Optional, Tuple

import yaml
def load_config():
    current_dir = os.path.dirname(__file__)  # Get the directory of the current script
    config_path = os.path.join(current_dir, "config.yaml")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # Get directories correct
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current working directory: {base_dir}")
    data_dir = os.path.join(base_dir, "toy_data")

    # Read parameters from command line arguments
    # filename = "j200_r0.25_Esn_kjContinuousBeta_m0"
    filename = sys.argv[1]

    print(f"Processing with {filename}")
    data_file = os.path.join(data_dir, f"{filename}.csv")
    if not os.path.isfile(data_file):
        print(f"Error: Data file {data_file} does not exist.")
        sys.exit(1)

    # Parameters
    config = load_config()
    print("Loaded config:")
    print(json.dumps(config, indent=4))

    # Number of independent optimization attempts in greedy ascent
    NStartpoints=config['NStartpoints']
    Niterations=config['Niterations']
    N_MCMC=config['N_MCMC']

    # BURN_IN=50
    # N_MCMC=200
    # OUTPUT_DIR='local_results'

    N_SHUFFLE=config['N_SHUFFLE']
    BURN_IN=config['BURN_IN']
    THINNING=config['THINNING']
    OUTPUT_DIR=config['OUTPUT_DIR']

    # Algo names
    sa_ebm_algo_names = config['SA_EBM_ALGO_NAMES']
    other_algo_names = config['OTHER_ALGO_NAMES']

    # Get true order and true stages dict
    with open(os.path.join(base_dir, "true_order_and_stages.json"), "r") as f:
        true_order_and_stages = json.load(f)
    true_order_dict = true_order_and_stages[filename]['true_order']
    true_stages = true_order_and_stages[filename]['true_stages']

    ###################################################################################
    # Step1: Our model
    ###################################################################################
    for algo_name in sa_ebm_algo_names:
        run_ebm(
            data_file=data_file,
            output_dir=OUTPUT_DIR,
            algorithm=algo_name,
            n_iter=N_MCMC,
            n_shuffle=N_SHUFFLE,
            burn_in=BURN_IN,
            thinning=THINNING,
            true_order_dict=true_order_dict,
            true_stages= true_stages,
            skip_heatmap = True,
            skip_traceplot = True,
        )

    ###################################################################################
    # Step2: Other models
    ###################################################################################
    # Convert and Extract data
    try:
        debm_output, data_matrix, non_diseased_ids = convert2debm_and_ucl(data_file, true_order_dict)
    except Exception as e:
        print(f"Error: Failed to convert {data_file}!")
        sys.exit(1)

    for algo_name in other_algo_names:
        if algo_name.startswith('ucl'):
            run_ucl(
                fname = filename, 
                output_dir=OUTPUT_DIR,
                data_matrix = data_matrix,
                algorithm = algo_name,
                true_stages = true_stages,
                true_order_dict=true_order_dict,
                n_iter = N_MCMC,
                greedy_n_init=NStartpoints,
                greedy_n_iter=Niterations,
            )
        else:
            if algo_name == 'debm':
                run_debm(
                    fname = filename,
                    algorithm = algo_name,
                    output_dir=OUTPUT_DIR,
                    DataIn = debm_output,
                    true_stages = true_stages,
                    true_order_dict = true_order_dict,
                )
            else:
                run_debm(
                    fname = filename,
                    algorithm = algo_name,
                    output_dir=OUTPUT_DIR,
                    DataIn = debm_output,
                    true_stages = true_stages,
                    true_order_dict = true_order_dict,
                    N_MCMC = N_MCMC,
                    NStartpoints = NStartpoints,
                    Niterations = Niterations,
                )