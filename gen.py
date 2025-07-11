from pysaebm import generate, get_params_path
import os
import numpy as np 
import json 
import yaml

def load_config():
    current_dir = os.path.dirname(__file__)  # Get the directory of the current script
    config_path = os.path.join(current_dir, "config.yaml")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def convert_np_types(obj):
    """Convert numpy types in a nested dictionary to Python standard types."""
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_np_types(obj.tolist())
    else:
        return obj

if __name__ == '__main__':

    # Get path to default parameters
    params_file = get_params_path()

    config = load_config()
    print("Loaded config:")
    print(json.dumps(config, indent=4))

    all_dicts = []

    for exp_name in config['EXPERIMENT_NAMES']:
        true_order_and_stages_dicts = generate(
                experiment_name = exp_name,
                params_file=params_file,
                js = config['JS'],
                rs = config['RS'],
                num_of_datasets_per_combination=config['N_VARIANTS'],
                output_dir='data',
                seed=config['GEN_SEED'],
                keep_all_cols = False,
            )
        all_dicts.append(true_order_and_stages_dicts)

    combined = {k: v for d in all_dicts for k, v in d.items()}
    combined = convert_np_types(combined)

    # Dump the JSON
    with open("true_order_and_stages.json", "w") as f:
        json.dump(combined, f, indent=2)