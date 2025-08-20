import os
import json
import pandas as pd
import re
import yaml
from tqdm import tqdm
import shutil
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

def extract_components(filename):
    # filename without "_results.json"
    name = filename.replace('_results.json', '')
    pattern = r'^j(\d+)_r([\d.]+)_E(.*?)_m(\d+)$'
    match = re.match(pattern, name)
    if match:
        return match.groups()  # returns tuple (J, R, E, M)
    return None

def generate_expected_files(config):
    """Generate all expected (algo, filename) tuples based on config"""
    expected = []
    for algo in config['SA_EBM_ALGO_NAMES'] + config['OTHER_ALGO_NAMES']:
        for J in config['JS']:
            for R in config['RS']:
                for E in config['EXPERIMENT_NAMES']:
                    for M in range(config['N_VARIANTS']):
                        fname = f"j{J}_r{R}_E{E}_m{M}_results.json"
                        expected.append((algo, fname))
    return set(expected)

def main():

    ALGONAMES = [
        'Conjugate Priors', "MLE", 'KDE', 'EM', 'Hard K-Means',
        'DEBM', 'DEBM GMM', 'UCL GMM', 'UCL KDE']

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    OUTPUT_DIR = config['OUTPUT_DIR']
    SA_EBM_ALGO_NAMES = config['SA_EBM_ALGO_NAMES']
    OTHER_ALGO_NAMES = config['OTHER_ALGO_NAMES']
    ALL_ALGOS = SA_EBM_ALGO_NAMES + OTHER_ALGO_NAMES
    JS = config['JS']
    RS = config['RS']
    EXPERIMENTS = config['EXPERIMENT_NAMES']
    N_VARIANTS = config['N_VARIANTS']

    titles = [
        "Exp 1: S + Ordinal kj (DM) + X (Normal)",
        "Exp 2: S + Ordinal kj (DM) + X (Non-Normal)",
        "Exp 3: S + Ordinal kj (Uniform) + X (Normal)",
        "Exp 4: S + Ordinal kj (Uniform) + X (Non-Normal)",
        "Exp 5: S + Continuous kj (Uniform)",
        "Exp 6: S + Continuous kj (Skewed)",
        "Exp 7: xi (Normal) + Continuous kj (Uniform)",
        "Exp 8: xi (Normal) + Continuous kj (Skewed)",
        "Exp 9: xi (Normal + Noise) + Continuous kj (Skewed)"
    ]

    # Normalize mapping dictionaries
    CONVERT_E_DICT = {k: v for k, v in zip(EXPERIMENTS, titles)}
    CONVERT_ALGO_DICT = {k.lower(): v for k, v in zip(ALL_ALGOS, ALGONAMES)}

    # Initialize tracking structures
    expected_files = generate_expected_files(config)
    found_files = set()
    missing_files = set()
    failed_files = []
    records = []

    which_tau_better = defaultdict(lambda: defaultdict(int))

    # Process all algorithms
    for algo in tqdm(ALL_ALGOS, desc="Processing algorithms"):
        algo_dir = os.path.join(OUTPUT_DIR, algo, "results")

        if not os.path.exists(algo_dir):
            print(f"\nWarning: Missing directory for {algo}")
            continue

        # Process all result files
        files = [f for f in os.listdir(algo_dir) if f.endswith('_results.json')]
        for fname in tqdm(files, desc=f"{algo}", leave=False):
            full_path = os.path.join(algo_dir, fname)

            # Track found files
            found_files.add((algo, fname))

            # Parse filename components
            components = extract_components(fname)
            if not components:
                failed_files.append((full_path, "Invalid filename format"))
                continue

            J, R, E, M = components
            try:
                J = int(J)
                R = float(R)
                M = int(M)
            except ValueError:
                failed_files.append((full_path, "Invalid numeric format in filename"))
                continue
                
            # Validate against config
            if J not in JS:
                failed_files.append((full_path, f"Invalid J value {J}"))
                continue
            if R not in RS:
                failed_files.append((full_path, f"Invalid R value {R}"))
                continue
            if E not in EXPERIMENTS:
                failed_files.append((full_path, f"Invalid experiment {E}"))
                continue
            if not (0 <= M < N_VARIANTS):
                failed_files.append((full_path, f"Invalid M value {M}"))
                continue

            # Load and validate JSON content
            try:
                with open(full_path, 'r') as f:
                    data = json.load(f)
                
                if 'kendalls_tau' not in data or 'mean_absolute_error' not in data:
                    failed_files.append((full_path, "Missing metrics in JSON"))
                    continue

                algo_pretty = CONVERT_ALGO_DICT.get(algo.lower(), algo)  # fallback to raw if not found
                E_pretty = CONVERT_E_DICT.get(E, E)

                if algo in SA_EBM_ALGO_NAMES:
                    ## Which Tau Better
                    ml_order_tau = data['kendalls_tau']
                    highest_ll_order_tau = data['kendalls_tau2']

                    if ml_order_tau > highest_ll_order_tau:
                        which_tau_better[algo]['ml_order_better'] += 1
                    elif ml_order_tau < highest_ll_order_tau:
                        which_tau_better[algo]['highest_ll_order_better'] += 1
                    else:
                        which_tau_better[algo]['equal'] += 1

                    # # MAE
                    # true_stages = data['true_stages']
                    # final_stage_post = data['stage_likelihood_posterior']
                    # n_participants = len(true_stages)
                    # ml_stages_soft = [
                    #     np.random.choice(len(final_stage_post[str(pid)]), p=final_stage_post[str(pid)]) + 1
                    #     if str(pid) in final_stage_post else 0
                    #     for pid in range(n_participants)
                    # ]
                    # mae_soft = mean_absolute_error(true_stages, ml_stages_soft)

                    kendalls_tau = highest_ll_order_tau
                    # mae_result = mae_soft
                    # qwk_result = cohen_kappa_score(true_stages, ml_stages_soft, weights='quadratic')

                else:
                    kendalls_tau = data['kendalls_tau']

                mae_result = data['mean_absolute_error']
                mae_diseased_result = data['mean_absolute_error_diseased']
                runtime = data['runtime']

                records.append({
                    'J': J,
                    'R': R,
                    'E': E_pretty,
                    'M': M,
                    'algo': algo_pretty,
                    'runtime': runtime,
                    'kendalls_tau': (1-kendalls_tau)/2,
                    'mae': mae_result,
                    'mae_diseased': mae_diseased_result
                })
            except json.JSONDecodeError:
                failed_files.append((full_path, "Invalid JSON format"))
            except Exception as e:
                failed_files.append((full_path, f"Unexpected error: {str(e)}"))
    
    # Calculate missing files
    missing_files = expected_files - found_files

    # Save results
    if records:
        df = pd.DataFrame(records)
        df = df.sort_values(by=['J', 'R', 'E', 'M', 'algo'])
        df.to_csv('all_results.csv', index=False)
        print(f"\nSaved {len(df)} valid records to all_results.csv")

    if which_tau_better:
        sorted_tau_data = {
            k: dict(sorted(v.items()))
            for k, v in which_tau_better.items()
        }
        with open ('which_tau_better.json', 'w') as f:
            json.dump(sorted_tau_data, f, indent=4)

    # Save diagnostics
    if missing_files:
        unique_missing_fnames = set([x[1].replace("_results.json", "") for x in missing_files])
        with open('missing_files.txt', 'w') as f:
            f.write("Algorithm, Filename\n")
            for algo, fname in sorted(missing_files):
                f.write(f"{algo}, {fname}\n")
        print(f"Logged {len(missing_files)} missing files to missing_files.txt")

        # Save NA_COMBINATIONS.txt 
        with open('na_combinations.txt', 'w') as f:
            print(f'Number of unique missing fnames: {len(unique_missing_fnames)}')
            for fname in sorted(unique_missing_fnames):
                f.write(f"{fname}\n")
        print(f"Logged {len(unique_missing_fnames)} unique missing files to na_combinations.txt")

        # Copy err and out logs 
        # Create the error_logs directory if it doesn't exist
        if not os.path.exists('error_logs'):
            os.makedirs('error_logs')
        
        ERR_LOGS = [f"eval_{x}.err" for x in unique_missing_fnames]
        OUT_LOGS = [f"eval_{x}.out" for x in unique_missing_fnames]
        LOG_LOGS = [f"eval_{x}.log" for x in unique_missing_fnames]
        # Copy each file from logs to error_logs
        for filename in ERR_LOGS + OUT_LOGS + LOG_LOGS:
            source_path = os.path.join('logs', filename)
            dest_path = os.path.join('error_logs', filename)
            try:
                shutil.copy2(source_path, dest_path)
            except FileNotFoundError:
                print(f"File not found: {filename}")
            except Exception as e:
                print(f"Error copying {filename}: {e}")
        print("Done copying files to error_logs folder")        
        
    if failed_files:
        with open('failed_files.txt', 'w') as f:
            f.write("Path, Reason\n")
            for path, reason in failed_files:
                f.write(f"{path}, {reason}\n")
        print(f"Logged {len(failed_files)} failed files to failed_files.txt")

if __name__ == '__main__':
    main()
