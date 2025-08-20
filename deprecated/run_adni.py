import os 
from pysaebm import run_ebm
import utils_adni
import yaml
import json 

meta_data = ['PTID', 'DX_bl', 'VISCODE', 'COLPROT']

select_biomarkers = ['MMSE_bl', 'Ventricles_bl', 'WholeBrain_bl', 
            'MidTemp_bl', 'Fusiform_bl', 'Entorhinal_bl', 
            'Hippocampus_bl', 'ADAS13_bl', 'PTAU_bl', 
            'TAU_bl', 'ABETA_bl', 'RAVLT_immediate_bl'
]

diagnosis_list = ['CN', 'EMCI', 'LMCI', 'AD']

raw = 'ADNIMERGE.csv'

def load_config():
    current_dir = os.path.dirname(__file__)  # Get the directory of the current script
    config_path = os.path.join(current_dir, "config.yaml")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

if __name__=='__main__':
    # Get directories correct
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current working directory: {base_dir}")

    OUTPUT_DIR = f'adni_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parameters
    config = load_config()
    print("Loaded config:")
    print(json.dumps(config, indent=4))

    # Number of independent optimization attempts in greedy ascent
    NStartpoints=config['NStartpoints']
    Niterations=config['Niterations']
    N_MCMC=config['N_MCMC']
    # N_MCMC = 100
    N_SHUFFLE=config['N_SHUFFLE']
    BURN_IN=config['BURN_IN']
    # BURN_IN = 10
    THINNING=config['THINNING']
    N_BOOTSTRAP=config['N_BOOTSTRAP']
    MCMC_SEED = config['MCMC_SEED']
    ADNI_MCMC_SEED = config['ADNI_MCMC_SEED']
    sa_ebm_algorithms = config['SA_EBM_ALGO_NAMES']

    raw = os.path.join(base_dir, raw)

    adni_filtered = utils_adni.get_adni_filtered(raw, meta_data, select_biomarkers, diagnosis_list)
    debm_output, data_matrix, df_long, participant_dx_dict, ordered_biomarkers = utils_adni.process_data(adni_filtered, ventricles_log=False, tau_log=False)

    for algorithm in sa_ebm_algorithms:
        results = run_ebm(
            data_file=os.path.join(base_dir, 'adni.csv'),
            algorithm=algorithm,
            output_dir=OUTPUT_DIR,
            n_iter=N_MCMC,
            n_shuffle=2,
            burn_in=BURN_IN,
            thinning=1,
            skip_heatmap=False,
            skip_traceplot=False,
            seed=ADNI_MCMC_SEED,
            save_results=True,
            save_details=True,
        )

    ucl_gmm_results = utils_adni.run_ucl_gmm(
        output_dir=OUTPUT_DIR,
        data_matrix = data_matrix,
        ordered_biomarkers=ordered_biomarkers,
        n_iter = N_MCMC,
        greedy_n_init=NStartpoints,
        greedy_n_iter=Niterations,
    )

    utils_adni.run_debm(
        output_dir=OUTPUT_DIR,
        algorithm = 'debm',
        DataIn = debm_output,
    )

    utils_adni.run_debm(
        output_dir=OUTPUT_DIR,
        algorithm = 'debm_gmm',
        DataIn = debm_output,
        N_MCMC = N_MCMC,
        NStartpoints = NStartpoints,
        Niterations = Niterations,
    )

    utils_adni.run_debm_with_bootstrap_and_plot(
        output_dir=OUTPUT_DIR,
        algorithm = 'debm',
        DataIn = debm_output,
        plot_title='DEBM Ordering Result',
        n_bootstraps = N_BOOTSTRAP
    )

    utils_adni.run_debm_with_bootstrap_and_plot(
        output_dir=OUTPUT_DIR,
        algorithm = 'debm_gmm',
        DataIn = debm_output,
        plot_title='DEBM GMM Ordering Result',
        N_MCMC = N_MCMC,
        NStartpoints = NStartpoints,
        Niterations = Niterations,
        n_bootstraps = N_BOOTSTRAP
    )