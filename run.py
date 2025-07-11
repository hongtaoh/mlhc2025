# Add this at the VERY TOP of your script
import os
import warnings
from functools import partialmethod
from tqdm import tqdm
import matplotlib as mpl
import time

# 1. Disable all progress bars
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
os.environ["PROGRESS_BAR"] = "0"  # Universal progress bar disable

# 2. Configure matplotlib before any other imports
mpl.use('Agg')  # Must come before pyplot import
import matplotlib.pyplot as plt

# 3. Global warning suppression
warnings.filterwarnings("ignore")

from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error

# Rest of imports
import pandas as pd 
import numpy as np 
import math 
import json 
from pyebm import debm
from pyebm import ebm
from typing import List, Dict, Tuple, Optional
from kde_ebm import mixture_model
from kde_ebm import mcmc
from kde_ebm.mixture_model import get_prob_mat
from scipy.stats import kendalltau
from collections import defaultdict, namedtuple

def save_json(outfname:str, data:Dict):
    with open(outfname, 'w') as f:
        json.dump(data, f, indent=4, default=convert_np_types)

def print_metrics(result: Dict):
    print(f"[{result['algorithm']}] QWK: {result['quadratic_weighted_kappa']:.3f}, "
          f"MAE: {result['mean_absolute_error']:.2f}, "
          f"Kendall's Tau: {result['kendalls_tau']:.3f}")

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

#* Define the EBM staging function
def ebm_staging(x,mixtures,samples):
    """
    Given a trained EBM (mixture_models,mcmc_samples), and correctly-formatted data, stage the data
    NOTE: To use CV-EBMs, you'll need to call this for each fold, then combine.
    Author: Neil P Oxtoby, UCL, September 2018
    """
    if type(mixtures[0]) is list:
        #* List of mixture models from cross-validation / bootstrapping
        n_cv = len(mixtures)
        prob_mat = []
        stages = []
        stage_likelihoods = []
        stages_expected = []
        for k in range(n_cv):
            #* Stage the data
            prob_mat.append(get_prob_mat(x, mixtures[k]))
            stages_k, stage_likelihoods_k = samples[k][0].stage_data(prob_mat[k])
            stages.append(stages_k)
            stage_likelihoods.append(stage_likelihoods_k)
            #* Average (expectation value) stage
            stages_expected_k = np.ndarray(shape=stages_k.shape)
            for kk in range(stages_expected_k.shape[0]):
                stages_expected_k[kk] = np.sum(stage_likelihoods_k[kk,:]*np.arange(1,stage_likelihoods_k.shape[1]+1,1))/np.sum(stage_likelihoods_k[kk,:]) - 1
            stages_expected.append(stages_expected_k)
    else:
        #* Stage the data
        prob_mat = get_prob_mat(x, mixtures)
        if type(samples[0]) is list:
            n_bs = len(samples)
            stages = []
            stage_likelihoods = []
            stages_expected = []
            for k in range(n_bs):
                #* Stage the data
                stages_k, stage_likelihoods_k = samples[k][0].stage_data(prob_mat)
                stages.append(stages_k)
                stage_likelihoods.append(stage_likelihoods_k)
                #* Average (expectation value) stage
                stages_expected_k = np.ndarray(shape=stages_k.shape)
                for kk in range(stages_expected_k.shape[0]):
                    stages_expected_k[kk] = np.sum(stage_likelihoods_k[kk,:]*np.arange(1,stage_likelihoods_k.shape[1]+1,1))/np.sum(stage_likelihoods_k[kk,:]) - 1
                stages_expected.append(stages_expected_k)
        else:
            stages, stage_likelihoods = samples[0].stage_data(prob_mat)
            #* Average (expectation value) stage
            stages_expected = np.ndarray(shape=stages.shape)
            for k in range(stages_expected.shape[0]):
                stages_expected[k] = np.sum(stage_likelihoods[k,:]*np.arange(1,stage_likelihoods.shape[1]+1,1))/np.sum(stage_likelihoods[k,:]) - 1
    # #* Average (expectation value) stage
    # stages_expected_n = np.sum(stage_likelihoods,axis=1)
    # stages_expected_ = np.average(stage_likelihoods_long_ml,axis=1,weights=np.arange(1,stage_likelihoods_long_ml.shape[1]+1,1))
    # stages_expected_ = stages_expected_/stages_expected_n
    
    return stages

def convert2debm_and_ucl(
    fname:str, correct_order:Dict[str, int]) -> Tuple[pd.DataFrame, np.ndarray]:
    """ Convert original data to debm and ucl ebm format 
    """
    df = pd.read_csv(fname)
    diseased_dict = dict(zip(df.participant, df.diseased))
    non_diseased_ids = non_diseased_ids = df.loc[~df.diseased, 'participant'].tolist()
    # biomarker names according to their position (ascending)
    desired_order = sorted(correct_order, key=correct_order.get)

    """To convert to DEBM format""" 
    dff = df.pivot(
        index='participant', columns='biomarker', values='measurement')
    # reorder the measurement multi-level according to the desired order
    dff = dff.reindex(columns=desired_order, level=1) 
    # remove column name (biomarker) to clean display
    dff.columns.name = None      
    # bring 'participant' back as a column  
    dff.reset_index(inplace=True)  
    # rename colname per DEBM requirement
    dff.rename(columns={'participant': 'PTID'}, inplace=True) 
    # add col of Diagnosis per DEBM requirement
    dff['Diagnosis'] = ['AD' if diseased_dict[x] else 'CN' for x in dff.PTID]
    debm_output = dff.copy()

    """To convert to UCL output""" 
    # Add Diseased col
    dff['Diseased'] = [int(diseased_dict[x]) for x in dff.PTID]
    # Drop cols
    dff.drop(columns=['PTID', 'Diagnosis'], inplace=True)
    data_matrix = dff.to_numpy()

    return debm_output, data_matrix, non_diseased_ids

def extract_info_ucl(data_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract information (in the format required by UCL) from datasets  
    """
    n_samples, n_features = data_matrix.shape[0], data_matrix.shape[1] - 1
    target_names = np.array(['CN, AD'])
    feature_names = np.array(['BM%i' % (x+1) for x in range(n_features)])

    # Convert data matrix to numpy array if it's not already
    data_matrix = np.array(data_matrix, dtype=np.float64)
    
    # Extract data and target
    data = data_matrix[:, :-1]
    target = data_matrix[:, -1].astype(int)
    
    return data, target, feature_names, target_names

def run_ucl(
    fname: str,
    output_dir: str,
    algorithm: str = "ucl_gmm",
    data_matrix: np.ndarray = None, 
    true_order_dict: Dict[str, int] = {},
    true_stages: List[int] = [],
    n_iter:int=1_0000, 
    greedy_n_iter:int=10, 
    greedy_n_init:int=5,
    non_diseased_ids:List[int]=[]
    ) -> Tuple[float, float, List[int]]:
    assert algorithm in ['ucl_gmm', 'ucl_kde'], f"Algorithm must be chosen from ['ucl_gmm', 'ucl_kde']!"

    start_time = time.time()

    results_folder = os.path.join(output_dir, algorithm, 'results')
    os.makedirs(results_folder, exist_ok=True)

    X, y, bmname, cname = extract_info_ucl(data_matrix)
    if algorithm == 'ucl_kde':
        mixture_models = mixture_model.fit_all_kde_models(X, y)
    else:
        mixture_models = mixture_model.fit_all_gmm_models(X, y)
    res = mcmc.mcmc(X, mixture_models, n_iter, greedy_n_iter, greedy_n_init)
    ml_stages = ebm_staging(
        x=X,
        mixtures=mixture_models,
        samples=res
    )
    qwk = cohen_kappa_score(true_stages, ml_stages, weights='quadratic')
    mae = mean_absolute_error(true_stages, ml_stages)
    mse = mean_squared_error(true_stages, ml_stages)
    rmse = math.sqrt(mse)

    n_participants = X.shape[0]
    ml_stages_diseased = []
    true_stages_diseased = []
    for pid in range(n_participants):
        if pid not in non_diseased_ids:
            ml_stages_diseased.append(ml_stages[pid])
            true_stages_diseased.append(true_stages[pid])
    
    qwk2 = cohen_kappa_score(true_stages_diseased, ml_stages_diseased, weights='quadratic')
    mae2 = mean_absolute_error(true_stages_diseased, ml_stages_diseased)
    mse2 = mean_squared_error(true_stages_diseased, ml_stages_diseased)
    rmse2 = math.sqrt(mse2)

    # assume our reformated data follows the order of BM1, BM2, BM3, ...
    # their ml_order is in this way
    # for example, [2 1 4 3 0]
    # means that BM3 (i.e., 2+1) is in order 1, 
    # BM2 (1+1) is in order 2, BM5(4+1) is in order 3, BM4(3+1) is in order 4
    # and the real order should be [0, 1, 2, 3, 4] if our biomarkers are sorted based on their order (ascending)

    # res.sort(reverse=True) because the original author used it:
    # https://github.com/ucl-pond/kde_ebm/blob/master/examples/basic_example.py
    res.sort(reverse=True)
    ml_order = res[0].ordering  
    tau, p_value = kendalltau(ml_order, range(0, len(ml_order)))

    # This is order by which biomarkers appear in the data_matrix
    desired_order = sorted(true_order_dict, key=true_order_dict.get)

    end_time = time.time()

    result_dict = {
        'algorithm': algorithm,
        "runtime": end_time - start_time,
        'N_MCMC': n_iter,
        "NStartpoints": greedy_n_init,
        "NIterations": greedy_n_iter,
        'kendalls_tau': tau,
        'p_value': p_value,
        "quadratic_weighted_kappa": qwk,
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "quadratic_weighted_kappa_diseased": qwk2,
        "mean_absolute_error_diseased": mae2,
        "mean_squared_error_diseased": mse2,
        "root_mean_squared_error_diseased": rmse2,
        'true_order': {bm: true_order_dict[bm] for bm in desired_order},
        'ml_order': {bm: ml_order[idx]+ 1 for idx, bm in enumerate(desired_order)},
        'ml_order_raw': ml_order,
        "true_stages": true_stages,
        'ml_stages': ml_stages,
        "true_stages_diseased": true_stages_diseased,
        "ml_stages_diseased": ml_stages_diseased
    }

    results_json = os.path.join(results_folder, f'{fname}_results.json')
    save_json(outfname = results_json, data=result_dict)
    print(f"Results saved to {results_json}")

    print_metrics(result_dict)

    return tau, p_value, mae, mae2

def run_debm(
    fname:str,
    algorithm:str,
    output_dir:str,
    DataIn:pd.DataFrame, 
    true_stages:List[int],
    true_order_dict:Dict[str, int],
    NStartpoints:Optional[int]=None, 
    Niterations:Optional[int]=None, 
    N_MCMC:Optional[int]=None,
    non_diseased_ids:List[int]=[]
    ):
    assert algorithm in ['debm', 'debm_gmm'], 'Algorithm not valid!'

    start_time = time.time()

    results_folder = os.path.join(output_dir, algorithm, 'results')
    os.makedirs(results_folder, exist_ok=True)

    if algorithm == 'debm':
        MethodOptions = namedtuple('MethodOptions', 'MixtureModel Bootstrap PatientStaging')
        my_method_options = MethodOptions(
            MixtureModel='GMMvv2',  # Default mixture model
            Bootstrap=0,  # No bootstrapping
            PatientStaging=['ml', 'p']  # 'ml' for discrete staging like Archetti 2019
        )

        # Now call debm.fit with these options
        ModelOutput, SubjTrainAll, SubjTestAll = debm.fit(
            DataIn=DataIn,
            MethodOptions=my_method_options,
            Factors=[]
        )
    else:
        MethodOptions = namedtuple('MethodOptions', 'NStartpoints Niterations N_MCMC MixtureModel Bootstrap PatientStaging')
        my_method_options = MethodOptions(
            NStartpoints=NStartpoints, 
            Niterations=Niterations, 
            N_MCMC=N_MCMC,
            MixtureModel='GMMvv2',  # or another mixture model option
            Bootstrap=0,  # or number of bootstrap iterations
            PatientStaging=['ml', 'p']  # 'ml' for discrete staging, 'exp' for continuous
        )

        # Now call ebm.fit with these options
        ModelOutput, SubjTrainAll, SubjTestAll = ebm.fit(
            DataIn=DataIn,
            MethodOptions=my_method_options,
            Factors=[]
        )
    ml_order = ModelOutput.MeanCentralOrdering
    ml_stages = SubjTrainAll[0]['Stages'].tolist()
    biomarkers_list = ModelOutput.BiomarkerList
    ml_order_dict = dict(zip(biomarkers_list, ml_order))

    # Sort by the keys
    ml_order_dict_sorted = sorted(ml_order_dict.items())
    # Sort by the keys 
    true_order_dict_sorted = sorted(true_order_dict.items())
    
    # Extract the values from the sorted tuples
    ml_order_values = [item[1] for item in ml_order_dict_sorted]
    true_order_values = [item[1] for item in true_order_dict_sorted]

    # Calculate Kendall's tau
    tau, p_value = kendalltau(ml_order_values, true_order_values)

    # Results based on ml_stages
    qwk = cohen_kappa_score(true_stages, ml_stages, weights='quadratic')
    mae = mean_absolute_error(true_stages, ml_stages)
    mse = mean_squared_error(true_stages, ml_stages)
    rmse = math.sqrt(mse)

    # Stages result for diseased participants only 
    n_participants = len(ml_stages)
    ml_stages_diseased = []
    true_stages_diseased = []
    for pid in range(n_participants):
        if pid not in non_diseased_ids:
            ml_stages_diseased.append(ml_stages[pid])
            true_stages_diseased.append(true_stages[pid])
    
    qwk2 = cohen_kappa_score(true_stages_diseased, ml_stages_diseased, weights='quadratic')
    mae2 = mean_absolute_error(true_stages_diseased, ml_stages_diseased)
    mse2 = mean_squared_error(true_stages_diseased, ml_stages_diseased)
    rmse2 = math.sqrt(mse2)

    # This is order by which biomarkers appear in the debm_output
    desired_order = sorted(true_order_dict, key=true_order_dict.get)

    end_time = time.time()
    
    result_dict = {
        'algorithm': algorithm,
        "runtime": end_time - start_time,
        'N_MCMC': N_MCMC,
        "NStartpoints": NStartpoints,
        "NIterations": Niterations,
        'kendalls_tau': tau,
        'p_value': p_value,
        "quadratic_weighted_kappa": qwk,
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "quadratic_weighted_kappa_diseased": qwk2,
        "mean_absolute_error_diseased": mae2,
        "mean_squared_error_diseased": mse2,
        "root_mean_squared_error_diseased": rmse2,
        'true_order': {bm: true_order_dict[bm] for bm in desired_order},
        'ml_order': {bm: ml_order[idx]+ 1 for idx, bm in enumerate(desired_order)},
        'ml_order_raw': ml_order,
        "true_stages": true_stages,
        'ml_stages': ml_stages,
        "true_stages_diseased": true_stages_diseased,
        "ml_stages_diseased": ml_stages_diseased
    }

    results_json = os.path.join(results_folder, f'{fname}_results.json')
    save_json(outfname = results_json, data=result_dict)
    print(f"Results saved to {results_json}")
    print_metrics(result_dict)

    return tau, p_value, mae, mae2