import pandas as pd 
import numpy as np 
import os 
from typing import List, Dict, Tuple, Optional
import copy 
import altair as alt
import run 
import matplotlib.pyplot as plt 
import seaborn as sns 
from pyebm import debm
from pyebm import ebm
from kde_ebm import mixture_model
from kde_ebm import mcmc
from collections import defaultdict, namedtuple, Counter
import time 

def get_adni_filtered(raw:str, meta_data:List[str], select_biomarkers:List[str], diagnosis_list:List[str]) -> pd.DataFrame:
    """Get the filtered data. 
    meta_data = ['PTID', 'DX_bl', 'VISCODE', 'COLPROT']

    select_biomarkers = ['MMSE_bl', 'Ventricles_bl', 'WholeBrain_bl', 
                'MidTemp_bl', 'Fusiform_bl', 'Entorhinal_bl', 
                'Hippocampus_bl', 'ADAS13_bl', 'PTAU_bl', 
                'TAU_bl', 'ABETA_bl', 'RAVLT_immediate_bl'
    ]

    diagnosis_list = ['CN', 'EMCI', 'LMCI', 'AD']
    """
    df = pd.read_csv(raw, usecols=meta_data + select_biomarkers)
    # 2. Filter to baseline and known diagnoses
    df = df[df['VISCODE'] == 'bl']
    df = df[df['DX_bl'].isin(diagnosis_list)]

    # 3. Convert biomarker columns to numeric (handles garbage strings like '--')
    for col in select_biomarkers:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Drop rows with any NaN in biomarkers
    df = df.dropna(subset=select_biomarkers).reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)
    print(len(df))
    if len(df.PTID.unique()) == len(df):
        print('No duplicates!')
    else:
        print('Data has duplicates!')
    
    # Print DX distribution
    counts = Counter(df['DX_bl'])
    total = sum(counts.values())

    for k, v in counts.items():
        perc = 100 * v / total
        print(f"{k}: {v} ({perc:.1f}%)")
    
    print('----------------------------------------------------')
    
    # Print Cohort distribution
    counts = Counter(df['COLPROT'])
    total = sum(counts.values())

    for k, v in counts.items():
        perc = 100 * v / total
        print(f"{k}: {v} ({perc:.1f}%)")

    return df 

def process_data(df:pd.DataFrame, ventricles_log:bool, tau_log:bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, str], List[str]]:
    """To get the required output for debm, ucl, and sa-ebm
    df: adni_filtered
    """
    df['PTID'] = range(len(df))
    # df['Diagnosis'] = ['MCI' if x in ['EMCI', 'LMCI'] else x for x in df.DX_bl]
    df['Diagnosis'] = df.DX_bl
    df.columns = df.columns.str.replace('_bl', '', regex=False)
    df['Diagnosis'] = df.DX
    if tau_log:
        df['TAU (log)'] = np.log10(df['TAU'])
        df['PTAU (log)'] = np.log10(df['PTAU'])
        df.drop(['TAU', 'PTAU'], axis=1, inplace=True)
    if ventricles_log:
        df['Ventricles (log)'] = np.log10(df['Ventricles'])
        df.drop(['Ventricles'], axis=1, inplace=True)
    participant_dx_dict = dict(zip(df.PTID, df.DX))
    df.drop(['VISCODE', 'COLPROT', 'DX'], axis=1, inplace=True)
    # for debm
    debm_output = df.copy()
    df.drop(['Diagnosis', 'PTID'], axis=1, inplace=True)
    # Ordered biomarkers, to match the ordering outputs later
    ordered_biomarkers = list(df.columns)
    df['diseased'] = [int(dx == 'AD') for dx in participant_dx_dict.values()]
    # df['diseased'] = [int(dx != 'CN') for dx in participant_dx_dict.values()]
    # for ucl
    data_matrix = copy.deepcopy(df.to_numpy())
    df['participant'] = range(len(df))
    df['diseased'] = [bool(x) for x in df.diseased]
    df_long = pd.melt(
        df,
        id_vars=['participant', 'diseased'],       # columns to keep fixed
        var_name='biomarker',              # name for former column names
        value_name='measurement'                 # name for the measured values
    )
    return debm_output, data_matrix, df_long, participant_dx_dict, ordered_biomarkers

def run_ucl_gmm(
        output_dir:str,
        data_matrix:np.ndarray, 
        ordered_biomarkers:List[str],
        n_iter:int=1_0000, 
        greedy_n_iter:int=10, 
        greedy_n_init:int=5,
    ):
    start = time.time()
    results_folder = os.path.join(output_dir, 'ucl_gmm', 'results')
    os.makedirs(results_folder, exist_ok=True)

    X, y, feature_names = data_matrix[:, :-1], data_matrix[:, -1].astype(int), np.array(ordered_biomarkers)
    mixture_models = mixture_model.fit_all_gmm_models(X, y)
    res = mcmc.mcmc(X, mixture_models, n_iter = n_iter,
                    greedy_n_init=greedy_n_init,
                    greedy_n_iter=greedy_n_iter,)
    ml_stages = run.ebm_staging(
        x=X,
        mixtures=mixture_models,
        samples=res
    )
    res.sort(reverse=True)
    ml_order = res[0].ordering 

    save_debm_heatmap(
        bootstrap_orderings=[sample.ordering for sample in res],
        biomarker_names=feature_names,
        folder_name=results_folder,
        file_name="ucl_gmm_heatmap",
        title="UCL GMM Ordering Result",
        mean_ordering=ml_order,
        plot_mean_order=True
    )

    end = time.time()

    result_dict = {
        'algorithm': 'UCL GMM',
        'runtime': end - start, 
        'N_MCMC': n_iter,
        "NStartpoints": greedy_n_init,
        "NIterations": greedy_n_iter,
        'ml_order': dict(zip(feature_names, [x+1 for x in ml_order])),
        'ml_order_raw': ml_order,
        'ml_stages': ml_stages
    }

    results_json = os.path.join(results_folder, f'results.json')
    run.save_json(outfname = results_json, data=result_dict)
    print(f"Results saved to {results_json}")

    return result_dict

def run_debm(
    algorithm:str,
    output_dir:str,
    DataIn:pd.DataFrame, 
    NStartpoints:Optional[int]=None, 
    Niterations:Optional[int]=None, 
    N_MCMC:Optional[int]=None,
    ):
    start = time.time()

    assert algorithm in ['debm', 'debm_gmm'], 'Algorithm not valid!'

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
    biomarker_names = ModelOutput.BiomarkerList

    end = time.time()

    result_dict = {
        'algorithm': algorithm,
        'runtime': end - start, 
        'N_MCMC': N_MCMC,
        "NStartpoints": NStartpoints,
        "NIterations": Niterations,
        'ml_order': dict(zip(biomarker_names, [x+1 for x in ml_order])),
        'ml_order_raw': ml_order,
        'ml_stages': ml_stages
    }

    results_json = os.path.join(results_folder, f'results.json')
    run.save_json(outfname = results_json, data=result_dict)
    print(f"Results saved to {results_json}")

    return result_dict

def save_debm_heatmap(
    bootstrap_orderings: List,  # List of orderings from bootstrap iterations
    biomarker_names: List[str],
    folder_name: str,
    file_name: str,
    title: str,
    mean_ordering: Optional[List[int]] = None,
    plot_mean_order: bool = True  # Whether to sort by mean ordering
):
    """
    Create a heatmap showing the positional variance of biomarkers across bootstrap iterations.
    Similar to what's shown in Archetti 2019 and the pyebm documentation.
    
    Args:
        bootstrap_orderings: List of orderings from each bootstrap iteration
        biomarker_names: List of biomarker names
        folder_name: Directory to save the plot
        file_name: Name of the output file
        title: Title for the plot
        mean_ordering: Mean ordering across all bootstraps (if not provided, will be calculated)
        plot_mean_order: If True, sort biomarkers by their mean position
    """
    os.makedirs(folder_name, exist_ok=True)
    
    n_biomarkers = len(biomarker_names)
    n_bootstraps = len(bootstrap_orderings)
    
    # Create a matrix to count occurrences of each biomarker at each position
    position_counts = np.zeros((n_biomarkers, n_biomarkers))
    
    # Count how many times each biomarker appears at each position
    for ordering in bootstrap_orderings:
        for position, biomarker_idx in enumerate(ordering):
            position_counts[biomarker_idx, position] += 1
    
    # Convert to DataFrame
    biomarker_position_df = pd.DataFrame(
        position_counts,
        index=biomarker_names,
        columns=range(1, n_biomarkers + 1)  # Stage positions 1 to N
    )
    
    # If mean ordering is provided or we want to plot by mean order
    if plot_mean_order:
        
        # Reorder biomarkers by mean ordering
        ordered_biomarker_names = [biomarker_names[i] for i in mean_ordering]
        biomarker_position_df = biomarker_position_df.loc[ordered_biomarker_names]
        
        # Add ordering numbers to biomarker names
        renamed_index = [f"{name} ({i+1})" for i, name in enumerate(ordered_biomarker_names)]
        biomarker_position_df.index = renamed_index
    
    # Normalize to show proportions/probabilities
    biomarker_position_df = biomarker_position_df.div(n_bootstraps)
    
    # Find the longest biomarker name
    max_name_length = max(len(name) for name in biomarker_position_df.index)
    
    # Dynamically adjust figure size
    fig_width = max(10, min(20, n_biomarkers * 0.5))  # Scale with number of biomarkers
    fig_height = max(8, min(15, n_biomarkers * 0.4))
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create heatmap
    sns.heatmap(
        biomarker_position_df,
        annot=True,
        cmap="Blues",  # Use Blues to match pyebm visualization
        linewidths=0.5,
        cbar_kws={'label': 'Probability'},
        fmt=".2f",
        vmin=0,
        vmax=1
    )
    
    plt.xlabel('Stage Position')
    plt.ylabel('Biomarker')
    plt.title(title)
    
    # Adjust y-axis ticks
    plt.yticks(rotation=0, ha='right')
    
    # Adjust margins
    plt.subplots_adjust(left=0.3 if max_name_length > 20 else 0.2)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{folder_name}/{file_name}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{folder_name}/{file_name}.png", bbox_inches="tight", dpi=300)
    plt.close()
    
    return biomarker_position_df

def run_debm_with_bootstrap_and_plot(
        algorithm:str,
        output_dir:str,
        DataIn:pd.DataFrame, 
        NStartpoints:Optional[int]=None, 
        Niterations:Optional[int]=None, 
        N_MCMC:Optional[int]=None, 
        n_bootstraps:int=100,
        plot_title:str="",
    ):
    """
    Run DEBM with bootstrap and create the heatmap visualization.
    """    
    start = time.time()

    results_folder = os.path.join(output_dir, algorithm, 'results')
    os.makedirs(results_folder, exist_ok=True)

    if algorithm == 'debm':
        MethodOptions = namedtuple('MethodOptions', 'MixtureModel Bootstrap PatientStaging')
        my_method_options = MethodOptions(
            MixtureModel='GMMvv2',  # Default mixture model
            Bootstrap=n_bootstraps,  # No bootstrapping
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
            Bootstrap=n_bootstraps,  # or number of bootstrap iterations
            PatientStaging=['ml', 'p']  # 'ml' for discrete staging, 'exp' for continuous
        )

        # Now call ebm.fit with these options
        ModelOutput, SubjTrainAll, SubjTestAll = ebm.fit(
            DataIn=DataIn,
            MethodOptions=my_method_options,
            Factors=[]
        )

    # Extract bootstrap orderings
    bootstrap_orderings = ModelOutput.CentralOrderings
    mean_ordering = ModelOutput.MeanCentralOrdering
    biomarker_names = ModelOutput.BiomarkerList
    
    # Create heatmap
    save_debm_heatmap(
        bootstrap_orderings=bootstrap_orderings,
        biomarker_names=biomarker_names,
        folder_name=results_folder,
        file_name="debm_bootstrap_heatmap",
        title=plot_title,
        mean_ordering=mean_ordering,
        plot_mean_order=True
    )

    print(f'Plots saved to file for {algorithm}!')
    
def plot_staging(ml_stages:List[int], participant_dx_dict:Dict[int, str], algorithm:str):

    # DataFrame preparation
    df = pd.DataFrame({
        'Stage': ml_stages,
        'Diagnosis': list(participant_dx_dict.values())
    })
    diagnosis_order = ['CN', 'EMCI', 'LMCI', 'AD']
    stage_range = list(range(df['Stage'].min(), df['Stage'].max() + 1))

    # Count table with missing combinations filled
    count_df = df.groupby(['Stage', 'Diagnosis']).size().reset_index(name='Count')
    all_combinations = pd.MultiIndex.from_product([stage_range, diagnosis_order], names=['Stage', 'Diagnosis'])
    count_df = count_df.set_index(['Stage', 'Diagnosis']).reindex(all_combinations, fill_value=0).reset_index()

    # Calculate stage totals for reference
    stage_totals = count_df.groupby('Stage')['Count'].sum().reset_index()
    stage_totals = stage_totals.rename(columns={'Count': 'Total'})
    count_df = pd.merge(count_df, stage_totals, on='Stage')

    # Calculate percentage for tooltips while keeping absolute counts for display
    count_df['Percentage'] = count_df['Count'] / count_df['Total']

    # Paul Tol's colorblind-friendly palette (scientific standard)
    color_scale = alt.Scale(
        domain=['CN', 'EMCI', 'LMCI', 'AD'],
        range=['#4477AA', '#66CCEE', '#228833', '#EE6677']  # blue, cyan, green, red
    )

    # Define the base chart with improved typography and sizing
    base = alt.Chart(count_df).properties(
        width=500,
        height=300,
        title={
            'text': f'Distribution of Disease Stages by Diagnosis, {algorithm}',
            'anchor': 'middle',
            'fontSize': 14,
            'dy': -10
        }
    )

    # Main stacked bar chart with absolute counts
    bars = base.mark_bar().encode(
        x=alt.X('Stage:O', 
                title='',
                axis=alt.Axis(
                    labelFontSize=11,
                    titleFontSize=12,
                    titleFont='Arial',
                    titlePadding=15,
                    grid=False
                )
        ),
        y=alt.Y('Count:Q', 
                title='Number of Participants',  # Changed to reflect absolute counts
                axis=alt.Axis(
                    labelFontSize=11,
                    titleFontSize=12,
                    titleFont='Arial',
                    grid=True,
                    gridOpacity=0.4,
                    titlePadding=15
                )
        ),
        color=alt.Color('Diagnosis:N', 
                        scale=color_scale,
                        legend=alt.Legend(
                            title=None,
                            labelFontSize=11,
                            symbolSize=100,
                            orient='top',
                            direction='horizontal',
                            columns=4
                        )
        ),
        order=alt.Order('Diagnosis:N', sort='ascending'),
        tooltip=[
            alt.Tooltip('Stage:O', title='Stage'),
            alt.Tooltip('Diagnosis:N', title='Diagnosis'),
            alt.Tooltip('Count:Q', title='Count'),
            alt.Tooltip('Percentage:Q', title='Percentage', format='.1%')
        ]
    )

    # Add text labels showing total sample sizes
    text = base.mark_text(
        align='center',
        baseline='bottom',
        dy=-5,
        fontSize=9
    ).encode(
        x='Stage:O',
        y=alt.value(20),  # Fixed position at the top
        text=alt.Text('Total:Q', format=',d'),
        tooltip=[
            alt.Tooltip('Stage:O', title='Stage'),
            alt.Tooltip('Total:Q', title='Total Sample Size', format=',d')
        ]
    )

    # Combine the chart elements
    final_chart = (bars + text).configure_view(
        stroke='lightgray',
        strokeWidth=0.5
    ).configure_axis(
        titleFont='Arial',
        labelFont='Arial',
        grid=True,
        gridColor='lightgray',
        gridOpacity=0.3,
        domain=True,
        domainColor='black',
        domainWidth=0.5,
        labelColor='black',
        titleColor='black'
    ).properties(
        padding={'left': 10, 'top': 30, 'right': 10, 'bottom': 40}
    )

    return final_chart