import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from memory_profiler import memory_usage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_wine, load_diabetes

import opns_pack.opns_np as op
import opns_pack.custom_gen_pairs as cgp
from opns_pack.opns import OPNs
from opns_sklearn.preprocessing import OPNsStandardScaler
from opns_sklearn.linear_model import Lasso, LinearRegression, LinearRegressionGradientDescent

# Global variables
FEATURE = []
INDEX = None


def find_best_model_by_r2(linear_metrics, lasso_metrics):
    r2_scores_info = [
        (linear_metrics['r2_a'], 'linear', 'a'),
        (linear_metrics['r2_ab'], 'linear', 'ab'),
        (lasso_metrics['r2_a'], 'lasso', 'a'),
        (lasso_metrics['r2_ab'], 'lasso', 'ab')
    ]

    best_score_info = max(r2_scores_info)
    
    best_r2 = best_score_info[0]
    best_model_name = best_score_info[1]
    best_suffix = best_score_info[2]
    
    if best_model_name == 'linear':
        source_dict = linear_metrics
    else:
        source_dict = lasso_metrics
        
    r2_key = f'r2_{best_suffix}'
    mse_key = f'mse_{best_suffix}'
    rmse_key = f'rmse_{best_suffix}'
    mae_key = f'mae_{best_suffix}'
    
    corresponding_r2 = source_dict[r2_key]
    corresponding_r2_std = source_dict[f'{r2_key}_std']
    corresponding_mse = source_dict[mse_key]
    corresponding_mse_std = source_dict[f'{mse_key}_std']
    corresponding_rmse = source_dict[rmse_key]
    corresponding_rmse_std = source_dict[f'{rmse_key}_std']
    corresponding_mae = source_dict[mae_key]
    corresponding_mae_std = source_dict[f'{mae_key}_std']

    return best_model_name, best_suffix, {
        'r2': f'{corresponding_r2:.4f} ± {corresponding_r2_std:.4f}',
        'mse': f'{corresponding_mse:.4f} ± {corresponding_mse_std:.4f}',
        'rmse': f'{corresponding_rmse:.4f} ± {corresponding_rmse_std:.4f}',
        'mae': f'{corresponding_mae:.4f} ± {corresponding_mae_std:.4f}'
    }


def run_experiment(alpha, nth_power, tri=0, test_size=[], dataset_name='', random_state=None):
    """Execute a single experiment workflow, return a dictionary of evaluation metrics"""
    assert dataset_name != '', 'Please check dataset_name.'
    global FEATURE
    try:
        # ============= Load Data =============
        from utils.functions import load_data
        if dataset_name == 'yacht':
            feature_names, X, y = load_data(dataset_name, file_type='data')
        else:
            feature_names, X, y = load_data(dataset_name, file_type='csv')

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scaler = MinMaxScaler(feature_range=(0, 1))
        y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # Create Dataset Object
        class TransDataset:
            def __init__(self, data, target, feature_names):
                self.data = data
                self.target = target
                self.feature_names = feature_names

        dataset = TransDataset(X_scaled, y, feature_names)
        
        # ============= Feature Engineering =============
        X_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        original_feature = X_df.columns.values
        X_df['zero'] = 0
        
        # Generate feature combinations 
        feature_to_pair_1 = np.array(cgp.linear_pair(original_feature))
        feature_to_pair_2 = np.array(cgp.all_pair(original_feature) + cgp.linear_pair(original_feature))

        # Build feature matrix
        X_1 = cgp.data_convert(X_df, feature_to_pair_1)
        if nth_power >= 2:
            X_2 = cgp.data_convert(X_df, feature_to_pair_2, poly=nth_power, tri=tri, linear_term=False, bias=False)
            X = op.hstack(X_1, X_2)
        elif tri != 0:
            X_2 = cgp.data_convert(X_df, feature_to_pair_2, tri=tri, linear_term=False, bias=False)
            X = op.hstack(X_1, X_2)
        else:
            X = X_1

        # ============= Data split =============
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size[0], random_state=random_state
        )

        # ============= OPNs Data Standard =============
        scaler = OPNsStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ============= Lasso Model =============
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train_scaled, y_train)
        y_pred = lasso.predict(X_test_scaled)
        y_pred_1 = lasso.predict(X_test_scaled, item=1)

        # Collect metrics of lasso model
        lasso_metrics = {
            'r2_a': r2_score(y_test, y_pred),
            'r2_ab': r2_score(y_test, y_pred_1),
            'mse_a': mean_squared_error(y_test, y_pred),
            'mse_ab': mean_squared_error(y_test, y_pred_1),
            'rmse_a': np.sqrt(mean_squared_error(y_test, y_pred)),
            'rmse_ab': np.sqrt(mean_squared_error(y_test, y_pred_1)),
            'mae_a': mean_absolute_error(y_test, y_pred),
            'mae_ab': mean_absolute_error(y_test, y_pred_1),
        }

        # ============= Feature analysis =============
        coefficients = lasso.coef_
        feature_to_pair = feature_to_pair_1.tolist()
        for _ in range(nth_power - 1):
            feature_to_pair += feature_to_pair_2.tolist()
        if tri != 0:
            feature_to_pair += feature_to_pair_2.tolist()

        feature_names = list(zip(feature_to_pair[::2], feature_to_pair[1::2]))
        selected_features = []
        for name, coef in zip(feature_names, coefficients):
            if coef != OPNs(0, 0):
                selected_features.append((name, coef))
        
        # ============= Linear model validation =============
        def split_list_by_zero(input_list, n):
            split_indices = []
            current_block_start = -1
            prev_has_zero = False

            for index, item in enumerate(input_list):
                current_has_zero = 'zero' in item
                if current_has_zero:
                    if not prev_has_zero:
                        current_block_start = index
                else:
                    if prev_has_zero:
                        split_indices.append(index - 1)
                        current_block_start = -1
                prev_has_zero = current_has_zero

            # Handle the case ending with 'zero'
            if current_block_start != -1:
                split_indices.append(len(input_list) - 1)

            max_possible_n = len(split_indices) + 1

            if n > max_possible_n:  
                print(f"\nCannot split into {n} lists, maximum possible is {max_possible_n}")
                return []

            # Generate split points
            splits_needed = n - 1
            selected_splits = split_indices[:splits_needed]

            # Split the list
            sublists = []
            start = 0
            for split in selected_splits:
                end = split + 1
                sublists.append(input_list[start:end])
                start = end
            sublists.append(input_list[start:])

            return sublists

        selected_features_name = [tup[0] for tup in selected_features]
        ndiv = nth_power if tri == 0 else nth_power + 1
        feature_list = split_list_by_zero(selected_features_name, ndiv)
        FEATURE.append(feature_list)

        # Feature transformation
        X_linear = cgp.data_convert(X_df, feature_list[0])
        for i in range(2, nth_power + 1):
            if i == nth_power and tri != 0:
                X_linear_temp = cgp.data_convert(X_df, feature_list[i], tri=tri, linear_term=False)
            else:
                X_linear_temp = cgp.data_convert(X_df, feature_list[i - 1], poly=i, poly_only=True, bias=False)
            X_linear = op.concatenate(X_linear, X_linear_temp, axis=1)

        # Linear model training
        X_linear_scaled = scaler.fit_transform(X_linear)
        X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(
            X_linear_scaled, y, test_size=test_size[0], random_state=random_state
        )
        linear = LinearRegressionGradientDescent()
        linear.fit(X_train_linear, y_train_linear)
        y_pred_linear = linear.predict(X_test_linear)
        y_pred_linear_1 = linear.predict(X_test_linear, item=1)

        # Collect linear model metrics
        linear_metrics = {
            'r2_a': r2_score(y_test_linear, y_pred_linear),
            'r2_ab': r2_score(y_test_linear, y_pred_linear_1),
            'mse_a': mean_squared_error(y_test_linear, y_pred_linear),
            'mse_ab': mean_squared_error(y_test_linear, y_pred_linear_1),
            'rmse_a': np.sqrt(mean_squared_error(y_test_linear, y_pred_linear)),
            'rmse_ab': np.sqrt(mean_squared_error(y_test_linear, y_pred_linear_1)),
            'mae_a': mean_absolute_error(y_test_linear, y_pred_linear),
            'mae_ab': mean_absolute_error(y_test_linear, y_pred_linear_1),
        }

        return {'lasso': lasso_metrics, 'linear': linear_metrics}

    except Exception as e:
        print(f"Experiment failed with random_state {random_state}: {str(e)}")
        return None


def analyze_memory(dataset_name, params):
    """Execute memory analysis and plot chart"""
    try:
        raw_data, exec_time = memory_usage(
            (run_experiment, (), {'dataset_name': dataset_name, **params}),
            interval=0.1,
            timestamps=True,
            retval=True,
            include_children=True,
            max_usage=False
        )

        # Data processing
        mem_curve = [m for m, t in raw_data]
        timestamps = [t for m, t in raw_data]
        start_time = timestamps[0]
        relative_times = [t - start_time for t in timestamps]
        peak_value = max(mem_curve)
        peak_time = relative_times[mem_curve.index(peak_value)]
        print(f"Start generate graph of Memory Usage")
        # Plot chart
        plt.figure(figsize=(12, 6))
        plt.plot(relative_times, mem_curve, label='Memory Usage', color='#2c7bb6')
        plt.scatter(peak_time, peak_value, color='red', label=f'Peak: {peak_value:.1f} MB')
        plt.title('Memory Usage Timeline')
        plt.xlabel('Time (s)')
        plt.ylabel('Memory (MB)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Memory analysis failed: {str(e)}")

def calculate_mean_std(model_metrics):
    linear_model, lasso_model = {}, {}

    metrics = ['r2', 'mse' ,'rmse', 'mae']
    model_list = ['lasso', 'linear']
    identity = None
    for model in model_list:
        temp = model_metrics[model]
        for k, v in temp.items():
            if model == 'linear':
                linear_model[k] = np.mean(v)
                linear_model[f'{k}_std'] = np.std(v)
            elif model == 'lasso':
                lasso_model[k] = np.mean(v)
                lasso_model[f'{k}_std'] = np.std(v)
                
    identity, best_suffix, best = find_best_model_by_r2(linear_metrics=linear_model, lasso_metrics=lasso_model)
    return identity, best_suffix, best


def get_index(model_metrics, identity, best_suffix):
    global INDEX
    global FEATURE
    max_value = max(model_metrics[identity][f'r2_{best_suffix}'])
    max_value_idx = model_metrics[identity][f'r2_{best_suffix}'].index(max_value)
    INDEX = max_value_idx
    
if __name__ == "__main__":
    import argparse
    import warnings
    import importlib.resources as pkg_resources
    import utils

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    with pkg_resources.files(utils).joinpath("configs/config.json").open("r", encoding="utf-8") as f:
        config_args = json.load(f)
    with pkg_resources.files(utils).joinpath("configs/default_params.json").open("r", encoding="utf-8") as f:
        default_args = json.load(f)
    with pkg_resources.files(utils).joinpath("configs/test_data_params.json").open("r", encoding="utf-8") as f:
        test_args = json.load(f)
    
    parser.add_argument('--dataset', type=str, default='default', choices=['abalone', 'bike', 'boston', 'concrete',
                                                                      'diabetes', 'energy_cooling', 'energy_heating',
                                                                      'folds', 'wine', 'yacht', 'default'], 
                                                                      help='Name of dataset. You can add your personal dataset to choice list.')
    parser.add_argument('--memory', action='store_true', help='Enable memory analysis (default: False)')

    args = parser.parse_args()

    if args.dataset in test_args.keys():
        params = test_args[args.dataset]
        print(f'Use pre-setting parameters about {args.dataset}.')
    else:
        params = default_args['default']
        print(f'Use default parameters.')
    
    # ============= Multiple experiment runs =============
    n_runs = config_args['n_runs']
    results = []
    time_result = []

    start_text = f"Start Running {n_runs} experiments"
    end_text = f"End Running {n_runs} experiments"

    print(start_text.center(100, '='))
    for seed in tqdm(range(n_runs), desc="Running single experiments"):
        start_time = time.time()
        result = run_experiment(**params, dataset_name=args.dataset, random_state=seed)
        if result:
            results.append(result)
            time_result.append(time.time() - start_time)

    # ============= Statistical results =============
    if results:
        # Initialize metrics collector
        metrics = {
            'lasso': {k: [] for k in results[0]['lasso']},
            'linear': {k: [] for k in results[0]['linear']}
        }
        
        # Aggregate data
        for res in results:
            for model in ['lasso', 'linear']:
                for k, v in res[model].items():
                    metrics[model][k].append(v)
        
        identity, best_suffix, best_result = calculate_mean_std(metrics)
        get_index(metrics, identity, best_suffix)

        if len(FEATURE[INDEX]) != 0:
            print()
            features_text = f"Selected Features"
            print(features_text.center(50, '='))
            for i, features in enumerate(FEATURE[INDEX]):
                if i != len(FEATURE[INDEX]) - 1:
                    print(f"Feature of {i + 1}th power: \n{FEATURE[INDEX][i]}\n")
                else:
                    print(f"Feature of tri power: \n{FEATURE[INDEX][i]}")

        print()
        perference_text = f"Performance (Mean ± Std)"
        print(perference_text.center(50, '='))
        for k, v in best_result.items():
            print(f"{k:8}: {v}")
        print(f"{'time':8}: {np.mean(time_result):.4f} ± {np.std(time_result):.4f}")
    else:
        print("\nAll experiments failed!")

    # ============= Memory analysis =============
    if args.memory:
        print()
        memory_text = f"Running memory analysis"
        print(memory_text.center(50, '='))
        analyze_memory(args.dataset, params)
    
    print(end_text.center(80, '='))
