# -*- coding: utf-8 -*-

import pandas as pd
import os

__all__ = ['load_data']

def load_data(dataset_name: str, file_type: str='csv'):
    assert dataset_name is not None and dataset_name != '', f'dataset_name: \'{dataset_name}\'. Please check it.'

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, '..', 'dataset')

    file_path = os.path.join(dataset_dir, f'{dataset_name}.{file_type}')
    dataset_loaders = {
        'abalone': _load_abalone,
        'bike': _load_bike,
        'boston': _load_boston,
        'concrete': _load_concrete,
        'diabetes': _load_diabetes,
        'energy_cooling': _load_cool,
        'energy_heating': _load_heat,
        'folds': _load_folds,
        'wine': _load_wine,
        'yacht': lambda df: _load_yacht(pd.read_csv(file_path, sep=r'\s+', header=0))
    }

    try:
        if dataset_name not in dataset_loaders:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        df = pd.read_csv(file_path)
        feature_names, X, y = dataset_loaders[dataset_name](df)

    except FileNotFoundError:
        raise FileNotFoundError(f"File not exist: {file_path}.")
    except ValueError as e:
        raise e
    return feature_names, X, y


def _load_abalone(df:pd.DataFrame):
    df['Sex'] = df['Sex'].map({'M': 1, 'F': -1, 'I': 0})
    feature_names = df.columns[:-1].tolist()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    return feature_names, X, y

def _load_bike(df:pd.DataFrame):
    feature_names = df.columns[:-1].tolist()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    return feature_names, X, y

def _load_boston(df:pd.DataFrame):
    pass

def _load_concrete(df:pd.DataFrame):
    feature_names = df.columns[:-1].tolist()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    return feature_names, X, y

def _load_diabetes(df:pd.DataFrame):
    feature_names = df.columns[:-1].tolist()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    return feature_names, X, y
    # print(pd.DataFrame(X, columns=feature_names))
    # print('0-0----------------------')
    # from sklearn.datasets import load_diabetes
    # dataset = load_diabetes()
    # X_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    # print(X_df)
    # exit()
    # feature_names = dataset.feature_names  # or list(X_df.columns)
    # X = X_df.values  # already excludes target column
    # y = dataset.target.reshape(-1, 1)
    # return feature_names, X, y

def _load_cool(df:pd.DataFrame):
    feature_names = df.columns[:-1].tolist()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    return feature_names, X, y

def _load_heat(df:pd.DataFrame):
    feature_names = df.columns[:-1].tolist()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    return feature_names, X, y
 
def _load_folds(df:pd.DataFrame):
    df = df.sample(frac=0.1, random_state=42)
    feature_names = df.columns[:-1].tolist()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    return feature_names, X, y

def _load_wine(df:pd.DataFrame):
    feature_names = df.columns[:-1].tolist()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    return feature_names, X, y

def _load_yacht(df:pd.DataFrame):
    feature_names = df.columns[:-1].tolist()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    return feature_names, X, y
