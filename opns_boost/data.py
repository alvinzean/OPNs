"""Centralized dataset loading, preprocessing, and OPNs conversion.

V2 notes
--------
This version uses dataset-specific metadata rather than assuming every file has
`features = all columns except the last one` and `target = last column`.
It keeps the original top-level `dataset/`, `OPNs/`, and `opns_sklearn/` folders
unchanged, and only centralizes how OPNs-Boost reads and converts datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import opns_pack.custom_gen_pairs as cgp
import opns_pack.opns_np as op
from opns_pack.opns import OPNs
from opns_pack.opns_matrix import OPNsMatrix


PairingMode = Literal["combinations", "permutations", "linear"]
TaskType = Literal["regression", "classification"]
FeatureScale = Literal["none", "standard", "minmax"]


@dataclass
class TabularDataset:
    name: str
    task_type: TaskType
    X: pd.DataFrame
    y: np.ndarray
    feature_names: list[str]
    target_name: str = "target"
    class_names: Optional[list[str]] = None


@dataclass
class PreparedOPNsData:
    dataset: TabularDataset
    X_opns: object
    y_model: object
    pair_features: list[str]
    candidate_pairs: int
    y_scaler: Optional[MinMaxScaler] = None


@dataclass
class PreparedRegressionFold:
    """One train/test fold after train-fitted feature preparation."""

    X_train_frame: pd.DataFrame
    X_test_frame: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_opns: object | None
    X_test_opns: object | None
    pair_features: list[str]
    candidate_pairs: int
    imputation_metadata: dict[str, object]


def dataset_dir() -> Path:
    return PROJECT_ROOT / "dataset"


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "missing")
    return df


def _encode_features(
    X: pd.DataFrame,
    categorical_maps: Optional[dict[str, dict]] = None,
    *,
    impute_missing: bool = True,
) -> pd.DataFrame:
    """Encode feature columns while optionally preserving numeric missingness.

    ``impute_missing=False`` is used only when a runner will fit the imputer
    after the cross-validation split. Dataset-specific ordinal maps are still
    deterministic and therefore may be applied before the split.
    """
    out = X.copy()
    categorical_maps = categorical_maps or {}
    for col, mapping in categorical_maps.items():
        if col in out.columns:
            out[col] = out[col].map(mapping)
    for col in out.columns:
        if out[col].dtype == "object" or str(out[col].dtype).startswith("category"):
            # Preserve missing cells when deferring imputation. Non-missing
            # categories receive a deterministic sorted-label encoding.
            missing = out[col].isna()
            encoder = LabelEncoder()
            encoded = pd.Series(np.nan, index=out.index, dtype=float)
            if (~missing).any():
                encoded.loc[~missing] = encoder.fit_transform(
                    out.loc[~missing, col].astype(str)
                ).astype(float)
            out[col] = encoded
    out = out.replace([np.inf, -np.inf], np.nan)
    if impute_missing:
        out = _clean_frame(out)
    return out.astype(float)


def fit_fold_local_median_imputer(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Fit numeric medians on ``X_train`` and apply them to both fold parts."""
    train = X_train.copy().replace([np.inf, -np.inf], np.nan)
    test = X_test.copy().replace([np.inf, -np.inf], np.nan)
    if list(train.columns) != list(test.columns):
        raise ValueError("Train/test feature columns differ.")

    non_numeric = [
        str(col)
        for col in train.columns
        if not pd.api.types.is_numeric_dtype(train[col])
    ]
    if non_numeric:
        raise TypeError(
            "Fold-local median imputation requires encoded numeric features; "
            f"non-numeric columns: {non_numeric}"
        )

    train_missing = train.isna().sum()
    test_missing = test.isna().sum()
    medians = train.median(axis=0, skipna=True)
    invalid = [str(col) for col in medians.index if pd.isna(medians[col])]
    if invalid:
        raise ValueError(
            "Training-fold median is undefined for all-missing columns: "
            f"{invalid}"
        )

    train = train.fillna(medians).astype(float)
    test = test.fillna(medians).astype(float)
    if train.isna().any().any() or test.isna().any().any():
        raise RuntimeError("Fold-local imputation left missing feature values.")

    columns_with_missing = [
        str(col)
        for col in train.columns
        if int(train_missing[col]) + int(test_missing[col]) > 0
    ]
    metadata: dict[str, object] = {
        "policy": "training_fold_numeric_median",
        "columns_with_missing": columns_with_missing,
        "medians": {str(col): float(medians[col]) for col in medians.index},
        "missing_train_by_column": {
            str(col): int(train_missing[col])
            for col in train_missing.index
            if int(train_missing[col]) > 0
        },
        "missing_test_by_column": {
            str(col): int(test_missing[col])
            for col in test_missing.index
            if int(test_missing[col]) > 0
        },
        "missing_train_total": int(train_missing.sum()),
        "missing_test_total": int(test_missing.sum()),
    }
    return train.reset_index(drop=True), test.reset_index(drop=True), metadata


def _maybe_scale_features(X: pd.DataFrame, scale: FeatureScale = "none") -> pd.DataFrame:
    if scale == "none":
        return X
    if scale == "standard":
        scaler = StandardScaler()
    elif scale == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown feature scaling mode: {scale}")
    values = scaler.fit_transform(X.to_numpy(dtype=float))
    return pd.DataFrame(values, columns=X.columns, index=X.index)


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path, **kwargs)


def _target_last(
    df: pd.DataFrame,
    name: str,
    task_type: TaskType,
    target_col=None,
    categorical_maps=None,
    *,
    defer_imputation: bool = False,
) -> TabularDataset:
    target_col = target_col if target_col is not None else df.columns[-1]
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]
    X = _encode_features(
        X,
        categorical_maps=categorical_maps,
        impute_missing=not defer_imputation,
    )
    if task_type == "classification":
        y = LabelEncoder().fit_transform(y_raw.astype(str) if y_raw.dtype == "object" else y_raw)
        class_names = [str(c) for c in np.unique(y_raw)]
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        class_names = None
    X.columns = [str(c) for c in X.columns]
    return TabularDataset(name=name, task_type=task_type, X=X, y=np.asarray(y), feature_names=list(X.columns), target_name=str(target_col), class_names=class_names)


def load_regression_dataset(
    name: str,
    data_root: Optional[Path] = None,
    *,
    defer_imputation: bool = False,
) -> TabularDataset:
    name = name.lower()
    data_root = data_root or dataset_dir()

    if name == "diabetes":
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=[str(c) for c in data.feature_names])
        y = np.asarray(data.target, dtype=float)
        return TabularDataset(name=name, task_type="regression", X=X, y=y, feature_names=list(X.columns), target_name="disease_progression")

    if name == "airfoil":
        df = _read_csv(data_root / "airfoil_self_noise.dat", sep=r"\s+", header=None)
        df.columns = ["Frequency", "Angle", "Chord", "Velocity", "Thickness", "SoundPressure"]
        return _target_last(df, name, "regression", "SoundPressure")

    if name == "yacht":
        df = _read_csv(data_root / "yacht_hydrodynamics.data", sep=r"\s+", header=None)
        df.columns = ["LongitudinalPosition", "PrismaticCoefficient", "LengthDisplacementRatio", "BeamDraughtRatio", "LengthBeamRatio", "FroudeNumber", "ResiduaryResistance"]
        return _target_last(df, name, "regression", "ResiduaryResistance")

    if name == "california":
        df = _read_csv(data_root / "California_House.csv")
        return _target_last(df, name, "regression", "median_house_value")

    if name == "concrete":
        df = _read_csv(data_root / "Concrete.csv")
        return _target_last(df, name, "regression", "strength")

    if name == "bike":
        df = _read_csv(data_root / "bike.csv")
        return _target_last(df, name, "regression", "cnt")

    if name == "boston":
        df = _read_csv(data_root / "boston_house.csv")
        return _target_last(df, name, "regression", "MEDV")

    if name == "abalone":
        df = _read_csv(data_root / "abalone.csv")
        # Preserve the mapping used in the original experimental scripts.
        sex_map = {"M": 1, "F": -1, "I": 0}
        return _target_last(df, name, "regression", "Rings", categorical_maps={"Sex": sex_map})

    if name in {"folds", "power_plant"}:
        df = _read_csv(data_root / "Folds5x2_pp.csv")
        return _target_last(df, "folds", "regression", "PE")

    if name in {"wine_quality", "wine_regression"}:
        df = _read_csv(data_root / "winequalityN.csv")
        # Match the historical LabelEncoder ordering explicitly while allowing
        # numeric missingness to remain until the training fold is known.
        if "type" in df.columns:
            df = df.copy()
            df["type"] = df["type"].astype(str).str.strip().str.lower()
        return _target_last(
            df,
            "wine_quality",
            "regression",
            "quality",
            categorical_maps={"type": {"red": 0.0, "white": 1.0}},
            defer_imputation=defer_imputation,
        )

    if name in {"energy_heating", "energy_cooling"}:
        df = _read_csv(data_root / "energy_efficiency_data.csv")
        feature_cols = [
            "Relative_Compactness", "Surface_Area", "Wall_Area", "Roof_Area",
            "Overall_Height", "Orientation", "Glazing_Area", "Glazing_Area_Distribution",
        ]
        target_col = "Heating_Load" if name == "energy_heating" else "Cooling_Load"
        X = _encode_features(df[feature_cols])
        y = df[target_col].to_numpy(dtype=float)
        return TabularDataset(name=name, task_type="regression", X=X, y=y, feature_names=feature_cols, target_name=target_col)

    # if name == "energy_cooling_with_heating":
    #     df = _read_csv(data_root / "energy_efficiency_data.csv")
    #     feature_cols = [
    #         "Relative_Compactness", "Surface_Area", "Wall_Area", "Roof_Area",
    #         "Overall_Height", "Orientation", "Glazing_Area", "Glazing_Area_Distribution",
    #         "Heating_Load",
    #     ]
    #     X = _encode_features(df[feature_cols])
    #     y = df["Cooling_Load"].to_numpy(dtype=float)
    #     return TabularDataset(
    #         name=name,
    #         task_type="regression",
    #         X=X,
    #         y=y,
    #         feature_names=feature_cols,
    #         target_name="Cooling_Load",
    #     )

    raise KeyError(
        f"Unknown regression dataset: {name}. Available: diabetes, airfoil, yacht, california, concrete, bike, boston, "
        "abalone, folds, wine_quality, energy_heating, energy_cooling."
    )


def load_classification_dataset(name: str, data_root: Optional[Path] = None) -> TabularDataset:
    name = name.lower()
    data_root = data_root or dataset_dir()

    sklearn_map = {
        "breast_cancer": load_breast_cancer,
        "iris": load_iris,
        "wine": load_wine,
        "digits": load_digits,
    }
    if name in sklearn_map:
        data = sklearn_map[name]()
        X = pd.DataFrame(data.data, columns=[str(c) for c in data.feature_names])
        y = np.asarray(data.target, dtype=int)
        class_names = [str(c) for c in getattr(data, "target_names", [])]
        return TabularDataset(name=name, task_type="classification", X=X, y=y, feature_names=list(X.columns), class_names=class_names)

    if name in {"pima_diabetes", "diabetes_classification"}:
        df = _read_csv(data_root / "diabetes.csv")
        return _target_last(df, "pima_diabetes", "classification", "Outcome")

    if name == "car":
        cols = [
            "buying",
            "maint",
            "doors",
            "persons",
            "lug_boot",
            "safety",
            "class",
        ]
        df = _read_csv(
            data_root / "car.data",
            header=None,
            names=cols,
        )

        feature_maps = {
            "buying": {
                "low": 0,
                "med": 1,
                "high": 2,
                "vhigh": 3,
            },
            "maint": {
                "low": 0,
                "med": 1,
                "high": 2,
                "vhigh": 3,
            },
            "doors": {
                "2": 0,
                "3": 1,
                "4": 2,
                "5more": 3,
            },
            "persons": {
                "2": 0,
                "4": 1,
                "more": 2,
            },
            "lug_boot": {
                "small": 0,
                "med": 1,
                "big": 2,
            },
            "safety": {
                "low": 0,
                "med": 1,
                "high": 2,
            },
        }
        target_map = {
            "unacc": 0,
            "acc": 1,
            "good": 2,
            "vgood": 3,
        }

        X = _encode_features(
            df[cols[:-1]],
            categorical_maps=feature_maps,
        )
        y = df["class"].map(target_map)
        if y.isna().any():
            unknown = sorted(
                df.loc[y.isna(), "class"].astype(str).unique().tolist()
            )
            raise ValueError(
                f"Unknown Car target categories: {unknown}"
            )

        return TabularDataset(
            name=name,
            task_type="classification",
            X=X,
            y=y.to_numpy(dtype=int),
            feature_names=cols[:-1],
            target_name="class",
            class_names=["unacc", "acc", "good", "vgood"],
        )
    if name == "zoo":
        cols = [
            "animal_name", "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator",
            "toothed", "backbone", "breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize", "type",
        ]
        df = _read_csv(data_root / "zoo.data", header=None, names=cols)
        X = _encode_features(df.drop(columns=["animal_name", "type"]))
        y = df["type"].to_numpy(dtype=int) - 1
        return TabularDataset(name=name, task_type="classification", X=X, y=y, feature_names=list(X.columns), target_name="type")

    if name == "balance":
        # UCI Balance Scale stores the class label in the first column.
        cols = ["class", "left_weight", "left_distance", "right_weight", "right_distance"]
        df = _read_csv(data_root / "balance-scale.data", header=None, names=cols)
        X = _encode_features(df[cols[1:]])
        y = LabelEncoder().fit_transform(df["class"])
        return TabularDataset(name=name, task_type="classification", X=X, y=y, feature_names=cols[1:], target_name="class")

    if name == "glass":
        df = _read_csv(data_root / "glass.data")  # file contains a header row
        return _target_last(df, name, "classification", "Class")

    if name == "seeds":
        df = _read_csv(data_root / "Seeds.txt", sep=r"\s+")  # file contains a header row
        return _target_last(df, name, "classification", "Class")

    if name == "heart":
        df = _read_csv(data_root / "heart.dat")  # file contains a header row
        return _target_last(df, name, "classification", "Class")

    if name == "bupa":
        df = _read_csv(data_root / "bupa.dat")  # file contains a header row
        return _target_last(df, name, "classification", "Class")

    raise KeyError(
        f"Unknown classification dataset: {name}. Available: breast_cancer, iris, wine, digits, pima_diabetes, "
        "car, zoo, balance, glass, seeds, heart, bupa."
    )


def build_pair_features(feature_names: list[str], mode: PairingMode = "combinations", max_original_features: Optional[int] = None) -> list[str]:
    names = list(feature_names)
    if max_original_features is not None:
        names = names[:max_original_features]
    if len(names) < 2:
        raise ValueError("At least two features are required for OPNs pairing.")
    if mode == "combinations":
        return cgp.all_pair(names)
    if mode == "permutations":
        return cgp.all_pair_repeat(names)
    if mode == "linear":
        return cgp.linear_pair(names)
    raise ValueError(f"Unknown pairing mode: {mode}")


def make_opns_features(
    X: pd.DataFrame,
    mode: PairingMode = "combinations",
    max_original_features: Optional[int] = None,
    scale_features: FeatureScale = "none",
):
    X_real = _maybe_scale_features(_encode_features(X), scale=scale_features)
    if mode == "linear" and "zero" not in X_real.columns:
        X_real = X_real.copy()
        X_real["zero"] = 0.0
    pair_features = build_pair_features(list(X_real.columns), mode=mode, max_original_features=max_original_features)

    # Fast path for the current Boost pipeline: cgp.data_convert constructs
    # Python OPNs objects row-by-row and then converts them back into the
    # two-array OPNsMatrix representation.  Here we build the two component
    # matrices directly.  This is equivalent for poly=1, tri=0, bias=False;
    # higher-order terms are handled later in core._expand_features.
    left_cols = pair_features[0::2]
    right_cols = pair_features[1::2]
    X_opns = OPNsMatrix()
    X_opns.set_matrix(
        X_real[left_cols].to_numpy(dtype=float, copy=True),
        X_real[right_cols].to_numpy(dtype=float, copy=True),
    )
    return X_opns, pair_features


def prepare_regression_fold(
    dataset: TabularDataset,
    train_idx,
    test_idx,
    *,
    need_opns: bool = True,
    fold_local_imputation: bool = False,
    pairing: PairingMode = "combinations",
    max_original_features: Optional[int] = None,
    scale_features: FeatureScale = "none",
) -> PreparedRegressionFold:
    """Prepare one regression fold without using held-out feature statistics."""
    if dataset.task_type != "regression":
        raise ValueError("prepare_regression_fold requires a regression dataset.")

    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    X_train = dataset.X.iloc[train_idx].copy()
    X_test = dataset.X.iloc[test_idx].copy()

    if fold_local_imputation:
        X_train, X_test, metadata = fit_fold_local_median_imputer(
            X_train, X_test
        )
    else:
        if X_train.isna().any().any() or X_test.isna().any().any():
            raise ValueError(
                "Missing feature values remain before fold preparation. "
                "Enable fold_local_imputation or load an already-clean matrix."
            )
        X_train = X_train.astype(float).reset_index(drop=True)
        X_test = X_test.astype(float).reset_index(drop=True)
        metadata = {
            "policy": "none_required",
            "columns_with_missing": [],
            "medians": {},
            "missing_train_by_column": {},
            "missing_test_by_column": {},
            "missing_train_total": 0,
            "missing_test_total": 0,
        }

    pair_features = build_pair_features(
        list(X_train.columns),
        mode=pairing,
        max_original_features=max_original_features,
    )
    X_train_opns = None
    X_test_opns = None
    if need_opns:
        X_train_opns, train_pair_features = make_opns_features(
            X_train,
            mode=pairing,
            max_original_features=max_original_features,
            scale_features=scale_features,
        )
        X_test_opns, test_pair_features = make_opns_features(
            X_test,
            mode=pairing,
            max_original_features=max_original_features,
            scale_features=scale_features,
        )
        if train_pair_features != pair_features or test_pair_features != pair_features:
            raise RuntimeError("Train/test OPNs pair-feature order is inconsistent.")

    return PreparedRegressionFold(
        X_train_frame=X_train,
        X_test_frame=X_test,
        y_train=np.asarray(dataset.y[train_idx], dtype=float).reshape(-1),
        y_test=np.asarray(dataset.y[test_idx], dtype=float).reshape(-1),
        X_train_opns=X_train_opns,
        X_test_opns=X_test_opns,
        pair_features=pair_features,
        candidate_pairs=len(pair_features) // 2,
        imputation_metadata=metadata,
    )


def make_opns_regression_target(y: np.ndarray, scale: bool = True):
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    y_scaler = None
    if scale:
        y_scaler = MinMaxScaler()
        y_scaled = y_scaler.fit_transform(y).reshape(-1)
    else:
        y_scaled = y.reshape(-1)
    y_opns = op.array([OPNs(float(v), 0.0) for v in y_scaled])
    return y_opns, y_scaler


def prepare_regression_data(
    name: str,
    pairing: PairingMode = "combinations",
    max_original_features: Optional[int] = None,
    scale_features: FeatureScale = "none",
) -> PreparedOPNsData:
    ds = load_regression_dataset(name)
    # print(ds.X.shape)
    # print(ds.X.columns)
    # print(ds.y.min(), ds.y.max(), ds.y.mean())
    # print(ds.target_name)
    X_opns, pair_features = make_opns_features(ds.X, mode=pairing, max_original_features=max_original_features, scale_features=scale_features)
    y_opns, y_scaler = make_opns_regression_target(ds.y, scale=True)
    return PreparedOPNsData(ds, X_opns, y_opns, pair_features, candidate_pairs=len(pair_features) // 2, y_scaler=y_scaler)


def prepare_classification_data(
    name: str,
    pairing: PairingMode = "combinations",
    max_original_features: Optional[int] = None,
    scale_features: FeatureScale = "none",
) -> PreparedOPNsData:
    ds = load_classification_dataset(name)
    X_opns, pair_features = make_opns_features(ds.X, mode=pairing, max_original_features=max_original_features, scale_features=scale_features)
    return PreparedOPNsData(ds, X_opns, np.asarray(ds.y, dtype=int), pair_features, candidate_pairs=len(pair_features) // 2, y_scaler=None)


def opns_train_val_test_split(X_opns, y, test_size=0.2, val_size=0.2, random_state=42, stratify=None):
    idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state, stratify=stratify)
    stratify_train_val = None if stratify is None else np.asarray(stratify)[train_val_idx]
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size, random_state=random_state, stratify=stratify_train_val)
    return (X_opns[train_idx], y[train_idx], X_opns[val_idx], y[val_idx], X_opns[test_idx], y[test_idx])


def repeated_splits(y, task_type: TaskType, n_splits: int = 5, n_repeats: int = 1, random_state: int = 42):
    y = np.asarray(y)
    idx = np.arange(len(y))
    if task_type == "classification":
        splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        return splitter.split(idx, y)
    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    return splitter.split(idx)


def repeated_holdout_splits(y, task_type: TaskType, n_splits: int = 20, test_size: float = 0.2, random_state: int = 42):
    y = np.asarray(y)
    idx = np.arange(len(y))
    if task_type == "classification":
        splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        return splitter.split(idx, y)
    splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    return splitter.split(idx)
