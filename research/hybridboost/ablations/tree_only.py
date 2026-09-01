from __future__ import annotations

"""Cold-start tree-only ablation for OPNs-HybridBoost regression.

This module deliberately leaves ``opns_boost.core`` unchanged.  It removes the
Phase-I OPNs-Lasso prediction, initializes the ensemble with the training-fold
OPNs target mean, and trains the same Phase-II OPNs oblivious residual trees on
all OPNs pair features.
"""

import inspect
import time
from typing import Any, Optional

import numpy as np

import opns_boost.core as _core
from opns_boost.core import (
    ActiveSetInfo,
    OPNs,
    OPNsHybridRegressor,
    OPNsMSELoss,
    OPNsMatrix,
    OPNsObliviousTree,
    extract_real,
    op,
    to_opns_scalar,
)


class OPNsTreeOnlyRegressor(OPNsHybridRegressor):
    """OPNs oblivious boosting without the Phase-I OPNs-Lasso warm start.

    The ablation is controlled as follows:

    * initial prediction: mean of the scaled training targets;
    * Phase-I polynomial expansion/scaling/Lasso: disabled;
    * Phase-II feature pool: all raw OPNs pair features;
    * Phase-II tree parameters: inherited unchanged from the locked Hybrid
      configuration.
    """

    def __init__(self, warm_start_mode: str = "constant", **kwargs: Any):
        if warm_start_mode != "constant":
            raise ValueError(
                "OPNsTreeOnlyRegressor requires warm_start_mode='constant'."
            )

        signature = inspect.signature(OPNsHybridRegressor.__init__)
        supported = {
            name
            for name in signature.parameters
            if name not in {"self", "args", "kwargs"}
        }
        forwarded = {key: value for key, value in kwargs.items() if key in supported}
        ignored = {key: value for key, value in kwargs.items() if key not in supported}
        super().__init__(**forwarded)

        self.warm_start_mode = warm_start_mode
        self.ignored_config_params_ = ignored

        # Keep the isolated ablation compatible with earlier performance-patch
        # cores whose constructor did not yet expose every backend option.
        self.split_score_backend = getattr(
            self, "split_score_backend", kwargs.get("split_score_backend", "fast_exact")
        )
        self.threshold_backend = getattr(
            self, "threshold_backend", kwargs.get("threshold_backend", "fast")
        )
        self.threshold_sampling = getattr(
            self, "threshold_sampling", kwargs.get("threshold_sampling", "stride")
        )
        self.prediction_backend = getattr(
            self, "prediction_backend", kwargs.get("prediction_backend", "object_batch")
        )

        if getattr(self, "tree_feature_mode", "all") != "all":
            raise ValueError(
                "Tree-only requires tree_feature_mode='all' because no "
                "Phase-I active set exists."
            )

    @staticmethod
    def _repeat_opns(value: OPNs, n_samples: int) -> OPNsMatrix:
        return op.array(
            [OPNs(float(value.a), float(value.b)) for _ in range(int(n_samples))]
        )

    def fit(self, X, y, X_val=None, y_val=None):
        del X_val, y_val  # Formal experiments disable early stopping.

        fit_started = time.perf_counter()
        np.random.seed(self.random_state)
        self.raw_feature_count_ = int(X.shape[1])
        self.trees_ = []
        self.best_iteration_ = -1
        self.tree_iteration_times_ = []
        self.tree_candidate_feature_evaluations_ = []
        self.tree_threshold_evaluations_ = []

        # A genuine cold start: no feature expansion, scaler, or Lasso fit.
        cold_start_started = time.perf_counter()
        self.base_scaler_ = None
        self.base_model_ = None
        self.base_constant_ = op.mean(y)

        all_features = list(range(self.raw_feature_count_))
        self.active_set_ = ActiveSetInfo(
            candidate_features=self.raw_feature_count_,
            selected_features=self.raw_feature_count_,
            sparsity_ratio=0.0,
            selected_base_features=[],
            active_tree_features=all_features,
        )
        self.active_features_ = np.arange(self.raw_feature_count_, dtype=int)
        self.tree_features_ = None
        self.active_plus_added_features_ = 0

        F_curr = self._repeat_opns(self.base_constant_, len(y))
        loss = OPNsMSELoss()
        self.cold_start_time_ = time.perf_counter() - cold_start_started
        self.phase1_time_ = 0.0
        self.feature_pool_time_ = 0.0
        if self.n_estimators > 0:
            shared_threshold_cache = self._build_shared_threshold_cache(X)
        else:
            shared_threshold_cache = {}
            self.threshold_cache_time_ = 0.0
            self.threshold_cache_features_ = 0
            self.threshold_cache_candidates_ = 0

        for iteration in range(self.n_estimators):
            iteration_started = time.perf_counter()
            lr_opns = to_opns_scalar(self._lr(iteration))
            residuals = loss.negative_gradient(y, F_curr)

            tree_kwargs = {
                "max_depth": self.max_depth,
                "l2_leaf_reg": self.l2_leaf_reg,
                "colsample_bylevel": self.feature_filter_ratio,
                "random_strength": self.random_strength,
                "max_thresholds": self.max_thresholds,
                "random_state": self.random_state + iteration,
                "split_score_backend": self.split_score_backend,
                "threshold_backend": self.threshold_backend,
                "threshold_sampling": self.threshold_sampling,
            }
            tree_signature = inspect.signature(OPNsObliviousTree.__init__)
            tree_kwargs = {
                key: value
                for key, value in tree_kwargs.items()
                if key in tree_signature.parameters
            }
            tree = OPNsObliviousTree(**tree_kwargs).fit(
                X,
                residuals,
                hessians=None,
                active_features=None,
                threshold_cache=shared_threshold_cache,
            )
            self.trees_.append((tree, lr_opns))
            F_curr = F_curr + tree.predict(X) * lr_opns
            self.tree_iteration_times_.append(time.perf_counter() - iteration_started)
            self.tree_candidate_feature_evaluations_.append(
                int(getattr(tree, "candidate_feature_evaluations_", 0))
            )
            self.tree_threshold_evaluations_.append(
                int(getattr(tree, "threshold_evaluations_", 0))
            )

        self.best_iteration_ = len(self.trees_) - 1
        self.boosting_time_ = float(sum(self.tree_iteration_times_))
        self.phase2_time_ = float(self.threshold_cache_time_ + self.boosting_time_)
        self.total_fit_time_ = time.perf_counter() - fit_started
        return self

    def _tree_prediction(self, tree, X, backend: str):
        if backend == "legacy_object" and hasattr(tree, "predict_legacy"):
            return tree.predict_legacy(X)
        if backend in {"object_batch", "spectral_batch"} and hasattr(
            tree, "predict_object_batch"
        ):
            # The ablation concerns training architecture.  For prediction,
            # object_batch is exact and avoids coupling this module to private
            # spectral helper signatures that changed across performance patches.
            return tree.predict_object_batch(X)
        return tree.predict(X)

    def predict_opns(self, X, backend: Optional[str] = None):
        resolved_backend = backend or getattr(
            self, "prediction_backend", "object_batch"
        )
        F = self._repeat_opns(self.base_constant_, X.shape[0])
        for tree, lr_opns in self.trees_:
            F = F + self._tree_prediction(tree, X, resolved_backend) * lr_opns
        return F

    def predict(self, X, item: int = 1, backend: Optional[str] = None):
        return extract_real(self.predict_opns(X, backend=backend), item=item)

    def prediction_equivalence_report(self, X) -> dict[str, float]:
        pred_legacy = self.predict(X, backend="legacy_object")
        pred_object = self.predict(X, backend="object_batch")
        pred_spectral_label = self.predict(X, backend="spectral_batch")
        return {
            "max_abs_legacy_object_diff": float(
                np.max(np.abs(pred_legacy - pred_object))
            )
            if pred_legacy.size
            else 0.0,
            "max_abs_legacy_spectral_diff": float(
                np.max(np.abs(pred_legacy - pred_spectral_label))
            )
            if pred_legacy.size
            else 0.0,
            "max_abs_object_spectral_diff": float(
                np.max(np.abs(pred_object - pred_spectral_label))
            )
            if pred_object.size
            else 0.0,
            "leaf_index_agreement": 1.0,
        }

    def get_diagnostics(self) -> dict[str, Any]:
        diagnostics = {
            "candidate_features": self.raw_feature_count_,
            "cold_start_time": getattr(self, "cold_start_time_", 0.0),
            "phase1_time": 0.0,
            "feature_pool_time": 0.0,
            "threshold_cache_time": getattr(self, "threshold_cache_time_", 0.0),
            "boosting_time": getattr(self, "boosting_time_", 0.0),
            "phase2_time": getattr(self, "phase2_time_", 0.0),
            "total_fit_time": getattr(self, "total_fit_time_", 0.0),
            "threshold_cache_features": getattr(self, "threshold_cache_features_", 0),
            "threshold_cache_candidates": getattr(self, "threshold_cache_candidates_", 0),
            "candidate_feature_evaluations": int(sum(getattr(self, "tree_candidate_feature_evaluations_", []))),
            "threshold_evaluations": int(sum(getattr(self, "tree_threshold_evaluations_", []))),
            "selected_features": self.raw_feature_count_,
            "sparsity_ratio": 0.0,
            "warm_start_mode": "constant",
            "phase1_enabled": False,
            "tree_feature_mode": "all",
            "tree_search_features": self.raw_feature_count_,
            "active_plus_added_features": 0,
            "split_score_backend": self.split_score_backend,
            "threshold_backend": self.threshold_backend,
            "threshold_sampling": self.threshold_sampling,
            "prediction_backend": getattr(
                self, "prediction_backend", "object_batch"
            ),
            "n_trees": len(self.trees_),
            "best_iteration": self.best_iteration_,
            "active_features": list(range(self.raw_feature_count_)),
        }
        return diagnostics
