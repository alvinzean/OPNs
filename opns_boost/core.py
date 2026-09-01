"""Core models for the refactored OPNs-HybridBoost project.

Design goals
------------
1. Keep the original `OPNs/` and `opns_sklearn/` implementations unchanged.
2. Move duplicated model code out of ad-hoc test files.
3. Expose sklearn-style regression and classification estimators.
4. Record active-pair statistics needed by the TKDE experiments:
   candidate pairs, selected K, sparsity ratio, tree count, and best iteration.

Important implementation note
-----------------------------
The original OPNs arithmetic is imported as-is.  This module does not redefine
OPNs algebra; it only centralizes the HybridBoost training logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
import math
import time
from typing import Iterable, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelBinarizer

import opns_pack.opns_np as op
from opns_pack.opns import OPNs
from opns_pack.opns_matrix import OPNsMatrix
from opns_module.preprocessing import OPNsStandardScaler
from ._phase1 import Lasso, LogisticRegression


OPNS_ONE = OPNs(0, -1)
OPNS_ZERO = OPNs(0, 0)

PREDICTION_BACKENDS = {"legacy_object", "object_batch", "spectral_batch"}


def _components_to_spectral(left, right) -> tuple[np.ndarray, np.ndarray]:
    """Map the implementation coordinates ``(a, b)`` to spectral ``(p, q)``.

    The OPNs algebra implemented in :mod:`opns` satisfies
    ``p = -(a + b)`` and ``q = a - b``.  In these coordinates addition and
    multiplication are channelwise real operations.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return -(left + right), left - right


def _spectral_to_components(p, q) -> tuple[np.ndarray, np.ndarray]:
    """Inverse spectral map from ``(p, q)`` to implementation ``(a, b)``."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * (q - p), -0.5 * (p + q)


def _validate_prediction_backend(backend: str) -> str:
    backend = str(backend)
    if backend not in PREDICTION_BACKENDS:
        raise ValueError(
            f"Unknown prediction_backend={backend!r}. "
            f"Expected one of: {sorted(PREDICTION_BACKENDS)}."
        )
    return backend


def to_opns_scalar(value: float) -> OPNs:
    """Map a real scalar to the scalar direction used by the existing code."""
    return value * OPNS_ONE


def _as_flat_opns(values):
    if isinstance(values, OPNsMatrix):
        return values.flatten()
    if isinstance(values, np.ndarray):
        return values.reshape(-1)
    return np.asarray(values).reshape(-1)


def opn_magnitude(x: OPNs) -> float:
    """Simple real-valued magnitude used only for feature-selection diagnostics.

    The model still trains with OPNs operations.  This helper is used to decide
    which Lasso coefficients are effectively non-zero and to report sparsity.
    """
    return float(abs(getattr(x, "a", 0.0)) + abs(getattr(x, "b", 0.0)))


def opn_vector_magnitudes(values) -> np.ndarray:
    flat = _as_flat_opns(values)
    return np.array([opn_magnitude(v) for v in flat], dtype=float)


def extract_real(values, item: int = 1) -> np.ndarray:
    """Extract real predictions/targets from an OPNs vector or matrix.

    item convention follows the regression experiments:
    - item=0: a + b
    - item=1: a
    - item=2: b
    """
    if isinstance(values, OPNsMatrix):
        flat = values.flatten()
    elif isinstance(values, np.ndarray) and values.size and isinstance(values.reshape(-1)[0], OPNs):
        flat = values.reshape(-1)
    elif isinstance(values, (list, tuple)) and values and isinstance(values[0], OPNs):
        flat = values
    else:
        return np.asarray(values).reshape(-1)

    item_map = {
        0: lambda z: z.a + z.b,
        1: lambda z: z.a,
        2: lambda z: z.b,
    }
    f = item_map.get(item, item_map[1])
    return np.array([f(z) for z in flat], dtype=float)


def opns_sigmoid(z):
    denom = OPNS_ONE + op.exp(-z)
    if hasattr(denom, "reciprocal"):
        return denom.reciprocal()
    return OPNS_ONE / denom


class OPNsMSELoss:
    def negative_gradient(self, y_true, y_pred):
        return y_true - y_pred

    def hessian(self, y_true, y_pred):
        return op.array([OPNS_ONE for _ in range(len(y_true))])

    def calc_loss(self, y_true, y_pred):
        res = y_true - y_pred
        return op.mean(res ** 2)


class OPNsLogLoss:
    def negative_gradient(self, y_true, y_pred_logits):
        probs = opns_sigmoid(y_pred_logits)
        return y_true - probs

    def hessian(self, y_true, y_pred_logits):
        probs = opns_sigmoid(y_pred_logits)
        return probs * (OPNS_ONE - probs)

    @staticmethod
    def logits_to_real(logits) -> np.ndarray:
        flat = _as_flat_opns(logits)
        return np.array([-(z.a + z.b) for z in flat], dtype=float)


class OPNsObliviousTree:
    """Symmetric/oblivious OPNs tree shared by regression and classification.

    At each depth, one feature and one threshold are selected and applied to all
    current leaves.  The split criterion is the sum of within-child OPNs gradient
    variances across all current leaves.  Leaf values use either a residual mean
    step (if Hessian is omitted) or Newton step (if Hessian is provided).
    """

    def __init__(
        self,
        max_depth: int = 4,
        l2_leaf_reg: float = 0.1,
        colsample_bylevel: float = 1.0,
        random_strength: float = 0.0,
        max_thresholds: int = 20,
        random_state: Optional[int] = None,
        split_score_backend: str = "fast_exact",
        threshold_backend: str = "ordered_exact",
        threshold_sampling: str = "stride",
    ):
        self.max_depth = max_depth
        self.l2_leaf_reg = to_opns_scalar(l2_leaf_reg)
        self.colsample_bylevel = colsample_bylevel
        self.random_strength = random_strength
        self.max_thresholds = max_thresholds
        self.random_state = random_state
        self.split_score_backend = split_score_backend
        self.threshold_backend = threshold_backend
        self.threshold_sampling = threshold_sampling
        self.splits: list[tuple[int, OPNs]] = []
        self.leaf_values: dict[int, OPNs] = {}

        # Lazily rebuilt after fit / unpickling.  The arrays eliminate
        # per-sample OPNs construction during prediction.
        self._prediction_cache_ready = False
        self._split_features = np.empty(0, dtype=np.int64)
        self._split_threshold_p = np.empty(0, dtype=float)
        self._split_threshold_q = np.empty(0, dtype=float)
        self._leaf_left = np.empty(0, dtype=float)
        self._leaf_right = np.empty(0, dtype=float)
        self._leaf_p = np.empty(0, dtype=float)
        self._leaf_q = np.empty(0, dtype=float)

    # def _thresholds(self, column):
    #     """Return candidate thresholds for one OPNs feature column.
    #
    #     The original implementation delegates to ``op.unique(column)``, which
    #     iterates through rows and constructs many temporary OPNs objects.  This
    #     fast path works directly on the two ndarray components stored inside
    #     OPNsMatrix and is therefore much cheaper.  The old path is still
    #     available by setting ``threshold_backend="opns"``.
    #     """
    #     if self.threshold_backend == "opns" or not isinstance(column, OPNsMatrix):
    #         vals = op.unique(column)
    #     else:
    #         left = np.asarray(column.left_matrix, dtype=float).reshape(-1)
    #         right = np.asarray(column.right_matrix, dtype=float).reshape(-1)
    #         pairs = np.column_stack([left, right])
    #         if pairs.shape[0] == 0:
    #             vals = op.array([])
    #         else:
    #             uniq = np.unique(pairs, axis=0)
    #             # Sort according to the OPNs ordering used by OPNs.__lt__.
    #             # x < y iff (x.a+x.b) > (y.a+y.b), tie: x.a < y.a.
    #             order = np.lexsort((uniq[:, 0], -(uniq[:, 0] + uniq[:, 1])))
    #             uniq = uniq[order]
    #             vals = OPNsMatrix()
    #             vals.set_matrix(uniq[:, 0], uniq[:, 1])
    #
    #     if len(vals) > self.max_thresholds:
    #         step = max(1, len(vals) // self.max_thresholds)
    #         return vals[::step]
    #     return vals

    def _unique_thresholds(self, column):
        """Return sorted unique OPNs thresholds using the configured backend."""
        if not isinstance(column, OPNsMatrix):
            return op.unique(column)

        if self.threshold_backend in {"opns", "fast", "ordered_exact"}:
            left = np.asarray(column.left_matrix, dtype=float).reshape(-1)
            right = np.asarray(column.right_matrix, dtype=float).reshape(-1)
            pairs = np.column_stack([left, right])

            if pairs.shape[0] == 0:
                return op.array([])

            uniq = np.unique(pairs, axis=0)
            order = np.lexsort(
                (
                    uniq[:, 0],
                    -(uniq[:, 0] + uniq[:, 1]),
                )
            )
            uniq = uniq[order]

            result = OPNsMatrix()
            result.set_matrix(uniq[:, 0], uniq[:, 1])
            return result

        if self.threshold_backend in {"fast_opns_unique", "legacy_swapped"}:
            return self._fast_opns_unique_thresholds(column)

        raise ValueError(
            f"Unknown threshold_backend={self.threshold_backend!r}. "
            "Expected 'ordered_exact'/'fast'/'opns', or the legacy "
            "'fast_opns_unique'/'legacy_swapped' backend."
        )

    def _thresholds(self, column):
        """Return candidate thresholds after uniqueness and sampling.

        For ``stride``, ``max_thresholds`` is a *target sampling budget*, not a
        strict cardinality cap.  The historical floor-stride grid is preserved
        because changing that grid measurably changes fitted trees on locked
        configurations.  The only removed candidate is the largest observed
        OPNs value when it is present in the sampled grid: under the routing
        rule ``x > theta`` it sends every training observation to the left and
        is therefore a provable global no-op.

        ``quantile`` remains an optional alternative that selects interior
        empirical quantiles.  Formal experiments use ``stride``.
        """
        if self.max_thresholds <= 0:
            raise ValueError("max_thresholds must be a positive integer.")

        vals = self._unique_thresholds(column)
        n_values = len(vals)

        # A constant column has no admissible observed-value threshold.
        if n_values <= 1:
            return vals[:0]

        if self.threshold_sampling == "stride":
            if n_values > self.max_thresholds:
                step = max(1, n_values // int(self.max_thresholds))
                candidates = vals[::step]

                # The largest ordered observation is included exactly when its
                # zero-based index is divisible by the stride.  Remove it
                # without changing any other point on the historical grid.
                if (n_values - 1) % step == 0:
                    candidates = candidates[:-1]
                return candidates

            # Without subsampling the last unique value is always the global
            # maximum and therefore a no-op threshold.
            return vals[:-1]

        if self.threshold_sampling == "quantile":
            return self._quantile_thresholds(column)

        raise ValueError(
            f"Unknown threshold_sampling={self.threshold_sampling!r}. "
            "Expected 'stride' or 'quantile'."
        )

    def _quantile_thresholds(self, column):
        """Select thresholds at interior empirical quantiles.

        The minimum and maximum observations are excluded because they usually
        create one-sided or empty-child splits. Duplicate selected observations
        are removed with the same uniqueness backend used by the tree.
        """
        n_samples = len(column)
        if n_samples <= 2:
            return self._unique_thresholds(column)[:0]

        n_candidates = min(int(self.max_thresholds), n_samples - 2)

        # Use the original OPNs ordering to avoid changing the algebraic
        # threshold semantics while testing only the sampling strategy.
        sorted_index = op.argsort(column)
        sorted_column = column[sorted_index]

        quantiles = np.linspace(0.0, 1.0, n_candidates + 2, dtype=float)[1:-1]
        positions = np.rint(quantiles * (n_samples - 1)).astype(int)
        positions = np.unique(np.clip(positions, 0, n_samples - 1))

        selected = sorted_column[positions]
        selected_unique = self._unique_thresholds(selected)

        # Rounding and duplicate observations can reduce the number of
        # candidates. That is intentional: never add arbitrary thresholds merely
        # to force the count back to max_thresholds.
        return selected_unique

    def _fast_opns_unique_thresholds(self, column):
        """Fast equivalent of ``op.unique`` for a 1-D OPNsMatrix.

        It preserves the original first-occurrence duplicate semantics,
        including the swapped pair check for ``(a, b)`` and ``(b, a)``.
        """
        left = np.asarray(column.left_matrix, dtype=float).reshape(-1)
        right = np.asarray(column.right_matrix, dtype=float).reshape(-1)

        unique_left: list[float] = []
        unique_right: list[float] = []

        # Equivalent to list.index(): retain only the first occurrence index.
        first_left_index: dict[float, int] = {}
        first_right_index: dict[float, int] = {}

        for a, b in zip(left, right):
            a = float(a)
            b = float(b)

            if (
                a in first_left_index
                and b in first_right_index
                and first_left_index[a] == first_right_index[b]
            ):
                continue

            if (
                b in first_left_index
                and a in first_right_index
                and first_left_index[b] == first_right_index[a]
            ):
                continue

            index = len(unique_left)
            unique_left.append(a)
            unique_right.append(b)
            first_left_index.setdefault(a, index)
            first_right_index.setdefault(b, index)

        result = OPNsMatrix()
        result.set_matrix(
            np.asarray(unique_left, dtype=float),
            np.asarray(unique_right, dtype=float),
        )

        if len(result) == 0:
            return result

        sorted_index = op.argsort(result)
        return result[sorted_index]

    @staticmethod
    def _opn_components_lt(a1: float, b1: float, a2: float, b2: float) -> bool:
        """Component version of OPNs.__lt__ for split scores."""
        da = a1 - a2
        db = b1 - b2
        s = da + db
        return (s > 0) or (s == 0 and da < 0)

    @staticmethod
    def _variance_score_components(a: np.ndarray, b: np.ndarray, mask: np.ndarray):
        """Exact OPNs variance components multiplied by group size.

        For z=(a,b), OPNs square gives z^2=(-2ab, -a^2-b^2).
        This computes n * mean((z - mean(z))^2) without constructing temporary
        OPNsMatrix objects.
        """
        n = int(np.sum(mask))
        if n <= 0:
            return 0.0, 0.0, False
        aa = a[mask]
        bb = b[mask]
        sum_a = float(np.sum(aa))
        sum_b = float(np.sum(bb))
        sum_ab = float(np.dot(aa, bb))
        sum_aa = float(np.dot(aa, aa))
        sum_bb = float(np.dot(bb, bb))
        left = -2.0 * (sum_ab - (sum_a * sum_b) / n)
        right = -((sum_aa - (sum_a * sum_a) / n) + (sum_bb - (sum_b * sum_b) / n))
        return left, right, True

    def _split_score_fast(self, node_indices, depth, is_right, grad_left, grad_right, rng):
        """Fast exact split score using grouped sums.

        This is equivalent to summing n_child * Var_OPNs(gradient_child)
        over all children produced by an oblivious split, but it avoids building
        one boolean mask per leaf and avoids constructing temporary OPNsMatrix
        objects.
        """
        group_ids = (node_indices << 1) + is_right.astype(np.int8, copy=False)
        n_groups = 2 ** (depth + 1)

        counts = np.bincount(group_ids, minlength=n_groups).astype(float)

        # An oblivious candidate is useful only if it refines at least one
        # currently non-empty parent node. Merely routing each parent wholly to
        # one side changes leaf identifiers but not the sample partition.
        child_counts = counts.reshape(-1, 2)
        refines_partition = np.any(
            (child_counts[:, 0] > 0.0) & (child_counts[:, 1] > 0.0)
        )
        if not refines_partition:
            return (0.0, 0.0), False

        valid = counts > 0
        if not np.any(valid):
            return (0.0, 0.0), False

        sum_a = np.bincount(group_ids, weights=grad_left, minlength=n_groups)
        sum_b = np.bincount(group_ids, weights=grad_right, minlength=n_groups)
        sum_ab = np.bincount(group_ids, weights=grad_left * grad_right, minlength=n_groups)
        sum_aa = np.bincount(group_ids, weights=grad_left * grad_left, minlength=n_groups)
        sum_bb = np.bincount(group_ids, weights=grad_right * grad_right, minlength=n_groups)

        c = counts[valid]
        sa = sum_a[valid]
        sb = sum_b[valid]
        sab = sum_ab[valid]
        saa = sum_aa[valid]
        sbb = sum_bb[valid]

        total_left = float(np.sum(-2.0 * (sab - (sa * sb) / c)))
        total_right = float(np.sum(-((saa - (sa * sa) / c) + (sbb - (sb * sb) / c))))

        if self.random_strength > 0:
            total_right += float(rng.normal(0.0, self.random_strength)) * 1e-3
        return (total_left, total_right), True

    def _candidate_features(self, active_features, rng) -> np.ndarray:
        active = np.asarray(active_features, dtype=int)
        if self.colsample_bylevel >= 1.0 or len(active) <= 1:
            return active
        n_keep = max(1, int(np.ceil(len(active) * self.colsample_bylevel)))
        return rng.choice(active, size=n_keep, replace=False)

    def _split_score(self, node_indices, depth, is_right, gradients, rng):
        total_score = OPNs(0, 0)
        valid = False
        for leaf_id in range(2 ** depth):
            leaf_mask = node_indices == leaf_id
            if not np.any(leaf_mask):
                continue
            left_mask = leaf_mask & (~is_right)
            right_mask = leaf_mask & is_right
            if np.any(left_mask) and np.any(right_mask):
                valid = True
            if np.any(left_mask):
                g_l = gradients[left_mask]
                total_score += op.var(g_l) * to_opns_scalar(int(np.sum(left_mask)))
            if np.any(right_mask):
                g_r = gradients[right_mask]
                total_score += op.var(g_r) * to_opns_scalar(int(np.sum(right_mask)))
        if valid and self.random_strength > 0:
            # Tiny stochastic tie-breaker / regularizer.  It is intentionally
            # small so that it perturbs only marginal split gains.
            total_score += OPNs(0, float(rng.normal(0.0, self.random_strength)) * 1e-3)
        return total_score, valid

    def fit(
        self,
        X,
        gradients,
        hessians=None,
        active_features: Optional[Iterable[int]] = None,
        threshold_cache: Optional[dict[int, OPNsMatrix]] = None,
    ):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        node_indices = np.zeros(n_samples, dtype=int)
        if active_features is None:
            active_features = np.arange(n_features)
        active_features = np.asarray(list(active_features), dtype=int)
        if len(active_features) == 0:
            active_features = np.arange(n_features)

        use_fast_score = (
            self.split_score_backend in {"fast", "fast_exact"}
            and isinstance(gradients, OPNsMatrix)
        )
        if use_fast_score:
            grad_left = np.asarray(gradients.left_matrix, dtype=float).reshape(-1)
            grad_right = np.asarray(gradients.right_matrix, dtype=float).reshape(-1)
        else:
            grad_left = grad_right = None

        # Threshold candidates depend only on X and the fixed feature pool, not
        # on residuals. A fold-level cache can therefore be reused exactly by
        # every boosting tree. Missing entries are filled defensively so callers
        # may also pass a partial cache.
        cache_reused = threshold_cache is not None
        if threshold_cache is None:
            threshold_cache = {}
        else:
            threshold_cache = dict(threshold_cache)
        for j in active_features:
            j = int(j)
            if j not in threshold_cache:
                threshold_cache[j] = self._thresholds(X[:, j])

        self.threshold_cache_reused_ = bool(cache_reused)
        self.active_feature_count_ = int(len(active_features))
        self.threshold_candidate_count_ = int(
            sum(len(threshold_cache[int(j)]) for j in active_features)
        )
        self.candidate_feature_evaluations_ = 0
        self.threshold_evaluations_ = 0

        for depth in range(self.max_depth):
            best_score = None
            best_split = None
            candidates = self._candidate_features(active_features, rng)
            self.candidate_feature_evaluations_ += int(len(candidates))
            for feat_idx in candidates:
                feat_idx = int(feat_idx)
                col = X[:, feat_idx]
                thresholds = threshold_cache[feat_idx]
                self.threshold_evaluations_ += int(len(thresholds))
                for threshold in thresholds:
                    is_right = col > threshold
                    if use_fast_score:
                        score, valid = self._split_score_fast(node_indices, depth, is_right, grad_left, grad_right, rng)
                    else:
                        score, valid = self._split_score(node_indices, depth, is_right, gradients, rng)
                    if not valid:
                        continue
                    if best_score is None:
                        best_score = score
                        best_split = (feat_idx, threshold)
                    elif use_fast_score:
                        if self._opn_components_lt(score[0], score[1], best_score[0], best_score[1]):
                            best_score = score
                            best_split = (feat_idx, threshold)
                    elif score < best_score:
                        best_score = score
                        best_split = (feat_idx, threshold)
            if best_split is None:
                break
            self.splits.append(best_split)
            node_indices = (node_indices << 1) | (X[:, best_split[0]] > best_split[1]).astype(int)

        for leaf_id in range(2 ** len(self.splits)):
            mask = node_indices == leaf_id
            if not np.any(mask):
                self.leaf_values[leaf_id] = OPNs(0, 0)
                continue
            grad_sum = op.sum(gradients[mask])
            if hessians is None:
                denom = to_opns_scalar(int(np.sum(mask))) + self.l2_leaf_reg
            else:
                denom = op.sum(hessians[mask]) + self.l2_leaf_reg
            self.leaf_values[leaf_id] = grad_sum / denom

        self._refresh_prediction_cache()
        return self

    def _refresh_prediction_cache(self) -> None:
        """Materialize split and leaf data as contiguous real arrays.

        Leaf values are cached both in implementation coordinates ``(a, b)``
        and in exact spectral coordinates ``(p, q)``.  No approximation is
        introduced: ``p=-(a+b)`` and ``q=a-b`` are the algebra isomorphism to
        the two real channels.
        """
        self._split_features = np.asarray(
            [int(feature) for feature, _ in self.splits],
            dtype=np.int64,
        )
        threshold_left = np.asarray(
            [float(threshold.a) for _, threshold in self.splits],
            dtype=float,
        )
        threshold_right = np.asarray(
            [float(threshold.b) for _, threshold in self.splits],
            dtype=float,
        )
        # Keep the native implementation coordinates for bitwise-identical
        # routing with OPNsMatrix.__gt__.  The p/q threshold cache is retained
        # for diagnostics and theoretical correspondence, but routing itself
        # uses the native comparator to avoid floating-point regrouping around
        # exact ties.
        self._split_threshold_left = threshold_left
        self._split_threshold_right = threshold_right
        self._split_threshold_p, self._split_threshold_q = _components_to_spectral(
            threshold_left, threshold_right
        )

        n_leaves = 1 << len(self.splits)
        self._leaf_left = np.empty(n_leaves, dtype=float)
        self._leaf_right = np.empty(n_leaves, dtype=float)
        for leaf_id in range(n_leaves):
            value = self.leaf_values.get(leaf_id, OPNS_ZERO)
            self._leaf_left[leaf_id] = float(value.a)
            self._leaf_right[leaf_id] = float(value.b)
        self._leaf_p, self._leaf_q = _components_to_spectral(
            self._leaf_left, self._leaf_right
        )
        self._prediction_cache_ready = True

    def _ensure_prediction_cache(self) -> None:
        if not getattr(self, "_prediction_cache_ready", False):
            self._refresh_prediction_cache()

    def apply_legacy(self, X) -> np.ndarray:
        """Original OPNsMatrix routing path, retained for ablation tests."""
        n_samples = X.shape[0]
        node_indices = np.zeros(n_samples, dtype=np.int64)
        for feat_idx, threshold in self.splits:
            node_indices = (node_indices << 1) | (X[:, feat_idx] > threshold).astype(np.int64)
        return node_indices

    def apply_component_batch(
        self,
        X_left: np.ndarray,
        X_right: np.ndarray,
    ) -> np.ndarray:
        """Vectorized routing exactly matching ``OPNsMatrix.__gt__``.

        The OPNs total order is not an algebraic channelwise operation.  To
        preserve the existing implementation *bit for bit* under float64, the
        route decision uses the same subtraction and summation order as
        ``OPNsMatrix.__gt__``::

            d_a = a_x - a_theta
            d_b = b_x - b_theta
            x > theta iff d_a + d_b < 0,
            or d_a + d_b == 0 and d_a > 0.

        This removes scalar OPNs construction while avoiding the rare tie
        changes caused by computing and comparing separately rounded p/q
        coordinates.
        """
        self._ensure_prediction_cache()
        X_left = np.asarray(X_left, dtype=float)
        X_right = np.asarray(X_right, dtype=float)
        node_indices = np.zeros(X_left.shape[0], dtype=np.int64)
        for depth, feat_idx in enumerate(self._split_features):
            delta_left = X_left[:, feat_idx] - self._split_threshold_left[depth]
            delta_right = X_right[:, feat_idx] - self._split_threshold_right[depth]
            delta_sum = delta_left + delta_right
            is_right = (delta_sum < 0.0) | (
                (delta_sum == 0.0) & (delta_left > 0.0)
            )
            node_indices = (node_indices << 1) | is_right.astype(
                np.int64, copy=False
            )
        return node_indices

    def apply_spectral(self, X_left: np.ndarray, X_right: np.ndarray) -> np.ndarray:
        """Compatibility alias for exact vectorized routing.

        Spectral p/q channels are used for leaf values and ensemble algebra.
        Tree routing is evaluated in native ``(a,b)`` coordinates because the
        total order must reproduce the established finite-precision comparator
        exactly.  See :meth:`apply_component_batch`.
        """
        return self.apply_component_batch(X_left, X_right)

    def predict_legacy(self, X):
        """Pre-optimization prediction with one OPNs object per sample."""
        node_indices = self.apply_legacy(X)
        return op.array(
            [self.leaf_values.get(int(idx), OPNs(0, 0)) for idx in node_indices]
        )

    def predict_object_batch(self, X):
        """Batched leaf gather returning one OPNsMatrix for all samples."""
        self._ensure_prediction_cache()
        node_indices = self.apply_legacy(X)
        result = OPNsMatrix()
        result.set_matrix(self._leaf_left[node_indices], self._leaf_right[node_indices])
        return result

    def predict_spectral_from_components(
        self,
        X_left: np.ndarray,
        X_right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return leaf predictions directly in the two spectral channels.

        ``X_left`` and ``X_right`` are the implementation coordinates used for
        exact vectorized routing.  The gathered leaf outputs are returned in
        the algebraically exact p/q spectral coordinates.
        """
        self._ensure_prediction_cache()
        node_indices = self.apply_component_batch(X_left, X_right)
        return self._leaf_p[node_indices], self._leaf_q[node_indices]

    def predict(self, X):
        """Compatibility prediction using batched leaf lookup.

        This method preserves the OPNsMatrix return type used during training,
        while removing the old per-sample OPNs object construction.
        """
        return self.predict_object_batch(X)


@dataclass
class ActiveSetInfo:
    candidate_features: int
    selected_features: int
    sparsity_ratio: float
    selected_base_features: list[int]
    active_tree_features: list[int]


class _FeatureExpansionMixin:
    poly_degree: int
    use_trig: bool

    def _expand_features(self, X):
        blocks = [X]
        for degree in range(2, self.poly_degree + 1):
            blocks.append(X ** degree)
        if self.use_trig:
            blocks.append(op.sin(X))
        return op.hstack(blocks) if len(blocks) > 1 else X

    def _derive_active_features(self, coef, raw_feature_count: int, threshold: float, fallback_ratio: float) -> ActiveSetInfo:
        mags = opn_vector_magnitudes(coef)
        selected_base = np.flatnonzero(mags > threshold)
        if len(selected_base) == 0 and len(mags) > 0:
            # Fallback for weak Lasso sparsity: keep top ratio by coefficient magnitude.
            k = max(1, int(np.ceil(raw_feature_count * fallback_ratio)))
            selected_base = np.argsort(mags)[-k:]
        active_tree = sorted(set(int(i % raw_feature_count) for i in selected_base))
        if not active_tree:
            active_tree = list(range(raw_feature_count))
        return ActiveSetInfo(
            candidate_features=int(raw_feature_count),
            selected_features=int(len(active_tree)),
            sparsity_ratio=1.0 - (len(active_tree) / max(1, raw_feature_count)),
            selected_base_features=[int(i) for i in selected_base],
            active_tree_features=[int(i) for i in active_tree],
        )


class OPNsHybridRegressor(_FeatureExpansionMixin, BaseEstimator, RegressorMixin):
    """Refactored regression estimator for OPNs-HybridBoost."""

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        l2_leaf_reg: float = 0.1,
        poly_degree: int = 2,
        use_trig: bool = False,
        lasso_alpha: float = 0.001,
        lasso_max_iter: int = 200,
        feature_filter_ratio: float = 1.0,
        active_threshold: float = 1e-10,
        random_strength: float = 0.0,
        lr_decay_type: str = "constant",
        # early_stopping_rounds: int = 20,
        early_stopping_rounds: Optional[int] = None,
        max_thresholds: int = 20,
        tree_feature_mode: str = "all",
        active_plus_ratio: float = 0.5,
        residual_gain_ratio: float = 0.25,
        residual_gain_max_thresholds: int = 10,
        residual_gain_gate_quantile: float = 0.25,
        residual_gain_stability_repeats: int = 5,
        residual_gain_subsample: float = 0.8,
        residual_gain_min_frequency: float = 0.6,
        residual_gain_crossfit_repeats: int = 5,
        residual_gain_crossfit_validation_fraction: float = 0.2,
        residual_gain_crossfit_min_frequency: float = 0.6,
        split_score_backend: str = "fast_exact",
        threshold_backend: str = "ordered_exact",
        threshold_sampling: str = "stride",
        prediction_backend: str = "object_batch",
        random_state: int = 42,
        verbose: bool = False,
        progress_interval: int = 100,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.l2_leaf_reg = l2_leaf_reg
        self.poly_degree = poly_degree
        self.use_trig = use_trig
        self.lasso_alpha = lasso_alpha
        self.lasso_max_iter = lasso_max_iter
        self.feature_filter_ratio = feature_filter_ratio
        self.active_threshold = active_threshold
        self.random_strength = random_strength
        self.lr_decay_type = lr_decay_type
        self.early_stopping_rounds = early_stopping_rounds
        self.max_thresholds = max_thresholds
        self.random_state = random_state
        self.verbose = bool(verbose)
        self.progress_interval = int(progress_interval)
        if self.progress_interval <= 0:
            raise ValueError("progress_interval must be a positive integer.")
        self.tree_feature_mode = tree_feature_mode
        self.active_plus_ratio = active_plus_ratio
        self.residual_gain_ratio = residual_gain_ratio
        self.residual_gain_max_thresholds = residual_gain_max_thresholds
        self.residual_gain_gate_quantile = residual_gain_gate_quantile
        self.residual_gain_stability_repeats = residual_gain_stability_repeats
        self.residual_gain_subsample = residual_gain_subsample
        self.residual_gain_min_frequency = residual_gain_min_frequency
        self.residual_gain_crossfit_repeats = residual_gain_crossfit_repeats
        self.residual_gain_crossfit_validation_fraction = residual_gain_crossfit_validation_fraction
        self.residual_gain_crossfit_min_frequency = residual_gain_crossfit_min_frequency
        self.split_score_backend = split_score_backend
        self.threshold_backend = threshold_backend
        self.threshold_sampling = threshold_sampling
        self.prediction_backend = _validate_prediction_backend(prediction_backend)

    def _lr(self, iteration: int) -> float:
        if self.lr_decay_type == "cosine":
            return self.learning_rate * 0.5 * (1.0 + np.cos(np.pi * iteration / max(1, self.n_estimators)))
        return self.learning_rate

    def _progress(self, message: str) -> None:
        """Emit one flushed progress line when verbose mode is enabled."""
        if self.verbose:
            print(f"    [OPNs-HybridBoost] {message}", flush=True)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        if seconds < 60.0:
            return f"{seconds:.1f}s"
        minutes, sec = divmod(seconds, 60.0)
        if minutes < 60.0:
            return f"{int(minutes)}m {sec:.0f}s"
        hours, minutes = divmod(minutes, 60.0)
        return f"{int(hours)}h {int(minutes)}m"

    @staticmethod
    def _as_component_matrix(values) -> tuple[np.ndarray, np.ndarray]:
        """Return the two OPNs components as two-dimensional float arrays."""
        if isinstance(values, OPNsMatrix):
            left = np.asarray(values.left_matrix, dtype=float)
            right = np.asarray(values.right_matrix, dtype=float)
        else:
            arr = np.asarray(values, dtype=float)
            left = arr
            right = np.zeros_like(arr, dtype=float)

        if left.ndim == 1:
            left = left.reshape(-1, 1)
        if right.ndim == 1:
            right = right.reshape(-1, 1)
        return left, right

    def _residual_feature_scores(self, X, residuals) -> np.ndarray:
        """Rank raw OPNs pair features by initial residual correlation.

        OPNs comparisons are driven primarily by the ordered-pair sum.  The
        screening signal therefore uses ``-(a+b)`` for both the feature and the
        initial residual.  This screening is performed only on the training
        fold and does not use validation/test targets.
        """
        x_left, x_right = self._as_component_matrix(X)
        r_left, r_right = self._as_component_matrix(residuals)

        x_signal = -(x_left + x_right)
        r_signal = -(r_left + r_right).reshape(-1)

        x_centered = x_signal - np.mean(x_signal, axis=0, keepdims=True)
        r_centered = r_signal - np.mean(r_signal)

        numerator = np.abs(x_centered.T @ r_centered)
        x_norm = np.sqrt(np.sum(x_centered * x_centered, axis=0))
        r_norm = float(np.sqrt(np.sum(r_centered * r_centered)))
        denominator = x_norm * r_norm

        scores = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > 1e-15,
        )
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _exact_component_route_values(
        feature_left: np.ndarray,
        feature_right: np.ndarray,
        threshold_left: float,
        threshold_right: float,
    ) -> np.ndarray:
        """Vectorized OPNs routing for raw implementation coordinates.

        The subtraction and summation order intentionally matches
        ``OPNsMatrix.__gt__``.  This preserves the native finite-precision
        ordering while avoiding per-sample OPNs object construction.
        """
        delta_left = feature_left - float(threshold_left)
        delta_right = feature_right - float(threshold_right)
        delta_sum = delta_left + delta_right
        return (delta_sum < 0.0) | (
            (delta_sum == 0.0) & (delta_left > 0.0)
        )

    @classmethod
    def _exact_component_route(
        cls,
        feature_left: np.ndarray,
        feature_right: np.ndarray,
        threshold: OPNs,
    ) -> np.ndarray:
        """Vectorized ``feature > threshold`` matching the native OPNs order."""
        return cls._exact_component_route_values(
            feature_left,
            feature_right,
            float(threshold.a),
            float(threshold.b),
        )

    def _rank_features_by_initial_residual_gain(
        self,
        X,
        residuals,
        candidate_features: np.ndarray,
        *,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Rank features by their best depth-one OPNs residual split.

        For feature ``j`` and screening thresholds ``Theta_j``, the method
        minimizes the exact OPNs within-child residual dispersion

            J_j* = min_theta sum_c |I_c| Var_OPNs(r_0[I_c]).

        The unsplit parent term is common to every feature. Therefore ordering
        features by increasing ``J_j*`` is exactly equivalent to ordering them
        by decreasing residual gain. Scores are compared with the established
        OPNs total order, not with an arbitrary real scalarization.

        ``show_progress=False`` is used by stability selection so repeated
        internal screenings do not flood the terminal.
        """
        candidate_features = np.unique(np.asarray(candidate_features, dtype=int))
        if candidate_features.size == 0:
            self.residual_gain_valid_features_ = 0
            self.residual_gain_screen_time_ = 0.0
            return candidate_features
        if not isinstance(X, OPNsMatrix) or not isinstance(residuals, OPNsMatrix):
            raise TypeError(
                "Residual-gain screening requires OPNsMatrix features and residuals."
            )
        threshold_budget = int(self.residual_gain_max_thresholds)
        if threshold_budget <= 0:
            raise ValueError("residual_gain_max_thresholds must be positive.")

        X_left, X_right = self._as_component_matrix(X)
        residual_left, residual_right = self._as_component_matrix(residuals)
        grad_left = residual_left.reshape(-1)
        grad_right = residual_right.reshape(-1)
        node_indices = np.zeros(X_left.shape[0], dtype=np.int64)
        rng = np.random.default_rng(self.random_state)

        screener = OPNsObliviousTree(
            max_depth=1,
            l2_leaf_reg=0.0,
            colsample_bylevel=1.0,
            random_strength=0.0,
            max_thresholds=threshold_budget,
            random_state=self.random_state,
            split_score_backend="fast_exact",
            threshold_backend=self.threshold_backend,
            threshold_sampling=self.threshold_sampling,
        )

        n_features = int(self.raw_feature_count_)
        best_left = np.full(n_features, np.nan, dtype=float)
        best_right = np.full(n_features, np.nan, dtype=float)
        best_threshold_left = np.full(n_features, np.nan, dtype=float)
        best_threshold_right = np.full(n_features, np.nan, dtype=float)
        valid = np.zeros(n_features, dtype=bool)
        started = time.perf_counter()
        total_candidates = int(candidate_features.size)
        if show_progress:
            self._progress(
                "Residual-gain screening: "
                f"features={total_candidates}, thresholds/feature={threshold_budget}."
            )

        for feature_idx in candidate_features:
            feature_idx = int(feature_idx)
            thresholds = screener._thresholds(X[:, feature_idx])
            feature_left = X_left[:, feature_idx]
            feature_right = X_right[:, feature_idx]
            best_score = None
            best_threshold = None

            for threshold in thresholds:
                is_right = self._exact_component_route(
                    feature_left, feature_right, threshold
                )
                if not np.any(is_right) or np.all(is_right):
                    continue
                score, score_valid = screener._split_score_fast(
                    node_indices,
                    0,
                    is_right,
                    grad_left,
                    grad_right,
                    rng,
                )
                if not score_valid:
                    continue
                if best_score is None or OPNsObliviousTree._opn_components_lt(
                    score[0], score[1], best_score[0], best_score[1]
                ):
                    best_score = score
                    best_threshold = threshold

            if best_score is not None and best_threshold is not None:
                best_left[feature_idx] = float(best_score[0])
                best_right[feature_idx] = float(best_score[1])
                best_threshold_left[feature_idx] = float(best_threshold.a)
                best_threshold_right[feature_idx] = float(best_threshold.b)
                valid[feature_idx] = True

        self.residual_gain_screen_time_ = time.perf_counter() - started
        self.residual_gain_valid_features_ = int(np.sum(valid[candidate_features]))
        self.residual_gain_best_score_left_ = best_left
        self.residual_gain_best_score_right_ = best_right
        self.residual_gain_best_threshold_left_ = best_threshold_left
        self.residual_gain_best_threshold_right_ = best_threshold_right

        ranked = candidate_features[valid[candidate_features]]
        if ranked.size == 0:
            self.residual_gain_ranked_features_ = np.empty(0, dtype=int)
            if show_progress:
                self._progress(
                    "Residual-gain screening completed: no valid feature, "
                    f"time={self._format_seconds(self.residual_gain_screen_time_)}."
                )
            return ranked

        def compare_features(i: int, j: int) -> int:
            i = int(i)
            j = int(j)
            if OPNsObliviousTree._opn_components_lt(
                best_left[i], best_right[i], best_left[j], best_right[j]
            ):
                return -1
            if OPNsObliviousTree._opn_components_lt(
                best_left[j], best_right[j], best_left[i], best_right[i]
            ):
                return 1
            return -1 if i < j else (1 if i > j else 0)

        ranked = np.asarray(
            sorted(ranked.tolist(), key=cmp_to_key(compare_features)),
            dtype=int,
        )
        self.residual_gain_ranked_features_ = ranked
        if show_progress:
            self._progress(
                "Residual-gain screening completed: "
                f"valid={ranked.size}/{total_candidates}, "
                f"time={self._format_seconds(self.residual_gain_screen_time_)}."
            )
        return ranked

    def _residual_gain_gate_partition(
        self,
        ranked_all: np.ndarray,
        active: np.ndarray,
        inactive: np.ndarray,
        gate_quantile: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float]:
        """Apply the active-calibrated OPNs gain gate to one screening run."""
        ranked_all = np.asarray(ranked_all, dtype=int)
        ranked_active = ranked_all[np.isin(ranked_all, active)]
        ranked_inactive = ranked_all[np.isin(ranked_all, inactive)]
        if ranked_active.size == 0 or ranked_inactive.size == 0:
            return (
                ranked_active,
                ranked_inactive,
                np.empty(0, dtype=int),
                -1,
                np.nan,
                np.nan,
            )

        gate_position = min(
            ranked_active.size - 1,
            max(0, int(np.ceil(gate_quantile * ranked_active.size)) - 1),
        )
        gate_feature = int(ranked_active[gate_position])
        gate_left = float(self.residual_gain_best_score_left_[gate_feature])
        gate_right = float(self.residual_gain_best_score_right_[gate_feature])

        qualified: list[int] = []
        for feature_idx in ranked_inactive:
            feature_idx = int(feature_idx)
            score_left = float(self.residual_gain_best_score_left_[feature_idx])
            score_right = float(self.residual_gain_best_score_right_[feature_idx])
            gate_is_strictly_better = OPNsObliviousTree._opn_components_lt(
                gate_left, gate_right, score_left, score_right
            )
            if not gate_is_strictly_better:
                qualified.append(feature_idx)

        return (
            ranked_active,
            ranked_inactive,
            np.asarray(qualified, dtype=int),
            gate_feature,
            gate_left,
            gate_right,
        )

    @staticmethod
    def _rank_features_from_component_scores(
        features: np.ndarray,
        score_left: np.ndarray,
        score_right: np.ndarray,
    ) -> np.ndarray:
        """Sort feature indices by the native OPNs score order."""
        features = np.asarray(features, dtype=int)
        if features.size == 0:
            return features

        def compare_features(i: int, j: int) -> int:
            i = int(i)
            j = int(j)
            if OPNsObliviousTree._opn_components_lt(
                float(score_left[i]),
                float(score_right[i]),
                float(score_left[j]),
                float(score_right[j]),
            ):
                return -1
            if OPNsObliviousTree._opn_components_lt(
                float(score_left[j]),
                float(score_right[j]),
                float(score_left[i]),
                float(score_right[i]),
            ):
                return 1
            return -1 if i < j else (1 if i > j else 0)

        return np.asarray(
            sorted(features.tolist(), key=cmp_to_key(compare_features)),
            dtype=int,
        )

    def _score_fixed_thresholds_on_holdout(
        self,
        X,
        residuals,
        candidate_features: np.ndarray,
        threshold_left: np.ndarray,
        threshold_right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
        """Evaluate training-selected thresholds on an independent holdout.

        For each feature, the threshold is fixed before the holdout is inspected.
        The returned ``positive_gain`` flag is true exactly when the holdout
        within-child OPNs dispersion is strictly smaller than the unsplit
        holdout parent dispersion under the native OPNs total order.
        """
        if not isinstance(X, OPNsMatrix) or not isinstance(residuals, OPNsMatrix):
            raise TypeError(
                "Cross-fitted residual-gain validation requires OPNsMatrix inputs."
            )

        candidate_features = np.unique(np.asarray(candidate_features, dtype=int))
        n_features = int(self.raw_feature_count_)
        score_left = np.full(n_features, np.nan, dtype=float)
        score_right = np.full(n_features, np.nan, dtype=float)
        valid = np.zeros(n_features, dtype=bool)
        positive_gain = np.zeros(n_features, dtype=bool)

        X_left, X_right = self._as_component_matrix(X)
        residual_left, residual_right = self._as_component_matrix(residuals)
        grad_left = residual_left.reshape(-1)
        grad_right = residual_right.reshape(-1)
        n_samples = int(grad_left.size)
        if n_samples < 2:
            return score_left, score_right, valid, positive_gain, (np.nan, np.nan)

        parent_mask = np.ones(n_samples, dtype=bool)
        parent_left, parent_right, parent_valid = (
            OPNsObliviousTree._variance_score_components(
                grad_left,
                grad_right,
                parent_mask,
            )
        )
        if not parent_valid:
            return score_left, score_right, valid, positive_gain, (np.nan, np.nan)

        node_indices = np.zeros(n_samples, dtype=np.int64)
        rng = np.random.default_rng(self.random_state)
        screener = OPNsObliviousTree(
            max_depth=1,
            l2_leaf_reg=0.0,
            colsample_bylevel=1.0,
            random_strength=0.0,
            max_thresholds=max(1, int(self.residual_gain_max_thresholds)),
            random_state=self.random_state,
            split_score_backend="fast_exact",
            threshold_backend=self.threshold_backend,
            threshold_sampling=self.threshold_sampling,
        )

        for feature_idx in candidate_features:
            feature_idx = int(feature_idx)
            threshold_a = float(threshold_left[feature_idx])
            threshold_b = float(threshold_right[feature_idx])
            if not np.isfinite(threshold_a) or not np.isfinite(threshold_b):
                continue

            is_right = self._exact_component_route_values(
                X_left[:, feature_idx],
                X_right[:, feature_idx],
                threshold_a,
                threshold_b,
            )
            if not np.any(is_right) or np.all(is_right):
                continue

            score, score_valid = screener._split_score_fast(
                node_indices,
                0,
                is_right,
                grad_left,
                grad_right,
                rng,
            )
            if not score_valid:
                continue

            score_left[feature_idx] = float(score[0])
            score_right[feature_idx] = float(score[1])
            valid[feature_idx] = True
            positive_gain[feature_idx] = OPNsObliviousTree._opn_components_lt(
                float(score[0]),
                float(score[1]),
                float(parent_left),
                float(parent_right),
            )

        return (
            score_left,
            score_right,
            valid,
            positive_gain,
            (float(parent_left), float(parent_right)),
        )

    def _prepare_tree_feature_pool(
        self,
        X,
        residuals,
        *,
        y=None,
        X_poly=None,
    ):
        """Resolve the fixed feature pool used by all residual trees.

        ``y`` and the unscaled expanded matrix ``X_poly`` are required only by
        ``residual_gain_crossfit``.  That mode refits the Phase-I sparse base
        learner inside each screening split so holdout residuals are genuinely
        out-of-fold rather than in-sample residuals from the outer-fold model.
        """
        mode = self.tree_feature_mode
        self.active_plus_added_features_ = 0
        self.residual_gain_added_features_ = 0
        self.residual_gain_screen_time_ = 0.0
        self.residual_gain_valid_features_ = 0
        self.residual_gain_active_valid_features_ = 0
        self.residual_gain_gate_qualified_features_ = 0
        self.residual_gain_gate_feature_ = -1
        self.residual_gain_gate_score_left_ = np.nan
        self.residual_gain_gate_score_right_ = np.nan
        self.residual_gain_max_added_features_ = 0
        self.residual_gain_stability_qualified_features_ = 0
        self.residual_gain_stability_mean_qualified_per_repeat_ = 0.0
        self.residual_gain_stability_selected_frequency_mean_ = 0.0
        self.residual_gain_stability_selected_frequency_min_ = 0.0
        self.residual_gain_stability_selected_frequency_max_ = 0.0
        self.residual_gain_stability_active_valid_mean_ = 0.0
        self.residual_gain_stability_inactive_valid_mean_ = 0.0
        self.residual_gain_stability_inactive_frequency_mean_ = 0.0
        self.residual_gain_stability_inactive_frequency_max_ = 0.0
        self.residual_gain_stability_subsample_size_ = 0
        self.residual_gain_crossfit_qualified_features_ = 0
        self.residual_gain_crossfit_mean_qualified_per_repeat_ = 0.0
        self.residual_gain_crossfit_selected_frequency_mean_ = 0.0
        self.residual_gain_crossfit_selected_frequency_min_ = 0.0
        self.residual_gain_crossfit_selected_frequency_max_ = 0.0
        self.residual_gain_crossfit_active_positive_mean_ = 0.0
        self.residual_gain_crossfit_inactive_positive_mean_ = 0.0
        self.residual_gain_crossfit_inactive_valid_mean_ = 0.0
        self.residual_gain_crossfit_train_size_ = 0
        self.residual_gain_crossfit_validation_size_ = 0
        self.residual_gain_crossfit_repeats_completed_ = 0
        self.residual_gain_crossfit_no_active_gate_repeats_ = 0

        if mode == "all":
            return None

        if mode == "active":
            return self.active_features_

        if mode == "auto":
            if self.raw_feature_count_ <= 100:
                return None
            return self.active_features_

        all_features = np.arange(self.raw_feature_count_, dtype=int)
        active = np.unique(np.asarray(self.active_features_, dtype=int))
        inactive = np.setdiff1d(all_features, active, assume_unique=False)

        if mode == "active_plus":
            ratio = float(self.active_plus_ratio)
            if not 0.0 <= ratio <= 1.0:
                raise ValueError("active_plus_ratio must be in [0, 1].")
            if inactive.size == 0 or ratio <= 0.0:
                return active

            n_add = min(inactive.size, max(1, int(np.ceil(ratio * inactive.size))))
            scores = self._residual_feature_scores(X, residuals)
            # Stable ordering makes repeated runs deterministic when scores tie.
            ranked_inactive = inactive[
                np.argsort(-scores[inactive], kind="mergesort")
            ]
            added = ranked_inactive[:n_add]
            pool = np.unique(np.concatenate([active, added])).astype(int)

            self.active_plus_added_features_ = int(len(pool) - len(active))
            self.active_plus_feature_scores_ = scores
            return pool

        if mode == "residual_gain_topk":
            ratio = float(self.residual_gain_ratio)
            if not 0.0 <= ratio <= 1.0:
                raise ValueError("residual_gain_ratio must be in [0, 1].")
            if inactive.size == 0 or ratio <= 0.0:
                return active

            n_add = min(inactive.size, max(1, int(np.ceil(ratio * inactive.size))))
            self.residual_gain_max_added_features_ = int(n_add)
            ranked_inactive = self._rank_features_by_initial_residual_gain(
                X, residuals, inactive
            )
            added = ranked_inactive[:n_add]
            pool = np.unique(np.concatenate([active, added])).astype(int)
            self.residual_gain_added_features_ = int(len(pool) - len(active))
            return pool

        if mode == "residual_gain_adaptive":
            ratio = float(self.residual_gain_ratio)
            gate_quantile = float(self.residual_gain_gate_quantile)
            if not 0.0 <= ratio <= 1.0:
                raise ValueError("residual_gain_ratio must be in [0, 1].")
            if not 0.0 < gate_quantile <= 1.0:
                raise ValueError(
                    "residual_gain_gate_quantile must be in (0, 1]."
                )
            if inactive.size == 0 or ratio <= 0.0:
                return active

            ranked_all = self._rank_features_by_initial_residual_gain(
                X, residuals, all_features
            )
            (
                ranked_active,
                ranked_inactive,
                qualified,
                gate_feature,
                gate_left,
                gate_right,
            ) = self._residual_gain_gate_partition(
                ranked_all, active, inactive, gate_quantile
            )

            self.residual_gain_active_valid_features_ = int(ranked_active.size)
            self.residual_gain_valid_features_ = int(ranked_inactive.size)
            n_add_max = min(
                inactive.size,
                max(1, int(np.ceil(ratio * inactive.size))),
            )
            self.residual_gain_max_added_features_ = int(n_add_max)
            self.residual_gain_gate_feature_ = int(gate_feature)
            self.residual_gain_gate_score_left_ = float(gate_left)
            self.residual_gain_gate_score_right_ = float(gate_right)
            self.residual_gain_gate_qualified_features_ = int(qualified.size)

            added = qualified[:n_add_max]
            pool = np.unique(np.concatenate([active, added])).astype(int)
            self.residual_gain_added_features_ = int(len(pool) - len(active))
            self._progress(
                "Adaptive gain gate: "
                f"qualified={qualified.size}, "
                f"added={self.residual_gain_added_features_}/{n_add_max}, "
                f"search_features={len(pool)}/{self.raw_feature_count_}."
            )
            return pool

        if mode == "residual_gain_stability":
            ratio = float(self.residual_gain_ratio)
            gate_quantile = float(self.residual_gain_gate_quantile)
            repeats = int(self.residual_gain_stability_repeats)
            subsample = float(self.residual_gain_subsample)
            min_frequency = float(self.residual_gain_min_frequency)

            if not 0.0 <= ratio <= 1.0:
                raise ValueError("residual_gain_ratio must be in [0, 1].")
            if not 0.0 < gate_quantile <= 1.0:
                raise ValueError(
                    "residual_gain_gate_quantile must be in (0, 1]."
                )
            if repeats <= 0:
                raise ValueError(
                    "residual_gain_stability_repeats must be positive."
                )
            if not 0.0 < subsample <= 1.0:
                raise ValueError("residual_gain_subsample must be in (0, 1].")
            if not 0.0 < min_frequency <= 1.0:
                raise ValueError(
                    "residual_gain_min_frequency must be in (0, 1]."
                )
            if inactive.size == 0 or ratio <= 0.0:
                return active

            n_samples = int(X.shape[0])
            if n_samples < 2:
                return active
            subsample_size = min(
                n_samples,
                max(2, int(np.ceil(subsample * n_samples))),
            )
            n_add_max = min(
                inactive.size,
                max(1, int(np.ceil(ratio * inactive.size))),
            )
            self.residual_gain_max_added_features_ = int(n_add_max)
            self.residual_gain_stability_subsample_size_ = int(subsample_size)

            pass_counts = np.zeros(self.raw_feature_count_, dtype=int)
            rank_sums = np.zeros(self.raw_feature_count_, dtype=float)
            active_valid_counts: list[int] = []
            inactive_valid_counts: list[int] = []
            qualified_counts: list[int] = []
            rank_penalty = float(inactive.size + 1)
            screening_started = time.perf_counter()
            rng = np.random.default_rng(self.random_state + 104729)

            self._progress(
                "Stability gain screening: "
                f"repeats={repeats}, subsample={subsample_size}/{n_samples}, "
                f"min_frequency={min_frequency:.2f}."
            )

            for _ in range(repeats):
                if subsample_size == n_samples:
                    sample_idx = np.arange(n_samples, dtype=int)
                else:
                    sample_idx = np.sort(
                        rng.choice(
                            n_samples,
                            size=subsample_size,
                            replace=False,
                        )
                    )

                ranked_all = self._rank_features_by_initial_residual_gain(
                    X[sample_idx],
                    residuals[sample_idx],
                    all_features,
                    show_progress=False,
                )
                (
                    ranked_active,
                    ranked_inactive,
                    qualified,
                    _,
                    _,
                    _,
                ) = self._residual_gain_gate_partition(
                    ranked_all, active, inactive, gate_quantile
                )

                active_valid_counts.append(int(ranked_active.size))
                inactive_valid_counts.append(int(ranked_inactive.size))
                qualified_counts.append(int(qualified.size))
                pass_counts[qualified] += 1

                # Average ordinal gain rank is used only after the OPNs gate.
                # Invalid features receive a conservative worst-rank penalty.
                rank_sums[inactive] += rank_penalty
                for rank_position, feature_idx in enumerate(
                    ranked_inactive, start=1
                ):
                    rank_sums[int(feature_idx)] += (
                        float(rank_position) - rank_penalty
                    )

            total_screen_time = time.perf_counter() - screening_started
            frequencies = pass_counts.astype(float) / float(repeats)
            mean_ranks = rank_sums / float(repeats)
            eligible = inactive[frequencies[inactive] >= min_frequency]

            ordered_eligible = np.asarray(
                sorted(
                    eligible.tolist(),
                    key=lambda feature_idx: (
                        float(mean_ranks[int(feature_idx)]),
                        -float(frequencies[int(feature_idx)]),
                        int(feature_idx),
                    ),
                ),
                dtype=int,
            )
            added = ordered_eligible[:n_add_max]
            pool = np.unique(np.concatenate([active, added])).astype(int)

            self.residual_gain_screen_time_ = float(total_screen_time)
            self.residual_gain_active_valid_features_ = float(
                np.mean(active_valid_counts)
            )
            self.residual_gain_valid_features_ = float(
                np.mean(inactive_valid_counts)
            )
            self.residual_gain_gate_qualified_features_ = int(eligible.size)
            self.residual_gain_stability_qualified_features_ = int(eligible.size)
            self.residual_gain_stability_mean_qualified_per_repeat_ = float(
                np.mean(qualified_counts)
            )
            self.residual_gain_stability_active_valid_mean_ = float(
                np.mean(active_valid_counts)
            )
            self.residual_gain_stability_inactive_valid_mean_ = float(
                np.mean(inactive_valid_counts)
            )
            inactive_frequencies = frequencies[inactive]
            self.residual_gain_stability_inactive_frequency_mean_ = float(
                np.mean(inactive_frequencies)
            )
            self.residual_gain_stability_inactive_frequency_max_ = float(
                np.max(inactive_frequencies)
            )
            self.residual_gain_added_features_ = int(len(pool) - len(active))

            selected_frequencies = frequencies[added]
            if selected_frequencies.size:
                self.residual_gain_stability_selected_frequency_mean_ = float(
                    np.mean(selected_frequencies)
                )
                self.residual_gain_stability_selected_frequency_min_ = float(
                    np.min(selected_frequencies)
                )
                self.residual_gain_stability_selected_frequency_max_ = float(
                    np.max(selected_frequencies)
                )

            self._progress(
                "Stability gain gate completed: "
                f"stable={eligible.size}, "
                f"added={self.residual_gain_added_features_}/{n_add_max}, "
                f"search_features={len(pool)}/{self.raw_feature_count_}, "
                f"time={self._format_seconds(total_screen_time)}."
            )
            return pool

        if mode == "residual_gain_crossfit":
            ratio = float(self.residual_gain_ratio)
            gate_quantile = float(self.residual_gain_gate_quantile)
            repeats = int(self.residual_gain_crossfit_repeats)
            validation_fraction = float(
                self.residual_gain_crossfit_validation_fraction
            )
            min_frequency = float(self.residual_gain_crossfit_min_frequency)

            if not 0.0 <= ratio <= 1.0:
                raise ValueError("residual_gain_ratio must be in [0, 1].")
            if not 0.0 < gate_quantile <= 1.0:
                raise ValueError(
                    "residual_gain_gate_quantile must be in (0, 1]."
                )
            if repeats <= 0:
                raise ValueError(
                    "residual_gain_crossfit_repeats must be positive."
                )
            if not 0.0 < validation_fraction < 1.0:
                raise ValueError(
                    "residual_gain_crossfit_validation_fraction must be in (0, 1)."
                )
            if not 0.0 < min_frequency <= 1.0:
                raise ValueError(
                    "residual_gain_crossfit_min_frequency must be in (0, 1]."
                )
            if inactive.size == 0 or ratio <= 0.0:
                return active
            if y is None or X_poly is None:
                raise ValueError(
                    "residual_gain_crossfit requires y and X_poly so the "
                    "Phase-I base learner can be refitted inside each "
                    "screening split."
                )

            n_samples = int(X.shape[0])
            if n_samples < 4:
                return active
            validation_size = min(
                n_samples - 2,
                max(2, int(np.ceil(validation_fraction * n_samples))),
            )
            train_size = n_samples - validation_size
            if train_size < 2:
                return active

            n_add_max = min(
                inactive.size,
                max(1, int(np.ceil(ratio * inactive.size))),
            )
            self.residual_gain_max_added_features_ = int(n_add_max)
            self.residual_gain_crossfit_train_size_ = int(train_size)
            self.residual_gain_crossfit_validation_size_ = int(validation_size)

            pass_counts = np.zeros(self.raw_feature_count_, dtype=int)
            rank_sums = np.zeros(self.raw_feature_count_, dtype=float)
            qualified_counts: list[int] = []
            active_positive_counts: list[int] = []
            inactive_positive_counts: list[int] = []
            inactive_valid_counts: list[int] = []
            no_active_gate_repeats = 0
            rank_penalty = float(inactive.size + 1)
            screening_started = time.perf_counter()
            rng = np.random.default_rng(self.random_state + 130363)

            self._progress(
                "Cross-fitted gain screening: "
                f"repeats={repeats}, train/validation="
                f"{train_size}/{validation_size}, "
                f"min_frequency={min_frequency:.2f}."
            )

            for repeat_idx in range(repeats):
                permutation = rng.permutation(n_samples)
                validation_idx = np.sort(permutation[:validation_size])
                screening_train_idx = np.sort(permutation[validation_size:])

                inner_scaler = OPNsStandardScaler()
                X_poly_train = inner_scaler.fit_transform(
                    X_poly[screening_train_idx]
                )
                X_poly_validation = inner_scaler.transform(
                    X_poly[validation_idx]
                )
                inner_base_model = Lasso(
                    alpha=self.lasso_alpha,
                    max_iter=self.lasso_max_iter,
                    adaptive_lr=True,
                )
                inner_base_model.fit(
                    X_poly_train,
                    y[screening_train_idx],
                )
                inner_active_set = self._derive_active_features(
                    inner_base_model.coef_,
                    raw_feature_count=self.raw_feature_count_,
                    threshold=self.active_threshold,
                    fallback_ratio=self.feature_filter_ratio,
                )
                inner_active = np.asarray(
                    inner_active_set.active_tree_features, dtype=int
                )
                inner_train_prediction = (
                    op.dot(X_poly_train, inner_base_model.coef_)
                    + inner_base_model.intercept_
                )
                inner_validation_prediction = (
                    op.dot(X_poly_validation, inner_base_model.coef_)
                    + inner_base_model.intercept_
                )
                inner_train_residuals = (
                    y[screening_train_idx] - inner_train_prediction
                )
                inner_validation_residuals = (
                    y[validation_idx] - inner_validation_prediction
                )

                self._rank_features_by_initial_residual_gain(
                    X[screening_train_idx],
                    inner_train_residuals,
                    all_features,
                    show_progress=False,
                )
                threshold_left = np.asarray(
                    self.residual_gain_best_threshold_left_, dtype=float
                ).copy()
                threshold_right = np.asarray(
                    self.residual_gain_best_threshold_right_, dtype=float
                ).copy()

                (
                    validation_left,
                    validation_right,
                    validation_valid,
                    validation_positive,
                    _,
                ) = self._score_fixed_thresholds_on_holdout(
                    X[validation_idx],
                    inner_validation_residuals,
                    all_features,
                    threshold_left,
                    threshold_right,
                )

                positive_active = inner_active[
                    validation_valid[inner_active]
                    & validation_positive[inner_active]
                ]
                positive_inactive = inactive[
                    validation_valid[inactive] & validation_positive[inactive]
                ]
                valid_inactive = inactive[validation_valid[inactive]]
                active_positive_counts.append(int(positive_active.size))
                inactive_positive_counts.append(int(positive_inactive.size))
                inactive_valid_counts.append(int(valid_inactive.size))

                rank_sums[inactive] += rank_penalty
                if positive_active.size == 0 or positive_inactive.size == 0:
                    qualified_counts.append(0)
                    if positive_active.size == 0:
                        no_active_gate_repeats += 1
                    continue

                ranked_active = self._rank_features_from_component_scores(
                    positive_active,
                    validation_left,
                    validation_right,
                )
                ranked_inactive = self._rank_features_from_component_scores(
                    positive_inactive,
                    validation_left,
                    validation_right,
                )
                gate_position = min(
                    ranked_active.size - 1,
                    max(
                        0,
                        int(np.ceil(gate_quantile * ranked_active.size)) - 1,
                    ),
                )
                gate_feature = int(ranked_active[gate_position])
                gate_left = float(validation_left[gate_feature])
                gate_right = float(validation_right[gate_feature])

                qualified: list[int] = []
                for feature_idx in ranked_inactive:
                    feature_idx = int(feature_idx)
                    gate_is_strictly_better = (
                        OPNsObliviousTree._opn_components_lt(
                            gate_left,
                            gate_right,
                            float(validation_left[feature_idx]),
                            float(validation_right[feature_idx]),
                        )
                    )
                    if not gate_is_strictly_better:
                        qualified.append(feature_idx)

                qualified_array = np.asarray(qualified, dtype=int)
                qualified_counts.append(int(qualified_array.size))
                pass_counts[qualified_array] += 1

                for rank_position, feature_idx in enumerate(
                    qualified_array, start=1
                ):
                    rank_sums[int(feature_idx)] += (
                        float(rank_position) - rank_penalty
                    )

            total_screen_time = time.perf_counter() - screening_started
            frequencies = pass_counts.astype(float) / float(repeats)
            mean_ranks = rank_sums / float(repeats)
            eligible = inactive[frequencies[inactive] >= min_frequency]
            ordered_eligible = np.asarray(
                sorted(
                    eligible.tolist(),
                    key=lambda feature_idx: (
                        -float(frequencies[int(feature_idx)]),
                        float(mean_ranks[int(feature_idx)]),
                        int(feature_idx),
                    ),
                ),
                dtype=int,
            )
            added = ordered_eligible[:n_add_max]
            pool = np.unique(np.concatenate([active, added])).astype(int)

            self.residual_gain_screen_time_ = float(total_screen_time)
            self.residual_gain_active_valid_features_ = float(
                np.mean(active_positive_counts)
            )
            self.residual_gain_valid_features_ = float(
                np.mean(inactive_valid_counts)
            )
            self.residual_gain_gate_qualified_features_ = int(eligible.size)
            self.residual_gain_crossfit_qualified_features_ = int(eligible.size)
            self.residual_gain_crossfit_mean_qualified_per_repeat_ = float(
                np.mean(qualified_counts)
            )
            self.residual_gain_crossfit_active_positive_mean_ = float(
                np.mean(active_positive_counts)
            )
            self.residual_gain_crossfit_inactive_positive_mean_ = float(
                np.mean(inactive_positive_counts)
            )
            self.residual_gain_crossfit_inactive_valid_mean_ = float(
                np.mean(inactive_valid_counts)
            )
            self.residual_gain_crossfit_repeats_completed_ = int(repeats)
            self.residual_gain_crossfit_no_active_gate_repeats_ = int(
                no_active_gate_repeats
            )
            self.residual_gain_added_features_ = int(len(pool) - len(active))

            selected_frequencies = frequencies[added]
            if selected_frequencies.size:
                self.residual_gain_crossfit_selected_frequency_mean_ = float(
                    np.mean(selected_frequencies)
                )
                self.residual_gain_crossfit_selected_frequency_min_ = float(
                    np.min(selected_frequencies)
                )
                self.residual_gain_crossfit_selected_frequency_max_ = float(
                    np.max(selected_frequencies)
                )

            self._progress(
                "Cross-fitted gain gate completed: "
                f"eligible={eligible.size}, "
                f"added={self.residual_gain_added_features_}/{n_add_max}, "
                f"search_features={len(pool)}/{self.raw_feature_count_}, "
                f"time={self._format_seconds(total_screen_time)}."
            )
            return pool

        raise ValueError(
            f"Unknown tree_feature_mode={mode!r}. "
            "Expected one of: 'active', 'active_plus', "
            "'residual_gain_topk', 'residual_gain_adaptive', "
            "'residual_gain_stability', 'residual_gain_crossfit', "
            "'all', 'auto'."
        )

    def _tree_active_features(self):
        return getattr(self, "tree_features_", None)

    def _build_shared_threshold_cache(self, X) -> dict[int, OPNsMatrix]:
        """Precompute exact threshold candidates once for one training fold.

        The selected thresholds are deterministic functions of ``X`` and the
        fixed tree feature pool. Reusing them across trees changes neither split
        candidates nor predictions, but removes hundreds of repeated uniqueness
        and ordering passes in long boosting runs.
        """
        active_features = self._tree_active_features()
        if active_features is None:
            active_features = np.arange(self.raw_feature_count_, dtype=int)
        active_features = np.asarray(list(active_features), dtype=int)
        if len(active_features) == 0:
            active_features = np.arange(self.raw_feature_count_, dtype=int)

        template = OPNsObliviousTree(
            max_depth=self.max_depth,
            l2_leaf_reg=self.l2_leaf_reg,
            colsample_bylevel=self.feature_filter_ratio,
            random_strength=self.random_strength,
            max_thresholds=self.max_thresholds,
            random_state=self.random_state,
            split_score_backend=self.split_score_backend,
            threshold_backend=self.threshold_backend,
            threshold_sampling=self.threshold_sampling,
        )
        started = time.perf_counter()
        cache = {
            int(j): template._thresholds(X[:, int(j)])
            for j in active_features
        }
        self.threshold_cache_time_ = time.perf_counter() - started
        self.threshold_cache_features_ = int(len(cache))
        self.threshold_cache_candidates_ = int(sum(len(v) for v in cache.values()))
        return cache

    def fit(self, X, y, X_val=None, y_val=None):
        fit_started = time.perf_counter()
        np.random.seed(self.random_state)
        self.raw_feature_count_ = int(X.shape[1])
        self.trees_ = []
        self.best_iteration_ = -1
        self.tree_iteration_times_ = []
        self.tree_candidate_feature_evaluations_ = []
        self.tree_threshold_evaluations_ = []

        phase1_started = time.perf_counter()
        X_poly = self._expand_features(X)
        self.base_scaler_ = OPNsStandardScaler()
        X_poly_scaled = self.base_scaler_.fit_transform(X_poly)
        stage_started = time.perf_counter()
        self.base_model_ = Lasso(alpha=self.lasso_alpha, max_iter=self.lasso_max_iter, adaptive_lr=True)
        self.base_model_.fit(X_poly_scaled, y)
        self.active_set_ = self._derive_active_features(
            self.base_model_.coef_,
            raw_feature_count=self.raw_feature_count_,
            threshold=self.active_threshold,
            fallback_ratio=self.feature_filter_ratio,
        )
        self.active_features_ = np.asarray(self.active_set_.active_tree_features, dtype=int)
        self._progress(
            "Lasso warm start completed in "
            f"{self._format_seconds(time.perf_counter() - stage_started)}; "
            f"active={len(self.active_features_)}/{self.raw_feature_count_}."
        )

        F_curr = op.dot(X_poly_scaled, self.base_model_.coef_) + self.base_model_.intercept_
        loss = OPNsMSELoss()
        initial_residuals = loss.negative_gradient(y, F_curr)
        self.phase1_time_ = time.perf_counter() - phase1_started
        stage_started = time.perf_counter()
        self.tree_features_ = self._prepare_tree_feature_pool(
            X,
            initial_residuals,
            y=y,
            X_poly=X_poly,
        )
        resolved_feature_count = (
            self.raw_feature_count_
            if self.tree_features_ is None
            else int(len(self.tree_features_))
        )
        self.feature_pool_time_ = time.perf_counter() - stage_started
        if self.tree_feature_mode not in {
            "residual_gain_adaptive",
            "residual_gain_stability",
            "residual_gain_crossfit",
        }:
            self._progress(
                "Tree feature pool: "
                f"search_features={resolved_feature_count}/{self.raw_feature_count_}."
            )

        if self.n_estimators > 0:
            shared_threshold_cache = self._build_shared_threshold_cache(X)
        else:
            shared_threshold_cache = {}
            self.threshold_cache_time_ = 0.0
            self.threshold_cache_features_ = 0
            self.threshold_cache_candidates_ = 0
        best_val_loss = None
        no_improve = 0
        boost_started = time.perf_counter()

        for i in range(self.n_estimators):
            iteration_started = time.perf_counter()
            lr_opns = to_opns_scalar(self._lr(i))
            residuals = loss.negative_gradient(y, F_curr)
            # tree = OPNsObliviousTree(
            #     max_depth=self.max_depth,
            #     l2_leaf_reg=self.l2_leaf_reg,
            #     colsample_bylevel=1.0,
            #     random_strength=self.random_strength,
            #     max_thresholds=self.max_thresholds,
            #     random_state=self.random_state + i,
            # ).fit(X, residuals, hessians=None, active_features=self.active_features_)
            tree = OPNsObliviousTree(
                max_depth=self.max_depth,
                l2_leaf_reg=self.l2_leaf_reg,
                colsample_bylevel=self.feature_filter_ratio,
                random_strength=self.random_strength,
                max_thresholds=self.max_thresholds,
                random_state=self.random_state + i,
                split_score_backend=self.split_score_backend,
                threshold_backend=self.threshold_backend,
                threshold_sampling=self.threshold_sampling,
            # ).fit(X, residuals, hessians=None, active_features=self.active_features_)
            # ).fit(X, residuals, hessians=None, active_features=None)
            ).fit(
                X,
                residuals,
                hessians=None,
                active_features=self._tree_active_features(),
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

            completed = i + 1
            if (
                self.verbose
                and (
                    completed == 1
                    or completed == self.n_estimators
                    or completed % self.progress_interval == 0
                )
            ):
                elapsed = time.perf_counter() - boost_started
                average = elapsed / max(completed, 1)
                eta = average * max(self.n_estimators - completed, 0)
                self._progress(
                    "Boosting "
                    f"{completed}/{self.n_estimators}, "
                    f"elapsed={self._format_seconds(elapsed)}, "
                    f"avg/tree={average:.3f}s, "
                    f"ETA={self._format_seconds(eta)}."
                )

        #     if X_val is not None and y_val is not None:
        #         val_pred = self.predict_opns(X_val)
        #         val_loss = op.abs(loss.calc_loss(y_val, val_pred))
        #         if best_val_loss is None or val_loss < best_val_loss:
        #             best_val_loss = val_loss
        #             self.best_iteration_ = i
        #             no_improve = 0
        #         else:
        #             no_improve += 1
        #         # if no_improve >= self.early_stopping_rounds:
        #         #     break
        #         if self.early_stopping_rounds is not None and no_improve >= self.early_stopping_rounds:
        #             break
        # if self.best_iteration_ >= 0:
        #     self.trees_ = self.trees_[: self.best_iteration_ + 1]
        # return self
            if self.early_stopping_rounds is not None and X_val is not None and y_val is not None:
                val_pred = self.predict_opns(X_val)
                val_loss = op.abs(loss.calc_loss(y_val, val_pred))

                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_iteration_ = i
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= self.early_stopping_rounds:
                    self._progress(
                        "Early stopping triggered at tree "
                        f"{i + 1}; best_iteration={self.best_iteration_ + 1}."
                    )
                    break

        if self.early_stopping_rounds is not None and self.best_iteration_ >= 0:
            keep = self.best_iteration_ + 1
            self.trees_ = self.trees_[:keep]
            self.tree_iteration_times_ = self.tree_iteration_times_[:keep]
            self.tree_candidate_feature_evaluations_ = self.tree_candidate_feature_evaluations_[:keep]
            self.tree_threshold_evaluations_ = self.tree_threshold_evaluations_[:keep]
        else:
            self.best_iteration_ = len(self.trees_) - 1

        self.boosting_time_ = float(sum(self.tree_iteration_times_))
        self.phase2_time_ = float(self.threshold_cache_time_ + self.boosting_time_)
        self.total_fit_time_ = time.perf_counter() - fit_started
        return self

    @staticmethod
    def _matrix_spectral_vectors(values) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(values, OPNs):
            left = np.asarray([values.a], dtype=float)
            right = np.asarray([values.b], dtype=float)
        else:
            if not isinstance(values, OPNsMatrix):
                values = op.array(values)
            left = np.asarray(values.left_matrix, dtype=float).reshape(-1)
            right = np.asarray(values.right_matrix, dtype=float).reshape(-1)
        return _components_to_spectral(left, right)

    def predict_spectral_components(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Exact batched ensemble prediction in the p/q spectral channels.

        The base learner is evaluated once with the established OPNs algebra.
        Tree routing is vectorized with the exact native OPNs order, while
        leaf lookup, learning-rate multiplication, and ensemble accumulation
        are performed in the two real spectral channels.
        """
        if not isinstance(X, OPNsMatrix):
            raise TypeError("spectral_batch prediction requires an OPNsMatrix input.")

        X_poly = self._expand_features(X)
        X_poly_scaled = self.base_scaler_.transform(X_poly)
        base = op.dot(X_poly_scaled, self.base_model_.coef_) + self.base_model_.intercept_
        F_p, F_q = self._matrix_spectral_vectors(base)
        F_p = F_p.copy()
        F_q = F_q.copy()

        X_left = np.asarray(X.left_matrix, dtype=float)
        X_right = np.asarray(X.right_matrix, dtype=float)
        for tree, lr_opns in self.trees_:
            tree_p, tree_q = tree.predict_spectral_from_components(
                X_left, X_right
            )
            lr_p, lr_q = _components_to_spectral(lr_opns.a, lr_opns.b)
            F_p += float(lr_p) * tree_p
            F_q += float(lr_q) * tree_q
        return F_p, F_q

    def predict_opns(self, X, backend: Optional[str] = None):
        backend = _validate_prediction_backend(backend or self.prediction_backend)
        X_poly = self._expand_features(X)
        X_poly_scaled = self.base_scaler_.transform(X_poly)
        F = op.dot(X_poly_scaled, self.base_model_.coef_) + self.base_model_.intercept_

        if backend == "legacy_object":
            for tree, lr_opns in self.trees_:
                F = F + tree.predict_legacy(X) * lr_opns
            return F

        if backend == "object_batch":
            for tree, lr_opns in self.trees_:
                F = F + tree.predict_object_batch(X) * lr_opns
            return F

        F_p, F_q = self.predict_spectral_components(X)
        left, right = _spectral_to_components(F_p, F_q)
        result = OPNsMatrix()
        result.set_matrix(left, right)
        return result

    def predict(self, X, item: int = 1, backend: Optional[str] = None):
        backend = _validate_prediction_backend(backend or self.prediction_backend)
        if backend != "spectral_batch":
            return extract_real(self.predict_opns(X, backend=backend), item=item)

        F_p, F_q = self.predict_spectral_components(X)
        if item == 0:
            return -F_p
        if item == 2:
            return -0.5 * (F_p + F_q)
        return 0.5 * (F_q - F_p)

    def prediction_equivalence_report(self, X) -> dict:
        """Compare all inference backends on the same fitted model."""
        pred_legacy = self.predict(X, item=1, backend="legacy_object")
        pred_object = self.predict(X, item=1, backend="object_batch")
        pred_spectral = self.predict(X, item=1, backend="spectral_batch")

        X_left = np.asarray(X.left_matrix, dtype=float)
        X_right = np.asarray(X.right_matrix, dtype=float)
        agreements = []
        for tree, _ in self.trees_:
            legacy_leaf = tree.apply_legacy(X)
            spectral_leaf = tree.apply_component_batch(X_left, X_right)
            agreements.append(float(np.mean(legacy_leaf == spectral_leaf)))

        return {
            "max_abs_legacy_object_diff": float(np.max(np.abs(pred_legacy - pred_object)))
            if pred_legacy.size
            else 0.0,
            "max_abs_legacy_spectral_diff": float(np.max(np.abs(pred_legacy - pred_spectral)))
            if pred_legacy.size
            else 0.0,
            "max_abs_object_spectral_diff": float(np.max(np.abs(pred_object - pred_spectral)))
            if pred_object.size
            else 0.0,
            "leaf_index_agreement": float(np.mean(agreements)) if agreements else 1.0,
        }

    # def get_diagnostics(self) -> dict:
    #     active = getattr(self, "active_set_", None)
    #     return {
    #         "candidate_features": getattr(active, "candidate_features", None),
    #         "selected_features": getattr(active, "selected_features", None),
    #         "sparsity_ratio": getattr(active, "sparsity_ratio", None),
    #         "n_trees": len(getattr(self, "trees_", [])),
    #         "best_iteration": getattr(self, "best_iteration_", None),
    #         "active_features": getattr(active, "active_tree_features", None),
    #     }
    def get_diagnostics(self) -> dict:
        active = getattr(self, "active_set_", None)

        mode = getattr(self, "tree_feature_mode", "all")
        if mode == "active":
            tree_search_features = getattr(active, "selected_features", None)
        elif mode in {
            "active_plus",
            "residual_gain_topk",
            "residual_gain_adaptive",
            "residual_gain_stability",
            "residual_gain_crossfit",
        }:
            pool = getattr(self, "tree_features_", None)
            tree_search_features = None if pool is None else int(len(pool))
        elif mode == "all":
            tree_search_features = getattr(active, "candidate_features", None)
        elif mode == "auto":
            if getattr(self, "raw_feature_count_", 0) <= 100:
                tree_search_features = getattr(active, "candidate_features", None)
            else:
                tree_search_features = getattr(active, "selected_features", None)
        else:
            tree_search_features = None

        return {
            "phase1_enabled": True,
            "phase1_time": getattr(self, "phase1_time_", 0.0),
            "feature_pool_time": getattr(self, "feature_pool_time_", 0.0),
            "threshold_cache_time": getattr(self, "threshold_cache_time_", 0.0),
            "boosting_time": getattr(self, "boosting_time_", 0.0),
            "phase2_time": getattr(self, "phase2_time_", 0.0),
            "total_fit_time": getattr(self, "total_fit_time_", 0.0),
            "threshold_cache_features": getattr(self, "threshold_cache_features_", 0),
            "threshold_cache_candidates": getattr(self, "threshold_cache_candidates_", 0),
            "threshold_target_budget": int(getattr(self, "max_thresholds", 0)),
            "threshold_policy": "ordered_exact_target_stride",
            "candidate_feature_evaluations": int(sum(getattr(self, "tree_candidate_feature_evaluations_", []))),
            "threshold_evaluations": int(sum(getattr(self, "tree_threshold_evaluations_", []))),
            "lasso_n_iter": getattr(getattr(self, "base_model_", None), "n_iter_", None),
            "lasso_best_iteration": getattr(getattr(self, "base_model_", None), "best_iteration_", None),
            "lasso_stop_reason": getattr(getattr(self, "base_model_", None), "stop_reason_", None),
            "lasso_objective_history_length": len(
                getattr(getattr(self, "base_model_", None), "objective_history_", [])
            ),
            "lasso_max_coordinate_residual_abs_p": getattr(
                getattr(self, "base_model_", None),
                "max_coordinate_residual_components_",
                {},
            ).get("abs_p"),
            "lasso_max_coordinate_residual_abs_q": getattr(
                getattr(self, "base_model_", None),
                "max_coordinate_residual_components_",
                {},
            ).get("abs_q"),
            "candidate_features": getattr(active, "candidate_features", None),
            "selected_features": getattr(active, "selected_features", None),
            "sparsity_ratio": getattr(active, "sparsity_ratio", None),
            "tree_feature_mode": getattr(self, "tree_feature_mode", None),
            "tree_search_features": tree_search_features,
            "active_plus_added_features": getattr(self, "active_plus_added_features_", 0),
            "residual_gain_added_features": getattr(self, "residual_gain_added_features_", 0),
            "residual_gain_valid_features": getattr(self, "residual_gain_valid_features_", 0),
            "residual_gain_active_valid_features": getattr(
                self, "residual_gain_active_valid_features_", 0
            ),
            "residual_gain_gate_qualified_features": getattr(
                self, "residual_gain_gate_qualified_features_", 0
            ),
            "residual_gain_max_added_features": getattr(
                self, "residual_gain_max_added_features_", 0
            ),
            "residual_gain_gate_feature": getattr(
                self, "residual_gain_gate_feature_", -1
            ),
            "residual_gain_gate_score_left": getattr(
                self, "residual_gain_gate_score_left_", np.nan
            ),
            "residual_gain_gate_score_right": getattr(
                self, "residual_gain_gate_score_right_", np.nan
            ),
            "residual_gain_screen_time": getattr(self, "residual_gain_screen_time_", 0.0),
            "residual_gain_stability_qualified_features": getattr(
                self, "residual_gain_stability_qualified_features_", 0
            ),
            "residual_gain_stability_mean_qualified_per_repeat": getattr(
                self, "residual_gain_stability_mean_qualified_per_repeat_", 0.0
            ),
            "residual_gain_stability_selected_frequency_mean": getattr(
                self, "residual_gain_stability_selected_frequency_mean_", 0.0
            ),
            "residual_gain_stability_selected_frequency_min": getattr(
                self, "residual_gain_stability_selected_frequency_min_", 0.0
            ),
            "residual_gain_stability_selected_frequency_max": getattr(
                self, "residual_gain_stability_selected_frequency_max_", 0.0
            ),
            "residual_gain_stability_active_valid_mean": getattr(
                self, "residual_gain_stability_active_valid_mean_", 0.0
            ),
            "residual_gain_stability_inactive_valid_mean": getattr(
                self, "residual_gain_stability_inactive_valid_mean_", 0.0
            ),
            "residual_gain_stability_inactive_frequency_mean": getattr(
                self, "residual_gain_stability_inactive_frequency_mean_", 0.0
            ),
            "residual_gain_stability_inactive_frequency_max": getattr(
                self, "residual_gain_stability_inactive_frequency_max_", 0.0
            ),
            "residual_gain_stability_subsample_size": getattr(
                self, "residual_gain_stability_subsample_size_", 0
            ),
            "residual_gain_crossfit_qualified_features": getattr(
                self, "residual_gain_crossfit_qualified_features_", 0
            ),
            "residual_gain_crossfit_mean_qualified_per_repeat": getattr(
                self, "residual_gain_crossfit_mean_qualified_per_repeat_", 0.0
            ),
            "residual_gain_crossfit_selected_frequency_mean": getattr(
                self, "residual_gain_crossfit_selected_frequency_mean_", 0.0
            ),
            "residual_gain_crossfit_selected_frequency_min": getattr(
                self, "residual_gain_crossfit_selected_frequency_min_", 0.0
            ),
            "residual_gain_crossfit_selected_frequency_max": getattr(
                self, "residual_gain_crossfit_selected_frequency_max_", 0.0
            ),
            "residual_gain_crossfit_active_positive_mean": getattr(
                self, "residual_gain_crossfit_active_positive_mean_", 0.0
            ),
            "residual_gain_crossfit_inactive_positive_mean": getattr(
                self, "residual_gain_crossfit_inactive_positive_mean_", 0.0
            ),
            "residual_gain_crossfit_inactive_valid_mean": getattr(
                self, "residual_gain_crossfit_inactive_valid_mean_", 0.0
            ),
            "residual_gain_crossfit_train_size": getattr(
                self, "residual_gain_crossfit_train_size_", 0
            ),
            "residual_gain_crossfit_validation_size": getattr(
                self, "residual_gain_crossfit_validation_size_", 0
            ),
            "residual_gain_crossfit_repeats_completed": getattr(
                self, "residual_gain_crossfit_repeats_completed_", 0
            ),
            "residual_gain_crossfit_no_active_gate_repeats": getattr(
                self, "residual_gain_crossfit_no_active_gate_repeats_", 0
            ),
            "split_score_backend": getattr(self, "split_score_backend", None),
            "threshold_backend": getattr(self, "threshold_backend", None),
            "threshold_sampling": getattr(self, "threshold_sampling", None),
            "prediction_backend": getattr(self, "prediction_backend", None),
            "n_trees": len(getattr(self, "trees_", [])),
            "best_iteration": getattr(self, "best_iteration_", None),
            "active_features": getattr(active, "active_tree_features", None),
        }


class _OPNsBinaryBooster(_FeatureExpansionMixin):
    """Binary OPNs booster with separable Phase-I and tree-feature modes.

    ``phase1_enabled=False`` uses the optimal constant Bernoulli logit on the
    training fold. ``tree_feature_mode`` controls whether residual trees search
    all OPNs pair features or only the Phase-I active set.
    """

    def __init__(
        self,
        n_estimators: int,
        learning_rate: float,
        max_depth: int,
        l2_leaf_reg: float,
        poly_degree: int,
        use_trig: bool,
        logistic_lr: float,
        logistic_max_iter: int,
        feature_filter_ratio: float,
        active_threshold: float,
        random_strength: float,
        lr_decay_type: str,
        early_stopping_rounds: Optional[int],
        max_thresholds: int,
        split_score_backend: str,
        threshold_backend: str,
        threshold_sampling: str,
        random_state: int,
        verbose: bool,
        phase1_enabled: bool = True,
        tree_feature_mode: str = "active",
    ):
        self.early_stopping_rounds = early_stopping_rounds
        self.__dict__.update(locals())
        self.__dict__.pop("self", None)
        self.trees_ = []
        self.best_iteration_ = -1

    def _lr(self, iteration: int) -> float:
        if self.lr_decay_type == "cosine":
            return self.learning_rate * 0.5 * (
                1.0 + np.cos(np.pi * iteration / max(1, self.n_estimators))
            )
        return self.learning_rate

    @staticmethod
    def _repeat_constant(value: OPNs, n_samples: int):
        matrix = OPNsMatrix()
        matrix.set_matrix(
            np.full(int(n_samples), float(value.a), dtype=float),
            np.full(int(n_samples), float(value.b), dtype=float),
        )
        return matrix

    def _tree_active_features(self):
        if self.tree_feature_mode == "all":
            return np.arange(self.raw_feature_count_, dtype=int)
        if self.tree_feature_mode == "active":
            return np.asarray(self.active_features_, dtype=int)
        raise ValueError(
            f"Unknown classification tree_feature_mode={self.tree_feature_mode!r}. "
            "Expected 'all' or 'active'."
        )

    def _build_shared_threshold_cache(self, X) -> dict[int, OPNsMatrix]:
        active_features = np.asarray(self._tree_active_features(), dtype=int)
        if active_features.size == 0:
            active_features = np.arange(self.raw_feature_count_, dtype=int)
        template = OPNsObliviousTree(
            max_depth=self.max_depth,
            l2_leaf_reg=self.l2_leaf_reg,
            colsample_bylevel=self.feature_filter_ratio,
            random_strength=self.random_strength,
            max_thresholds=self.max_thresholds,
            random_state=self.random_state,
            split_score_backend=self.split_score_backend,
            threshold_backend=self.threshold_backend,
            threshold_sampling=self.threshold_sampling,
        )
        started = time.perf_counter()
        cache = {
            int(j): template._thresholds(X[:, int(j)])
            for j in active_features
        }
        self.threshold_cache_time_ = time.perf_counter() - started
        self.threshold_cache_features_ = int(len(cache))
        self.threshold_cache_candidates_ = int(sum(len(v) for v in cache.values()))
        return cache

    def _initial_logits(self, X):
        if getattr(self, "base_model_", None) is None:
            return self._repeat_constant(self.base_constant_, X.shape[0])
        X_poly = self._expand_features(X)
        X_poly_scaled = self.base_scaler_.transform(X_poly)
        return op.dot(X_poly_scaled, self.base_model_.weights) + self.base_model_.bias

    def fit(self, X, y_real, X_val=None, y_val_real=None):
        fit_started = time.perf_counter()
        np.random.seed(self.random_state)
        y_real = np.asarray(y_real, dtype=int).reshape(-1)
        y_opns = op.array([OPNS_ONE if int(v) == 1 else OPNs(0, 0) for v in y_real])
        self.raw_feature_count_ = int(X.shape[1])
        self.trees_ = []
        self.best_iteration_ = -1
        self.tree_iteration_times_ = []
        self.tree_candidate_feature_evaluations_ = []
        self.tree_threshold_evaluations_ = []

        phase1_started = time.perf_counter()
        if self.phase1_enabled:
            X_poly = self._expand_features(X)
            self.base_scaler_ = OPNsStandardScaler()
            X_poly_scaled = self.base_scaler_.fit_transform(X_poly)
            self.base_model_ = LogisticRegression(
                learning_rate=self.logistic_lr,
                max_iter=self.logistic_max_iter,
                adapt_lr=True,
            )
            self.base_model_.fit(X_poly_scaled, y_real)
            self.active_set_ = self._derive_active_features(
                self.base_model_.weights,
                raw_feature_count=self.raw_feature_count_,
                threshold=self.active_threshold,
                fallback_ratio=self.feature_filter_ratio,
            )
            self.active_features_ = np.asarray(
                self.active_set_.active_tree_features, dtype=int
            )
            F_curr = op.dot(X_poly_scaled, self.base_model_.weights) + self.base_model_.bias
            self.base_constant_ = None
            self.base_logit_ = math.nan
        else:
            eps = 1e-12
            positive_rate = float(np.clip(np.mean(y_real), eps, 1.0 - eps))
            self.base_logit_ = float(np.log(positive_rate / (1.0 - positive_rate)))
            self.base_constant_ = to_opns_scalar(self.base_logit_)
            self.base_model_ = None
            self.base_scaler_ = None
            self.active_set_ = ActiveSetInfo(
                candidate_features=self.raw_feature_count_,
                selected_features=self.raw_feature_count_,
                sparsity_ratio=0.0,
                selected_base_features=list(range(self.raw_feature_count_)),
                active_tree_features=list(range(self.raw_feature_count_)),
            )
            self.active_features_ = np.arange(self.raw_feature_count_, dtype=int)
            F_curr = self._repeat_constant(self.base_constant_, X.shape[0])
        self.phase1_time_ = time.perf_counter() - phase1_started

        pool_started = time.perf_counter()
        self.tree_features_ = self._tree_active_features()
        self.feature_pool_time_ = time.perf_counter() - pool_started

        if self.n_estimators > 0:
            shared_threshold_cache = self._build_shared_threshold_cache(X)
        else:
            shared_threshold_cache = {}
            self.threshold_cache_time_ = 0.0
            self.threshold_cache_features_ = 0
            self.threshold_cache_candidates_ = 0

        loss = OPNsLogLoss()
        best_val = float("inf")
        no_improve = 0
        for i in range(self.n_estimators):
            iteration_started = time.perf_counter()
            lr_opns = to_opns_scalar(self._lr(i))
            gradients = loss.negative_gradient(y_opns, F_curr)
            hessians = loss.hessian(y_opns, F_curr)
            tree = OPNsObliviousTree(
                max_depth=self.max_depth,
                l2_leaf_reg=self.l2_leaf_reg,
                colsample_bylevel=self.feature_filter_ratio,
                random_strength=self.random_strength,
                max_thresholds=self.max_thresholds,
                random_state=self.random_state + i,
                split_score_backend=self.split_score_backend,
                threshold_backend=self.threshold_backend,
                threshold_sampling=self.threshold_sampling,
            ).fit(
                X,
                gradients,
                hessians=hessians,
                active_features=self.tree_features_,
                threshold_cache=shared_threshold_cache,
            )
            self.trees_.append((tree, lr_opns))
            F_curr = F_curr + tree.predict_object_batch(X) * lr_opns
            self.tree_iteration_times_.append(time.perf_counter() - iteration_started)
            self.tree_candidate_feature_evaluations_.append(
                int(getattr(tree, "candidate_feature_evaluations_", 0))
            )
            self.tree_threshold_evaluations_.append(
                int(getattr(tree, "threshold_evaluations_", 0))
            )

            if (
                self.early_stopping_rounds is not None
                and X_val is not None
                and y_val_real is not None
            ):
                proba = self.predict_proba_real(X_val)
                eps = 1e-15
                proba = np.clip(proba, eps, 1 - eps)
                val_loss = -np.mean(
                    y_val_real * np.log(proba)
                    + (1 - y_val_real) * np.log(1 - proba)
                )
                if val_loss < best_val:
                    best_val = val_loss
                    self.best_iteration_ = i
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self.early_stopping_rounds:
                    break

        if self.early_stopping_rounds is not None and self.best_iteration_ >= 0:
            keep = self.best_iteration_ + 1
            self.trees_ = self.trees_[:keep]
            self.tree_iteration_times_ = self.tree_iteration_times_[:keep]
            self.tree_candidate_feature_evaluations_ = (
                self.tree_candidate_feature_evaluations_[:keep]
            )
            self.tree_threshold_evaluations_ = self.tree_threshold_evaluations_[:keep]
        else:
            self.best_iteration_ = len(self.trees_) - 1

        self.boosting_time_ = float(sum(self.tree_iteration_times_))
        self.phase2_time_ = float(self.threshold_cache_time_ + self.boosting_time_)
        self.total_fit_time_ = time.perf_counter() - fit_started
        return self

    def predict_logits(self, X):
        F = self._initial_logits(X)
        for tree, lr_opns in self.trees_:
            F = F + tree.predict_object_batch(X) * lr_opns
        return F

    def predict_proba_real(self, X):
        logits = OPNsLogLoss.logits_to_real(self.predict_logits(X))
        logits = np.clip(logits, -40.0, 40.0)
        return np.clip(1.0 / (1.0 + np.exp(-logits)), 0.0, 1.0)

    def get_diagnostics(self) -> dict:
        active = getattr(self, "active_set_", None)
        tree_features = np.asarray(getattr(self, "tree_features_", []), dtype=int)
        return {
            "phase1_enabled": bool(self.phase1_enabled),
            "phase1_time": float(getattr(self, "phase1_time_", 0.0)),
            "feature_pool_time": float(getattr(self, "feature_pool_time_", 0.0)),
            "threshold_cache_time": float(getattr(self, "threshold_cache_time_", 0.0)),
            "boosting_time": float(getattr(self, "boosting_time_", 0.0)),
            "phase2_time": float(getattr(self, "phase2_time_", 0.0)),
            "total_fit_time": float(getattr(self, "total_fit_time_", 0.0)),
            "candidate_feature_evaluations": int(
                sum(getattr(self, "tree_candidate_feature_evaluations_", []))
            ),
            "threshold_evaluations": int(
                sum(getattr(self, "tree_threshold_evaluations_", []))
            ),
            "threshold_cache_features": int(
                getattr(self, "threshold_cache_features_", 0)
            ),
            "threshold_cache_candidates": int(
                getattr(self, "threshold_cache_candidates_", 0)
            ),
            "candidate_features": getattr(active, "candidate_features", None),
            "selected_features": getattr(active, "selected_features", None),
            "sparsity_ratio": getattr(active, "sparsity_ratio", None),
            "tree_feature_mode": self.tree_feature_mode,
            "tree_search_features": int(tree_features.size),
            "n_trees": len(getattr(self, "trees_", [])),
            "best_iteration": getattr(self, "best_iteration_", None),
            "split_score_backend": getattr(self, "split_score_backend", None),
            "threshold_backend": getattr(self, "threshold_backend", None),
            "threshold_sampling": getattr(self, "threshold_sampling", None),
            "active_features": getattr(active, "active_tree_features", None),
            "base_logit": float(getattr(self, "base_logit_", math.nan)),
        }


class OPNsHybridClassifier(BaseEstimator, ClassifierMixin):
    """One-vs-rest OPNs-HybridBoost classifier with ablation switches."""

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.2,
        max_depth: int = 4,
        l2_leaf_reg: float = 1.0,
        poly_degree: int = 2,
        use_trig: bool = False,
        logistic_lr: float = 0.1,
        logistic_max_iter: int = 500,
        feature_filter_ratio: float = 1.0,
        active_threshold: float = 1e-10,
        random_strength: float = 1.0,
        lr_decay_type: str = "cosine",
        early_stopping_rounds: Optional[int] = None,
        max_thresholds: int = 20,
        split_score_backend: str = "fast_exact",
        threshold_backend: str = "ordered_exact",
        threshold_sampling: str = "stride",
        random_state: int = 42,
        verbose: bool = False,
        phase1_enabled: bool = True,
        tree_feature_mode: str = "active",
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.l2_leaf_reg = l2_leaf_reg
        self.poly_degree = poly_degree
        self.use_trig = use_trig
        self.logistic_lr = logistic_lr
        self.logistic_max_iter = logistic_max_iter
        self.feature_filter_ratio = feature_filter_ratio
        self.active_threshold = active_threshold
        self.random_strength = random_strength
        self.lr_decay_type = lr_decay_type
        self.early_stopping_rounds = early_stopping_rounds
        self.max_thresholds = max_thresholds
        self.split_score_backend = split_score_backend
        self.threshold_backend = threshold_backend
        self.threshold_sampling = threshold_sampling
        self.random_state = random_state
        self.verbose = verbose
        self.phase1_enabled = phase1_enabled
        self.tree_feature_mode = tree_feature_mode

    def _booster_params(self):
        return dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            l2_leaf_reg=self.l2_leaf_reg,
            poly_degree=self.poly_degree,
            use_trig=self.use_trig,
            logistic_lr=self.logistic_lr,
            logistic_max_iter=self.logistic_max_iter,
            feature_filter_ratio=self.feature_filter_ratio,
            active_threshold=self.active_threshold,
            random_strength=self.random_strength,
            lr_decay_type=self.lr_decay_type,
            early_stopping_rounds=self.early_stopping_rounds,
            max_thresholds=self.max_thresholds,
            split_score_backend=self.split_score_backend,
            threshold_backend=self.threshold_backend,
            threshold_sampling=self.threshold_sampling,
            random_state=self.random_state,
            verbose=self.verbose,
            phase1_enabled=self.phase1_enabled,
            tree_feature_mode=self.tree_feature_mode,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        fit_started = time.perf_counter()
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.estimators_ = []
        y_val = None if y_val is None else np.asarray(y_val).astype(int)

        if self.n_classes_ <= 2:
            y_bin = (y == self.classes_[-1]).astype(int)
            y_val_bin = None if y_val is None else (y_val == self.classes_[-1]).astype(int)
            booster = _OPNsBinaryBooster(**self._booster_params())
            booster.fit(X, y_bin, X_val=X_val, y_val_real=y_val_bin)
            self.estimators_.append(booster)
        else:
            lb = LabelBinarizer()
            Y = lb.fit_transform(y)
            Y_val = None if y_val is None else lb.transform(y_val)
            for k in range(self.n_classes_):
                booster = _OPNsBinaryBooster(
                    **{
                        **self._booster_params(),
                        "random_state": self.random_state + k * 1000,
                    }
                )
                booster.fit(
                    X,
                    Y[:, k],
                    X_val=X_val,
                    y_val_real=None if Y_val is None else Y_val[:, k],
                )
                self.estimators_.append(booster)
        self.total_fit_time_ = time.perf_counter() - fit_started
        return self

    def predict_proba(self, X):
        if len(self.estimators_) == 1:
            p = self.estimators_[0].predict_proba_real(X)
            return np.column_stack([1 - p, p])
        probs = np.column_stack(
            [est.predict_proba_real(X) for est in self.estimators_]
        )
        row_sums = probs.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums[:, None]

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def get_diagnostics(self) -> dict:
        if not getattr(self, "estimators_", None):
            return {}
        ds = [est.get_diagnostics() for est in self.estimators_]
        selected = np.asarray([d.get("selected_features", 0) for d in ds], dtype=float)
        search = np.asarray([d.get("tree_search_features", 0) for d in ds], dtype=float)
        candidate = float(ds[0].get("candidate_features", 0) or 0)
        return {
            "phase1_enabled": bool(self.phase1_enabled),
            "tree_feature_mode": self.tree_feature_mode,
            "candidate_features": int(candidate),
            "selected_features_mean": float(np.mean(selected)),
            "selected_features_min": float(np.min(selected)),
            "selected_features_max": float(np.max(selected)),
            "selected_features_sum": float(np.sum(selected)),
            "active_retention_mean": float(np.mean(selected / max(candidate, 1.0))),
            "tree_search_features_mean": float(np.mean(search)),
            "tree_search_features_sum": float(np.sum(search)),
            "sparsity_ratio_mean": float(
                np.mean([d.get("sparsity_ratio", 0) for d in ds])
            ),
            "n_trees_mean": float(np.mean([d.get("n_trees", 0) for d in ds])),
            "n_binary_estimators": int(len(ds)),
            "phase1_time": float(sum(d.get("phase1_time", 0.0) for d in ds)),
            "feature_pool_time": float(
                sum(d.get("feature_pool_time", 0.0) for d in ds)
            ),
            "threshold_cache_time": float(
                sum(d.get("threshold_cache_time", 0.0) for d in ds)
            ),
            "boosting_time": float(sum(d.get("boosting_time", 0.0) for d in ds)),
            "phase2_time": float(sum(d.get("phase2_time", 0.0) for d in ds)),
            "total_fit_time": float(getattr(self, "total_fit_time_", 0.0)),
            "candidate_feature_evaluations": int(
                sum(d.get("candidate_feature_evaluations", 0) for d in ds)
            ),
            "threshold_evaluations": int(
                sum(d.get("threshold_evaluations", 0) for d in ds)
            ),
            "threshold_cache_candidates": int(
                sum(d.get("threshold_cache_candidates", 0) for d in ds)
            ),
            "per_class": ds,
        }
