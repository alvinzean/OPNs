"""Verified Phase-I solvers used internally by OPNs-HybridBoost.

The function and class bodies in this module are extracted from the
frozen HybridBoost linear-model source with SHA-256:

    c15f88c19c58355d87ec9b42a33e15bceedb62f4da7fa973c1a4bd8bbd4a8891

Only module-level imports are adapted to the public ``opns_pack``
package layout. The existing ``opns_module.linear_model`` implementation
is intentionally left unchanged for OPNs-LR compatibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import opns_pack.custom_gen_pairs as cgp
import opns_pack.opns_np as op

from opns_pack.opns import OPNs
from opns_pack.opns_matrix import OPNsMatrix


OPNS_ONE = OPNs(0, -1)
OPNS_ZERO = OPNs(0, 0)


def array_to_dataframe(array, column_prefix='Column'):
    """
    将任意形状的 ndarray 或list 转换为 DataFrame。
    :param array: 要转换的 ndarray 或 list。
    :param column_prefix: 列名前缀，默认值为 'Column'。
    :return:  转换后的 Pandas DataFrame。
    """
    if isinstance(array, list):
        array = np.array(array)
    if array.ndim == 1:  # 检查 array 的维度
        df = pd.DataFrame(array, columns=[f'{column_prefix}1'])
    elif array.ndim == 2:
        num_columns = array.shape[1]
        columns = [f'{column_prefix}{i + 1}' for i in range(num_columns)]
        df = pd.DataFrame(array, columns=columns)
    else:
        raise ValueError("只支持一维和二维的 ndarray 转换为 DataFrame")
    return df


class LogisticRegression:

    @classmethod
    def sigmoid(cls, z):
        return (OPNs(0, -1) + op.exp(-z)).reciprocal()

    @classmethod
    def softmax(cls, z):
        exp_z = op.exp(z - op.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def __init__(self, learning_rate=1, max_iter=1000, decay_rate=0.9, multiclass=False, weights=None, bias=None,
                 adapt_lr=True,
                 tol=1e-4, adapt_max_iter=True, pred_proba_bias=False):
        """
        :param learning_rate: 初始学习率
        :param max_iter: 最大迭代次数
        :param multiclass: False为二分类，True为多分类
        :param adapt_lr: 是否使用自适应学习率
        :param tol: 收敛容忍度，用于提前终止迭代
        :param adapt_max_iter: 是否自适应最大迭代次数
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.multiclass = multiclass
        self.weights = weights
        self.bias = bias
        self.classes_ = None
        self.adapt_lr = adapt_lr
        self.tol = tol
        self.adapt_max_iter = adapt_max_iter
        self.pred_proba_bias = pred_proba_bias
        self.decay_rate = decay_rate

    def get_params(self, deep=True):
        return {'learning_rate': self.learning_rate, 'max_iter': self.max_iter, 'multiclass': self.multiclass,
                'weights': self.weights, 'bias': self.bias, 'tol': self.tol, 'adapt_max_iter': self.adapt_max_iter}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        """
        训练模型参数
        :param X: OPNs特征矩阵
        :param y: 常规实数标签向量
        """
        n_samples, n_features = X.shape
        self.classes_, y = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        self.classes_ = np.arange(n_classes)
        for i in range(1, n_classes):
            if self.classes_[i] != self.classes_[i - 1] + 1:
                raise ValueError('标签类别数组 {} 确保相邻类别的步长为1'.format(self.classes_))

        if n_classes > 2:
            self.multiclass = True
        else:
            y = op.array([elem * OPNs(0, -1) for elem in y])

        # 初始化权重和偏置
        if self.multiclass:
            self.weights = op.zeros((n_features, n_classes))
            self.bias = op.zeros(n_classes)
            y_one_hot = op.zeros((n_samples, n_classes))
            y_one_hot[np.arange(n_samples), y] = OPNs(0, -1)
        else:
            self.weights = op.zeros(n_features)
            self.bias = OPNs(0, 0)

        prev_loss = OPNs(0, -float('inf'))
        prev_grad_norm = OPNs(0, -float('inf'))

        # 训练过程
        for iteration in range(self.max_iter):
            linear_model = op.dot(X, self.weights) + self.bias
            if self.multiclass:
                y_predicted = self.softmax(linear_model)
                dw = (1 / n_samples) * op.dot(X.T, (y_predicted - y_one_hot))
                db = (1 / n_samples) * op.sum(y_predicted - y_one_hot, axis=0)
            else:
                y_predicted = self.sigmoid(linear_model)
                dw = (1 / n_samples) * op.dot(X.T, (y_predicted - y))
                db = (1 / n_samples) * op.sum(y_predicted - y)

            # 更新权重和偏置
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 如果启用自适应学习率
            loss = self.compute_loss(y_predicted, y_one_hot if self.multiclass else y)
            if self.adapt_lr:
                # 计算损失函数
                if loss > prev_loss:  # 如果损失增加，减小学习率
                    # print('学习率减小。。。')
                    self.learning_rate *= self.decay_rate
                prev_loss = loss

            # 判断收敛
            grad_norm = op.linalg_norm(dw) + op.abs(db)
            if grad_norm < 100 * self.tol * OPNs(0, -1):  # 梯度变小
                break

            # 如果启用自适应最大迭代次数
            if self.adapt_max_iter and op.abs(prev_grad_norm - grad_norm) < self.tol * OPNs(0, -1):
                self.max_iter = iteration + 1
                break

            prev_grad_norm = grad_norm

    def compute_loss(self, y_predicted, y_true):
        """计算损失函数"""
        if self.multiclass:
            return -op.mean(op.sum(y_true * op.log(y_predicted), axis=1))
        else:
            return -op.mean(y_true * op.log(y_predicted) + (OPNs(0, -1) - y_true) * op.log(OPNs(0, -1) - y_predicted))

    def predict(self, X):
        """
        :param X: OPNs特征矩阵
        :return: 样本的分类结果
        """
        linear_model = op.dot(X, self.weights) + self.bias
        if self.multiclass:
            y_predicted = self.softmax(linear_model)
            y_predicted_class = op.argmax(y_predicted, axis=1)
        else:
            y_predicted = self.sigmoid(linear_model)
            y_predicted_class = (y_predicted > OPNs(0, -0.5)).astype(int)
        return y_predicted_class

    def predict_proba(self, X):
        """
        :param X: OPNs特征矩阵
        :return: 各个类别的预测概率
        """
        linear_model = op.dot(X, self.weights) + self.bias
        if self.multiclass:
            y_opn_predicted = self.softmax(linear_model)
            y_predicted = np.array([[-x.b for x in row] for row in y_opn_predicted])
        else:
            y_predicted = self.sigmoid(linear_model)
            y_opn_predicted = op.vstack(OPNs(0, -1) - y_predicted, y_predicted).T
            if self.pred_proba_bias:
                y_predicted = np.array([[-x.a for x in row] for row in y_opn_predicted])
            else:
                y_predicted = np.array([[-(x.a + x.b) for x in row] for row in y_opn_predicted])
        return y_predicted


class Lasso:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, learning_rate=0.01, adaptive_lr=True, adaptive_iter=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.adaptive_lr = adaptive_lr
        self.adaptive_iter = adaptive_iter
        self.coef_ = None
        self.intercept_ = None

    @classmethod
    def _soft_threshold(cls, rho, alpha):
        return op.sign(rho) * max(op.abs(rho) - alpha * OPNs(0, -1), OPNs(0, 0))

    def fit(self, X, y):
        """Fit the Lasso model with cyclic coordinate shrinkage.

        The update order and stopping rules match the historical implementation.
        A residual cache replaces the repeated full matrix product inside every
        coordinate update; this is algebraically equivalent to the original
        Gauss--Seidel sweep and substantially reduces Phase-I cost.

        Notes
        -----
        The coordinate formula omits division by the empirical column second
        moment.  HybridBoost applies :class:`OPNsStandardScaler` before calling
        this estimator, for which that second moment is the OPNs identity.  A
        caller that supplies unstandardized data must account for the column
        denominator externally or use a solver designed for that setting.
        """
        n_samples, n_features = X.shape
        if self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if self.tol < 0:
            raise ValueError("tol must be non-negative.")

        if isinstance(y, (list, np.ndarray)):
            y = array_to_dataframe(y, column_prefix='y')
            if y.shape[1] % 2 != 0:
                y['New_Column'] = 0
            y = cgp.data_convert(y, y.columns.tolist(), bias=False)
        elif isinstance(y, pd.Series):
            y = y.to_frame()
            if y.shape[1] % 2 != 0:
                y['New_Column'] = 0
            y = cgp.data_convert(y, y.columns.tolist(), bias=False)
        y = y.reshape(-1)

        self.coef_ = op.zeros(n_features)
        self.intercept_ = OPNs(0, 0)

        X_mean = op.mean(X, axis=0)
        y_mean = op.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Coordinate descent uses a closed coordinate update.  The historical
        # learning_rate/adaptive_lr arguments remain for API compatibility but
        # deliberately do not alter the fitted coefficients.
        prev_loss = None
        best_loss = None
        best_coef = None
        best_intercept = None
        best_iteration = -1
        self.objective_history_ = []
        self.n_iter_ = 0
        self.stop_reason_ = "max_iter"

        # Current centered residual r = y_c - X_c beta.  During a coordinate
        # update, add back the old contribution, solve the coordinate, and
        # subtract the new contribution.  This preserves cyclic Gauss--Seidel
        # semantics without recomputing X_c @ beta for every feature.
        residual = y_centered - X_centered @ self.coef_

        for iteration in range(self.max_iter):
            coef_old = self.coef_.__copy__()

            for j in range(n_features):
                old_coef = self.coef_[j]
                residual = residual + X_centered[:, j] * old_coef
                rho = op.dot(X_centered[:, j], residual) / n_samples
                new_coef = self._soft_threshold(rho, self.alpha)
                self.coef_[j] = new_coef
                residual = residual - X_centered[:, j] * new_coef

            self.intercept_ = y_mean - op.dot(X_mean, self.coef_)

            loss = (
                0.5 * op.mean(residual ** 2)
                + self.alpha * op.sum(op.abs(self.coef_))
            )
            self.objective_history_.append(loss.__copy__())
            self.n_iter_ = iteration + 1

            if best_loss is None or loss < best_loss:
                best_loss = loss.__copy__()
                best_coef = self.coef_.__copy__()
                best_intercept = self.intercept_.__copy__()
                best_iteration = iteration + 1

            if (
                prev_loss is not None
                and op.abs(loss - prev_loss)
                < self.tol * OPNs(0, -1)
            ):
                self.stop_reason_ = "objective_change"
                break

            if (
                op.linalg_norm(self.coef_ - coef_old, ord=1)
                < self.tol * OPNs(0, -1)
            ):
                self.stop_reason_ = "coefficient_change"
                break

            prev_loss = loss

        if self.adaptive_iter and best_coef is not None:
            self.coef_ = best_coef
            self.intercept_ = best_intercept

        self.best_iteration_ = int(best_iteration)
        self.best_objective_ = (
            None if best_loss is None else best_loss.__copy__()
        )

        # Auditable fixed-point diagnostic for the returned snapshot.  This is
        # recorded, not used as an additional stopping rule, so the established
        # early-stopped model semantics remain unchanged.
        final_residual = y_centered - X_centered @ self.coef_
        max_coordinate_residual = OPNs(0, 0)
        max_abs_a = 0.0
        max_abs_b = 0.0
        max_abs_p = 0.0
        max_abs_q = 0.0

        for j in range(n_features):
            residual_without_j = (
                final_residual
                + X_centered[:, j] * self.coef_[j]
            )
            rho = (
                op.dot(X_centered[:, j], residual_without_j)
                / n_samples
            )
            coordinate_update = self._soft_threshold(rho, self.alpha)
            delta = coordinate_update - self.coef_[j]
            abs_delta = op.abs(delta)
            if abs_delta > max_coordinate_residual:
                max_coordinate_residual = abs_delta.__copy__()

            delta_a = float(delta.a)
            delta_b = float(delta.b)
            delta_p = -(delta_a + delta_b)
            delta_q = delta_a - delta_b
            max_abs_a = max(max_abs_a, abs(delta_a))
            max_abs_b = max(max_abs_b, abs(delta_b))
            max_abs_p = max(max_abs_p, abs(delta_p))
            max_abs_q = max(max_abs_q, abs(delta_q))

        self.max_coordinate_residual_ = max_coordinate_residual
        self.max_coordinate_residual_components_ = {
            "abs_a": float(max_abs_a),
            "abs_b": float(max_abs_b),
            "abs_p": float(max_abs_p),
            "abs_q": float(max_abs_q),
        }
        self.final_objective_ = (
            0.5 * op.mean(final_residual ** 2)
            + self.alpha * op.sum(op.abs(self.coef_))
        )

        return self

    def predict(self, X, item=0):
        """Predict using the Lasso model."""
        y_opn_predicted = op.dot(X, self.coef_) + self.intercept_
        # Mapping of item to attribute selection logic
        item_map = {
            0: lambda x: x.a,
            1: lambda x: x.a + x.b,
            2: lambda x: x.b
        }
        func = item_map.get(item, lambda x: x.a)  # Default to item=0 if not in map
        if y_opn_predicted.ndim == 1:
            y_predicted = np.array([func(x) for x in y_opn_predicted])
        else:
            y_predicted = np.array([[func(x) for x in row] for row in y_opn_predicted])
        return y_predicted

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "learning_rate": self.learning_rate,
            "adaptive_lr": self.adaptive_lr,
            "adaptive_iter": self.adaptive_iter,
            # "coef_": self.coef_,
            # "intercept_": self.intercept_
        }

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


__all__ = ["Lasso", "LogisticRegression"]
