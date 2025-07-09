import pandas as pd
import numpy as np

import opns_pack.opns_np as op
from opns_pack.opns import OPNs
import opns_pack.custom_gen_pairs as cgp


def array_to_dataframe(array, column_prefix='Column'):
    """
    Convert any shape of ndarray or list to DataFrame.
    :param array: ndarray or list to be converted.
    :param column_prefix: Column name prefix, default is 'Column'.
    :return: Converted Pandas DataFrame.
    """
    if isinstance(array, list):
        array = np.array(array)
    if array.ndim == 1:  # Check the dimension of the array
        df = pd.DataFrame(array, columns=[f'{column_prefix}1'])
    elif array.ndim == 2:
        num_columns = array.shape[1]
        columns = [f'{column_prefix}{i + 1}' for i in range(num_columns)]
        df = pd.DataFrame(array, columns=columns)
    else:
        raise ValueError("Only one-dimensional and two-dimensional ndarrays are supported for conversion to DataFrame")
    return df


class LinearRegression:
    def __init__(self, coefficient=None, intercept=None):
        self.coefficient = coefficient
        self.intercept = intercept

    def get_params(self, deep=None):
        return {'coefficient': self.coefficient, 'intercept': self.intercept}

    def fit(self, X, y):
        if isinstance(y, (list, np.ndarray)):
            y = array_to_dataframe(y, column_prefix='y')
        if isinstance(y, pd.Series):  # Convert Series to DataFrame
            y = y.to_frame()
        if y.shape[1] % 2 != 0:  # Add a new column with value 0
            y['New_Column'] = 0
        y = cgp.data_convert(y, y.columns.tolist(), bias=False)  # Convert real-valued dataset y to OPNs
        X_b = op.c_[X, op.ones((X.shape[0], 1))]  # Adding bias term (column of ones) to the input features
        theta_best = op.recursive_blockwise_inverse(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = theta_best[-1]
        self.coefficient = theta_best[:-1]
        return self

    def predict(self, X, item=0):
        y_opn_predicted = op.dot(X, self.coefficient) + self.intercept
        # Mapping of item to attribute selection logic
        item_map = {
            0: lambda x: x.a + x.b,
            1: lambda x: x.a,
            2: lambda x: x.b
        }
        func = item_map.get(item, lambda x: x.a + x.b)  # Default to item=0 if not in map
        y_predicted = np.array([[func(x) for x in row] for row in y_opn_predicted])
        return y_predicted


class LinearRegressionGradientDescent:
    def __init__(self, bias_position='first', learning_rate=0.01, max_iters=2000, tol=1e-4,  # Loss change tolerance (early stopping condition)
                 optimizer='adam',  # Options: 'sgd', 'adam', 'rmsprop'
                 beta1=0.9,  # Adam momentum parameter
                 beta2=0.999,  # Adam RMS parameter
                 epsilon=1e-8  # Prevent division by zero
                 ):
        self.bias_position = bias_position
        self.lr = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weights = None
        self.loss_history = []  # Record loss changes

    def get_params(self, deep=True):
        """
        Parameters: deep (bool): Whether to return a copy of nested parameters (preserves interface consistency, not actually used).
        Returns: dict: Parameter dictionary containing all hyperparameters defined in the constructor.
        """
        return {
            'bias_position': self.bias_position,
            'learning_rate': self.lr,
            'max_iters': self.max_iters,
            'tol': self.tol,
            'optimizer': self.optimizer,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon
        }

    def set_params(self, **params):
        """
        Parameters: **params: Parameter dictionary, keys must match those returned by get_params().
        Returns: self: Returns the updated model instance.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def fit(self, X, y):
        # Data preprocessing (maintain original logic)
        if isinstance(y, (list, np.ndarray)):
            y = array_to_dataframe(y, column_prefix='y')
        if isinstance(y, pd.Series):
            y = y.to_frame()
        if y.shape[1] % 2 != 0:
            y['New_Column'] = 0
        y = cgp.data_convert(y, y.columns.tolist(), bias=False)

        # Add bias column
        if self.bias_position == 'first':
            X_b = op.c_[op.ones(X.shape[0]).reshape((-1, 1)), X]
        else:
            X_b = op.c_[X, op.ones(X.shape[0])]

        # Initialize parameters and optimizer state
        self.weights = op.zeros(X_b.shape[1]).reshape((-1, 1))
        m = op.zeros_like(self.weights)  # Momentum (Adam)
        v = op.zeros_like(self.weights)  # RMS (Adam)
        t = 0  # Time step (Adam)

        # Gradient descent (dynamic iteration)
        for iter in range(self.max_iters):
            y_pred = X_b @ self.weights
            error = y_pred - y
            gradient = X_b.T @ error / (len(X_b) * OPNs(0, -1))

            # ====== Adaptive learning rate (Adam optimizer) ======
            if self.optimizer == 'adam':
                t += 1
                m = self.beta1 * m + (1 - self.beta1) * gradient
                v = self.beta2 * v + (1 - self.beta2) * (gradient ** 2)
                m_hat = m / ((1 - self.beta1 ** t) * OPNs(0, -1))
                v_hat = v / ((1 - self.beta2 ** t) * OPNs(0, -1))
                update = self.lr * m_hat / (op.sqrt(v_hat) + self.epsilon * OPNs(0, -1))
            elif self.optimizer == 'rmsprop':
                v = self.beta2 * v + (1 - self.beta2) * (gradient ** 2)
                update = self.lr * gradient / (op.sqrt(v) + self.epsilon * OPNs(0, -1))
            else:  # Standard SGD
                update = self.lr * gradient

            # ====== Update weights ======
            self.weights -= update

            # ====== Calculate loss (for early stopping) ======
            current_loss = op.mean(error ** 2)  # Mean squared error
            self.loss_history.append(current_loss)  # Assume OPN for loss

            # ====== Early stopping check ======
            # if iter > 10 and op.abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol * OPN(0, -1):
            if iter > 10 and op.abs(self.loss_history[-2] - self.loss_history[-1]) < OPNs(self.tol, -self.tol):
                print('loss_history[-2], loss_history[-1]: ', self.loss_history[-2], self.loss_history[-1])
                print('self.loss_history[-2] - self.loss_history[-1]):',
                      op.abs(self.loss_history[-2] - self.loss_history[-1]))
                print(f"Early stopping at iteration {iter}")
                break

    def predict(self, X, item=0):
        if self.bias_position == 'first':
            X_b = op.c_[op.ones(X.shape[0]).reshape((-1, 1)), X]
        else:
            X_b = op.c_[X, op.ones(X.shape[0]).reshape((-1, 1))]
        y_opn_predicted = X_b @ self.weights
        item_map = {
            0: lambda x: x.a + x.b,
            1: lambda x: x.a,
            2: lambda x: x.b
        }
        func = item_map.get(item, lambda x: x.a + x.b)  # Default to item=0 if not in map
        y_predicted = np.array([[func(x) for x in row] for row in y_opn_predicted])
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
        """Fit the Lasso model using coordinate descent."""
        # Validate input
        n_samples, n_features = X.shape
        if isinstance(y, (list, np.ndarray)):
            y = array_to_dataframe(y, column_prefix='y')
        if isinstance(y, pd.Series):  # Convert Series to DataFrame
            y = y.to_frame()
        if y.shape[1] % 2 != 0:  # Add a new column with value 0
            y['New_Column'] = 0
        y = cgp.data_convert(y, y.columns.tolist(), bias=False)  # Convert real-valued dataset y to OPNs
        y = y.reshape(-1)

        # Initialize coefficients
        self.coef_ = op.zeros(n_features)
        self.intercept_ = OPNs(0, 0)

        # Precompute some values
        X_mean = op.mean(X, axis=0)
        y_mean = op.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        lr = self.learning_rate
        prev_loss = OPNs(0, -float('inf'))
        coef_old = None

        for iteration in range(self.max_iter):
            coef_old = self.coef_.__copy__()
            for j in range(n_features):
                residual = y_centered - (X_centered @ self.coef_ - X_centered[:, j] * self.coef_[j])
                rho = op.dot(X_centered[:, j], residual) / n_samples
                self.coef_[j] = self._soft_threshold(rho, self.alpha)

            # Update intercept
            self.intercept_ = y_mean - op.dot(X_mean, self.coef_)

            # Calculate loss
            loss = 0.5 * op.mean((y_centered - X_centered @ self.coef_) ** 2) + self.alpha * op.sum(op.abs(self.coef_))

            if op.abs(loss - prev_loss) < self.tol * OPNs(0, -1):
                # print(f"Break1: Converged after {iteration} iterations.")
                break

            # Check convergence
            if op.linalg_norm(self.coef_ - coef_old, ord=1) < self.tol * OPNs(0, -1):  # Consider termination condition
                # print(f"Break2: Converged after {iteration} iterations.")
                break

            # Adaptive learning rate
            if self.adaptive_lr:
                if loss < prev_loss:
                    lr *= 1.5  # Increase learning rate slightly (this variable can be manually adjusted)
                else:
                    lr *= 0.5  # Decrease learning rate significantly
            prev_loss = loss

            # Store best iteration for early stopping
            if self.adaptive_iter and loss < prev_loss:
                self.max_iter = iteration + 1
                # print(f"Break2: Converged after {iteration} iterations.")
                break

        # If adaptive_iter is enabled, finalize using the best iteration
        if self.adaptive_iter:
            self.coef_ = coef_old

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
            "coef_": self.coef_,
            "intercept_": self.intercept_
        }

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
