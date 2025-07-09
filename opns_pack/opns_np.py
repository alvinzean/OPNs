import copy
import functools
import numpy as np
import collections.abc

from opns_pack.opns import OPNs
from opns_pack.opns_matrix import OPNsMatrix
from opns_pack import opns_math

# To avoid calculation errors caused by floating-point inaccuracies, replace a=0 with abs(a) <= __zero_threshold__

zero_threshold = 1e-8


class Concatenator:
    def __getitem__(self, arrays):
        # Ensure input is a tuple
        if not isinstance(arrays, tuple):
            arrays = (arrays,)
        return concatenate(arrays[0], arrays[1], axis=1)


# Simplified implementation to concatenate two OPNsMatrixes horizontally, used like np.c_[arr1, arr2]
c_ = Concatenator()


# Interface consistent with Numpy --------------------------------------------------------
def array(mat=None):
    """
    Pass in real number arrays or OPNs arrays to create OPNsMatrix. Currently only supports 1D and 2D.
    Canceled the initialization array method of OPNsMatrix, changed to unified creation in this interface
    :param mat: Real number array or OPNs array
    :return: OPNsMatrix
    """
    # Group the input matrix elements in pairs sequentially to form OPNs
    # Put the first element of each OPNs in one matrix, and the second element in another matrix
    if isinstance(mat, OPNsMatrix):
        return copy.copy(mat)  # OPNsMatrix internally implements __copy__ with deep copy to ensure references are not passed

    elif isinstance(mat, (np.ndarray, list)):
        first_mat = []
        second_mat = []
        if isinstance(mat[0], OPNs):  # 1D, and contains OPNs
            for o in mat:
                first_mat.append(o.a)
                second_mat.append(o.b)
        elif isinstance(mat[0], (float, int, np.integer)):  # 1D, and contains numbers
            first_mat = mat[::2]
            second_mat = mat[1::2]
            if np.array(mat).shape[0] % 2 != 0:  # Odd number of elements, pad with 0 at the end
                second_mat = np.append(second_mat, 0)

        elif isinstance(mat[0], OPNsMatrix):  # 2D, outer layer is list or ndarray, inner layer is OPNsMatrix
            new_mat = array(np.array(mat))
            return new_mat

        elif isinstance(mat[0], (np.ndarray, list)):  # 2D
            if isinstance(mat[0][0], OPNs):  # 2D, and contains OPNs
                for row in mat:
                    first_row = []
                    second_row = []
                    for o in row:
                        first_row.append(o.a)
                        second_row.append(o.b)
                    first_mat.append(first_row)
                    second_mat.append(second_row)
            elif isinstance(mat[0][0], (float, int, np.integer)):  # 2D, and contains numbers
                darray_mat = np.array(mat)
                first_mat = darray_mat[:, ::2]
                second_mat = darray_mat[:, 1::2]
                if darray_mat.shape[1] % 2 != 0:  # Odd number of columns, pad with 0 column at the end
                    second_mat = np.column_stack((second_mat, np.zeros((darray_mat.shape[0], 1))))
        new_mat = OPNsMatrix()
        new_mat.set_matrix(first_mat, second_mat)
        return new_mat
    elif mat is None:
        return OPNsMatrix()


def mean(matrix, axis=None):
    """
    Mean value
    axis = None, calculate the overall mean of all elements
    axis = 0, calculate the mean of each column in the matrix
    axis = 1, calculate the mean of each row in the matrix
    or the mean of the array
    :param matrix:
    :param axis:
    :return:
    """
    if isinstance(matrix, OPNsMatrix):
        return matrix.mean(axis)
    else:
        print("This functionality is not yet implemented")


def average(a, axis=None, weights=None, returned=False):
    """
    Calculate the weighted average of an array. If no weights are specified, calculate the ordinary arithmetic mean.
    :param a: The array or object to be calculated.
    :param axis: Specify along which axis to calculate the average. By default, calculate the average of the entire array.
    :param weights: Weight array, consistent with the shape of a. Used to calculate weighted average.
    :param returned: If True, returns a tuple containing the weighted average and the sum of weights.
    :return: When returned=False, returns a scalar or array representing the average. When returned=True, returns a tuple (average, sum_of_weights).
    """
    # Convert input to numpy array for easier manipulation
    a = array(a)

    # If weights are not provided, calculate normal average
    if weights is None:
        # Calculate mean along the given axis
        avg = mean(a, axis=axis)
        if returned:
            # Return average and total weights (assumed as count of elements)
            count = a.shape[axis] if axis is not None else a.size
            return avg, count
        return avg

    # Convert weights to numpy array
    weights = np.array(weights)

    # Ensure weights shape matches the input array
    if weights.shape != a.shape:
        raise ValueError("weights and a must have the same shape.")

    # Calculate the weighted sum and weight total along the given axis
    weighted_sum = sum(a * weights, axis=axis)
    sum_of_weights = sum(weights, axis=axis)

    # Avoid division by zero
    # External processing: extract the zero property of OPNs instances and convert to numpy ndarray
    def extract_zero_property(opn_objects):
        return np.array([opn.zero for opn in opn_objects])
    zero_list = sum_of_weights
    if np.any(extract_zero_property(zero_list)):
        raise ZeroDivisionError("Sum of weights must not be zero.")

    # Calculate weighted average
    avg = weighted_sum / sum_of_weights

    if returned:
        # Return average and total weights
        return avg, sum_of_weights

    return avg


def sin(matrix):
    if isinstance(matrix[0], OPNsMatrix):
        # Process 2D matrix
        return array([[opns_math.sin(x) for x in row] for row in matrix])
    else:
        # Process 1D matrix
        return array([opns_math.sin(x) for x in matrix])


def abs(x):
    """
    Calculate the absolute value of the input, supports OPNs and array inputs.

    Parameters:
    :param x : OPNs or list-like
        A single OPNs value or iterable object (such as list, tuple, etc.) as input.

    Returns:
    OPNs or list
        The absolute value of the input. For array input, returns a list where each element is the absolute value.
    """
    # If input is an iterable object (e.g., list or tuple)
    if isinstance(x, (np.ndarray, list, tuple, OPNsMatrix)):
        return array([abs(el) for el in x])  # Recursively call abs on each element
    else:
        return x if x >= OPNs(0, 0) else -x  # For a single OPNs, directly return its absolute value


def linalg_norm(x, ord=None):  # To be migrated to opns_np.linalg (module structure needs to be reorganized)
    """
    Calculate the norm of a vector or matrix, similar functionality to np.linalg.norm.

    Parameters:
    :param x : array_like
        Input vector or matrix (list, tuple, or 2D list).
    :param ord : {None, 1, 2, inf, -inf}, optional
        Type of norm to calculate. Default calculates L2 norm.
        - None: Calculate L2 norm (default, Euclidean norm)
        - 1: Calculate L1 norm
        - inf: Calculate L∞ norm
        - -inf: Calculate L-∞ norm (minimum absolute value)
        - 'fro': For matrices, calculate Frobenius norm

    Returns:
    norm_value : float
        The norm of the input object.
    """

    if isinstance(x[0], (list, OPNsMatrix)):  # If input is a matrix (2D list)
        if ord is None or ord == 'fro':  # Frobenius norm
            return sum(sum(el ** 2 for el in row) for row in x)
        elif ord == float('inf'):  # L∞ norm (sum by rows)
            return max(sum(abs(el) for el in row) for row in x)
        elif ord == -float('inf'):  # L-∞ norm (minimum absolute value by rows)
            return min(max(abs(el) for el in row) for row in x)
        else:
            raise ValueError("Unsupported norm order for matrix")

    else:  # Input is a vector (1D list)
        if ord is None or ord == 2:  # L2 norm (Euclidean norm)
            return opns_math.sqrt(sum(el ** 2 for el in x))
        elif ord == 1:  # L1 norm
            return sum(abs(el) for el in x)
        elif ord == float('inf'):  # L∞ norm
            return max(abs(el) for el in x)
        elif ord == -float('inf'):  # L-∞ norm (minimum absolute value)
            return min(abs(el) for el in x)
        else:
            raise ValueError("Unsupported norm order for vector")


def copysign(opns_1, opns_2):
    """
    Modify and return the sign of the first parameter based on the sign of the second parameter
    :return:
    """
    if (opns_2 < OPNs(0, 0) < opns_1) or (opns_2 > OPNs(0, 0) > opns_1):  # If they have opposite signs
        return -opns_1
    else:
        return opns_1


def transpose(matrix):
    if isinstance(matrix, OPNsMatrix):
        return matrix.T


def dot(matrix1, matrix2):
    if isinstance(matrix1, OPNsMatrix) and isinstance(matrix2, OPNsMatrix):
        return matrix1.dot(matrix2)


def eye(n):
    """
    Create identity matrix
    :param n: Number of rows and columns
    :return: Identity matrix
    """
    return array([[-1 if j == 2 * i + 1 else 0 for j in range(n * 2)] for i in range(n)])


def sqrt(matrix):
    if isinstance(matrix, OPNsMatrix):
        if len(matrix.shape) == 1:
            result = [sqrt(single_opn) for single_opn in matrix]
            return array(result)
        else:
            rows = []
            for row in matrix:
                rows.append([sqrt(single_opn) for single_opn in row])
            return array(rows)
    elif isinstance(matrix, OPNs):
        return matrix ** 0.5
    else:
        print(f"sqrt function raise error: type error")


def diag(matrix):
    return array([matrix[i, i] for i in range(len(matrix))])


def sort(mat, axis=-1):
    """
    Perform ascending comparison on 1D and 2D OPNsMatrix, return sorted results.
    :param mat:
    :param axis: Default is -1, sort along the last dimension, for 2D it means sorting within each row
    :return:
    """
    if mat.ndim == 1:  # If it's a 1D array
        return mat[argsort(mat, axis=axis)]
    elif mat.ndim == 2:  # If it's a 2D array
        if axis == 0:
            return mat[argsort(mat, axis=axis), np.arange(mat.shape[1])[None, :]]
        elif axis == 1 or axis == -1:
            return mat[np.arange(mat.shape[0])[:, None], argsort(mat, axis=axis)]


def max(mat, axis=None, keepdims=False):
    """
    Array maximum value, currently only implements maximum values for 1D and 2D arrays
    Uses the same method as np.max, but returns an ndarray of OPNs elements
    :param mat:
    :param axis: None, find maximum among all elements; 0, row direction i.e., maximum in each column; 1, column direction i.e., maximum in each row
    :param keepdims:
    :return: Corresponding OPNs or OPNsMatrix
    """
    if axis is None:  # Find maximum among all elements
        flat = mat.flatten()  # Flatten
        return flat[argsort(flat)[-1]]  # Sort indices then directly take, high efficiency
    elif axis == 0:  # Row direction i.e., maximum in each column
        result = mat[argmax(mat, axis=axis), np.arange(mat.shape[1])]
        if keepdims:
            return result.reshape((1, -1))  # One row n columns
        else:
            return result
    elif axis == 1 or axis == -1:  # Column direction i.e., maximum in each row
        result = mat[np.arange(mat.shape[0]), argmax(mat, axis=axis)]
        if keepdims:
            return result.reshape((-1, 1))  # n rows one column
        else:
            return result
    raise Exception("This operation is not yet implemented")


def min(mat, axis=None, keepdims=False):
    """
   Array minimum value, currently only implements minimum values for 1D and 2D arrays
   Uses the same method as np.max, but returns an ndarray of OPNs elements
   :param mat:
   :param axis: None, find minimum among all elements; 0, row direction i.e., minimum in each column; 1, column direction i.e., minimum in each row
   :param keepdims:
   :return: Corresponding OPNs or OPNsMatrix
   """

    if axis is None:  # Find minimum among all elements
        flat = mat.flatten()  # Flatten
        return flat[argsort(flat)[-1]]  # Sort indices then directly take, high efficiency
    elif axis == 0:  # Row direction i.e., minimum in each column
        return mat[np.arange(mat.shape[0]), argmax(mat, axis=axis)]
    elif axis == 1 or axis == -1:  # Column direction i.e., minimum in each row
        return mat[argmax(mat, axis=axis), np.arange(mat.shape[1])]
    raise Exception("This operation is not yet implemented")


def argsort(mat, *args, **kwargs):
    """
    Can directly call np.argsort
    """
    return np.argsort(mat, *args, **kwargs)


def argmax(mat, axis=None):
    """
    Perform ascending comparison on OPNs arrays, return indices of new order
    Currently only implements sorting for 1D and 2D arrays
    axis defaults to None, i.e., flatten and sort
    :param mat:
    :param axis: None, sort all elements; 0, sort along row direction, i.e., sort each column; 1, sort along column direction, i.e., sort each row
    :return: A number indicating the position of maximum value; or a list
    """
    if axis is None:  # Flatten and compare
        return argsort(mat.flatten())[-1]
    elif axis == 0:  # Row direction sorting, i.e., sort within each column. The last row of sorted result is the maximum value
        return argsort(mat, axis=axis)[-1]
    elif axis == 1 or axis == -1:  # Column direction sorting, i.e., sort within each row. The last column of sorted result is the maximum value.
        return argsort(mat, axis=axis)[:, -1]
    else:
        raise Exception(f"opns_np.argmax high dimension axis={axis} not yet implemented")


def argmin(mat, axis=None):
    """
    Find the index of minimum value in 1D and 2D OPNsMatrix
    Currently only implements sorting for 1D and 2D arrays
    axis defaults to None, i.e., flatten and sort
    :param mat:
    :param axis: None, sort all elements; 0, sort along row direction, i.e., sort each column; 1, sort along column direction, i.e., sort each row
    :return: A number indicating the position of minimum value; or a list
    """
    if axis is None:  # Flatten and compare
        return argsort(mat.flatten())[0]
    elif axis == 0:  # Row direction sorting, i.e., sort within each column. The first row of sorted result is the minimum value
        return argsort(mat, axis=axis)[0]
    elif axis == 1 or axis == -1:  # Column direction sorting, i.e., sort within each row. The first column of sorted result is the minimum value. Since high dimensions not implemented, for compatibility, -1 temporarily represents 1
        return argsort(mat, axis=axis)[:, 0]
    else:
        raise Exception(f"opns_np.argmin high dimension axis={axis} not yet implemented")


def zeros(shape):
    result = OPNsMatrix()
    result.set_matrix(np.zeros(shape, dtype=float), np.zeros(shape, dtype=float))
    return result


def zeros_like(matrix):
    result = OPNsMatrix()
    result.set_matrix(np.zeros(matrix.shape), np.zeros(matrix.shape))
    return result


def ones(shape):
    result = OPNsMatrix()
    result.set_matrix(np.zeros(shape, dtype=float), -np.ones(shape, dtype=float))
    return result


def ones_like(matrix):
    result = OPNsMatrix()
    result.set_matrix(np.zeros(matrix.shape), -np.ones(matrix.shape, dtype=float))
    return result


def outer(matrix1, matrix2):
    return matrix1.outer(matrix2)


def unique(matrix, return_counts=False):
    """
    Find all unique elements in an array and optionally return their occurrence counts. These arrays are sorted in ascending order by element values
    Currently only implements processing for 1D arrays
    :param matrix: Input OPNsMatrix 1D or 2D array
    :param return_counts: Whether to return the occurrence counts of unique elements
    """
    if len(matrix.shape) == 1:
        unique_first_entry = []
        unique_second_entry = []

        times = []
        for i in range(matrix.shape[0]):
            # Check if this OPNs exists in the unique list and whether the two OPNs elements are at consistent positions in the two lists
            if matrix[i].a in unique_first_entry and matrix[i].b in unique_second_entry and unique_first_entry.index(
                    matrix[i].a) == unique_second_entry.index(matrix[i].b):
                index = unique_first_entry.index(matrix[i].a)
                times[index] = times[index] + 1
            elif matrix[i].b in unique_first_entry and matrix[i].a in unique_second_entry and unique_first_entry.index(
                    matrix[i].b) == unique_second_entry.index(
                matrix[i].a):
                index = unique_first_entry.index(matrix[i].b)
                times[index] = times[index] + 1
            else:  # If the two elements are at inconsistent positions, it means they are not the same OPNs
                unique_first_entry.append(matrix[i].a)
                unique_second_entry.append(matrix[i].b)
                times.append(1)
        result = OPNsMatrix()
        result.set_matrix(unique_first_entry, unique_second_entry)
        sorted_index = argsort(result)
        if return_counts:
            return result[sorted_index], np.array(times)[sorted_index]
        return result[sorted_index]


def size(matrix):
    """
    Number of elements
    """
    if isinstance(matrix, OPNsMatrix):
        return matrix.size()
    else:
        return np.size(matrix)


def sign(x, threshold=OPNs(0, -1e-8)):
    """
    Custom sign function simulating np.sign functionality.

    Parameters:
    - x: Input value or array
    - threshold: Threshold for judging close to zero, default is 1e-8

    Returns:
    - Sign values similar to np.sign: 1 (positive), -1 (negative), 0 (close to zero)
    """
    # Handle cases where input is a single value or array
    if isinstance(x, OPNsMatrix):
        return array([sign(val, threshold) for val in x])

    if x > threshold:
        return OPNs(0, -1)
    elif x < -threshold:
        return OPNs(0, 1)
    else:
        return OPNs(0, 0)


def concatenate(matrix1, matrix2, axis=0):
    """
    Concatenate two matrices
    :param matrix1:
    :param matrix2:
    :param axis: 0 means vertical concatenation (stack along row direction from top to bottom, number of columns must be the same), 1 means horizontal concatenation (along column direction from left to right, number of rows must be the same)
    """
    return matrix1.concatenate(matrix2, axis)


def _parse_input(matrices):
    """Uniformly convert input to a list of matrices"""
    if len(matrices) == 1 and isinstance(matrices[0], (list, tuple)):  # If input is a single list/tuple, extract internal matrices
        return list(matrices[0])
    else:  # If input is multiple independent arguments, directly convert to list
        return list(matrices)


def _validate_shapes(matrices, axis):
    """Check whether matrix shapes allow concatenation"""
    if not matrices:
        raise ValueError("Input matrix list is empty")

    ref_shape = matrices[0].shape  # Get reference dimensions of the first matrix
    ref_dim = ref_shape[1] if axis == 0 else ref_shape[0]

    for mat in matrices:
        if axis == 0:  # Vertical concatenation: check if number of columns is consistent
            if mat.shape[1] != ref_dim:
                raise ValueError(f"Number of columns inconsistent: expected {ref_dim}, actual {mat.shape[1]}")
        elif axis == 1:  # Horizontal concatenation: check if number of rows is consistent
            if mat.shape[0] != ref_dim:
                raise ValueError(f"Number of rows inconsistent: expected {ref_dim}, actual {mat.shape[0]}")
        else:
            raise ValueError("axis must be 0 (vertical) or 1 (horizontal)")


def vstack(*matrices):
    """
    Vertically concatenate multiple matrices (stack by rows)
    :param  matrices: Can pass multiple matrices or a single matrix list, e.g.: vstack(mat1, mat2) or vstack([mat1, mat2])
    :return  Concatenated matrix
    """
    matrices = _parse_input(matrices)
    _validate_shapes(matrices, axis=0)
    return array(np.vstack(matrices))


def hstack(*matrices):
    """
    Horizontally concatenate multiple matrices (stack by columns)
    :param  matrices: Can pass multiple matrices or a single matrix list, e.g.: hstack(mat1, mat2) or hstack([mat1, mat2])
    :return  Concatenated matrix
    """
    matrices = _parse_input(matrices)
    _validate_shapes(matrices, axis=1)
    return array(np.hstack(matrices))


def block(blocks):
    """
    Recursively implement block matrix combination, supporting nested structures.
    :param  blocks: (list or array) Input block structure
    :return  Combined matrix
    """
    if isinstance(blocks, list):
        if not blocks:
            return zeros((0, 0))  # Empty matrix

        # Determine if it's vertical concatenation (sub-elements are lists)
        if isinstance(blocks[0], list):
            # Recursively process each row, vertical concatenation
            sub_matrices = [block(sub) for sub in blocks]
            # Check if all sub-matrices have consistent number of columns
            n_cols = sub_matrices[0].shape[1] if sub_matrices else OPNs(0, 0)
            for sm in sub_matrices[1:]:
                if sm.shape[1] != n_cols:
                    raise ValueError("Inconsistent number of columns for vertical concatenation")
            # Vertical concatenation
            return vstack(sub_matrices)
        else:
            # Recursively process each block, horizontal concatenation
            sub_matrices = [block(b) for b in blocks]
            # Check if all sub-matrices have consistent number of rows
            n_rows = sub_matrices[0].shape[0] if sub_matrices else 0
            for sm in sub_matrices[1:]:
                if sm.shape[0] != n_rows:
                    raise ValueError("Inconsistent number of rows for horizontal concatenation")
            # Horizontal concatenation
            return hstack(sub_matrices)
    else:
        # Single block, directly convert to array, to be modified
        return np.atleast_2d(blocks)


def inv(matrix):
    """
    Find inverse matrix (Gaussian elimination method for inversion)
    """
    identity_mat = zeros_like(matrix)  # Initialize an all-zero matrix
    # Set diagonal elements to 1
    for i in range(matrix.shape[0]):
        # print(identity_mat[i][i])
        identity_mat[i, i] = OPNs(0, -1)
    # Put the identity matrix after the original matrix
    merge_matrix = hstack(matrix, identity_mat)
    for i in range(len(merge_matrix)):
        # Set the i-th element of the i-th row to 1
        ele = merge_matrix[i, i]
        for k in range(len(merge_matrix[i])):
            merge_matrix[i, k] = merge_matrix[i, k] / ele
        # Perform elimination on other rows
        for j in range(len(merge_matrix)):
            if j != i:
                ele = merge_matrix[j, i]
                for k in range(len(merge_matrix[j])):
                    merge_matrix[j, k] = merge_matrix[j, k] - ele * merge_matrix[i, k]
    inv_mat = merge_matrix[:, len(matrix):]
    return inv_mat


def inverse(X):
    """
    Manually implement matrix inversion (Gauss-Jordan elimination method)
    Input: OPNs square matrix X
    Output: Inverse matrix X_inv
    """
    # Check if input is a square matrix
    if X.shape[0] != X.shape[1]:
        raise ValueError("Matrix must be a square matrix to find inverse")

    n = X.shape[0]
    # Create augmented matrix [X | I]
    augmented = hstack(X, eye(n))

    for col in range(n):
        # Find pivot element (row with maximum absolute value) in current column
        pivot_row = argmax(abs(augmented[col:, col])) + col
        pivot_row = pivot_row.item()

        # Swap current row and pivot row
        temp_row = augmented[col].__copy__()
        augmented[col] = augmented[pivot_row]
        augmented[pivot_row] = temp_row

        # Normalize pivot row (make pivot element 1)
        pivot = augmented[col, col]
        augmented[col] = augmented[col] / pivot

        # Eliminate current column elements in other rows
        for row in range(n):
            if row != col:
                factor = augmented[row, col]
                augmented[row] -= factor * augmented[col]

    # Extract inverse matrix part (right side of augmented matrix)
    X_inv = augmented[:, n:]
    return X_inv


def custom_inv(matrix):
    """
    Manually implement small matrix inversion (supports n=1, 2, 3)
    :param  matrix: Input square matrix (dimension <= 3)
    :return  Inverse matrix
    """
    n = matrix.shape[0]
    assert matrix.shape == (n, n), "Input must be a square matrix"

    if n == 1:  # Scalar inversion
        return array([[OPNs(0, -1) / matrix[0, 0]]])

    elif n == 2:  # 2x2 matrix explicit formula
        a, b = matrix[0, 0], matrix[0, 1]
        c, d = matrix[1, 0], matrix[1, 1]
        det = a * d - b * c
        return array([[d, -b], [-c, a]]) / det

    elif n == 3:  # 3x3 matrix adjugate method
        # Calculate determinant
        det = (
                matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]) -
                matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0]) +
                matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
        )

        # Calculate adjugate matrix (transpose of cofactor matrix)
        cofactors = zeros((3, 3))
        cofactors[0, 0] = matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]
        cofactors[0, 1] = -(matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
        cofactors[0, 2] = matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0]
        cofactors[1, 0] = -(matrix[0, 1] * matrix[2, 2] - matrix[0, 2] * matrix[2, 1])
        cofactors[1, 1] = matrix[0, 0] * matrix[2, 2] - matrix[0, 2] * matrix[2, 0]
        cofactors[1, 2] = -(matrix[0, 0] * matrix[2, 1] - matrix[0, 1] * matrix[2, 0])
        cofactors[2, 0] = matrix[0, 1] * matrix[1, 2] - matrix[0, 2] * matrix[1, 1]
        cofactors[2, 1] = -(matrix[0, 0] * matrix[1, 2] - matrix[0, 2] * matrix[1, 0])
        cofactors[2, 2] = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

        adjugate = cofactors.T
        return adjugate / det

    else:
        raise ValueError("Only matrices with n <= 3 are supported")


def recursive_blockwise_inverse(M, threshold=3):
    """
    Recursive blockwise inverse method (Case 2: D is invertible), automatic blocking, no need to specify submatrix dimensions.
    :param  M: Input OPNs square matrix
    :param  threshold: Threshold for direct inversion, when matrix dimension is less than or equal to this value, invert directly
    :return  Inverse matrix
    """
    n = M.shape[0]
    assert M.shape[0] == M.shape[1], "Matrix must be square"

    # Base case: matrix is small enough, invert directly
    if n <= threshold:
        return custom_inv(M)

    # Blocking strategy: set D as the bottom-right 1x1 submatrix
    k = n - 1  # A has dimension k x k, D is 1x1

    A = M[:k, :k]
    B = M[:k, k:]
    C = M[k:, :k]
    D = M[k:, k:]

    try:
        D_inv = recursive_blockwise_inverse(D, threshold)  # Inverse of D (recursive processing)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError(f"D{D} is singular, cannot compute inverse.")

    # Calculate Schur complement S = A - B @ D^{-1} @ C
    S = A - B @ D_inv @ C

    try:
        S_inv = recursive_blockwise_inverse(S, threshold)  # Recursively compute inverse of S
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Schur complement is singular.")

    # Calculate each block
    top_left = S_inv
    top_right = -S_inv @ B @ D_inv
    bottom_left = -D_inv @ C @ S_inv
    bottom_right = D_inv + D_inv @ C @ S_inv @ B @ D_inv

    # Combine inverse matrix
    inv_mat = array(np.block([[top_left, top_right], [bottom_left, bottom_right]]))
    return inv_mat




def var(matrix, axis=None):
    """
    Calculate the variance of the array
    :param matrix:
    :param axis: 0 returns variances for each column. 1 returns variances for each row
    """
    mat_mean = mean(matrix, axis)  # Calculate mean
    variance = mean((matrix - mat_mean) ** 2, axis)  # Variance
    return variance


def std(matrix, axis=None):
    """
    Calculate the standard deviation of the array
    :param matrix:
    :param axis: 0 returns standard deviations for each column. 1 returns standard deviations for each row
    """
    variance = var(matrix, axis)
    std_dev = sqrt(variance)  # Calculate standard deviation
    return std_dev


def prod(matrix):
    """
    Product of all elements
    Axis-wise product is not yet implemented, currently calculates the product of all elements in a 1D array
    """
    if len(matrix.shape) == 1:
        result = OPNs(0, -1)
        for ele in matrix:
            result = result * ele
        return result
    raise Exception("This operation is not yet implemented")


def exp(mat):
    """
    Custom implementation similar to np.exp function, currently only supports multi-dimensional OPNs matrices as input
    """
    if isinstance(mat, OPNsMatrix) and mat.ndim == 1:
        # Process 1D OPNs list, recursively apply to each element
        return array([exp(element) for element in mat])
    elif isinstance(mat, OPNsMatrix) and mat.ndim == 2:
        # Process OPNs matrix
        return array([[exp(element) for element in row] for row in mat])
    else:
        # Process single value
        return opns_math.exp(mat)


def log(matrix):
    """
    Logarithm for 1D arrays
    :param matrix:
    :return:
    """
    if len(matrix.shape) == 1:
        return array([opns_math.log(ele) for ele in matrix])
    raise Exception("This operation is not yet implemented")

# OPNs-specific interfaces --------------------------------------------------------

class OPNsMatrixGenerate:
    """
    OPNs matrix generator
    """

    def __init__(self, matrix):
        """
        Initialization, pass in a real number matrix
        """
        if matrix.shape[1] % 2 != 0:
            self.matrix_new = np.column_stack((matrix, np.zeros((matrix.shape[0], 1))))
        else:
            self.matrix_new = matrix[:]

    def generate(self, pair):
        """
        Pass in pairing method, e.g., [(2, 0), (1, 3)], to generate OPNsMatrix
        :param pair: Pairing method
        :return: OPNsMatrix
        """
        flat_pair = [item for sublist in pair for item in sublist]  # Flatten grouped pairs into a continuous list

        return array(self.matrix_new[:, flat_pair])


def pdf(x, loc=OPNs(0, 0), scale=OPNs(0, -1)):
    """
    Probability density function of normal distribution. Interface: scipy.stats.norm.pdf
    f(x | μ, σ) = (1 / sqrt(2πσ²)) * exp( - ((x-μ)²) / (2σ²))
    :param x: Value for calculation
    :param loc: μ Mean value
    :param scale: σ Standard deviation
    """
    return exp(-(x - loc) ** 2 / (2 * (scale ** 2))) / (scale * np.sqrt(2 * np.pi))  # Not sure why there's a negative sign here, the original formula doesn't have it


def make_matrix_from_two_matrix(first_entry, second_entry):
    result = OPNsMatrix()
    result.set_matrix(first_entry, second_entry)
    return result


def sum(lst, axis=None):
    """
    All sum operations must use this method
    :param lst: OPNs matrix (vector), or generator
    :param axis: 0 means one sum for each column, resulting in a total of num_columns results. 1 means num_rows results
    :return: Sum of all values
    """
    result = 0
    if isinstance(lst, OPNsMatrix):
        return lst.sum(axis=axis)
    elif isinstance(lst, collections.abc.Generator):
        for el in lst:
            result += el
        return result
    else:
        raise Exception("Currently only supports sum operations for OPNs matrices (vectors) and generators")


def all_lt(matrix, opns_1):
    """
    Determine if all elements in the matrix are less than a value
    Initial flag is True, stops the loop and returns False as soon as a greater value is encountered
    :param matrix: Input matrix
    :param opns_1: Value to compare against
    :return: Boolean result
    """
    flag = True
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > opns_1:  # Break as soon as any element is greater
                flag = False
                break
    return flag


def compare(opns_1, opns_2):
    """
    Return a value that can be used for comparison
    :param opns_1: Ordered pair 1
    :param opns_2: Ordered pair 2
    :return: Comparable value (1 if opns_1 >= opns_2, -1 otherwise)
    """
    if opns_1 >= opns_2:
        return 1
    else:
        return -1


def opns_to_2_num(matrix):
    """
    Decompose OPNs matrix into real number matrices
    :param matrix: Input OPNs matrix
    :return: Decomposed real number matrices
    """
    return matrix.opns_to_2_num()


def opns_to_1_num(matrix):
    """
    Add OPNs matrix elements pairwise and merge into a real number matrix
    :param matrix: Input OPNs matrix
    :return: Merged real number matrix
    """
    return matrix.opns_to_1_num()


def distance_to_row(matrix1, matrix2):
    """
    Calculate the distance between matrix1 and each row of matrix2
    Used for KNN distance calculation
    :param matrix1: First matrix
    :param matrix2: Second matrix (distances calculated to each row)
    :return: Distance matrix
    """
    # Matrix A as a whole - each row of matrix B
    diff_first_entry = matrix1.left_matrix - matrix2.left_matrix[:, np.newaxis]  # Result is a 3D matrix
    diff_second_entry = matrix1.right_matrix - matrix2.right_matrix[:, np.newaxis]

    # Calculate first_entry + second_entry
    sum_entry = diff_first_entry + diff_second_entry

    # Define condition
    condition = (sum_entry > 0) | ((sum_entry == 0) & (diff_first_entry < 0))

    # Use numpy's where function to select elements based on condition
    _first_array = np.where(condition, -diff_first_entry, diff_first_entry)
    _second_array = np.where(condition, -diff_second_entry, diff_second_entry)

    result = OPNsMatrix()
    result.first_entry = np.sum(_first_array, axis=2)  # Calculate the sum of each 2D array in the 3D array
    result.second_entry = np.sum(_second_array, axis=2)
    return result


def jacobi(A, iterations=5000):
    n = len(A)
    last_X = None
    X = eye(n)
    D = copy.copy(A)

    for sss in range(iterations):
        max_val = D[0][1]
        max_pos = (0, 1)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(D[i, j]) > max_val:
                    max_val = abs(D[i, j])
                    max_pos = (i, j)
        i, j = max_pos
        # To prevent D[i, i] == D[j, j], add an epsilon = 1e-9
        epsilon = 1e-9
        if D[i, i] - D[j, j] < OPNs(0, -epsilon):
            theta = OPNs(0, -np.pi / 4)
        else:
            theta = opns_math.atan((2 * D[i, j]) / (D[i, i] - D[j, j]) + OPNs(0, -epsilon)) / 2

        U = eye(n)
        U[i, i] = opns_math.cos(theta)
        U[j, j] = opns_math.cos(theta)
        U[i, j] = -opns_math.sin(theta)
        U[j, i] = opns_math.sin(theta)

        D = U.T.dot(D).dot(U)
        X = X.dot(U)

        if last_X is not None:
            change_matrix = X - last_X
            if (abs(change_matrix)).sum() <= OPNs(0, -epsilon):
                break
        last_X = X
    return diag(D), X


def generate_random_matrix(m, n, dtype='float', low=0, high=1):
    """
    Generate an m×n random matrix for testing purposes
    :param
        n (int): Number of rows
        m (int): Number of columns
        dtype (str): Data type, can be 'float' or 'int'
        low (float/int): Minimum value for random numbers
        high (float/int): Maximum value for random numbers
    :return: Generated random matrix
    """
    # Check input validity
    assert n > 0 and m > 0, "Rows and columns must be positive integers"
    assert low < high, "Minimum value must be less than maximum value"

    # Generate matrix based on data type
    if dtype == 'float':
        matrix = np.random.uniform(low=low, high=high, size=(m, n))
    elif dtype == 'int':
        matrix = np.random.randint(low=low, high=high, size=(m, n))
    else:
        raise ValueError("dtype must be 'float' or 'int'")

    return matrix

