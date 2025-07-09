import copy
import numpy as np
from opns_pack.opns import OPNs


class OPNsMatrix:
    """
    OPNs matrix class
    """
    # To avoid calculation errors caused by floating-point inaccuracies,
    # a is considered zero if abs(a) <= __zero_threshold__
    __zero_threshold__ = 1e-8

    @classmethod
    def set_zero_threshold(cls, threshold):
        cls.__zero_threshold__ = threshold

    @property
    def shape(self):
        return self.__left_matrix.shape

    @property
    def size(self):
        return np.size(self.__left_matrix)

    @property
    def T(self):
        result = OPNsMatrix()   # Calculate the squared difference between each element and the mean
        result.__left_matrix = self.__left_matrix.T
        result.__right_matrix = self.__right_matrix.T
        return result

    @property
    def ndim(self):
        return self.__left_matrix.ndim

    @property
    def left_matrix(self):
        return self.__left_matrix

    @property
    def right_matrix(self):
        return self.__right_matrix

    @left_matrix.setter
    def left_matrix(self, matrix):
        nd_mat = np.array(matrix)  # Convert everything to numpy
        if np.issubdtype(nd_mat.dtype, np.number):  # Let numpy check if all elements are numeric
            self.__left_matrix = np.array(matrix, dtype=float)
        else:
            raise ValueError(f"{matrix} element is not number!")

    @right_matrix.setter
    def right_matrix(self, matrix):
        nd_mat = np.array(matrix)  # Convert everything to numpy
        if np.issubdtype(nd_mat.dtype, np.number):  # Let numpy check if all elements are numeric
            self.__right_matrix = np.array(matrix, dtype=float)
        else:
            raise ValueError(f"matrix element is not number: {matrix}")

    def __init__(self):
        self.__left_matrix = None
        self.__right_matrix = None

    def __copy__(self):
        """
        Must perform deep copy to ensure the new object does not modify the content of the original object
        """
        new_obj = self.__class__()
        new_obj.__left_matrix = np.copy(self.__left_matrix)
        new_obj.__right_matrix = np.copy(self.__right_matrix)
        return new_obj

    def __len__(self):
        return len(self.__left_matrix)

    def __str__(self):
        """
        Print result
        Output all contents
        """
        text = ""
        text = text + "["
        if len(self.__left_matrix.shape) == 2:
            for i in range(self.__left_matrix.shape[0]):
                if i == 0:
                    text = text + "["
                else:
                    text = text + " ["
                for j in range(self.__left_matrix.shape[1]):
                    text = text + f"({self.__left_matrix[i][j]}, {self.__right_matrix[i][j]})"
                    if j != self.__left_matrix.shape[1] - 1:
                        text = text + " "
                if i == self.__left_matrix.shape[0] - 1:
                    text = text + "]"
                else:
                    text = text + "]\n"
        else:
            for i in range(self.__left_matrix.shape[0]):
                text = text + f"({self.__left_matrix[i]}, {self.__right_matrix[i]})"
                if i != self.__left_matrix.shape[0] - 1:
                    text = text + " "
        text = text + "]"
        return text

    def __getitem__(self, item):
        result = OPNsMatrix()
        if isinstance(item, tuple):  # If the input is a tuple [ , ]
            if isinstance(item[0], int) and isinstance(item[1], int):
                return OPNs(self.__left_matrix[item], self.__right_matrix[item])
            elif (isinstance(item[0], int) and isinstance(item[1], slice)) or (
                    isinstance(item[0], slice) and isinstance(item[1], int)):
                result.__left_matrix = np.array(self.__left_matrix[item], dtype=float)
                result.__right_matrix = np.array(self.__right_matrix[item], dtype=float)
            elif isinstance(item[0], slice) and isinstance(item[1], slice):
                result.__left_matrix = self.__left_matrix[item]
                result.__right_matrix = self.__right_matrix[item]
            elif item[0] is None or item[1] is None:  # Not strict, mainly to handle the case like [:, None]
                result.__left_matrix = self.__left_matrix[item]
                result.__right_matrix = self.__right_matrix[item]
            else:
                result.__left_matrix = self.__left_matrix[item]
                result.__right_matrix = self.__right_matrix[item]
                
        elif isinstance(item, int):
            if len(self.__left_matrix.shape) == 2:  # It is a matrix
                if isinstance(self.__left_matrix[item], np.ndarray):
                    result.__left_matrix = np.array(self.__left_matrix[item], dtype=float)
                    result.__right_matrix = np.array(self.__right_matrix[item], dtype=float)
                else:
                    return OPNs(self.__left_matrix[item], self.__right_matrix[item])
            else:  # It is a list
                result = OPNs(self.__left_matrix[item], self.__right_matrix[item])
        elif isinstance(item, slice) or isinstance(item, list) or isinstance(item, np.ndarray):
            result.__left_matrix = self.__left_matrix[item]
            result.__right_matrix = self.__right_matrix[item]
        return result

    def __setitem__(self, key, value):
        if isinstance(key, tuple):  # If the input is a tuple [ , ]
            if isinstance(key[0], int) and isinstance(key[1], int):  # both are integers
                self.__left_matrix[key], self.__right_matrix[key] = value.a, value.b
            else:  # When both elements in key are lists
                if isinstance(value, OPNs):
                    self.__left_matrix[key] = value.a
                    self.__right_matrix[key] = value.b
                else:
                    self.__left_matrix[key] = value.__left_matrix
                    self.__right_matrix[key] = value.__right_matrix
        elif isinstance(key, int) and self.ndim == 1:
            self.__left_matrix[key], self.__right_matrix[key] = value.a, value.b
        elif isinstance(key, int) and self.ndim == 2:
            self.__left_matrix[key], self.__right_matrix[key] = value.__left_matrix, value.__right_matrix
        else:
            raise Exception(f"This operation is not implemented yet")

    def __eq__(self, other):
        """
        Equal to ==
        """
        if isinstance(other, OPNsMatrix):
            return np.array_equal(self.__left_matrix, other.__left_matrix) and np.array_equal(self.__right_matrix,
                                                                                              other.__right_matrix)
        elif isinstance(other, OPNs):
            return np.logical_and(self.__left_matrix == other.a, self.__right_matrix == other.b)
        raise Exception(f"== cannot compare OPNsMatrix and {type(other)}")

    def __lt__(self, other):
        """
        Less than <
        """
        if isinstance(other, OPNsMatrix):
            sub_left = self.__left_matrix - other.__left_matrix
            sub_right = self.__right_matrix - other.__right_matrix
            
            return np.logical_or((sub_left + sub_right) > 0, np.logical_and(sub_left + sub_right == 0, sub_left < 0))
        elif isinstance(other, OPNs):
            sub_left = self.__left_matrix - other.a
            sub_right = self.__right_matrix - other.b
            return np.logical_or((sub_left + sub_right) > 0, np.logical_and(sub_left + sub_right == 0, sub_left < 0))
        raise Exception(f"< cannot compare OPNsMatrix and {type(other)}")

    def __le__(self, other):
        """
        Less than or equal to <=
        """
        if isinstance(other, OPNsMatrix) or isinstance(other, OPNs):
            return np.logical_or(self.__lt__(other), self.__eq__(other))
        raise Exception(f"<= cannot compare OPNsMatrix and {type(other)}")

    def __gt__(self, other):
        """
        Greater than >
        """
        if isinstance(other, OPNsMatrix):
            sub_left = self.__left_matrix - other.__left_matrix
            sub_right = self.__right_matrix - other.__right_matrix
            
            # (a + b < 0) or (a + b == 0 and a > 0)
            return np.logical_or((sub_left + sub_right) < 0, np.logical_and(sub_left + sub_right == 0, sub_left > 0))
        elif isinstance(other, OPNs):
            sub_left = self.__left_matrix - other.a
            sub_right = self.__right_matrix - other.b
            return np.logical_or((sub_left + sub_right) < 0, np.logical_and(sub_left + sub_right == 0, sub_left > 0))
        raise Exception(f"> cannot compare OPNsMatrix and {type(other)}")

    def __gl__(self, other):
        """
        Greater than or equal to >=
        """
        if isinstance(other, OPNsMatrix) or isinstance(other, OPNs):
            return np.logical_or(self.__gt__(other), self.__eq__(other))
        raise Exception(f">= cannot compare OPNsMatrix and {type(other)}")

    def __add__(self, other):
        """
        Addition + +
        :param other:
        :return:
        """
        if isinstance(other, OPNsMatrix):
            result = OPNsMatrix()
            result.__left_matrix = self.__left_matrix + other.__left_matrix
            result.__right_matrix = self.__right_matrix + other.__right_matrix
            return result
        elif isinstance(other, OPNs):
            result = OPNsMatrix()
            result.__left_matrix = self.__left_matrix + other.a
            result.__right_matrix = self.__right_matrix + other.b
            return result
        raise Exception(f"Addition not supported between: {OPNsMatrix} and {type(other)}")

    def __radd__(self, other):
        """
        The built-in sum() function starts with 0, so the first time it calls __add__,
        it actually tries to add 0 (an integer) with your custom object. This typically
        leads to an error because Python doesn't know how to add an int and your object.

        To solve this, you should define the __radd__ method in your class. This method
        specifies how to perform addition when your object appears on the right-hand side
        and an object of a different type (such as an int) appears on the left.

        A common practice is to check in __radd__ whether other == 0. If so, return a copy
        of the current object. This way, when sum() tries to add 0 and your object, it simply
        returns your object and avoids errors.
        :param other:
        :return:
        """
        if isinstance(other, (int, float)) and other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtraction -
        :param other:
        :return:
        """
        if isinstance(other, OPNsMatrix):
            result = OPNsMatrix()
            result.__left_matrix = self.__left_matrix - other.__left_matrix
            result.__right_matrix = self.__right_matrix - other.__right_matrix
            return result
        elif isinstance(other, OPNs):
            result = OPNsMatrix()
            result.__left_matrix = self.__left_matrix - other.a
            result.__right_matrix = self.__right_matrix - other.b
            return result
        raise Exception(f"The subtrahend is neither an OPNs nor an OPNsMatrix: {self}, {other}")

    def __rsub__(self, other):
        if isinstance(other, OPNs):
            result = OPNsMatrix()
            result.__left_matrix = other.a - self.__left_matrix
            result.__right_matrix = other.b - self.__right_matrix
            return result
        raise Exception(f"Subtraction must be between an OPNs and an OPNsMatrix: {other}, {self} ")

    def __neg__(self):
        """
        Unary negation operator
        :return:
        """
        result = OPNsMatrix()
        result.__left_matrix = - self.__left_matrix
        result.__right_matrix = - self.__right_matrix
        return result

    def __mul__(self, other):
        """
        Element-wise multiplication or scalar multiplication *
        (a, b) * (c, d) = (-ad-bc, -ac-bd)
        :param other:
        :return:
        """
        result = OPNsMatrix()
        if isinstance(other, OPNsMatrix):  # Element-wise matrix multiplication
            result.__left_matrix = -self.__left_matrix * other.__right_matrix - self.__right_matrix * other.__left_matrix
            result.__right_matrix = -self.__left_matrix * other.__left_matrix - self.__right_matrix * other.__right_matrix
        elif isinstance(other, (int, float)):  # Scalar multiplication
            result.__left_matrix = other * self.__left_matrix
            result.__right_matrix = other * self.__right_matrix
        elif isinstance(other, OPNs):  # OPNs × OPNsMatrix
            result.__left_matrix = -self.__left_matrix * other.b - self.__right_matrix * other.a
            result.__right_matrix = -self.__left_matrix * other.a - self.__right_matrix * other.b
        elif isinstance(other, np.ndarray):  # Element-wise multiplication with real-valued array
            result.__left_matrix = self.__left_matrix * other
            result.__right_matrix = self.__right_matrix * other
        else:
            raise Exception(f"Unsupported multiplication: {type(other)} * OPNsMatrix")
        return result

    def __rmul__(self, other):
        if isinstance(other, int) and other == 1:
            return self
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Division
        :param other:
        :return:
        """
        if isinstance(other, OPNs):
            return self.__mul__(1 / other)
        elif isinstance(other, OPNsMatrix):
            # Element-wise matrix division
            # Division is equivalent to multiplication by the reciprocal
            # First compute the reciprocal
            reciprocal = other.reciprocal()
            # Then multiply by the reciprocal
            return self.__mul__(reciprocal)
        elif isinstance(other, np.ndarray):  # Element-wise division by real-valued array
            result = OPNsMatrix()
            result.__left_matrix = self.__left_matrix / other
            result.__right_matrix = self.__right_matrix / other
            return result
        raise Exception("Other division types are not yet implemented")

    def __rtruediv__(self, other):
        """
        Called when this matrix is the divisor
        """
        if isinstance(other, (int, float)):  # Divided by a number: number × reciprocal
            return self.reciprocal().multi(other)
        return other.__truediv__(self)

    def __matmul__(self, other):
        """
        Matrix multiplication operator @
        """
        return self.dot(other)

    def reciprocal(self):
        """
        Compute the reciprocal of each element in the OPNs matrix
        Equivalent to 1 / matrix
        """
        # Create new zero matrices c and d with the same shape as a and b
        c = np.zeros_like(self.__left_matrix, dtype=float)
        d = np.zeros_like(self.__right_matrix, dtype=float)

        # Condition: elements are equal or negatives of each other
        condition = np.logical_or(self.__left_matrix == self.__right_matrix,
                                  self.__left_matrix == -self.__right_matrix)

        # If equal or opposite, keep original values for c and d
        c[condition] = self.__left_matrix[condition]
        d[condition] = self.__right_matrix[condition]

        # Otherwise
        not_condition = np.logical_not(condition)

        # c = a / (a^2 - b^2)
        c[not_condition] = self.__left_matrix[not_condition] / (
                self.__left_matrix[not_condition] ** 2 - self.__right_matrix[not_condition] ** 2)

        # d = b / (b^2 - a^2); Note: denominator must be non-zero to avoid division by zero
        d[not_condition] = self.__right_matrix[not_condition] / (
                self.__right_matrix[not_condition] ** 2 - self.__left_matrix[not_condition] ** 2)

        reciprocal = OPNsMatrix()
        reciprocal.__left_matrix = c
        reciprocal.__right_matrix = d
        return reciprocal

    def __pow__(self, power):
        """
        Power operation
        """
        result = OPNsMatrix()
        result.__left_matrix = self.__left_matrix
        result.__right_matrix = self.__right_matrix

        while power > 1:
            result = result.multi(self)
            power = power - 1
        return result

    def __abs__(self):
        # According to the definition, if a + b > 0 or (a + b == 0 and a < 0), return (-a, -b)
        result = OPNsMatrix()
        condition = (self.__left_matrix + self.__right_matrix > 0) | (
                (self.__left_matrix + self.__right_matrix == 0) & (self.__left_matrix < 0))

        # Use numpy.where to select elements based on the condition
        result.__left_matrix = np.where(condition, -self.__left_matrix, self.__left_matrix)
        result.__right_matrix = np.where(condition, -self.__right_matrix, self.__right_matrix)

        return result

    def add(self, other: 'OPNsMatrix'):
        return self.__add__(other)

    def sub(self, other: 'OPNsMatrix'):
        return self.__sub__(other)

    def multi(self, other):
        return self.__mul__(other)

    def dot(self, other):
        """
        Matrix multiplication
        OPNs multiplication: (a, b) * (c, d) = (-ad - bc, -ac - bd)
        :param other:
        :return:
        """
        if isinstance(other, OPNsMatrix):
            if len(self.__left_matrix.shape) == 1 and len(other.__left_matrix.shape) == 1:  # Two 1D arrays: inner
                # product
                a = -np.dot(self.__right_matrix, other.__left_matrix) - np.dot(self.__left_matrix, other.__right_matrix)
                b = -np.dot(self.__left_matrix, other.__left_matrix) - np.dot(self.__right_matrix, other.__right_matrix)
                return OPNs(a, b)
            elif len(self.__left_matrix.shape) == 1 or len(other.__left_matrix.shape) == 1:  # One is 1D, one is 2D
                result = OPNsMatrix()
                result.__left_matrix = -np.dot(self.__right_matrix, other.__left_matrix) - np.dot(self.__left_matrix,
                                                                                                  other.__right_matrix)
                result.__right_matrix = -np.dot(self.__left_matrix, other.__left_matrix) - np.dot(self.__right_matrix,
                                                                                                  other.__right_matrix)
                return result
            elif other.__left_matrix.shape[0] == self.__left_matrix.shape[1] or other.__left_matrix.shape[1] == \
                    self.__left_matrix.shape[0]:  # Check matrix dimensions
                result = OPNsMatrix()
                result.__left_matrix = -np.dot(self.__right_matrix, other.__left_matrix) - np.dot(self.__left_matrix,
                                                                                                  other.__right_matrix)
                result.__right_matrix = -np.dot(self.__left_matrix, other.__left_matrix) - np.dot(self.__right_matrix,
                                                                                                  other.__right_matrix)
                return result
            else:
                raise Exception(f"Incompatible matrix dimensions for multiplication: {self.__left_matrix.shape}, "
                                f"{other.__left_matrix.shape}")
        raise Exception(f"Cannot multiply: operands are not both OPNsMatrix: {self}, {other}")

    def outer(self, other):
        result = OPNsMatrix()
        # Get dimensions of the two vectors
        dim1 = self.__left_matrix.shape[0]
        dim2 = other.__left_matrix.shape[0]

        # Create result matrices
        result_first = np.zeros((dim1, dim2))
        result_second = np.zeros((dim1, dim2))

        # Compute outer product
        for i in range(dim1):
            for j in range(dim2):
                a, b = self.__left_matrix[i], self.__right_matrix[i]
                c, d = other.__left_matrix[j], other.__right_matrix[j]
                result_first[i, j] = -a * d - b * c
                result_second[i, j] = -a * c - b * d
        result.__left_matrix = result_first
        result.__right_matrix = result_second
        return result

    def sum(self, axis=None, keepdims=False):
        if axis is None:
            return OPNs(np.sum(self.__left_matrix), np.sum(self.__right_matrix))
        result = OPNsMatrix()
        result.__left_matrix = np.sum(self.__left_matrix, axis=axis, keepdims=keepdims)
        result.__right_matrix = np.sum(self.__right_matrix, axis=axis, keepdims=keepdims)
        return result

    def flatten(self):
        """
        Flatten into a 1D array
        """
        if len(self.shape) == 1:
            return self
        else:  # Higher-dimensional
            result = OPNsMatrix()
            result.__left_matrix = self.__left_matrix.flatten()
            result.__right_matrix = self.__right_matrix.flatten()
            return result

    def reshape(self, shape):
        result = OPNsMatrix()
        result.__left_matrix = self.__left_matrix.reshape(shape)
        result.__right_matrix = self.__right_matrix.reshape(shape)
        return result

    
    def mean(self, axis=None):
        if axis is None:  # Compute the mean of all elements
            return OPNs(np.mean(self.__left_matrix), np.mean(self.__right_matrix))
        elif axis == 0:  # Compute the mean of each column
            result = OPNsMatrix()
            result.__left_matrix = np.mean(self.__left_matrix, axis=0)
            result.__right_matrix = np.mean(self.__right_matrix, axis=0)
            return result
        elif axis == 1:  # Compute the mean of each row
            result = OPNsMatrix()
            result.__left_matrix = np.mean(self.__left_matrix, axis=1)
            result.__right_matrix = np.mean(self.__right_matrix, axis=1)
            return result
        else:
            raise ValueError("Invalid axis value. Use None, 0, or 1.")

    def all(self, condition):
        return all(item for row in self.data for item in row)

    def opn_to_1_num(self):
        return - (self.__left_matrix + self.__right_matrix)

    def opn_to_2_num(self):
        # Initialize an empty list to store results
        result = []
        # Loop over each column
        if len(self.__left_matrix.shape) == 2:
            for i in range(self.__left_matrix.shape[1]):
                # Append the corresponding columns from a and b to the result
                result.append(self.__left_matrix[:, i])
                result.append(self.__right_matrix[:, i])
            return np.column_stack(result)
        else:
            for i in range(self.__left_matrix.shape[0]):
                result.append(self.__left_matrix[i])
                result.append(self.__right_matrix[i])
            return np.array(result, dtype=float)
        

    def set_matrix(self, left_matrix, right_matrix):
        """
        Set the two matrices. Since they must be numpy arrays, assignment must go through this method.
        """
        # Failsafe: convert to numpy arrays if not already
        nd_left_mat = np.array(left_matrix)  # Convert all to numpy
        nd_right_mat = np.array(right_matrix)
        if np.issubdtype(nd_left_mat.dtype, np.number) and np.issubdtype(nd_right_mat.dtype,
                                                                         np.number):  # Ensure all elements are numeric
            self.__left_matrix = np.array(left_matrix, dtype=float)
            self.__right_matrix = np.array(right_matrix, dtype=float)
        else:
            raise ValueError(f"left_matrix or right_matrix element is not number: {left_matrix}, {right_matrix}")

    def num_div_matrix(self, num):
        # Number divided by the matrix, equivalent to multiplying by the reciprocal of each element
        # Create mask for elements where a == b or a == -b, no modification needed
        mask = np.logical_or(self.__left_matrix == self.__right_matrix, self.__left_matrix == -self.__right_matrix)
        result = OPNsMatrix()
        result.__left_matrix = np.where(mask, self.__left_matrix,
                                        self.__left_matrix / (self.__left_matrix ** 2 - self.__right_matrix ** 2))
        result.__right_matrix = np.where(mask, self.__right_matrix,
                                         self.__right_matrix / (self.__right_matrix ** 2 - self.__left_matrix ** 2))
        

    def complex_square_root_euclidean(self):
        """
        Compute Euclidean distance using the complex number square root method.
        The Euclidean distance between a + bi and c + di is sqrt((a - c)^2 + (b - d)^2)
        :return:
        """
        return np.sum(np.sqrt((self.__left_matrix - self.__right_matrix) ** 2), axis=1)  # Sum columns

    def distance(self, other):
        """
        Distance used in KNN. Generalized metric for OPNs.
        :param other:
        :return: OPNsMatrix
        """
        result = abs(self - other)
        result.__left_matrix = np.sum(result.__left_matrix, axis=1)
        result.__right_matrix = np.sum(result.__right_matrix, axis=1)
        return result

    def concatenate(self, other, axis=0):
        """
        Concatenate two matrices.
        :param other: The matrix to append
        :param axis: 0 for vertical concatenation (stack rows, number of columns must match),
                 1 for horizontal concatenation (stack columns, number of rows must match)
        """
        result = OPNsMatrix()
        result.__left_matrix = np.concatenate((self.__left_matrix, other.__left_matrix), axis=axis)
        result.__right_matrix = np.concatenate((self.__right_matrix, other.__right_matrix), axis=axis)
        return result
    
