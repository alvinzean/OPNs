import random
import itertools
import numpy as np

import opns_pack.opns_np as op
import opns_pack.opns_math as omath
from opns_pack.opns import OPNs


def random_pair(feature):  # Random pairing
    feature_copy = feature.copy()
    if len(feature_copy) % 2 != 0:
        feature_copy.pop()
    random.shuffle(feature_copy)
    return feature_copy


def all_pair(feature):
    """
    All pairwise combinations of a set of features
    :param feature: A set of features
    :return: A 1D list containing all pairings without repetition
    """
    pairings = list(itertools.combinations(feature, 2))
    new_arr = [item for pairing in pairings for item in pairing]
    return new_arr


def enumerate_pair(feature):
    """
    Enumerate all pairing combinations of a feature set
    :param feature: A set of features
    :return: A 2D list of all pairing permutations
    """

    def remove_elements(a, n, m):
        b = a[:n] + a[n + 1:m] + a[m + 1:]
        return b

    def comb(new_comb, tocomb_list):
        if len(tocomb_list) == 2:
            new_comb.append(tocomb_list[0])
            new_comb.append(tocomb_list[1])
            comb_list.append(new_comb)
            return comb_list
        for i in range(1, len(tocomb_list)):
            new_comb.append(tocomb_list[0])
            new_comb.append(tocomb_list[i])
            comb(new_comb, remove_elements(tocomb_list, 0, i))
            new_comb = new_comb[:-len(tocomb_list)]

    # Convert input to a list to ensure consistent handling
    feature_list = list(feature)

    if len(feature_list) % 2 != 0:
        feature_list.pop()
    comb_list = []
    comb([], feature_list)
    return np.array(comb_list)


def linear_pair(feature, position='odd'):  # Each feature is paired with zero; ensure a column named 'zero' exists in
    # the DataFrame before using
    """
    :param feature: An ordered list of features to be paired, e.g., [1, 2, 3, 4]
    :param position: Position to insert zero; 'odd' means insert at odd indices, 'even' at even indices
    :return: A list of paired features like [(1, 0), (2, 0)...] or [(0, 1), (0, 2)...]
    """
    new_arr = []
    if position == 'odd':  # Insert at odd indices
        for i in range(len(feature)):
            new_arr.append(feature[i])
            new_arr.append('zero')
    elif position == 'even':  # Insert at even indices
        for i in range(len(feature)):
            new_arr.append('zero')
            new_arr.append(feature[i])
    else:
        raise ValueError("Position must be 'odd' or 'even'")
    return new_arr


def remove_pairs(original_pairs, pairs_to_remove):
    """
    Remove specified pairs from the original pairing list
    :param original_pairs: Original list of pairs
    :param pairs_to_remove: List of pairs to be removed
    :return: Resulting list after removal
    """
    # original_pairs, pairs_to_remove = list(original_pairs), list(pairs_to_remove)
    pairs = list(zip(pairs_to_remove[::2], pairs_to_remove[1::2]))
    result = []
    for i in range(0, len(original_pairs), 2):
        pair = (original_pairs[i], original_pairs[i+1])
        if pair not in pairs:
            result.extend(pair)
    return result


def data_convert(X, ori_feature, poly=1, tri=0, poly_only=False, linear_term=True, bias=False):
    """
    Generate an OPNs feature matrix and label vector based on a specified pairing scheme
    :param poly_only: Whether to include only polynomial terms; if True, do not set linear_term
    :param X: Dataset containing only feature columns (as a DataFrame)
    :param ori_feature: A list of features to pair, ordered like [1, 2, 3, 4] → pairs to [(1, 2), (3, 4)]
    :param poly: Power coefficient, e.g., 3 means include 1st, 2nd, and 3rd power terms
    :param tri: Trigonometric function selection: 0 = none, 1 = sin, 2 = cos, 3 = tan
    :param linear_term: Whether to include linear (1st power) terms
    :param bias: Whether to add a bias term (0, -1)
    :return: OPNsMatrix
    """

    def flatten_specific_list(input_data):
        """
        Flatten list of tuples [(1,2), (3,4)] into [1,2,3,4] if format matches; otherwise return original input
        Conditions:
          1. Input must be a list
          2. Each element must be a tuple
          3. Each tuple must contain exactly two integers
        """
        # Check whether the input is a list or tuple
        if not isinstance(input_data, (list, tuple)):
            return input_data

        # Iterate over each element in the list
        for item in input_data:
            # Check if the element is a tuple (or list) and has a length of 2
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                return input_data
            # Check if each element in the tuple is not itself a tuple
            for element in item:
                if isinstance(element, tuple):
                    return input_data

        # If all conditions are met, flatten the list of pairs
        return [num for pair in input_data for num in pair]
    feature = flatten_specific_list(ori_feature)
    data_set = X.copy()
    data_set = data_set[feature]
    dx_model = [[OPNs(0, 0) for x in range(int(data_set.shape[1] / 2))] for x in range(data_set.shape[0])]
    x_row = int(data_set.shape[1] / 2)
    for i in range(x_row):
        for j in range(data_set.shape[0]):
            dx_model[j][i] = OPNs(data_set.iloc[j, i * 2], data_set.iloc[j, i * 2 + 1])

    for exp in range(poly, 1, -1):  # Add higher-order polynomial terms
        if poly_only and exp != poly:
            pass
        else:
            data_set_sq = X.copy()
            comb_list_sq = feature.copy()
            data_set_sq = data_set_sq[comb_list_sq]
            dx_suq = [[OPNs(0, 0) for x in range(int(data_set_sq.shape[1] / 2))] for x in range(data_set_sq.shape[0])]
            for i in range(int(data_set_sq.shape[1] / 2)):
                for j in range(data_set_sq.shape[0]):
                    dx_suq[j][i] = OPNs(data_set_sq.iloc[j, i * 2], data_set_sq.iloc[j, i * 2 + 1]) ** exp
            if poly_only:
                dx_model = dx_suq
            else:
                dx_model = [a + b for a, b in zip(dx_model, dx_suq)]

    if tri != 0:  # Add sin, cos, or tan terms
        data_set_tri = X.copy()
        comb_list_tri = feature.copy()
        data_set_tri = data_set_tri[comb_list_tri]
        dx_tri = [[OPNs(0, 0) for x in range(int(data_set_tri.shape[1] / 2))] for x in range(data_set_tri.shape[0])]
        for i in range(int(data_set_tri.shape[1] / 2)):
            for j in range(data_set_tri.shape[0]):
                if tri == 1:
                    dx_tri[j][i] = omath.sin(OPNs(data_set_tri.iloc[j, i * 2], data_set_tri.iloc[j, i * 2 + 1]))
                elif tri == 2:
                    dx_tri[j][i] = omath.cos(OPNs(data_set_tri.iloc[j, i * 2], data_set_tri.iloc[j, i * 2 + 1]))
                elif tri == 3:
                    dx_tri[j][i] = omath.tan(OPNs(data_set_tri.iloc[j, i * 2], data_set_tri.iloc[j, i * 2 + 1]))
        dx_model = [a + b for a, b in zip(dx_model, dx_tri)]

    if bias:
        for row in dx_model:
            row.insert(len(dx_model[0]), OPNs(0, -1))

    if not linear_term:
        dx_model = [row[x_row:] for row in dx_model]
    return op.array(dx_model)


if __name__ == '__main__':
    import pandas as pd

    data = {
        'Feature1': [1, 2, 3, 4, 5],
        'Feature2': [10, 20, 30, 40, 50],
        'Feature3': [100, 200, 300, 400, 500],
        'Feature4': [1000, 2000, 3000, 4000, 5000],
        'Feature5': [10000, 20000, 30000, 40000, 50000]
    }
    df = pd.DataFrame(data)
    print(df)
    Feature = df.columns.tolist()
    print(Feature)
    print(all_pair(Feature))
    print(remove_pairs(all_pair(Feature), ['Feature1', 'Feature2', 'Feature3', 'Feature4']))
    X = data_convert(df, Feature, tri=1, linear_term=False)
    print(X)
    print(omath.sin(OPNs(100, 1000)))

