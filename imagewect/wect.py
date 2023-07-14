import numpy as np

def compute_wect(img_matrix, filtration, fe="MAX", verbose=False):
    """
    Computes the weighted Euler characteristic transform (WECT) of an image.

    Args:
        img_matrix (np.ndarray): The image matrix.
        filtration (list): The height filtration to use for computing the WECT.
        fe (str, optional): The function extension to use for computing weights. Defaults to "MAX".
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        list: The computed WECT.
    """
    weights = np.concatenate(img_matrix)
    wect = []
    for s in filtration:
        height = s.data
        weight = __get_simplex_weight(s, weights, fe)
        
        if len(wect) == 0:
            wect.append([height, weight])
            if verbose:
                print(f"new change point, height = {height}\n", s, "curr =", weight)
            continue

        if height == wect[-1][0]:
            wect[-1][1] += (-1)**(s.dimension())*weight
            if verbose:
                print(s, "weight +=", (-1)**(s.dimension())*weight, "now", wect[-1][1])
        else:
            wect.append([height, wect[-1][1]+weight])
            if verbose:
                print(f"new change point, height = {height}\n", s, "curr =", weight)
    
    return wect

def __contains_zero_weights(s, weights):
    """
    Checks if a simplex contains any zero-weighted vertices.

    Args:
        s (dionysus.Simplex): The simplex.
        weights (np.ndarray): The weights array.

    Returns:
        bool: True if the simplex contains zero-weighted vertices, False otherwise.
    """
    for i in range(0,s.dimension()+1):
        if weights[s[i]] == 0:
            return True
    return False

def __get_maxfe_weight(s, weights):
    """
    Computes the weight of a simplex using the MAX function extension.

    Args:
        s (dionysus.Simplex): The simplex.
        weights (np.ndarray): The weights array.

    Returns:
        float: The weight of the simplex.
    """
    if __contains_zero_weights(s, weights):
        return 0

    d = s.dimension()
    if d == 0:
        return weights[s[0]]
    if d == 1:
        return max([weights[s[0]], weights[s[1]]])
    return max([weights[s[0]], weights[s[1]], weights[s[2]]])
    
def __get_simplex_weight(s, weights, fe):
    """
    Computes the weight of a simplex based on the function extension specified.

    Args:
        s (dionysus.Simplex): The simplex.
        weights (np.ndarray): The weights array.
        fe (str): The function extension to use for computing weights.

    Returns:
        float: The weight of the simplex.
    """
    if fe == "MAX":
        return __get_maxfe_weight(s, weights)
    elif fe == "MIN":
        return __get_minfe_weight(s, weights)
    elif fe == "AVG":
        return __get_avgfe_weight(s, weights)
    elif fe == "EC":
        return 1
    else:
        raise NotImplementedError(f"unknown function extension \"{fe}\"")
    

def __get_minfe_weight(s, weights):
    """
    Computes the weight of a simplex using the MIN function extension.

    Args:
        s (dionysus.Simplex): The simplex.
        weights (np.ndarray): The weights array.

    Returns:
        float: The weight of the simplex.
    """
    if __contains_zero_weights(s, weights):
        return 0
    
    d = s.dimension()
    if d == 0:
        return weights[s[0]]
    elif d == 1:
        return min([weights[s[0]], weights[s[1]]])
    else:
        return min([weights[s[0]], weights[s[1]], weights[s[2]]])


def __get_avgfe_weight(s, weights):
    """
    Computes the weight of a simplex using the AVG function extension.

    Args:
        s (dionysus.Simplex): The simplex.
        weights (np.ndarray): The weights array.

    Returns:
        float: The weight of the simplex.
    """
    if __contains_zero_weights(s, weights):
        return 0
    
    d = s.dimension()
    if d == 0:
        return weights[s[0]]
    elif d == 1:
        return sum([weights[s[0]], weights[s[1]]]) / 2
    else:
        return sum([weights[s[0]], weights[s[1]], weights[s[2]]]) / 3
    

def vectorize_wect(wect, height_vals):
    """
    Vectorizes the WECT by computing its values at specified height values.

    Args:
        wect (list): The WECT.
        height_vals (list): The height values at which to compute the WECT.

    Returns:
        np.ndarray: The vectorized WECT.
    """
    ind = 0
    chi_vals = []
    running_chi = 0
    for val in height_vals:
        while((ind != len(wect)) and (wect[ind][0] <= val)):
            running_chi = wect[ind][1]
            ind += 1
        chi_vals.append(running_chi)

    return np.asarray(chi_vals)


def distance_between_wects_multidirectional(wects1, wects2, a, b):
    """
    Computes the distance between two sets of nonvectorized multidirectional WECTs.

    Args:
        wects1 (list): The first set of WECTs.
        wects2 (list): The second set of WECTs.
        a (float): The starting height value.
        b (float): The ending height value.

    Returns:
        float: The multidirectional distance between the two sets of WECTs.
    """
    assert(len(wects1) == len(wects2))
    diffs = []
    for d in range(0, len(wects1)):
        diffs.append(distance_between_wects_unidirectional(wects1[d], wects2[d], a, b))
    
    return max(diffs)


def distance_between_wects_unidirectional(wect1, wect2, a, b):
    """
    Computes the distance between two nonvectorized unidirectional WECTs (WECF).

    Args:
        wect1 (list): The first WECT.
        wect2 (list): The second WECT.
        a (float): The starting height value.
        b (float): The ending height value.

    Returns:
        float: The unidirectional distance between the two WECTs.
    """
    x = a
    curr_wect1, curr_wect2 = 0, 0
    diff = 0

    while (x <= b):
        if len(wect1) == 0:
            w1_next_x = b+1
        else:
            w1_next_val = wect1[0][1]
            w1_next_x = wect1[0][0]
        if len(wect2) == 0:
            w2_next_x = b+2
        else:
            w2_next_val = wect2[0][1]
            w2_next_x = wect2[0][0]


        if w1_next_x == w2_next_x:
            new_x = w1_next_x
        else:
            if w1_next_x < w2_next_x:
                new_x = w1_next_x
            else:
                new_x = w2_next_x

        diff += (new_x - x) * abs(curr_wect1 - curr_wect2)
        x = new_x

        if w1_next_x == w2_next_x:
            curr_wect1 = w1_next_val
            wect1 = wect1[1:]
            curr_wect2 = w2_next_val
            wect2 = wect2[1:]
        else:
            if w1_next_x < w2_next_x:
                curr_wect1 = w1_next_val
                wect1 = wect1[1:]
            else:
                curr_wect2 = w2_next_val
                wect2 = wect2[1:]
    
    return diff
