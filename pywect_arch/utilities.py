import numpy as np
from tqdm import tqdm

import dionysus as d
from dionysus import Simplex

def print_filtration(filtration):
    for simplex in filtration:
        try:
            print(simplex, simplex.weight)
        except:
            print(simplex)

def get_array_index(vertex, mod_val):
    """
    Returns the array index corresponding to vertex number assigned by Dionysus.

    :param vertex: vertex number assigned by Dionysus
    :param mod_val: value used to compute the array indices (usually number of columns in array)
    :type vertex: integer
    :type mod_val: integer
    :return: returns (floor(vertex/mod_val), vertex%mod_val)
    :rtype: 2-tuple

    """
    return (vertex//mod_val, vertex%mod_val)

def projection_function(direction, arr_shape):
    """
    Construct the height filtration for a given direction

    :param dir: direction in which to construct the filtration
    :param arr_shape: shape of 2D grid
    :type dir: numpy.ndarray with shape (2,)
    :type arr_shape: 2-tuple
    :return: height filtration along direction 'dir' using freudenthal triangulation
    :rtype: dionysus.Filtration()

    """
    new_arr = np.zeros(arr_shape)

    i_bar = np.mean(np.arange(arr_shape[0]))
    j_bar = np.mean(np.arange(arr_shape[1]))

    r = 0.0
    for i in range(new_arr.shape[0]):
        for j in range(new_arr.shape[1]):
            z_i = i - i_bar
            z_j = j - j_bar
            z_mag = np.sqrt(z_i**2 + z_j**2)
            if z_mag > r:
                r = z_mag
    for i in range(arr_shape[0]):
        for j in range(arr_shape[1]):
            z_i = i - i_bar
            z_j = j - j_bar
            new_arr[i,j] = (z_i/r)*direction[0] + (z_j/r)*direction[1]

    return d.fill_freudenthal(new_arr.astype('f4'))

def get_simplex_weight(weight_arr, simplex, weight_function, zero_flag=True):
    verts = []
    weight = -np.Inf

    if weight_function == "MAX":
        weight_decided = False
        for vertex in simplex.__iter__():
            verts.append(vertex)
            if (not zero_flag) or (not weight_decided):
                (i,j) = get_array_index(vertex, weight_arr.shape[1])

            if zero_flag:
                if weight_arr[i,j] == 0:
                    weight = 0
                    weight_decided = True
            if (not weight_decided) and (weight < weight_arr[i,j]):
                weight = weight_arr[i,j]

    elif weight_function == "MEAN":
        weight_sum = 0.0
        for vertex in simplex.__iter__():
            verts.append(vertex)
            (i,j) = get_array_index(vertex, weight_arr.shape[1])
            weight_sum += weight_arr[i,j]
        weight = weight_sum / simplex.__len__()

    return verts, weight

def get_weighted_filtration(filtration, weight_arr, weight_function, zero_flag=True):
    """
    Get filtration with both height and weight function values.

    :param filtration: previously computed height filtration
    :param weighted_arr: array representing the weight function
    :type filtration: dionysus.Filtration()
    :type weight_arr: numpy.ndarray
    :return: filtration with both height and weight function values.
    :rtype: list

    """
    weighted_filtration = []
    for simplex in filtration:
        verts,weight = get_simplex_weight(weight_arr, simplex, weight_function, zero_flag)
        weighted_simplex = WeightedSimplex(verts, simplex.data, weight)
        weighted_filtration.append(weighted_simplex)

    return weighted_filtration


class WeightedSimplex(Simplex):
    """
    Custom class to store Height and Weight 
    function value with Simplex information.

    """
    def __init__(self, verts, value, weight):
        Simplex.__init__(self, verts, value)
        self.weight = weight

