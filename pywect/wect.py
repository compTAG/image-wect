import numpy as np

from .utilities import projection_function, get_weighted_filtration

def get_multidirectional_wects(image_arr, num_steps=50, num_directions=8, threshold_vals=None, weight_function="MAX", ec_only=False):
    """
    Calculate the Weighted Euler Curve Transform
    for a given number of directions and threshold values.

    :param image_arr: image array representing the weight function
    :param num_steps: number of steps at which to compute the filtration
    :param num_directions: number of directions in which to compute the filtration
    :param weight_function: the weight function extension to use when computing filtration
    :param threshold_vals: threshold values at which the Weighted Euler Characteristic is computed
    :param ec_only: flag that only computes the Euler Curve Transform without weights
    :type image_arr: numpy.ndarray of dimension 2
    :type threshold_vals: numpy.ndarray with shape (num_steps,1)
    :type ec_only: boolean
    :return: returns (Weighted) Euler Curves for given input in each direction
    :rtype: numpy.ndarray with shape (num_directions, num_steps)

    """
    
    sample_theta = np.expand_dims(np.linspace(-np.pi, np.pi, num=num_directions), axis=1)
    directions = np.concatenate((np.cos(sample_theta), np.sin(sample_theta)), axis=1)
    if threshold_vals is None:
        threshold_vals = np.linspace(-1.0, 1.0, num=num_steps)
    WECTs = np.zeros((directions.shape[0], num_steps))

    for i,direction in enumerate(directions):
        WECTs[i,:] = get_unidirectional_wect(image_arr, direction, threshold_vals, weight_function=weight_function, ec_only=ec_only)

    return WECTs

def get_unidirectional_wect(image_arr, direction, threshold_vals, weight_function="MAX", ec_only=False):
    """
    Calculate the Weighted Euler Curve Transform
    for the given direction and threshold values.

    :param image_arr: image array representing the weight function
    :param direction: direction to construct the height filtration
    :param threshold_vals: threshold values at which the Weighted Euler Characteristic is computed
    :param ec_only: flag that only computes the Euler Curve Transform without weights
    :type direction: numpy.ndarray with shape (2,)
    :type image_arr: numpy.ndarray of dimension 2
    :type threshold_vals: numpy.ndarray with shape (num_steps,1)
    :type ec_only: boolean
    :return: returns (Weighted) Euler Curve for given input
    :rtype: numpy.ndarray with shape (num_steps,1)

    """
    height_filtration = projection_function(direction, image_arr.shape)
    if ec_only:
        image_arr_copy = image_arr.copy()
        image_arr_copy[image_arr_copy != 0] = 1
        weighted_filtration = get_weighted_filtration(height_filtration, image_arr_copy, weight_function)
    else:
        weighted_filtration = get_weighted_filtration(height_filtration, image_arr, weight_function, zero_flag=True)
    assert(height_filtration.__len__() == weighted_filtration.__len__())

    change_points = []
    for simplex in weighted_filtration:
        if len(change_points) == 0:
            change_points.append([simplex.data, simplex.weight])
            continue

        height = simplex.data
        weight = simplex.weight
        if height == change_points[-1][0]:
            change_points[-1][1] += (-1)**(simplex.dimension())*weight
        else:
            change_points.append([height, weight])

    ind = 0
    chi_vals = []
    running_chi = 0
    for val in threshold_vals:
        while((ind != len(change_points)) and (change_points[ind][0] <= val)):
            running_chi += change_points[ind][1]
            ind += 1
        chi_vals.append(running_chi)

    return np.asarray(chi_vals)
