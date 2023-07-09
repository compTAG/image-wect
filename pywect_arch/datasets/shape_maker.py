import itertools
import random
import numpy as np
import scipy.stats as stats

def make_square(grid_size = 64, distribution = "uniform", center=(32, 32), side_length=35, mu=None, sigma=None):
    points_in_shape = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if (abs(x-center[0]) <= side_length//2) and (abs(y-center[1]) <= side_length//2)]
    return __make_image(points_in_shape, distribution, grid_size, mu=mu, sigma=sigma)
    
def make_circle(grid_size = 64, distribution = "uniform", center = (32, 32), radius = 20, mu=None, sigma=None):
    points_in_shape = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if ((x-center[0])**2 + (y-center[1])**2 <= radius**2)]
    return __make_image(points_in_shape, distribution, grid_size, mu=mu, sigma=sigma)

def make_favorite_tetris(distribution = "uniform", mu=None, sigma=None):
    pts_ls = np.load("pywect/datasets/favorite_tetris.npy").tolist()
    points_in_shape = [tuple(point) for point in pts_ls]
    return __make_image(points_in_shape, distribution, 64, mu=mu, sigma=sigma)

def make_random_tetris(grid_size = 64, distribution = "uniform", center = (32, 32), side_length = 35, mu=None, sigma=None):
    num_subsquares = 8
    sub_square_side = side_length // num_subsquares * 3

    center_x, center_y = center

    centers = [(center_x, center_y)]
    points_in_shape = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if (abs(x-center_x) <= sub_square_side//2) and (abs(y-center_y) <= sub_square_side//2)]
    
    pad_factor = 0.05
    pad = (grid_size*pad_factor, grid_size - (grid_size*pad_factor))

    while len(centers) < num_subsquares:
        new_center_choice = random.uniform(0,1)
        basis = random.choice(centers)
        center = None
        if new_center_choice < 0.25:            # left
            center = __check_candidate_pt((basis[0] - sub_square_side, basis[1]), points_in_shape, pad)
        elif new_center_choice >= 0.25 and new_center_choice < 0.5:            # right
            center = __check_candidate_pt((basis[0] + sub_square_side, basis[1]), points_in_shape, pad)
        elif new_center_choice >= 0.5 and new_center_choice < 0.75:            # up
            center = __check_candidate_pt((basis[0], basis[1] + sub_square_side), points_in_shape, pad)
        else:            # down
            center = __check_candidate_pt((basis[0], basis[1] - sub_square_side), points_in_shape, pad)
            
        if center is None:
            pass
        else:
            centers.append(center)
            new_pts_in_shape = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if (abs(x-center[0]) <= sub_square_side//2) and (abs(y-center[1]) <= sub_square_side//2)]
            for item in new_pts_in_shape:
                points_in_shape.append(item)

    return __make_image(points_in_shape, distribution, grid_size, mu=mu, sigma=sigma)

def __check_candidate_pt(cand_pt, points_in_shape, pad):
    if cand_pt in points_in_shape:
        return None
    elif cand_pt[0] < pad[0] or cand_pt[0] > pad[1]:
        return None
    elif cand_pt[1] < pad[0] or cand_pt[1] > pad[1]:
        return None
    else:
        return cand_pt
    

def __recenter(points_in_shape, center):
    x_tot, y_tot = 0, 0
    for item in points_in_shape:
        x_tot += item[0]
        y_tot += item[1]
    
    x_avg = x_tot // len(points_in_shape)
    y_avg = y_tot // len(points_in_shape)

    centered = []
    for item in points_in_shape:
        centered.append((item[0] + (x_avg - center[0]), item[1] + (y_avg - center[1])))
    
    return centered

    

def make_clusters(grid_size = 64, distribution = "uniform", center = (32, 32), side_length = 35, mu=None, sigma=None):
    interval = grid_size // 4
    sub_square_side = side_length // 5 * 2
    centers = [center, (interval, interval), (interval, 3*interval), (3*interval, interval), (3*interval, 3*interval)]

    points_in_shape = []
    for center in centers:
        new_points_in_shape = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if (abs(x-center[0]) <= sub_square_side//2) and (abs(y-center[1]) <= sub_square_side//2)]
        for item in new_points_in_shape:
            points_in_shape.append(item)

    return __make_image(points_in_shape, distribution, grid_size, mu=mu, sigma=sigma)


def make_swiss_cheese(grid_size = 64, distribution = "uniform", center = (32, 32), side_length = 35, mu=None, sigma=None):
    interval = grid_size // 16
    sub_square_side = side_length // 5 * 2 + 3
    fac1, fac2 = 3, 13
    centers = [center, 
               (fac1*interval, center[1]), 
               (fac2*interval, center[1]), 
               (center[0], fac1*interval),
               (center[0], fac2*interval),
               (fac1*interval, fac1*interval), 
               (fac1*interval, fac2*interval), 
               (fac2*interval, fac1*interval), 
               (fac2*interval, fac2*interval)]

    points_in_shape = []
    for center in centers:
        new_points_in_shape = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if (abs(x-center[0]) <= sub_square_side//2) and (abs(y-center[1]) <= sub_square_side//2)]
        for item in new_points_in_shape:
            points_in_shape.append(item)

    return __make_image(points_in_shape, distribution, grid_size, mu=mu, sigma=sigma)


def make_annulus(grid_size = 64, distribution = "uniform", center = (32, 32), outer_radius = 22, inner_radius = 10, mu=None, sigma=None):
    assert(outer_radius > inner_radius)
    points_in_shape = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if (((x-center[0])**2 + (y-center[1])**2 >= inner_radius**2) and ((x-center[0])**2 + (y-center[1])**2 <= outer_radius**2) )]
    return __make_image(points_in_shape, distribution, grid_size, mu=mu, sigma=sigma)


def __make_image(points_in_shape, distribution, grid_size, inverted=False, mu=0.5, sigma=0.17):
    mask = __make_mask(points_in_shape, grid_size, inverted)
    if distribution in ["uniform"]:
        if mu is not None or sigma is not None:
            raise(RuntimeWarning("creating uniform image, but mu or sigma parameters were set"))
        return __make_uniform_image(mask)
    elif distribution in ["normal"]:
        return __make_gaussian_image(mask, mu, sigma)
    else:
        raise NotImplementedError(f"Unknown distribution {distribution}")


def __make_mask(points_in_shape, grid_size, inverted = False):
    if inverted:
        mask = np.ones((grid_size, grid_size))
        v = 0
    else:
        mask = np.zeros((grid_size, grid_size))
        v = 1
    for point in points_in_shape:
        mask[point] = v
    return mask


def __make_uniform_image(mask):
    channel = np.random.uniform(0, 1, (mask.shape[0], mask.shape[0]))
    return np.multiply(mask, channel)

def __make_gaussian_image(mask, _mu, _sigma):
    lower, upper = 0, 1
    mu, sigma = _mu, _sigma
    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    vals = X.rvs(mask.shape[0]*mask.shape[0])
    channel = np.reshape(vals, (mask.shape[0],mask.shape[0]))
    
    return np.multiply(mask, channel)