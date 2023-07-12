import itertools

import numpy as np
import dionysus as d
import scipy.stats as stats
import matplotlib.pyplot as plt

from .wect import compute_wect, discretize_wect

class Shape:
    def __init__(self, grid_size, mask) -> None:
        self.grid_size = grid_size
        self.location_matrix = self.__compute_location_matrix()
        self.mask = mask


    def __compute_location_matrix(self) -> np.ndarray:
        mid = self.grid_size//2
        vals = np.linspace(-1*mid, mid, self.grid_size)
        location_matrix = np.zeros((self.grid_size, self.grid_size, 2))

        for i in range(0, self.grid_size):
            for j in range(0, self.grid_size):
                location_matrix[j][i][0] = vals[i]
                location_matrix[j][i][1] = np.flip(vals)[j]

        return location_matrix
    
    def get_heights(self, direction):
        assert(len(direction) == 2)
        heights = np.zeros((self.grid_size, self.grid_size))
        
        for i in range(0, self.grid_size):
            for j in range(0, self.grid_size):
                heights[i][j] = np.dot(self.location_matrix[i][j], direction)

        return heights
    
    def get_height_filtration(self, direction):
        height_filtration = d.fill_freudenthal(self.get_heights(direction))
        
        return height_filtration



class OnePixel(Shape):
    def __init__(self, grid_size) -> None:
        self.mask = np.zeros((grid_size, grid_size))
        self.mask[0][0] = 1
        super().__init__(grid_size=grid_size, mask=self.mask)


class Channel:
    def __init__(self, channel) -> None:
        self.channel = channel

class UniformDistribution:
    def __init__(self, grid_size, a=0, b=1) -> None:
        self.grid_size = grid_size
        self.a, self.b = a, b

    def make_channel(self):
        return np.random.uniform(self.a, self.b, (self.grid_size, self.grid_size))

class NormalDistribution:
    def __init__(self, grid_size, mu=0.5, sigma=0.17, a=0, b=1) -> None:
        self.grid_size = grid_size
        self.mu, self.sigma = mu, sigma
        self.a, self.b = a, b

    def make_channel(self):
        X = stats.truncnorm(
            (self.a - self.mu) / self.sigma, (self.b - self.mu) / self.sigma, loc=self.mu, scale=self.sigma)
        vals = X.rvs(self.grid_size*self.grid_size)
        return np.reshape(vals, (self.grid_size,self.grid_size))

class Image:
    def __init__(self, shape, channel) -> None:
        self.shape = shape
        self.channel = channel
        self.img = np.multiply(self.shape.mask, channel)
        self.wect = None
        self.discretized_wect = None

    def show(self, colorbar=True, axes=True, savePath=None):
        plt.imshow(self.img, vmin=0, vmax=1, extent=[-self.shape.grid_size/2., self.shape.grid_size/2., -self.shape.grid_size/2., self.shape.grid_size/2. ])
        if colorbar:
            plt.colorbar()
        if not axes:
            plt.axis("off")
        if savePath is not None:
            plt.savefig(f"square_annulus.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def get_wect(self, hf):
        if self.wect is None:
            return compute_wect(self.img, hf)
        return self.wect
    
    def get_discretized_wect(self, hf, height_vals):
        if self.discretized_wect is None:
            if self.wect is None:
                self.wect = compute_wect(self.img, hf)
            return discretize_wect(self.wect, height_vals)
        return self.discretized_wect


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

def make_square(grid_size=65, center=(32,32), side_length=35):
    pts = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if (abs(x-center[0]) <= side_length//2) and (abs(y-center[1]) <= side_length//2)]
    mask = __make_mask(pts, grid_size)
    return Shape(grid_size, mask)

def make_circle(grid_size = 65, center = (32, 32), radius = 20):
    pts = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if ((x-center[0])**2 + (y-center[1])**2 <= radius**2)]
    mask = __make_mask(pts, grid_size)
    return Shape(grid_size, mask)

def make_favorite_tetris(distribution = "uniform"):
    pts_ls = np.load("imagewect/favorite_tetris.npy").tolist()
    pts = [tuple(point) for point in pts_ls]
    mask = __make_mask(pts, 65)
    return Shape(65, mask)

def make_annulus(grid_size = 65, center = (32, 32), outer_radius = 22, inner_radius = 10, mu=None, sigma=None):
    assert(outer_radius > inner_radius)
    pts = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if (((x-center[0])**2 + (y-center[1])**2 >= inner_radius**2) and ((x-center[0])**2 + (y-center[1])**2 <= outer_radius**2) )]
    mask = __make_mask(pts, grid_size)
    return Shape(65, mask)


def _linf(p1, p2):
    return max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))

def make_square_annulus(grid_size = 65, center = (32, 32), outer_side_length = 38, inner_side_length = 16):
    assert(outer_side_length > inner_side_length)
    # pts = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if ( ((abs(x-center[0]) <= outer_side_length//2) and (abs(x-center[0]) >= inner_side_length//2)) and ((abs(y-center[1]) <= outer_side_length//2) and (abs(y-center[1]) >= inner_side_length//2)) )]
    pts = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if \
             _linf((x,y), center) <= outer_side_length//2 and _linf((x,y), center) >= inner_side_length//2]
    
    mask = __make_mask(pts, grid_size)
    return Shape(65, mask)

def make_clusters(grid_size = 65, center = (32, 32), side_length = 35):
    interval = grid_size // 4
    sub_square_side = side_length // 5 * 2
    centers = [center, (interval, interval), (interval, 3*interval), (3*interval, interval), (3*interval, 3*interval)]

    pts = []
    for center in centers:
        new_points_in_shape = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if (abs(x-center[0]) <= sub_square_side//2) and (abs(y-center[1]) <= sub_square_side//2)]
        for item in new_points_in_shape:
            pts.append(item)

    mask = __make_mask(pts, grid_size)
    return Shape(65, mask)

def make_swiss_cheese(grid_size = 65, center = (32, 32), side_length = 35):
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

    pts = []
    for center in centers:
        new_points_in_shape = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if (abs(x-center[0]) <= sub_square_side//2) and (abs(y-center[1]) <= sub_square_side//2)]
        for item in new_points_in_shape:
            pts.append(item)

    mask = __make_mask(pts, grid_size, inverted=True)
    return Shape(65, mask)