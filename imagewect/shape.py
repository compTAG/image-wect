import itertools
import random

import numpy as np
import dionysus as d
import scipy.stats as stats
import matplotlib.pyplot as plt

from .wect import compute_wect, vectorize_wect

class Shape:
    """
    Represents a binary mask of a shape in a grid.

    Attributes:
        grid_size (int): The size of the grid.
        location_matrix (np.ndarray): The location matrix of the shape.
        mask (np.ndarray): The mask of the shape.

    Methods:
        __compute_location_matrix(): Computes the location matrix of the shape.
        get_heights(direction): Calculates the heights of the shape in the given direction.
        get_height_filtration(direction): Generates the height filtration of the shape in the given direction.
    """
    def __init__(self, grid_size, mask) -> None:
        """
        Initializes a Shape instance.

        Args:
            grid_size (int): The size of the grid.
            mask (np.ndarray): The mask of the shape.
        """
        self.grid_size = grid_size
        self.location_matrix = self.__compute_location_matrix()
        self.mask = mask


    def __compute_location_matrix(self) -> np.ndarray:
        """
        Computes the location matrix (x,y coordinates of pixels) of the shape.

        Returns:
            np.ndarray: The location matrix of the shape.
        """
        mid = self.grid_size//2
        vals = np.linspace(-1*mid, mid, self.grid_size)
        location_matrix = np.zeros((self.grid_size, self.grid_size, 2))

        for i in range(0, self.grid_size):
            for j in range(0, self.grid_size):
                location_matrix[j][i][0] = vals[i]
                location_matrix[j][i][1] = np.flip(vals)[j]

        return location_matrix
    
    def get_heights(self, direction):
        """
        Calculates the heights of the shape in the given direction.

        Args:
            direction (np.ndarray): A 1d array of two elements representing the direction vector.

        Returns:
            np.ndarray: The heights of the shape.
        """
        assert(len(direction) == 2)
        heights = np.zeros((self.grid_size, self.grid_size))
        
        for i in range(0, self.grid_size):
            for j in range(0, self.grid_size):
                heights[i][j] = np.dot(self.location_matrix[i][j], direction)

        return heights
    
    def get_height_filtration(self, direction):
        """
        Generates the height filtration of the shape in the given direction.

        Args:
            direction (np.ndarray): A 1d array of two elements representing the direction vector.

        Returns:
            dionysus.Filtration: The height filtration of the shape.
        """
        height_filtration = d.fill_freudenthal(self.get_heights(direction))
        
        return height_filtration


def make_shape_from_image(img):
    """
        Creates a shape object from a grid image.

        Args:
            img (np.ndarray): The image matrix

        Returns:
            Shape: The shape underlying the image
        """
    mask = np.zeros((img.shape[0], img.shape[0]))
    mask[img != 0] = 1
    return Shape(mask.shape[0], mask)


class UniformDistribution:
    """
    Represents a uniform distribution of pixel intesnities over a grid.

    Attributes:
        grid_size (int): The size of the grid.
        a (float): The lower bound of the distribution.
        b (float): The upper bound of the distribution.

    Methods:
        __init__(grid_size, a=0, b=1): Initializes a UniformDistribution instance.
        make_channel(): Generates a channel with values drawn from the uniform distribution.
    """
    def __init__(self, grid_size, a=0, b=1) -> None:
        """
        Initializes a UniformDistribution instance.

        Args:
            grid_size (int): The size of the grid.
            a (float, optional): The lower bound of the distribution. Defaults to 0.
            b (float, optional): The upper bound of the distribution. Defaults to 1.
        """
        self.grid_size = grid_size
        self.a, self.b = a, b

    def make_channel(self):
        """
        Generates a grid channel of pixel intensities with values drawn from the uniform distribution.

        Returns:
            np.ndarray: The channel with values drawn from the uniform distribution.
        """
        channel = np.zeros((self.grid_size, self.grid_size))
        for i in range(0, channel.shape[0]):
            for j in range(0, channel.shape[1]):
                rn = random.uniform(0,1) 
                channel[i][j] = rn + (1 - rn) * 1e-10 
        return channel

class NormalDistribution:
    """
    Represents a normal distribution of pixel intensities over a grid.

    Attributes:
        grid_size (int): The size of the grid.
        mu (float): The mean of the distribution.
        sigma (float): The standard deviation of the distribution.
        a (float): The lower bound of the truncation.
        b (float): The upper bound of the truncation.

    Methods:
        __init__(grid_size, mu=0.5, sigma=0.17, a=0, b=1): Initializes a NormalDistribution instance.
        make_channel(): Generates a channel with values drawn from the normal distribution.
    """
    def __init__(self, grid_size, mu=0.5, sigma=0.17, a=0, b=1) -> None:
        """
        Initializes a NormalDistribution instance.

        Args:
            grid_size (int): The size of the grid.
            mu (float, optional): The mean of the distribution. Defaults to 0.5.
            sigma (float, optional): The standard deviation of the distribution. Defaults to 0.17.
            a (float, optional): The lower bound of the truncation. Defaults to 0.
            b (float, optional): The upper bound of the truncation. Defaults to 1.
        """
        self.grid_size = grid_size
        self.mu, self.sigma = mu, sigma
        self.a, self.b = a, b

    def make_channel(self):
        """
        Generates  a grid channel of pixel intensities with values drawn from the normal distribution.

        Returns:
            np.ndarray: The channel with values drawn from the normal distribution.
        """
        X = stats.truncnorm(
            (self.a - self.mu) / self.sigma, (self.b - self.mu) / self.sigma, loc=self.mu, scale=self.sigma)
        vals = X.rvs(self.grid_size*self.grid_size)
        return np.reshape(vals, (self.grid_size,self.grid_size))

class Image:
    """
    Represents a grid image.

    Attributes:
        shape (Shape): The shape of the image.
        channel (Channel): The channel of the image.
        img (np.ndarray): The image data.
        wect: The persistent homology of the image using the WECT algorithm.
        vectorized_wect: The vectorized version of the persistent homology.

    Methods:
        __init__(shape, channel): Initializes an Image instance.
        show(colorbar=True, axes=True, savePath=None): Displays the image.
        get_wect(hf): Computes the persistent homology of the image using the WECT algorithm.
        get_vectorized_wect(hf, height_vals): Computes the vectorized version of the persistent homology.
    """
    def __init__(self, shape, channel) -> None:
        """
        Initializes an Image instance.

        Args:
            shape (Shape): The shape of the nonzero pixels of the image.
            channel (np.ndarray): The channel of pixel intensities of the image.
        """
        self.shape = shape
        self.channel = channel
        self.img = np.multiply(self.shape.mask, channel)
        self.wect = None
        self.vectorized_wect = None

    def show(self, colorbar=True, axes=True, savePath=None):
        """
        Displays the image.

        Args:
            colorbar (bool, optional): Whether to show the colorbar. Defaults to True.
            axes (bool, optional): Whether to show the axes. Defaults to True.
            savePath (str, optional): The file path to save the image. Defaults to None.
        """
        plt.imshow(self.img, vmin=0, vmax=1, extent=[-self.shape.grid_size/2., self.shape.grid_size/2., -self.shape.grid_size/2., self.shape.grid_size/2. ])
        if colorbar:
            plt.colorbar()
        if not axes:
            plt.axis("off")
        if savePath is not None:
            plt.savefig(f"{savePath}.pdf", format="pdf", bbox_inches="tight")
        plt.show()

    def get_wect(self, hf):
        """
        Computes non-vectorized WECT of the image in a direction.

        Args:
            hf: The height filtration in a direction.

        Returns:
            list(tuple): returns a list of tuples representing the (height value, WEC) of the WECF
        """
        if self.wect is None:
            return compute_wect(self.img, hf)
        return self.wect
    
    def get_vectorized_wect(self, hf, height_vals):
        """
        Computes the vectorized version of the WECT.

        Args:
            hf: The height filtration.
            height_vals: The height values for discretization.

        Returns:
            np.ndarray: An array of shape (1,len(height_vals)) that returns the value of the WECT at the given height values
        """
        if self.vectorized_wect is None:
            if self.wect is None:
                self.wect = compute_wect(self.img, hf)
            return vectorize_wect(self.wect, height_vals)
        return self.vectorized_wect


def __make_mask(points_in_shape, grid_size, inverted = False):
    """
    Creates a mask based on the points in a shape.

    Args:
        points_in_shape (list): The points that belong to the shape.
        grid_size (int): The size of the grid.
        inverted (bool, optional): Whether the mask should be inverted. Defaults to False.

    Returns:
        np.ndarray: The mask of the shape.
    """
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
    """
    Creates a square shape.

    Args:
        grid_size (int, optional): The size of the grid. Defaults to 65.
        center (tuple, optional): The center coordinates of the square. Defaults to (32, 32).
        side_length (int, optional): The side length of the square. Defaults to 35.

    Returns:
        Shape: The square shape.
    """
    pts = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if (abs(x-center[0]) <= side_length//2) and (abs(y-center[1]) <= side_length//2)]
    mask = __make_mask(pts, grid_size)
    return Shape(grid_size, mask)

def make_circle(grid_size = 65, center = (32, 32), radius = 20):
    """
    Creates a circular shape.

    Args:
        grid_size (int, optional): The size of the grid. Defaults to 65.
        center (tuple, optional): The center coordinates of the circle. Defaults to (32, 32).
        radius (int, optional): The radius of the circle. Defaults to 20.

    Returns:
        Shape: The circular shape.
    """
    pts = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if ((x-center[0])**2 + (y-center[1])**2 <= radius**2)]
    mask = __make_mask(pts, grid_size)
    return Shape(grid_size, mask)

def make_favorite_tetris():
    """
    Creates a shape based on a standardly formatted tetris.

    Returns:
        Shape: The shape based on the favorite_tetris.npy file.
    """
    pts_ls = np.load("imagewect/favorite_tetris.npy").tolist()
    pts = [tuple(point) for point in pts_ls]
    mask = __make_mask(pts, 65)
    return Shape(65, mask)

def make_annulus(grid_size = 65, center = (32, 32), outer_radius = 22, inner_radius = 10):
    """
    Creates an annulus shape.

    Args:
        grid_size (int, optional): The size of the grid. Defaults to 65.
        center (tuple, optional): The center coordinates of the annulus. Defaults to (32, 32).
        outer_radius (int, optional): The outer radius of the annulus. Defaults to 22.
        inner_radius (int, optional): The inner radius of the annulus. Defaults to 10.

    Returns:
        Shape: The annulus shape.
    """
    assert(outer_radius > inner_radius)
    pts = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if (((x-center[0])**2 + (y-center[1])**2 >= inner_radius**2) and ((x-center[0])**2 + (y-center[1])**2 <= outer_radius**2) )]
    mask = __make_mask(pts, grid_size)
    return Shape(65, mask)


def __linf(p1, p2):
    """
    Calculates the L-infinity distance between two points.

    Args:
        p1 (tuple): The coordinates of the first point.
        p2 (tuple): The coordinates of the second point.

    Returns:
        float: The L-infinity distance between the two points.
    """
    return max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))

def make_square_annulus(grid_size = 65, center = (32, 32), outer_side_length = 38, inner_side_length = 16):
    """
    Creates a square annulus shape.

    Args:
        grid_size (int, optional): The size of the grid. Defaults to 65.
        center (tuple, optional): The center coordinates of the annulus. Defaults to (32, 32).
        outer_side_length (int, optional): The outer side length of the annulus. Defaults to 38.
        inner_side_length (int, optional): The inner side length of the annulus. Defaults to 16.

    Returns:
        Shape: The square annulus shape.
    """
    assert(outer_side_length > inner_side_length)# pts = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if ( ((abs(x-center[0]) <= outer_side_length//2) and (abs(x-center[0]) >= inner_side_length//2)) and ((abs(y-center[1]) <= outer_side_length//2) and (abs(y-center[1]) >= inner_side_length//2)) )]
    pts = [(x,y) for x,y in itertools.product(range(grid_size),range(grid_size)) if \
             __linf((x,y), center) <= outer_side_length//2 and __linf((x,y), center) >= inner_side_length//2]
    
    mask = __make_mask(pts, grid_size)
    return Shape(65, mask)

def make_clusters(grid_size = 65, center = (32, 32), side_length = 35):
    """
    Creates a Clusters shape.

    Args:
        grid_size (int, optional): The size of the grid. Defaults to 65.
        center (tuple, optional): The center coordinates of the clusters. Defaults to (32, 32).
        side_length (int, optional): The side length of each sub-square in the clusters. Defaults to 35.

    Returns:
        Shape: The clusters shape.
    """
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
    """
    Creates a Swiss cheese shape.

    Args:
        grid_size (int, optional): The size of the grid. Defaults to 65.
        center (tuple, optional): The center coordinates of the Swiss cheese. Defaults to (32, 32).
        side_length (int, optional): The side length of each sub-square in the Swiss cheese. Defaults to 35.

    Returns:
        Shape: The Swiss cheese shape.
    """
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