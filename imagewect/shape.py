import numpy as np
import dionysus as d
import scipy.stats as stats
import matplotlib.pyplot as plt

from .wect import compute_wect

class Shape:
    def __init__(self, grid_size, mask) -> None:
        self.grid_size = grid_size
        self.location_matrix = self.__compute_location_matrix()
        self.mask = mask


    def __compute_location_matrix(self) -> np.ndarray:
        mid = round(self.grid_size/2)
        vals = np.linspace(-1*round(mid)+1, round(mid)-1, self.grid_size)
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

    def show(self):
        plt.imshow(self.img, vmin=0, vmax=1)
        plt.colorbar()
        # plt.axis("off")
        plt.show()

    def compute_wect(self, hf):
        return compute_wect(self.img, hf)