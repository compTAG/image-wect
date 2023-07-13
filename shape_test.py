import numpy as np

from imagewect.shape import Shape, OnePixel, Image, UniformDistribution, NormalDistribution, \
    make_square, make_square_annulus, make_annulus, make_clusters, make_swiss_cheese
from imagewect.wect import compute_wect

import matplotlib.pyplot as plt

square = make_square_annulus()
m = square.mask
print(m)

ud = UniformDistribution(65)
n17 = NormalDistribution(65)
n25 = NormalDistribution(65, sigma=0.25)
n50 = NormalDistribution(65, sigma=0.5)

img = Image(square, ud.make_channel())
print(np.count_nonzero(img.img))
img.show()
