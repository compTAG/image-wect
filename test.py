import numpy as np

from imagewect.shape import Shape, OnePixel, Image, UniformDistribution

shape = OnePixel(3)
# print(shape.location_matrix[0][0])
# print(shape.location_matrix[0][1])
# print(shape.location_matrix[0][2])
# print(shape.get_heights(np.array([1,0])))

hf = shape.get_height_filtration(np.array([1,0]))


ud = UniformDistribution(3, 0, 1)

img = Image(shape, ud.make_channel())

s = Shape(3, np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]]))
img = Image(s, np.array([[0.75, 0.25, 0], [0.25, 0.75, 0], [0.25, 0.75, 0]]))
cp = img.compute_wect(s.get_height_filtration(np.array([1,0])))


print("\n\nWECT:", cp)
