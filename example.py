import numpy as np
import matplotlib.pyplot as plt

from imagewect.shape import Image, UniformDistribution, make_square, make_circle, make_shape_from_image
from imagewect.wect import compute_wect, vectorize_wect

####################################################################
### Compute WECT on example shapes and images in one direction
####################################################################

# make the shape and distribution objects
square = make_square(65)
uniform_distribution = UniformDistribution(65)

# make image
square_img = Image(square, uniform_distribution.make_channel())
square_img.show()

# compute a height filtration for the square
direction = np.array([1,0])
square_height_filtration = square.get_height_filtration(direction)

# compute the wect...
square_wect = square_img.get_wect(square_height_filtration)
# ...or vectorized wect
discretization_vals = np.linspace(-45, 45, num=91)
square_vectorized_wect = square_img.get_vectorized_wect(square_height_filtration, discretization_vals)

# and do the same for another shape...
circle = make_circle(65)
circle_img = Image(circle, uniform_distribution.make_channel())
circle_height_filtration = circle.get_height_filtration(direction)
circle_wect =  circle_img.get_wect(circle_height_filtration)

# ... and then compute a proper distance between the WECTs on the interval -32 to 32
from imagewect.wect import distance_between_wects_unidirectional
distance = distance_between_wects_unidirectional(square_wect, circle_wect, -32, 32)
print("Distance between Uniform Square and Uniform Circle WECTs:", distance)


####################################################################
### Compute WECT on our own image
####################################################################

# example image
img = np.array([[0.75, 0.25, 0], [0.25, 0.75, 0], [0.25, 0.75, 0]])
plt.imshow(img)
plt.show()

# create the shape
shape = make_shape_from_image(img)

# this time we will compute the wect from 8 directions from the circle
num_dirs = 8
sample_theta = np.expand_dims(np.linspace(0, 2*np.pi, num=num_dirs+1), axis=1)[:-1]
directions = np.concatenate((np.cos(sample_theta), np.sin(sample_theta)), axis=1)

height_vals = np.linspace(-2, 2, num=5)
vectorized_wects = np.zeros((num_dirs, len(height_vals)))
for i, direction in enumerate(directions):
    # compute the height filtration
    height_filtration = shape.get_height_filtration(direction)

    # compute the wect
    wect = compute_wect(img, height_filtration)
    vectorized_wect = vectorize_wect(wect, height_vals)
    vectorized_wects[i] = vectorized_wect

# let's make a plot of the WECF in each direction
fig, ax = plt.subplots()
for i, dw in enumerate(vectorized_wects):
    ax.plot(height_vals, dw, label=f"{i+1}")
ax.legend(title="Directions")
plt.show()
