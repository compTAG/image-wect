import numpy as np
import matplotlib.pyplot as plt

from imagewect.shape import Image, UniformDistribution, make_square, make_circle, make_shape_from_image
from imagewect.wect import compute_wect
from imagewect.visualization import build_wect, prep_wect, create_polygons, plot_wect, convert_svg_to_png, image_to_matrix, resize_image, rgb_plot_wect



####################################################################
### Visualize the WECT on example shapes and images
####################################################################

# example image
img = np.array([[0.75, 0.25, 0], [0.25, 0.75, 0], [0.25, 0.75, 0]])
plt.imshow(img)
#plt.show()

# build the WECT
wect = build_wect(img,360)

# create the polygons
polygons = create_polygons(wect)

# create the svg file
plot_wect(polygons,'wect_plot.svg')

#convert the svg to a png (optional)
convert_svg_to_png('wect_plot.svg','wect_plot.png')


###################################################################
## Mona Lisa example
###################################################################

# resize the image
square_mona = resize_image('mona_lisa.jpg', 'sq_mona.jpg',(100,100))

# build the matrix
mona_lisa_matrix = image_to_matrix('sq_mona.jpg')

# now the WECT
mona_wect = build_wect(mona_lisa_matrix, 16)

# create the polygons
mona_polygons = create_polygons(mona_wect)

# create the svg
plot_wect(mona_polygons, 'mona_wect.svg')

convert_svg_to_png('mona_wect.svg','mona_wect.png')






