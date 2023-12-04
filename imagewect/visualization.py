import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import cairosvg

from imagewect.shape import make_shape_from_image
from imagewect.wect import compute_wect

####################################################################
### Grayscale WECT plots
####################################################################

def build_wect(img, num_dirs=16):
    """
    Builds the WECT
    IMPORTANT: this usage of the term WECT differs from the main code. Throughout this file, we use WECF to refer to the weighted euler characteristic function for a single direction and WECT to refer to the ensemble of WECFs for all listed directions.

    Args:
        img (np.ndarray): The input image.
        num_dirs (int): The number of directions. Defaults to 16.

    Returns:
        wect (dict): The WECT, stored as a dictionary of dictionaries.
    """
    shape = make_shape_from_image(img)

    # Construct the direction vectors
    sample_theta = np.expand_dims(np.linspace(0, 2*np.pi, num=num_dirs+1), axis=1)[:-1]
    directions = np.concatenate((np.cos(sample_theta), np.sin(sample_theta)), axis=1)

    # Build the WECT. The keys for wect are directions and their values are dictionaries corresponding to the WECF in a given direction.
    wect = {}
    for i in directions:
        # Convert the NumPy array to a tuple
        direction_key = tuple(i)

        height_filtration = shape.get_height_filtration(i)
        wecf = dict(compute_wect(img, height_filtration))
        wect[direction_key] = wecf
    return wect

def prep_wect(wect):
    """
    Prepares the WECT for the interpolation step.
    First normalizes the height values.

    Args:
        wect (dict): The WECT.
    
    Returns:
        prepped_wect (dict): The prepared WECT.
    """
    # Normalize the height values to be from 0 to 1
    normalized_wect = {}
    for dir in wect:
        wecf = {}
        max_key = max(wect[dir].keys())
        min_key = min(wect[dir].keys())
        key_range = max_key - min_key

        for h in wect[dir]:
            normalized_h = (h - min_key) / key_range
            wecf[normalized_h] = wect[dir][h]

        normalized_wect[dir] = wecf

    return normalized_wect

def find_range(wect):
    """
    Finds the range of values appearing in the WECT.

    Args:
        wect (dict): The WECT.

    Returns:
        ran (tuple): The range of values appearing in the WECFs.
    """

    all_values = [value for wecf in wect.values() for value in wecf.values()]

    largest_value = max(all_values)
    smallest_value = min(all_values)
    ran = (smallest_value,largest_value)
    
    return ran

def number_to_color(x, ran, color_map = 'viridis'):
    """
    Converts a number to a color for use in a heat map.

    Args:
        x (float): A number.
        ran (tuple): The range of numbers that can appear.

    Returns:
        color: The color corresponding to x.
    """
    # Normalize x to be in the range [0, 1]
    normalized_value = (x- ran[0]) / (ran[1]-ran[0])

    # Choose a colormap
    colormap = plt.get_cmap(color_map)

    # Map the normalized value to a color in the chosen colormap
    RGBA_color = colormap(normalized_value)

    # Convert RGBA values to hexadecimal
    color = "#{:02x}{:02x}{:02x}".format(
        int(RGBA_color[0] * 255),
        int(RGBA_color[1] * 255),
        int(RGBA_color[2] * 255)
)
    return color

def get_coords(dir, h):
    """
    Gets the coordinates for a given height value in a given direction. Essentially converts circular coordinates to cartesian coordinates (with a translation).

    Args:
        dir (list): A direction vector
        h (float): A normalized height value 

    Returns:
        coords (tuple): The (x,y) coordinates
    """
    # The image we are creating will be 200 by 200 units.
    coords = (100 + 100 * (1-h) * dir[0], 100 + 100 * (1-h) * dir[1])
    return coords

def create_polygons(wect, color_map = 'viridis'):
    """
    Creates the polygons with their corresponding gradient fills to be drawn in the svg file.

    Args:
        wect (dict): The WECT.
        color_map: The colormap, defaults to viridis.
    
    Returns:
        polygons (dict): A dictionary indexed by polygons with values color gradients.
    """
    # First prepare the WECT by normalizing the height values
    prepped_wect = prep_wect(wect)
    # Find the range of values appearing in the WECT
    ran = find_range(wect)
    directions = list(prepped_wect.keys())
    N = len(directions)

    polygons = {}
    for i, current_dir in enumerate(directions):
        next_dir = directions[(i+1) % N]
        heights = list(prepped_wect[current_dir].keys())
        for j, current_height in enumerate(heights[:-1]):
            next_height = heights[j+1]
            #Build the polygon
            polygon = (get_coords(current_dir,current_height), get_coords(current_dir,next_height), get_coords(next_dir,next_height), get_coords(next_dir, current_height))
            #Add the polygon to the dictionary with the color for the polygon to be filled with as value
            polygons[polygon] = number_to_color(prepped_wect[current_dir][current_height], ran, color_map)
    
    return polygons

def plot_wect(polygons, output_filename='wect_plot.svg'):
    """
    Visualizes the WECT by creating an SVG image, adding polygons, and filling them with a single color without borders.

    Args:
        polygons (dict): A dictionary indexed by polygons with values as single color codes.
        output_filename (str): The name of the output SVG file. Defaults to 'output.svg'.
    """
    svg_content = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
    svg_content += '<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">\n'

    for i, (polygon, fill_color) in enumerate(polygons.items()):
        poly_elem = f'<polygon points="{polygon[0][0]},{polygon[0][1]} {polygon[1][0]},{polygon[1][1]} {polygon[2][0]},{polygon[2][1]} {polygon[3][0]},{polygon[3][1]}"'
        poly_elem += f' fill="{fill_color}" stroke="none" />\n'  # Set stroke to none to hide the border

        # Add the polygon element to the SVG content
        svg_content += poly_elem

    svg_content += '</svg>'

    # Save the SVG content to the specified file
    with open(output_filename, 'w') as svg_file:
        svg_file.write(svg_content)

####################################################################
### Image manipulation tools
####################################################################

def convert_svg_to_png(input_file, output_file):
    """
    Converts an svg to a png.

    Args:
        input_file (string): The filename of the svg
        output_file (string): The filename for the output
    """
    cairosvg.svg2png(url=input_file, write_to=output_file, output_width=3000, output_height=3000)

def resize_image(input_path, output_path, new_size):
    """
    Resizes an image to a specified size.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path to save the resized image.
        new_size (tuple): Target size (width, height) for the resized image.
    """
    # Open the image using Pillow
    img = Image.open(input_path)

    # Resize the image to the new size
    resized_img = img.resize(new_size, Image.BICUBIC)

    # Save the resized image
    resized_img.save(output_path)

def image_to_matrix(image_path):
    """
    Converts an image to a NumPy array of its grayscale values.

    Args:
        image_path (string): The path to the image.

    Returns:
        matrix (NumPy array): The matrix corresponding to the grayscale image
    """
    # Open the image using Pillow
    img = Image.open(image_path)

    # Convert the image to a grayscale image
    gray_img = img.convert('L')

    # Convert the grayscale image to a NumPy array
    matrix = np.array(gray_img)

    return matrix

def image_to_rgb_matrices(image_path):
    """
    Converts an image to matrices of red, green, and blue values.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        red_matrix, green_matrix, blue_matrix: Matrices of red, green, and blue values.
    """
    # Open the image using Pillow
    img = Image.open(image_path)

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Separate the RGB channels
    red_matrix = img_array[:, :, 0]
    green_matrix = img_array[:, :, 1]
    blue_matrix = img_array[:, :, 2]

    return red_matrix, green_matrix, blue_matrix

####################################################################
### RGB WECT plots
####################################################################

def rgb_plot_wect(image_path, num_dirs = 16):

    red_matrix, green_matrix, blue_matrix = image_to_rgb_matrices(image_path)

    # find the range of red, green, and blue values in the original image
    red_range = (np.min(red_matrix), np.max(red_matrix))
    green_range = (np.min(green_matrix), np.max(green_matrix))
    blue_range = (np.min(blue_matrix), np.max(blue_matrix))
    color_ranges = [red_range, green_range, blue_range]

    # build the wects
    red_wect = build_wect(red_matrix, num_dirs)
    green_wect = build_wect(green_matrix, num_dirs)
    blue_wect = build_wect(blue_matrix, num_dirs)
    wects = [red_wect, green_wect, blue_wect]

    for color_index in range(3):
        cmap = rgb_colormap(color_ranges[color_index], color_index)
        polygons = create_polygons(wects[color_index], cmap)
        plot_wect(polygons, f'wect_{color_index}.svg')
        convert_svg_to_png(f'wect_{color_index}.svg', f'wect_{color_index}.png')
    
    red_wect_plot = Image.open('wect_0.png')
    green_wect_plot = Image.open('wect_1.png')
    blue_wect_plot = Image.open('wect_2.png')

    # Create a blank image with the same size and mode as the individual channels
    rgb_wect_plot = Image.new('RGB', red_wect_plot.size)

    # Merge the individual channels into the RGB image
    rgb_wect_plot.paste(red_wect_plot, (0, 0))
    rgb_wect_plot.paste(green_wect_plot, (0, 0))
    rgb_wect_plot.paste(blue_wect_plot, (0, 0))

    rgb_wect_plot.save('rgb_wect_plot.png')



def rgb_colormap(color_range, color_index):
    """
    Defines a custom colormap for red, green or blue values.

    Args:
        color_range (tuple): The range of the specified color appearing in the original image.
        color_index (int): 0 for red, 1 for green, 2 for blue.
    
    Returns:
        custom_colormap: The color map
    """

    color0 = [0, 0, 0]
    color0[color_index] = color_range[0]/255
    color1 = [0, 0, 0]
    color1[color_index] = color_range[1]/255
    colors = [color0, color1]

    custom_colormap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)
    return custom_colormap
