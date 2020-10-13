import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def run_pipeline():
    """
    # Run Pipeline
    This function is the spine of project, so here all the functions are runned.
    """
    raw_img = read_img('img/input_1.png')
    gray_img = change_img_color(raw_img, cv2.COLOR_BGR2GRAY)
    edge_img = apply_filter(gray_img, 'sobel')
    # show_img(edge_img, 'cv')
    (r_table, border_values) = build_r_table(edge_img)
    has_object = detect_object('img/objects.png', r_table)
    # find object into image


def read_img(path):
    """
    # Read Image
    This function make the image reading.
    """
    return cv2.imread(path)


def change_img_color(img, color_transformation):
    """
    # Change Color Image
    This function change the color of image.
    """
    return cv2.cvtColor(img, color_transformation)


def apply_filter(img, name=''):
    """
    # Apply Filter
    Select a filter to make convolution process.

    By default we use prewitt filter.
    """
    if name == 'sobel':
        h_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        h_filtered_image = make_convolution(img, h_kernel)

        v_kernel = np.array([[-1, -2, -1], [0, 0, 0], [-1, -2, -1]])
        v_filtered_image = make_convolution(img, v_kernel)

        aux_img = h_filtered_image + v_filtered_image
    else:
        h_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        h_filtered_image = make_convolution(img, h_kernel)

        v_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        v_filtered_image = make_convolution(img, v_kernel)

        aux_img = h_filtered_image + v_filtered_image

    return aux_img


def make_convolution(img, kernel):
    """
    # Make Convolution
    https://en.wikipedia.org/wiki/Convolution
    """
    aux = np.zeros_like(img)

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            aux[i][j] = np.sum(img[i-1:i+2, j-1:j+2] * kernel)
    return aux


def show_img(img, param=''):
    """
    # Show Image
    This function plot image with opencv or matplotlib.

    By default we use matplotlib.
    """
    if param == 'cv':
        cv2.imwrite('result_img/result.png', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        plt.imshow(img)
        plt.show()


def build_r_table(img):
    """
    # Build R Table
    Building r table acording edge image
    """
    border_values = find_border_values(img)
    r_table = define_default_r_table(len(border_values))
    yc, xc = (img.shape[0]/2, img.shape[1]/2)

    keys_r_table = list(r_table.keys())

    for values in border_values:
        (y, x, value) = values
        r = math.sqrt((x-xc)**2 + (y-yc)**2)
        alpha = math.atan((y-yc)/(x-xc)) if x != xc else 0

        angle = math.atan2(y, x) * 180 / math.pi
        angle = find_nearest(keys_r_table, angle)

        r_table[angle].append([r, alpha])

    # angle, r, alpha
    clean_r_table = list(filter(lambda values_r_table: len(
        values_r_table[1]) > 0, r_table.items()))

    return clean_r_table, border_values


def find_border_values(img):
    """
    # Find border values
    Mapping all x, y and edge values of the image
    """
    coordinates = []
    for i in range(0, img.shape[0]):
        for j in range(1, img.shape[1]):
            coordinates.append((i, j, img[i][j])) if img[i][j] != 0 else ...
    return coordinates


def define_default_r_table(k):
    """
    # Define Default R-Table
    """
    r_table = defaultdict(list)
    gradient = 0
    r_table[gradient]
    while (gradient < 180):
        gradient = round(gradient + 180/k, 2)
        r_table[gradient]
    return r_table


def find_nearest(array, value):
    """
    # Find nearest values
    Find the value closest on the list
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def detect_object(test_image, r_table):
    """
    # Detect Object
    Reads an image that contains several things, including the detected object.
    """
    return True


if __name__ == "__main__":
    run_pipeline()
