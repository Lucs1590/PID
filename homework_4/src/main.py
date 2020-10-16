import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


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

    test_img = read_img('img/objects.png')
    object_locations = detect_object(test_img, r_table, 2, 5)
    if object_locations:
        paint_image(test_img, object_locations)
    else:
        print('Erro ao encontrar o objeto na imagem!')
    return object_locations[0]


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
    # y, x, values
    border_values = find_border_values(img)
    # angle: [r, alpha]
    r_table = define_default_r_table(len(border_values))
    yc, xc = (img.shape[0]/2, img.shape[1]/2)

    keys_r_table = list(r_table.keys())

    alpha_list = list(map(
        lambda bv: math.atan((bv[0]-yc)/(bv[1]-xc)) if bv[1] != xc else 0, border_values))

    r_list = list(map(
        lambda bv: math.sqrt((bv[1]-xc)**2 + (bv[0]-yc)**2), border_values))

    angle_list = list(map(
        lambda bv: find_nearest(keys_r_table, math.atan2(bv[0], bv[1]) * 180 / math.pi), border_values))

    idx = 0
    while idx < len(angle_list):
        r_table[angle_list[idx]].append([r_list[idx], alpha_list[idx]])
        idx += 1
    # clean_r_table = list(filter(lambda values_r_table: len(
    #     values_r_table[1]) > 0, r_table.items()))

    return r_table, border_values


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


def detect_object(img, r_table, max_scale, max_rotation, range_best=0):
    """
    # Detect Object
    Reads an image that contains several things, including the detected object.
    """
    gray_img = change_img_color(img, cv2.COLOR_BGR2GRAY)
    edge_img = apply_filter(gray_img, 'prewitt')
    borders = find_border_values(edge_img)
    M = {}
    # M[x,y,S,rotation] = 0
    S_range = range(0, max_scale)
    rotation_range = range(0, max_rotation)

    keys_r_table = list(r_table.keys())
    y_x_list = list(map(lambda bv: (bv[0], bv[1]), list(filter(
        lambda bv: bv[2], borders))))
    angle_list = list(map(
        lambda yx: find_nearest(keys_r_table, math.atan2(yx[0], yx[1]) * 180 / math.pi), y_x_list))
    del keys_r_table
    r_alpha_list = list(map(lambda angle: r_table[angle], angle_list))
    del angle_list
    complete_list = list(filter(
        lambda values: values[1], list(map(
            lambda coord, r_alpha: [coord, r_alpha], y_x_list, r_alpha_list))))
    del y_x_list
    del r_alpha_list

    complete_list = remove_dimensions(complete_list)

    complete_list = add_rotation_scale(complete_list, rotation_range)
    complete_list = add_rotation_scale(complete_list, S_range)

    # x + r * S * math.sin(alpha + rotation)
    xc_list = list(map(
        lambda value: value[0][1] + value[1][0] * value[3] * math.sin(value[1][1] + value[2]), complete_list))
    # y + r * S * math.sin(alpha + rotation)
    yc_list = list(map(
        lambda value: value[0][0] + value[1][0] * value[3] * math.sin(value[1][1] + value[2]), complete_list))

    idx = 0
    while idx < len(complete_list):
        index = (int(xc_list[idx]), int(yc_list[idx]), int(
            complete_list[idx][3]), int(complete_list[idx][2]))
        M[index] = M[index] + 1 if index in list(M.keys()) else 1
        idx += 1

    if range_best:
        best_values = Counter(M).most_common(range_best)
    else:
        best_values = [list(M.keys())[list(M.values()).index(max(M.values()))]]

    return best_values if best_values else None


def remove_dimensions(list_r_alpha):
    """
    # Remove dimension
    In other words, we broke lists with more than one r and alpha.
    """
    idx = 0
    complete_list = []

    while idx < len(list_r_alpha):
        idx_2 = 0
        curr_value = list_r_alpha[idx]
        while idx_2 < len(curr_value[-1]):
            complete_list.append(
                [curr_value[0], curr_value[-1][idx_2]])
            idx_2 += 1
        idx += 1
    return complete_list


def add_rotation_scale(list_of_values, range_r_s):
    """
    # Add Rotation or Scale
    """
    complete_list = []
    aux_list = [list_of_values for i in range_r_s]
    idx = 0
    while idx < len(range_r_s):
        idx_2 = 0
        curr_list = aux_list[idx]
        while idx_2 < len(curr_list):
            curr_list_cp = curr_list[idx_2].copy()
            curr_list_cp.append(idx)
            complete_list.append(curr_list_cp)
            idx_2 += 1
        idx += 1
    return complete_list


def paint_image(img, object_locations):
    """
    # Paint Image
    Paint image with the location of objects
    """
    pass


if __name__ == "__main__":
    run_pipeline()
