import cv2
import numpy as np
import matplotlib.pyplot as plt


def run_pipeline():
    """
    # Run Pipeline
    This function is the spine of project, so here all the functions are runned.
    """
    raw_img = read_img('phone_0.png')
    gray_img = change_img_color(raw_img, cv2.COLOR_BGR2GRAY)
    edge_img = apply_filter(gray_img, 'sobel')
    show_img(edge_img, 'cv')


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

    By default we use roberts filter.
    """
    if name == 'sobel':
        h_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        h_filtered_image = make_convolution(img, h_kernel)

        v_kernel = np.array([[-1, -2, -1], [0, 0, 0], [-1, -2, -1]])
        v_filtered_image = make_convolution(img, v_kernel)

        aux_img = h_filtered_image + v_filtered_image
    elif name == 'prewitt':
        h_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        h_filtered_image = make_convolution(img, h_kernel)

        v_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        v_filtered_image = make_convolution(img, v_kernel)

        aux_img = h_filtered_image + v_filtered_image
    else:
        h_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        h_filtered_image = make_convolution(img, h_kernel)

        v_kernel = np.array([[-1, -2, -1], [0, 0, 0], [-1, -2, -1]])
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


if __name__ == "__main__":
    run_pipeline()
