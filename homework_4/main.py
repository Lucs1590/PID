import cv2


def run_pipeline():
    """
    ## Run Pipeline
    This function is the spine of project, so here all the functions are runned.
    """
    raw_img = read_img('')
    gray_img = change_img_color(raw_img, cv2.COLOR_BGR2GRAY)


def read_img(path):
    """
    ## Read Image
    This function make the image reading.
    """
    return cv2.imread(path)


def change_img_color(img, color_transformation):
    """
    ## Change Color Image
    This function change the color of image.
    """
    return cv2.cvtColor(img, color_transformation)


if __name__ == "__main__":
    run_pipeline()
