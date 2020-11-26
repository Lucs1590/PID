import os
import cv2
import random
import glob
import gdown
import tarfile
import zipfile
import shutil

import numpy as np

from os import path
from matplotlib import pyplot as plt
# from google.colab.patches import cv2_imshow
from mtcnn import MTCNN
from skimage import feature
from sklearn.svm import LinearSVC

""" Main Function """


def main():
    _path = '/home/brito/Documentos/Mestrado/PDI/codigos/homework_5'
    # path.abspath(os.getcwd())
    print('Dataset verify')
    load_dataset(_path)
    _path = _path + '/arface'

    print('Run MTCNN')
    (faces_desc_mtcnn, labels_desc_mtcnn) = detect_faces_mtcnn(
        _path + '/face', _path + '/mtcnn_detect')

    print('Run LBP')
    faces_desc_lbp = run_lbp()
    (faces_desc_lbp_2, labels_desc_lbp_2, model) = run_lbp_2(
        _path + '/face', _path + '/lbp_detect')
    # classify_lbp(_path + '/lbp_detect', model)


""" Download Dataset """


def load_dataset(_path):
    destination = _path + '/arface'
    dataset_file = destination + '/arface.zip'

    if not path.isdir(destination):
        os.mkdir(destination)

    if not path.isdir(destination + '/mtcnn_detect'):
        os.mkdir(destination + '/mtcnn_detect')

    if not path.isdir(destination + '/lbp_detect'):
        os.mkdir(destination + '/lbp_detect')

    if not path.isdir(destination + '/face'):
        get_dataset(dataset_file, destination)
    elif path.isfile(dataset_file):
        unzip_file(dataset_file, destination)
    else:
        if path.isdir(destination + '/face'):
            print('Dataset is set!')
        else:
            raise OSError(
                'the default directory of Python is not found.')


def get_dataset(output, destination):
    url = 'https://drive.google.com/u/2/uc?export=download&confirm=HiLF&id=1BQuEQfmMiA_cEYmvkQDCnYAJd_TT3rCk'
    try:
        gdown.download(url, output, quiet=False)
        unzip_file(output, destination)

    except Exception:
        raise ConnectionError(
            'you need to be connected to some internet network to download the database.')


def unzip_file(_file, destination):
    with zipfile.ZipFile(_file, 'r') as zip_ref:
        zip_ref.extractall(destination)
    os.remove(_file)


""" Detect Faces """


def detect_faces_mtcnn(_path, destination):
    faces = []
    labels = []

    for _file in glob.glob(path.join(_path, "*.bmp")):
        img = cv2.cvtColor(cv2.imread(_file), cv2.COLOR_BGR2RGB)
        detected_face = MTCNN().detect_faces(img)
        if detected_face:
            labels.append(_file.split(os.path.sep)[-1].split('-')[1])
            faces.append(detected_face)
            copy_file(_file, destination)

    detected = list(map(lambda aqv: aqv.split(
        os.path.sep)[-1].split('-')[-1].split('_')[0], glob.glob(path.join(destination, "*.bmp"))))
    count_detected = {i: detected.count(i) for i in detected}
    print(count_detected)
    return faces, labels


def copy_file(file, destination):
    if not path.isfile(destination):
        shutil.copy2(file, destination)


"""Local Binary Pattern (LBP)"""


class LocalBinaryPatterns:
    def __init__(self, num_points, radius):
        self.numPoints = num_points
        self.radius = radius

    def describe(self, image, eps=1e-5):
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist


def run_lbp_2(_path, destination):
    desc = LocalBinaryPatterns(24, 8)
    faces = []
    labels = []

    # train
    for _file in glob.glob(path.join(_path, "*.bmp")):
        gray_img = cv2.cvtColor(cv2.imread(_file), cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray_img)

        if hist:
            labels.append(_file.split(os.path.sep)[-1].split('-')[1])
            faces.append(hist)
            copy_file(_file, destination)

    model = LinearSVC(C=100.0, random_state=42)
    model.fit(faces, labels)

    return faces, labels, model


def classify_lbp(_path, model):
    desc = LocalBinaryPatterns(24, 8)

    for _file in glob.glob(path.join(_path, "*.bmp")):
        image = cv2.imread(_file)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray_img)
        prediction = model.predict(hist.reshape(1, -1))

        # display the image and the prediction
        cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3)
        cv2.imshow("Image", image)
        cv2.waitKey(0)


def run_lbp(_path):
    images = []

    for _file in glob.glob(path.join(_path, "*.bmp")):
        img = cv2.cvtColor(cv2.imread(_file), cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
        height, width = img_gray.shape
        img_lbp = np.zeros((height, width), np.uint8)

        for i in range(0, height):
            for j in range(0, width):
                img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
        images.append(img_lbp)

    return img_lbp


def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []

    val_ar.append(get_pixel(img, center, x-1, y-1))
    val_ar.append(get_pixel(img, center, x-1, y))
    val_ar.append(get_pixel(img, center, x-1, y + 1))
    val_ar.append(get_pixel(img, center, x, y + 1))
    val_ar.append(get_pixel(img, center, x + 1, y + 1))
    val_ar.append(get_pixel(img, center, x + 1, y))
    val_ar.append(get_pixel(img, center, x + 1, y-1))
    val_ar.append(get_pixel(img, center, x, y-1))

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val


def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass

    return new_value


""" Applying Filters """

if __name__ == "__main__":
    main()
