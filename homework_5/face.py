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


""" Main Function """


def main():
    _path = '/home/brito/Documentos/Mestrado/PDI/codigos/homework_5'
    # path.abspath(os.getcwd())
    load_dataset(_path)
    _path = _path + '/arface'
    faces_desc_mtcnn = detect_faces_mtcnn(
        _path + '/face', _path + '/mtcnn_detect')
    faces_desc_lbp = run_lbp()


""" Download Dataset """


def load_dataset(_path):
    destination = _path + '/arface'
    dataset_file = destination + '/arface.zip'

    if not path.isdir(destination):
        os.mkdir(destination)

    if not path.isdir(destination + '/mtcnn_detect'):
        os.mkdir(destination + '/mtcnn_detect')

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
    label = []

    for arquivo in glob.glob(path.join(_path, "*.bmp")):
        img = cv2.cvtColor(cv2.imread(arquivo), cv2.COLOR_BGR2RGB)
        detected_face = MTCNN().detect_faces(img)
        if detected_face:
            label.append(arquivo.split('/')[-1])
            faces.append(detected_face)
            copy_file(arquivo, destination)

    return faces


def copy_file(file, destination):
    shutil.copy2(file, destination)


"""Local Binary Pattern (LBP)"""


def run_lbp(_path):
    images = []

    for arquivo in glob.glob(path.join(_path, "*.bmp")):
        img = cv2.cvtColor(cv2.imread(arquivo), cv2.COLOR_BGR2RGB)
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


if __name__ == "__main__":
    main()
