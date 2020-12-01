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
    print('INFO: Dataset verify')
    load_dataset(_path)
    _path = _path + '/arface'

    print('INFO: Run MTCNN')
    # (faces_desc_mtcnn, labels_desc_mtcnn) = detect_faces_mtcnn(
    #    _path + '/face', _path + '/mtcnn_detect')

    print('INFO: Divide dataset')
    divide_dataset(_path, 80, 20)

    print('INFO: Run LBP')
    (faces_desc_lbp, labels_desc_lbp, model) = run_lbp(_path + '/training')
    # classify_lbp(_path + '/lbp_detect', model)


""" Download Dataset """


def load_dataset(_path):
    destination = _path + '/arface'
    dataset_file = destination + '/arface.zip'

    if not path.isdir(destination):
        os.mkdir(destination)

    if not path.isdir(destination + '/mtcnn_detect'):
        os.mkdir(destination + '/mtcnn_detect')

    if not path.isdir(destination + '/training'):
        os.mkdir(destination + '/training')
        os.mkdir(destination + '/test')

    if not path.isdir(destination + '/face'):
        get_dataset(dataset_file, destination)
    elif path.isfile(dataset_file):
        unzip_file(dataset_file, destination)
    else:
        if path.isdir(destination + '/face'):
            print('INFO: Dataset is set!')
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
    print('INFO:', count_detected)
    return faces, labels


def copy_file(file, destination):
    if not path.isfile(destination):
        shutil.copy2(file, destination)


def plot_poits(_image, detected_face):
    if len(detected_face):
        x1, y1, x2, y2 = detected_face[0]['box']
        _image = cv2.rectangle(_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for point in detected_face[0]['keypoints'].values():
            x, y = point
            _image = cv2.circle(_image, (x, y), radius=1,
                                color=(0, 0, 255), thickness=3)
    return _image


""" Divide Dataset """


def divide_dataset(_path, percentage_train=80, percentage_test=20):
    pictures = glob.glob(path.join(_path + '/mtcnn_detect', "*.bmp")).copy()
    total_images = len(pictures)

    percentage_train = (percentage_train/100)
    percentage_test = (percentage_test/100)

    if percentage_train + percentage_test < 1 \
            or percentage_train + percentage_test > 1:
        raise ValueError('invalid train/test percentage.')

    i = 0
    while i < int(total_images*percentage_train):
        rand_image = random.choice(pictures)
        copy_file(rand_image, _path+'/training')
        pictures.remove(rand_image)
        i += 1

    i = 0
    while i < len(pictures):
        rand_image = random.choice(pictures)
        copy_file(rand_image, _path+'/test')
        pictures.remove(rand_image)
        i += 1


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


def run_lbp(_path):
    desc = LocalBinaryPatterns(24, 8)
    faces = []
    labels = []

    # train
    for _file in glob.glob(path.join(_path, "*.bmp")):
        gray_img = cv2.cvtColor(cv2.imread(_file), cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray_img)

        if len(hist) > 0:
            labels.append(_file.split(os.path.sep)[-1].split('-')[1])
            faces.append(hist)

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


""" Applying Filters """

if __name__ == "__main__":
    main()
