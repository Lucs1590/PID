import os
import cv2
import random
import glob
import gdown
import tarfile
import zipfile
import shutil

import numpy as np
import tensorflow as tf

from os import path
from matplotlib import pyplot as plt
# from google.colab.patches import cv2_imshow
from PIL import Image
from mtcnn import MTCNN
from skimage import feature
from natsort import natsorted
from sklearn.svm import LinearSVC
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.models import Model

""" Metrics import """
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from CMC import CMC
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc
""" Main Function """


def main():
    os.environ['DISPLAY'] = ':0'
    _path = '/home/brito/Documentos/Mestrado/PDI/codigos/homework_5'
    # path.abspath(os.getcwd())
    print('INFO: Dataset verify')
    load_dataset(_path)
    _path = _path + '/arface'

    print('INFO: Run MTCNN')
    # (faces_desc_mtcnn, labels_desc_mtcnn) = detect_faces_mtcnn(
    #     _path + '/face', _path + '/mtcnn_detect')

    print('INFO: Divide dataset (FACE)')
    divide_dataset(_path, _path + '/mtcnn_detect', 80, 20)

    print('INFO: Run LBP (FACE)')
    (faces_desc_lbp, labels_desc_lbp, lbp_model) = run_lbp(_path + '/training')

    print('INFO: Classifing Images (LBP - FACE)')
    classify_lbp(_path, lbp_model)

    print('INFO: Run VGGFACE (FACE)')
    (faces_desc_vgg, labels_desc_vgg, vgg_model) = run_vgg('resnet50', _path)

    print('INFO: Classifing Images (VGGFACE2 - FACE)')
    classify_vgg(_path, vgg_model, labels_desc_vgg)

    print('INFO: Compare images')
    compare_images(_path, vgg_model)


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

    if not path.isdir(destination + '/lbp-detected'):
        os.mkdir(destination + '/lbp-detected')

    if not path.isdir(destination + '/vgg-detected'):
        os.mkdir(destination + '/vgg-detected')

    if not path.isdir(destination + '/equalized'):
        os.mkdir(destination + '/equalized')

    if not path.isdir(destination + '/face'):
        get_dataset(dataset_file, destination,
                    'https://drive.google.com/u/2/uc?export=download&confirm=HiLF&id=1BQuEQfmMiA_cEYmvkQDCnYAJd_TT3rCk')
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
    pictures = glob.glob(path.join(_path, "*.bmp")).copy()
    pictures = natsorted(pictures)

    for _file in pictures:
        img = cv2.cvtColor(cv2.imread(_file), cv2.COLOR_BGR2RGB)
        detected_face = MTCNN().detect_faces(img)
        if detected_face:
            # cv2.imwrite('/home/brito/Documentos/Mestrado/PDI/codigos/homework_5/4.bmp', cv2.cvtColor(plot_poits(img, detected_face), cv2.COLOR_RGB2BGR))
            labels.append(
                ''.join(_file.split(os.path.sep)[-1].split('-')[0:2]))
            faces.append(detected_face)
            save_file(detected_face, img, destination,
                      _file.split(os.path.sep)[-1])

    detected = list(map(lambda aqv: ''.join(aqv.split(os.path.sep)[-1].split(
        '-')[::2]).split('_')[0], glob.glob(path.join(destination, "*.bmp"))))

    count_detected = {i: detected.count(i) for i in detected}
    print('INFO:', count_detected)
    return faces, labels


def save_file(detected_face, img, destination, file_name, required_size=(224, 224)):
    x1, y1, width, height = detected_face[0]['box']
    x2, y2 = x1 + width, y1 + height
    y1 = y1 if y1 >= 0 else 0
    y2 = y2 if y2 >= 0 else 0
    x1 = x1 if x1 >= 0 else 0
    x2 = x2 if x2 >= 0 else 0

    y1 = y1 if y1 <= img.shape[0] else img.shape[0]
    y2 = y2 if y2 <= img.shape[0] else img.shape[0]
    x1 = x1 if x1 <= img.shape[1] else img.shape[1]
    x2 = x2 if x2 <= img.shape[1] else img.shape[1]

    face = img[y1:y2, x1:x2]
    image = Image.fromarray((face).astype(np.uint8))
    image = image.resize(required_size)
    image.save(destination+os.path.sep+file_name)


def copy_file(file, destination):
    if not path.isfile(destination):
        shutil.copy2(file, destination)


def has_files(_path):
    files = glob.glob(path.join(_path, "*.bmp")).copy()
    return True if len(files) else False


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


def divide_dataset(_path, _path_destination, percentage_train=80, percentage_test=20):
    pictures = glob.glob(path.join(_path_destination, "*.bmp")).copy()
    training_path = _path+'/training'
    test_path = _path+'/test'

    total_images = len(pictures)

    percentage_train = (percentage_train/100)
    percentage_test = (percentage_test/100)

    if has_files(training_path):
        print('Dataset already is divided!')
        return 0

    if percentage_train + percentage_test < 1 \
            or percentage_train + percentage_test > 1:
        raise ValueError('invalid train/test percentage.')

    i = 0
    while i < int(total_images*percentage_train):
        rand_image = random.choice(pictures)
        copy_file(rand_image, training_path)
        pictures.remove(rand_image)
        i += 1

    i = 0
    while i < len(pictures):
        rand_image = random.choice(pictures)
        copy_file(rand_image, test_path)
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
    desc = LocalBinaryPatterns(40, 8)
    faces = []
    labels = []
    pictures = glob.glob(path.join(_path, "*.bmp")).copy()
    pictures = natsorted(pictures)

    for _file in pictures:
        try:
            gray_img = cv2.cvtColor(cv2.imread(_file), cv2.COLOR_BGR2GRAY)
            hist = desc.describe(gray_img)

            if len(hist) > 0:
                labels.append(
                    ''.join(_file.split(os.path.sep)[-1].split('-')[0:2]))
                faces.append(hist)
        except Exception:
            pass

    model = LinearSVC(C=100.0, random_state=42)
    model.fit(faces, labels)

    return faces, labels, model


def classify_lbp(_path, model):
    desc = LocalBinaryPatterns(40, 8)
    hit = 0
    miss = 0
    hist_list = []
    label_list = []
    predicted_values_list = []

    pictures = glob.glob(path.join(_path+'/test', "*.bmp")).copy()
    pictures = natsorted(pictures)

    for _file in pictures:
        image = cv2.imread(_file)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray_img)
        prediction = model.predict(hist.reshape(1, -1))
        correct_class = ''.join(_file.split(os.path.sep)[-1].split('-')[0:2])

        if prediction[0] == correct_class:
            hit += 1
        else:
            miss += 1

        cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3)
        cv2.imwrite(_path+'/lbp-detected/'+_file.split(os.path.sep)[-1], image)

        hist_list.append(hist.reshape(1, -1))
        label_list.append(correct_class)
        predicted_values_list.append(prediction[0])

    calcule_f1(predicted_values_list, label_list)
    data = np.array(hist_list).squeeze()
    score = model.decision_function(data)
    making_cmc(score, label_list)
    compute_precision_recall(label_list, score)
    make_roc_curve(label_list, score)

    print(hit, miss)
    return hit, miss


""" VGGFACE """


def run_vgg(_model, _path):
    pictures = glob.glob(path.join(_path+'/training', "*.bmp")).copy()
    pictures = natsorted(pictures)

    faces = []
    labels = []

    for _file in pictures:
        try:
            img = cv2.cvtColor(cv2.imread(_file), cv2.COLOR_BGR2RGB)
            labels.append(
                ''.join(_file.split(os.path.sep)[-1].split('-')[0:2]))
            faces.append(img)
        except Exception:
            pass

    encoded_labels = OneHotEncoder().fit_transform(
        np.array(labels).reshape(-1, 1)).toarray()

    model = define_vgg_model(_model, len(np.unique(np.array(labels))))
    model.fit(np.array(faces), encoded_labels, epochs=5)

    return faces, labels, model


def define_vgg_model(_mode, num_class):
    base_model = VGGFace(model='resnet50', include_top=False)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='elu', name='fc1')(x)
    x = tf.keras.layers.Dense(196, activation='elu', name='fc2')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    predictions = tf.keras.layers.Dense(
        num_class, activation='softmax', name='predictions')(x)
    model = Model(base_model.input, predictions)

    for layer in model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def classify_vgg(_path, model, labels):
    hit = 0
    miss = 0
    label_list = []
    score_list = []
    predicted_values_list = []

    pictures = glob.glob(path.join(_path+'/test', "*.bmp")).copy()
    pictures = natsorted(pictures)

    encoder = OneHotEncoder()
    encoded_labels = encoder.fit_transform(
        np.array(labels).reshape(-1, 1)).toarray()
    encoder.fit_transform(np.array(labels).reshape(-1, 1))
    labels = encoder.inverse_transform(encoded_labels)

    for _file in pictures:
        image = cv2.imread(_file)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
        samples = np.expand_dims(rgb_img, axis=0)
        samples = preprocess_input(samples, version=2)
        prediction = model.predict(samples)
        correct_class = ''.join(_file.split(os.path.sep)[-1].split('-')[0:2])

        idx_best_prediction = np.argmax(prediction[0])
        best_prediction = labels[idx_best_prediction][0]

        if best_prediction == correct_class:
            hit += 1
        else:
            miss += 1

        cv2.putText(image, best_prediction, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imwrite(_path+'/vgg-detected/'+_file.split(os.path.sep)[-1], image)

        label_list.append(correct_class)
        score_list.append(prediction)
        predicted_values_list.append(best_prediction)

    calcule_f1(predicted_values_list, label_list)
    making_cmc(np.array(score_list).squeeze(), label_list)
    compute_precision_recall(label_list, np.array(score_list).squeeze())
    make_roc_curve(label_list, np.array(score_list).squeeze())

    print(hit, miss)
    return hit, miss


def compare_images(_path, model, image1=None, image2=None):
    pictures = glob.glob(path.join(_path + '/mtcnn_detect', "*.bmp")).copy()
    pic1 = image1 if image1 else random.choice(pictures)
    print(pic1)
    pic2 = image2 if image2 else random.choice(pictures)
    print(pic2)

    rand_image1 = cv2.cvtColor(cv2.imread(
        pic1), cv2.COLOR_BGR2RGB).astype('float32')
    rand_image2 = cv2.cvtColor(cv2.imread(
        pic2), cv2.COLOR_BGR2RGB).astype('float32')

    sample1 = np.expand_dims(rand_image1, axis=0)
    sample1 = preprocess_input(sample1, version=2)

    sample2 = np.expand_dims(rand_image2, axis=0)
    sample2 = preprocess_input(sample2, version=2)

    prediction1 = model.predict(sample1)
    prediction2 = model.predict(sample2)

    is_match(prediction1, prediction2)


def is_match(known_embedding, candidate_embedding, thresh=0.5):
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))


""" Metrics """


def compute_precision_recall(label, score):
    Y = OneHotEncoder().fit_transform(np.array(label).reshape(-1, 1)).toarray()

    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(Y.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(
            Y[:, i], score[:, i])
        average_precision[i] = average_precision_score(Y[:, i], score[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(
        Y.ravel(), score[:, :Y.shape[1]].ravel())
    average_precision["micro"] = average_precision_score(
        Y, score[:, :Y.shape[1]], average="micro")

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]))
    plt.show()


def calcule_f1(predicted, true):
    f1_value = f1_score(true, predicted, average='macro')
    print('F1 Score: ', f1_value)
    return f1_value


def making_cmc(values, keys):
    i = 0
    default_array = []
    cmc_dict = {}

    while i < 5:
        default_array.append(keys.index(random.choice(keys)))
        i += 1

    i = 0
    while i < len(default_array):
        cmc_dict[keys[default_array[i]]] = values[[
            default_array[i]]].squeeze().tolist()
        i += 1

    cmc = CMC(cmc_dict)
    cmc.plot(title='CMC', xlabel='Rank Score', ylabel='Recognition Rate')


def eer(fpr_list, tpr_list):
    err = brentq(lambda x: 1. - x - interp1d(fpr_list, tpr_list)(x), 0., 1.)
    print(err)
    return err


def make_roc_curve(label, score):
    Y = OneHotEncoder().fit_transform(np.array(label).reshape(-1, 1)).toarray()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(Y.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(
        Y.ravel(), score[:, :Y.shape[1]].ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    eer(fpr["micro"], tpr["micro"])


""" Applying Filters """


def histogram_equalization(_path):
    pictures = glob.glob(path.join(_path, "*.bmp")).copy()
    pictures = natsorted(pictures)

    for _file in pictures:
        img = cv2.cvtColor(cv2.imread(_file), cv2.COLOR_BGR2HSV)
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
        image = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        cv2.imwrite(_path+'/equilized/'+_file.split(os.path.sep)[-1], image)


if __name__ == "__main__":
    main()
