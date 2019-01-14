import numpy as np
import matplotlib.pyplot as plt 
from glob import glob
import time
import cv2
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from similarity_classifier import SimilarityClassifier

SUPORTED_FORMATS = ['jpg', 'jpeg', 'gif', 'png']

def get_file_names(path):
    files = []
    [files.extend(glob(path+'/*.'+extension)) for extension in SUPORTED_FORMATS]
    return files

def get_image_label(img_name):
    return img_name.split('/')[-1].split('-')[0]

def get_image_name(img_name):
    return img_name.split('/')[-1].split('.')[0]

def load_image(file_name):
    return cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)

def load_image_as_vector(file_name):
    img = load_image(file_name)
    return img.reshape(img.size)

def load_dataset(path):
    X = []
    y = []
    files = get_file_names(path)
    for img in files:
        label = get_image_label(img)
        face = load_image_as_vector(img)
        X.append(face)
        y.append(label)
    return X, y

def split_dataset(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def classify(X, y, classifier=LogisticRegression(), test_size=0.3):
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=test_size)
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return classifier, score, cm

def cross_validation(X, y, classifier=LogisticRegression(), test_size=0.3, k=10):
    avg_score = 0
    avg_time = 0
    for i in range(k):
        t0 = time.time()
        _, score, _ = classify(X, y, classifier, test_size)
        avg_time += time.time() - t0
        avg_score += score
    return avg_score/k, avg_time/k

def recognize_image(file_name, model, pca):
    img = load_image_as_vector(file_name)
    img_pca = pca.transform(img.reshape(1,-1))
    pred = model.predict(img_pca)
    label = get_image_label(file_name)
    return label, pred[0]

def evaluate_recognition(dataset_path, classifier, use_pca=True, n_components=None):
    X, y = load_dataset(dataset_path)
    if use_pca:
        X = PCA(n_components=n_components).fit_transform(X)
    score, time = cross_validation(X, y, classifier)
    print('#'*50)
    print('On dataset: %s' % dataset_path)
    print('using PCA: ' + ('YES' if use_pca else 'NO'))
    print('classifier: %s' % classifier.__class__.__name__)
    print('10 fold cross-validation results:')
    print('  Average Score: %s' % score)
    print('  Average Time : %s' % time)
    print('')
