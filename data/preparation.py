
import sys
import gc
import numpy as np
import cv2
from config import *
# for carbage collection
gc.enable()


# FLAGS - Args
# ----image_path
# ----on_cloude

image_path = None


def training_data(data_multiplied, on_cloud=False):
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from pandas import read_pickle
    from data.augmentation import augmentation

    pickle_file = cloud_path if on_cloud else data_path
    # load data
    train_data = read_pickle(pickle_file)

    # data spliting and organization 
    features = train_data['data']
    labels = train_data['fine_labels']

    assert len(features) == len(labels)

    # reshape and convert to gray
    features = np.reshape(features, [-1, data_input_channel, data_width, data_hight]) 
    features = np.transpose(features, [0, 2, 3, 1])
    features = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in features]

    # data augmentation
    features, labels = augmentation(features, labels, data_multiplied)

    # reshape for standarization process
    features = np.reshape(features, [len(features), -1])

    # Standardize features
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # OneHotEncoder lables
    onehot_encoder = OneHotEncoder(sparse=False)
    labels = np.reshape(labels, [-1, 1])
    labels = onehot_encoder.fit_transform(labels)

    # free the memory
    del train_data, scaler, onehot_encoder,

    return features, labels


def predict_image(image_path):
    #  assumbtion
    # - input image is 32 * 32
    # - assert image_path is not None

    # gray image
    image = cv2.imread(image_path, 0)
    # Standardize features
    # pass

    return np.reshape(image, (32, 32, 1))

