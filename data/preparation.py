
import sys
import gc
import numpy as np
import cv2
# for carbage collection
gc.enable()


# %%
# TODO:
# - move all Strings to configuration file 
# - handle in the image is bigger than 32 * 32


# ##flags
# ----image_path
# ----on_cloude

image_path = None

def training_data(on_cloud=False) :
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from pandas import read_pickle
    from data.augmentation import augmentation

    pickle_file = '/floyd/input/data/train' if on_cloud  else'./dataset/train'
    # load data
    train_data = read_pickle(pickle_file)

    # data spliting and organization
    features = train_data['data']
    labels = train_data['fine_labels']

    assert len(features) == len(labels)

    # reshape and convert to gray
    features = np.reshape(features, [-1, 3, 32, 32])
    features = np.transpose(features, [0, 2, 3, 1])
    features = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in features]
    
    # data augmentation
    features, labels = augmentation(features, labels, 3)


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
    del train_data, features, labels, scaler, onehot_encoder,

    return features , labels

def predict_image(image_path):
    #  assumbtion 
    # - input image is 32 * 32 
    # - assert image_path is not None 

    # gray image
    image = cv2.imread(image_path,0) 

    return np.reshape( image , (32,32,1) )

    # Standardize features




