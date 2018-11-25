# %%
# lib. import section
from imgaug import augmenters as iaa
import imgaug as ia
import gc
import sys
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from pandas import read_pickle
import tensorflow as tf
print('tensorflow.__version__', tf.__version__)
# '1.10.0'
# for carbage collection
gc.enable()


# Dataset augmentation
seq = iaa.SomeOf((0,4), [
    iaa.Noop(),
    iaa.Fliplr(0.5),  # horizontal flips
    # Small gaussian blur with random sigma between 0 and 0.5.
    iaa.GaussianBlur(sigma=(0, 3)),
    iaa.Multiply((0.2, 2)),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.5, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02*255), per_channel=0.2),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Sometimes(0.7,
                  iaa.Affine(
                      scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                      translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                      rotate=(-25, 25),
                      shear=(-8, 8)
                  )
                  ),
], random_order=True)  # apply augmenters in random order

# %%
# TODO:
# check if the batch normalization works well in training and testing phase => https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
#

# %%
# load data
train_data = read_pickle('/floyd/input/data/train')

# data spliting and organization
features = train_data['data']
labels = train_data['fine_labels']

assert len(features) == len(labels)

# OneHotEncoder lables
onehot_encoder = OneHotEncoder(sparse=False)
labels = np.reshape(labels, [-1, 1])
labels = onehot_encoder.fit_transform(labels)


# reshape and convert to gray
features = np.reshape(features, [-1, 32, 32, 3])
features = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in features]

# data augmantation

old_labels = labels
old_features = features

for i in range(3):
    features = np.concatenate(
        (features, seq.augment_images(old_features)), axis=0)
    print(features.shape)
    labels = np.concatenate((labels, old_labels), axis=0)

del old_labels, old_features

# reshape for standarization process
features = np.reshape(features, [len(features), -1])

# Standardize features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
np.save('features', features)
np.save('labels', labels)
# cross validation spliting
X_train, X_validation, y_train, y_validation = train_test_split(
    features, labels, test_size=0.2, random_state=42)

# reshaping input data
X_train = np.reshape(X_train, [-1, 32, 32, 1])
X_validation = np.reshape(X_validation, [-1, 32, 32, 1])


# free the memory
del train_data, features, labels, scaler, onehot_encoder,


tf.reset_default_graph()

# %%
# initialization veriables
n_classes = 100
epochs = 250
learning_rate = 0.001
batch_size = 100

# %%
# declear dynamic data placeholder
x = tf.placeholder(tf.float32, [None, 32, 32, 1])
y = tf.placeholder(tf.float32, [None, n_classes], name='y')

# Config.
is_training = tf.placeholder(tf.bool, name='is_training')
keep_prob = tf.constant(0.60, tf.float32)


# %%


def batch_norm(x, simple=False):
    """
    batch normalization function for convolutions + folly connected layers 
        :param x: input neurons after weights mutiplied to it 
        :param simple=False: IF simple THEN get normalization for on axes [0]
    """
    axes = [0] if simple else [0, 1, 2]
    # batch_mean, batch_variance = tf.nn.moments(x, axes=axes)
    # scale = tf.Variable())
    scale = tf.Variable(tf.ones(x.get_shape().as_list()[-1]))
    offset = tf.Variable(tf.zeros(x.get_shape().as_list()[-1]))
    x, _, _ = tf.nn.fused_batch_norm(
        x, scale, offset, epsilon=0.001, data_format='NHWC')
    return x


def conv_layer(x, w, b, strides=1):

    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = batch_norm(x)
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)

    return maxpool(x)


def maxpool(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def dropout(x):
    return tf.cond(is_training, lambda: tf.nn.dropout(x, keep_prob=keep_prob), lambda: x)


weights = {
    'wc1': tf.get_variable('w0', shape=(5, 5, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': tf.get_variable('w1', shape=(5, 5, 32, 64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc3': tf.get_variable('w2', shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer()),
    'wc4': tf.get_variable('w3', shape=(1, 1, 128, 256), initializer=tf.contrib.layers.xavier_initializer()),
    'wf': tf.get_variable('wf', shape=(2*2*256, 256), initializer=tf.contrib.layers.xavier_initializer()),
    'wf2': tf.get_variable('wf2', shape=(256, 200), initializer=tf.contrib.layers.xavier_initializer()),
    'wf3': tf.get_variable('wf3', shape=(200, 160), initializer=tf.contrib.layers.xavier_initializer()),


    'out': tf.get_variable('wout', shape=(160, n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'bc1': tf.get_variable('b0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('b1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('b2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bc4': tf.get_variable('b3', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'bf': tf.get_variable('bf', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'bf2': tf.get_variable('bf2', shape=(200), initializer=tf.contrib.layers.xavier_initializer()),
    'bf3': tf.get_variable('bf3', shape=(160), initializer=tf.contrib.layers.xavier_initializer()),

    'out': tf.get_variable('bout', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}


def model(x, weights, biases):

    # convolution layers
    conv1 = conv_layer(x, weights['wc1'], biases['bc1'])

    conv2 = conv_layer(conv1, weights['wc2'], biases['bc2'])

    conv3 = conv_layer(conv2, weights['wc3'], biases['bc3'])

    conv4 = conv_layer(conv3, weights['wc4'], biases['bc4'])

    # flatten
    # - Reshape conv4 output to fit fully connected layer input
    flatten_layer = tf.reshape(
        conv4, [-1, weights['wf'].get_shape().as_list()[0]])

    # Fully connected layers
    layer_1 = tf.add(tf.matmul(flatten_layer, weights['wf']), biases['bf'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.layers.dropout(layer_1)
#     layer_1 = dropout(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['wf2']), biases['bf2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.layers.dropout(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['wf3']), biases['bf3'])
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.layers.dropout(layer_3)


#     layer_2 = dropout(layer_2)

    # output layer
    return tf.add(tf.matmul(layer_3, weights['out']), biases['out'])


# Loss and Optimizer Nodes
logits = model(x, weights, biases)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate Model Node
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialization of graph variables
init = tf.initializers.global_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# %%
with tf.Session() as sess:
    sess.run(init, feed_dict={is_training: True})
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    summary = tf.summary.FileWriter('./Output', sess.graph)

    # max size of dataset
    max_len = X_train.shape[0]
    print(max_len)

    for i in range(epochs):
        for batch in range(max_len//batch_size):
            batch_x = X_train[batch *
                              batch_size:min((batch+1)*batch_size, max_len)]
            batch_y = y_train[batch *
                              batch_size:min((batch+1)*batch_size, max_len)]

            # print('.', end=" ")

            opt = sess.run(optimizer, feed_dict={
                           x: batch_x, y: batch_y, is_training: True})

            loss, acc = sess.run([cost, accuracy], feed_dict={
                                 x: batch_x, y: batch_y, is_training: False})

        print("\nIter " + str(i) + ", Loss= " +
              "{:.6f}".format(loss) + ", Training Accuracy= " +
              "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={
                                        x: X_validation, y: y_validation, is_training: False})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:", "{:.5f}".format(test_acc))
    # Save the variables to disk.
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in file: %s" % save_path)
    summary.close()
