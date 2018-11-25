# lib. import section
import gc
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from data.preparation import training_data
import argparse
# for carbage collection
gc.enable()


# %%
# TODO:
# - move all Strings or veriables to config. file
# - handle if the predected image is bigger than 32 * 32



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud', action="store_true", dest="cloud")
    args = parser.parse_args()
    on_cloud = args.cloud

# training data preparation
features, labels = training_data(on_cloud=on_cloud)

# cross validation spliting
X_train, X_validation, y_train, y_validation = train_test_split(
    features, labels, test_size=0.2, random_state=42)

del features, labels

# reshaping input data
X_train = np.reshape(X_train, [-1, 32, 32, 1])
X_validation = np.reshape(X_validation, [-1, 32, 32, 1])


tf.reset_default_graph()

# initialization veriables
n_classes = 100
epochs = 250
learning_rate = 0.001
batch_size = 64


# declear dynamic data placeholder
x = tf.placeholder(tf.float32, [None, 32, 32, 1], name='x')
y = tf.placeholder(tf.float32, [None, n_classes], name='y')

# Config.
is_training = tf.placeholder(tf.bool, name='is_training')
keep_prob = tf.constant(0.60, tf.float32)


def batch_norm(x, simple=False):
    """
    batch normalization function for convolutions + folly connected layers 
        :param x: input neurons after weights mutiplied to it 
        :param simple=False: IF simple THEN get normalization for on axes [0]
    """
    scale = tf.Variable(tf.ones(x.get_shape().as_list()[-1]))
    offset = tf.Variable(tf.zeros(x.get_shape().as_list()[-1]))
    x, _, _ = tf.nn.fused_batch_norm(
        x, scale, offset, epsilon=0.001, data_format='NHWC')
    return x


def conv_layer(x, w, wr, b, strides=1):

    xr = x

    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = batch_norm(x)
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)

    xr = tf.nn.conv2d(xr, wr, strides=[1, strides, strides, 1], padding='SAME')
    xr = batch_norm(xr)
    x = tf.add(x,  xr)


    return maxpool(x)


def maxpool(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def dropout(x):
    return tf.cond(is_training, lambda: tf.nn.dropout(x, keep_prob=keep_prob), lambda: x)


weights = {
    'wc1': tf.get_variable('w0', shape=(5, 5, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'wc1r': tf.get_variable('w0r', shape=(5, 5, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),


    'wc2': tf.get_variable('w1', shape=(5, 5,  32, 64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc2r': tf.get_variable('w1r', shape=(5, 5,  32, 64), initializer=tf.contrib.layers.xavier_initializer()),

    'wc3': tf.get_variable('w2', shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer()),
    'wc3r': tf.get_variable('w2r', shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer()),

    'wc4': tf.get_variable('w3', shape=(1, 1, 128, 256), initializer=tf.contrib.layers.xavier_initializer()),
    'wc4r': tf.get_variable('w3r', shape=(1, 1, 128, 256), initializer=tf.contrib.layers.xavier_initializer()),

    'wf': tf.get_variable('wf', shape=(2*2*256, 256), initializer=tf.contrib.layers.xavier_initializer()),
    'wf2': tf.get_variable('wf2', shape=(256, 512), initializer=tf.contrib.layers.xavier_initializer()),
    'wf3': tf.get_variable('wf3', shape=(512, 256), initializer=tf.contrib.layers.xavier_initializer()),
    'wf4': tf.get_variable('wf4', shape=(256, 170), initializer=tf.contrib.layers.xavier_initializer()),

    'out': tf.get_variable('wout', shape=(170, n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'bc1': tf.get_variable('b0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('b1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('b2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bc4': tf.get_variable('b3', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'bf': tf.get_variable('bf', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'bf2': tf.get_variable('bf2', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
    'bf3': tf.get_variable('bf3', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'bf4': tf.get_variable('bf4', shape=(170), initializer=tf.contrib.layers.xavier_initializer()),

    'out': tf.get_variable('bout', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}


def model(x, weights, biases):

    # convolution layers
    conv1 = conv_layer(x, weights['wc1'], weights['wc1r'], biases['bc1'])

    conv2 = conv_layer(conv1, weights['wc2'], weights['wc2r'], biases['bc2'])

    conv3 = conv_layer(conv2, weights['wc3'], weights['wc3r'], biases['bc3'])

    conv4 = conv_layer(conv3, weights['wc4'], weights['wc4r'], biases['bc4'])

    # flatten
    # - Reshape conv4 output to fit fully connected layer input
    flatten_layer = tf.reshape(
        conv4, [-1, weights['wf'].get_shape().as_list()[0]])

    # Fully connected layers
    layer_1 = tf.add(tf.matmul(flatten_layer, weights['wf']), biases['bf'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.layers.dropout(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['wf2']), biases['bf2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.layers.dropout(layer_2, rate=0.4)

    layer_3 = tf.add(tf.matmul(layer_2, weights['wf3']), biases['bf3'])
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.layers.dropout(layer_3, rate=0.4)

    layer_4 = tf.add(tf.matmul(layer_3, weights['wf4']), biases['bf4'])
    layer_4 = tf.nn.relu(layer_4)
    layer_4 = tf.layers.dropout(layer_4, rate=0.4)

    # output layer
    return tf.add(tf.matmul(layer_4, weights['out']), biases['out'])


# Loss and Optimizer Nodes
logits = model(x, weights, biases)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y, name='logits'), name='cost')

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predictions = tf.argmax(logits, 1, name="predictions")

# Evaluate Model Node
correct_prediction = tf.equal(predictions, tf.argmax(y, 1))

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

    for i in range(epochs):
        for batch in range(max_len//batch_size):
            batch_x = X_train[batch *
                              batch_size:min((batch+1)*batch_size, max_len)]
            batch_y = y_train[batch *
                              batch_size:min((batch+1)*batch_size, max_len)]

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
    np.save('plt_data', {'train_loss': train_loss, 'train_accuracy': train_accuracy,
                         'test_loss': test_loss, 'test_accuracy': test_accuracy})
    summary.close()
