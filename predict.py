from data.preparation import predict_image
import tensorflow as tf
import argparse


image_path = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', dest="image")
    args = parser.parse_args()
    image_path = args.image

print('image_path', image_path)

sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./checkpoint/model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./checkpoint'))

graph = tf.get_default_graph()
# declear dynamic data placeholder

x = graph.get_tensor_by_name("Placeholder:0")
# y = graph.get_tensor_by_name("y:0")

logits = graph.get_operation_by_name("softmax_cross_entropy_with_logits")

#Now, access the op that you want to run. 
# logits = graph.get_operation_by_name("softmax_cross_entropy_with_logits:0")
# for i in sess.graph.get_operations():
#     print(i.name)

# print the loaded variable
# weight = sess.run([softmax_cross_entropy_with_logits])
# print('W = ', weight)
# print('b = ', bias)


# n_classes = 100
# epochs = 250
# learning_rate = 0.001
# batch_size = 100

# # Config.
# is_training = tf.placeholder(tf.bool, name='is_training')


# def batch_norm(x, simple=False):
#     """
#     batch normalization function for convolutions + folly connected layers 
#         :param x: input neurons after weights mutiplied to it 
#         :param simple=False: IF simple THEN get normalization for on axes [0]
#     """
#     axes = [0] if simple else [0, 1, 2]
#     # batch_mean, batch_variance = tf.nn.moments(x, axes=axes)
#     # scale = tf.Variable())
#     scale = tf.Variable(tf.ones(x.get_shape().as_list()[-1]))
#     offset = tf.Variable(tf.zeros(x.get_shape().as_list()[-1]))
#     x, _, _ = tf.nn.fused_batch_norm(
#         x, scale, offset, epsilon=0.001, data_format='NHWC')
#     return x


# def conv_layer(x, w, b, strides=1):

#     x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
#     x = batch_norm(x)
#     x = tf.nn.bias_add(x, b)
#     x = tf.nn.relu(x)

#     return maxpool(x)


# def maxpool(x, k=2):
#     return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# def dropout(x):
#     return tf.cond(is_training, lambda: tf.nn.dropout(x, keep_prob=keep_prob), lambda: x)


# weights = {
#     'wc1': tf.get_variable('w0', shape=(5, 5, 1, 32)),
#     'wc2': tf.get_variable('w1', shape=(5, 5, 32, 64)),
#     'wc3': tf.get_variable('w2', shape=(3, 3, 64, 128)),
#     'wc4': tf.get_variable('w3', shape=(1, 1, 128, 256)),
#     'wf': tf.get_variable('wf', shape=(2*2*256, 256)),
#     'wf2': tf.get_variable('wf2', shape=(256, 200)),
#     'wf3': tf.get_variable('wf3', shape=(200, 160)),


#     'out': tf.get_variable('wout', shape=(160, n_classes)),
# }
# biases = {
#     'bc1': tf.get_variable('b0', shape=(32)),
#     'bc2': tf.get_variable('b1', shape=(64)),
#     'bc3': tf.get_variable('b2', shape=(128)),
#     'bc4': tf.get_variable('b3', shape=(256)),
#     'bf': tf.get_variable('bf', shape=(256)),
#     'bf2': tf.get_variable('bf2', shape=(200)),
#     'bf3': tf.get_variable('bf3', shape=(160)),

#     'out': tf.get_variable('bout', shape=(n_classes)),
# }


# def model(x, weights, biases):

#     # convolution layers
#     conv1 = conv_layer(x, weights['wc1'], biases['bc1'])

#     conv2 = conv_layer(conv1, weights['wc2'], biases['bc2'])

#     conv3 = conv_layer(conv2, weights['wc3'], biases['bc3'])

#     conv4 = conv_layer(conv3, weights['wc4'], biases['bc4'])

#     # flatten
#     # - Reshape conv4 output to fit fully connected layer input
#     flatten_layer = tf.reshape(
#         conv4, [-1, weights['wf'].get_shape().as_list()[0]])

#     # Fully connected layers
#     layer_1 = tf.add(tf.matmul(flatten_layer, weights['wf']), biases['bf'])
#     layer_1 = tf.nn.relu(layer_1)
#     layer_1 = tf.layers.dropout(layer_1)
# #     layer_1 = dropout(layer_1)

#     layer_2 = tf.add(tf.matmul(layer_1, weights['wf2']), biases['bf2'])
#     layer_2 = tf.nn.relu(layer_2)
#     layer_2 = tf.layers.dropout(layer_2)

#     layer_3 = tf.add(tf.matmul(layer_2, weights['wf3']), biases['bf3'])
#     layer_3 = tf.nn.relu(layer_3)
#     layer_3 = tf.layers.dropout(layer_3)


# #     layer_2 = dropout(layer_2)

#     # output layer
#     return tf.add(tf.matmul(layer_3, weights['out']), biases['out'])


# # Loss and Optimizer Nodes
# logits = model(x, weights, biases)






print(sess.run(logits,feed_dict={x:[ predict_image(image_path)]}))
