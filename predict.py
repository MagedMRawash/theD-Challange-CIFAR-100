from data.preparation import predict_image
import tensorflow as tf
import argparse
from sklearn.externals.joblib import load

image_path = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', dest="image")
    args = parser.parse_args()
    image_path = args.image

onehot = load('./checkpoint/onehot_encoder.joblib')
print('image_path', image_path)

# sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./checkpoint/model.ckpt.meta')


graph = tf.get_default_graph()
# declear dynamic data placeholder


#Now, access the op that you want to run. 
input_image = [predict_image(image_path)]
with tf.Session() as sess:
    saver.restore(sess,tf.train.latest_checkpoint('./checkpoint'))
    x = graph.get_tensor_by_name("x:0")
    # y = graph.get_tensor_by_name("y:0")

    logits = graph.get_operation_by_name("predictions")
    
    result = sess.run(logits,feed_dict={x: input_image }) 
    onehot.inverse_transform(result)
    print( result )
