import input_data
import model
import tensorflow as tf

data = input_data.read_data_sets('MNIST', one_hot=True)

x = tf.placeholder("float", [None, 784])

sess = tf.Session()

with tf.variable_scope("regression"):
    y1, variables = model.regression(x)

saver = tf.train.Saver(variables)
saver.restore(sess, "data/regression.ckpt")

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)

saver = tf.train.Saver(variables)
saver.restore(sess, "data/convolutional.ckpt")


y_ = tf.placeholder(tf.float32, [None, 10], name='y')

print(data.test.labels[:2])

print(sess.run(y2, feed_dict={x: data.test.images[:2], keep_prob:1.0}))
print(sess.run(y1, feed_dict={x: data.test.images[:2]}))

# correct_prediction = tf.equal(tf.argmax(y2, 1), tf.argmax(y_, 1))
#
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels, keep_prob:1.0}))
#
# correct_prediction_reg = tf.equal(tf.argmax(y1, 1), tf.argmax(y_, 1))
#
# accuracy_reg = tf.reduce_mean(tf.cast(correct_prediction_reg, tf.float32))
#
# print(sess.run(accuracy_reg, feed_dict={x: data.test.images, y_: data.test.labels, keep_prob:1.0}))