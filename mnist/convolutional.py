import os

import input_data
import model
import tensorflow as tf

data = input_data.read_data_sets('MNIST', one_hot=True)

with tf.variable_scope("convolutional"):
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    keep_prob = tf.placeholder(tf.float32)
    y, variables = model.convolutional(x, keep_prob)

# 参数解释见regression.py
y_ = tf.placeholder(tf.float32, [None, 10], name='y')
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)

with tf.Session() as sess:
    merged_summary_op = tf.summary.merge_all()
    summary_write = tf.summary.FileWriter('mnist_log/1', sess.graph)
    summary_write.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(3000):
        batch = data.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        # 在训练时可以dropout，但是在做预测的时候不可以
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})

    print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels, keep_prob:1.0}))

    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'convolutional.ckpt'),
        write_meta_graph=False, write_state=False
    )
    print("Saved:", path)




















