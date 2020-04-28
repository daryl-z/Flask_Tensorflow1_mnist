import os

import input_data
import model
import tensorflow as tf

# 加载数据集
data = input_data.read_data_sets('MNIST', one_hot=True)


with tf.variable_scope("regression"):
    # 定义一个简单的线性回归
    x = tf.placeholder(tf.float32, [None, 784])
    y, variables = model.regression(x)


# 真实的标签
y_ = tf.placeholder("float", [None, 10])

# 定义交叉熵损失-分类问题
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 训练目标是最小化损失
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# y和y_的维度都是是[n, 10]，分别代表预测标签和真实标签
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver(variables)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        batch_xs, batch_ys = data.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels}))

    # 保存模型
    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),
        write_meta_graph=False, write_state=False
    )
    print("Saved:", path)




















