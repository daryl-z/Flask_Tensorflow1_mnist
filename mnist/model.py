import tensorflow as tf


def regression(x):
    # 线性回归
    W = tf.Variable(tf.zeros([784, 10]), name="W")
    b = tf.Variable(tf.zeros([10]), name="b")
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    return y, [W, b]


def convolutional(x, keep_prob):

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, [1,1,1,1], padding='SAME')

    def max_pool_2x2(x):
        # 最大池化，窗口大小2x2
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 加载的数据集维度是[-1, 784]的格式
    x_image = tf.reshape(x, [-1, 28, 28, 1]) # 标准的图片数据格式[数量， 宽度， 高度， 通道]
    # 卷积核
    W_conv1 = weight_variable([5, 5, 1, 32]) # 核大小为5x5，输入通道为1，这个必须和数据集的通道一致，输出通道为32
    # 权重
    b_conv1 = bias_variable([32])
    # 激活函数
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # 池化
    h_pool1 = max_pool_2x2(h_conv1) # [-1, 14, 14, 32]，卷积是使用"SAME"填充，不改变尺寸，池化缩小一半

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)  # [-1, 7, 7, 64]

    # 全连接层
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # [-1, 1024]

    # dropout, keep_prob是每个元素保留下来的概率
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # [-1, 10]

    return y, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]



















