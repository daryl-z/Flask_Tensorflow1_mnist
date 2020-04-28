import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

import matplotlib.pyplot as plt

from mnist import model

x = tf.placeholder("float", [None, 784])
sess = tf.Session()

with tf.variable_scope("regression"):
    y1, variables = model.regression(x)

saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/regression.ckpt")

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)

saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional.ckpt")


def regression(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


app = Flask(__name__)


@app.route('/api/mnist', methods=['post'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    print(input.shape)
    plt.imshow(input.reshape(28, 28))
    plt.show()
    output_reg = regression(input)
    output_con = convolutional(input)
    print(output_reg)
    print(output_con)
    return jsonify(results=[output_reg, output_con])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.debug = True
    app.run()






