#!/usr/bin/env python3
from flask import *
from basic_framework import *
import numpy as np
app = Flask(__name__)

with tf.name_scope('normal'):
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        tf.summary.scalar('dropout_keep_probability', keep_prob)

    main_net = le_net(x, keep_prob, 'LeNet')
    max_arg = tf.arg_max(main_net, 1)

sess = tf.Session()
saver_path = './checkpoints/normal/normal.ckpt'
saver = tf.train.Saver()
saver.restore(sess, saver_path)


def recognize_image(input_image):
    k = sess.run(max_arg, feed_dict={x: input_image, keep_prob: 1})
    return k


@app.route('/')
def index():
    return render_template_string(open('hand_write.html').read())


@app.route('/recognize', methods=['POST'])
def recognize():
    input_image = request.get_json()
    arr = np.array(input_image)
    return jsonify(recognize_image(arr.reshape([-1, 28*28])).tolist())


if __name__ == '__main__':
    app.run(debug=True)
