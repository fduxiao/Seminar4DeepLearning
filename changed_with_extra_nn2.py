from basic_framework import *
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np


def main():
    with tf.name_scope('changed_with_extra_nn2'):
        y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
        x = tf.placeholder(tf.float32, [None, 784], name='x')
        nstep = tf.Variable(0, trainable=False, name='step')
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            tf.summary.scalar('dropout_keep_probability', keep_prob)

        p, q = pickle.load(open('./static/matrix2.pkl', 'rb'))
        p = tf.constant(p, dtype=tf.float32, shape=[28, 28])
        q = tf.constant(q, dtype=tf.float32, shape=[28, 28])
        encoded = permute(x, p, q)

        with tf.name_scope('p_inverse'):
            p_inverse = weight_variable((28, 28))
            variable_summaries(p_inverse)

        with tf.name_scope('q_inverse'):
            q_inverse = weight_variable((28, 28))
            variable_summaries(q_inverse)

        decoded = permute(encoded, p_inverse, q_inverse)

        pictures = pickle.load(open('./static/pictures.pkl', 'rb'))
        for l, im in pictures.items():
            e = encode_with_p_q(im)
            d = permute(e, p_inverse, q_inverse)
            tf.summary.image('image_%d' % l, tf.reshape(d, [-1, 28, 28, 1]), max_outputs=20)

        with tf.name_scope('LeNet'):
            x_image = tf.reshape(decoded, [-1, 28, 28, 1])
            cnn1 = cnn_layer(x_image, [5, 5, 1, 32], "cnn1")
            cnn2 = cnn_layer(cnn1, [5, 5, 32, 64], "cnn2")
            cnn2_flat = tf.reshape(cnn2, [-1, 7 * 7 * 64])

            nn1 = nn_layer(cnn2_flat, 7 * 7 * 64, 1024, 'nn1')
            nn1_drop = tf.nn.dropout(nn1, keep_prob)

            nn2 = nn_layer(nn1_drop, 1024, 10, 'nn2')
            nn2_drop = tf.nn.dropout(nn2, keep_prob)

        rate = 0.001
        with tf.name_scope('train_affair'):
            with tf.name_scope('cross_entropy'):
                diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=nn2_drop)
                with tf.name_scope('total'):
                    cross_entropy = tf.reduce_mean(diff)
            tf.summary.scalar('cross_entropy', cross_entropy)

            with tf.name_scope('penalty'):
                cnn1_flat = tf.reshape(cnn1, [-1, 14*14, 32])
                _, var1 = tf.nn.moments(cnn1_flat, axes=[1])
                std1 = tf.sqrt(var1)
                penalty1 = tf.reduce_mean(tf.reshape(std1, [-1]))
                variable_summaries(penalty1)

                cnn2_flat = tf.reshape(cnn2, [-1, 7*7, 64])
                _, var2 = tf.nn.moments(cnn2_flat, axes=[1])
                std2 = tf.sqrt(var2)
                penalty2 = tf.reduce_mean(tf.reshape(std2, [-1]))
                variable_summaries(penalty2)

            loss = cross_entropy / tf.exp(penalty1)
            tf.summary.scalar('loss', loss)

            with tf.name_scope('train'):
                train_step = tf.train.AdamOptimizer(rate).minimize(loss)

            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(nn2_drop, 1))
                with tf.name_scope('accuracy'):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    sess = tf.Session()
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

    train_writer = tf.summary.FileWriter('./tensorboard/changed_with_extra_nn2', sess.graph)
    saver = tf.train.Saver()
    saver_path = './checkpoints/changed_with_extra_nn2.ckpt'
    if os.path.isfile(saver_path+'.meta'):
        saver.restore(sess, saver_path)
    else:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

    start = sess.run(nstep)
    n_max_step = 20000
    for i in range(start, n_max_step):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        if i % 500 == 0:
            sess.run(tf.assign(nstep, i))
            saver.save(sess, saver_path)
            # saver2 = tf.train.Saver()
            # saver2.save(sess, './checkpoints/changed_with_extra_nn2_%d.ckpt' % i)
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        train_writer.add_summary(summary, i)

    sess.run(tf.assign(nstep, n_max_step))
    saver.save(sess, saver_path)
    print("test accuracy %g" % sess.run(accuracy, feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    main()
