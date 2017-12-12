from basic_framework import *
from tensorflow.examples.tutorials.mnist import input_data
import os


def main():
    with tf.name_scope('changed_with_LeNet'):
        y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
        x = tf.placeholder(tf.float32, [None, 784], name='x')
        nstep = tf.Variable(0, trainable=False, name='step')
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            tf.summary.scalar('dropout_keep_probability', keep_prob)

        encoded_x = encode_with_p_q(x)
        main_net = le_net(encoded_x, keep_prob, 'LeNet')
        train_step, accuracy = train_affair(y_, main_net, 'train_affair')

    merged = tf.summary.merge_all()

    sess = tf.Session()
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

    train_writer = tf.summary.FileWriter('./tensorboard/changed_with_LeNet', sess.graph)
    saver = tf.train.Saver()
    saver_path = './checkpoints/changed_with_LeNet.ckpt'
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
            # saver2.save(sess, './checkpoints/changed_with_LeNet_%d.ckpt' % i)
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        train_writer.add_summary(summary, i)

    sess.run(tf.assign(nstep, n_max_step))
    saver.save(sess, saver_path)
    print("test accuracy %g" % sess.run(accuracy, feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    main()
