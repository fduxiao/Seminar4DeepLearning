from basic_framework import *
from tensorflow.examples.tutorials.mnist import input_data
import os


def main():
    with tf.name_scope('refine'):
        y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
        x = tf.placeholder(tf.float32, [None, 784], name='x')
        nstep = tf.Variable(0, trainable=False, name='step')
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            tf.summary.scalar('dropout_keep_probability', keep_prob)

        encoded = encode_with_p_q(x)

        with tf.name_scope('p_inverse'):
            p_inverse = weight_variable((28, 28))
            variable_summaries(p_inverse)
            tf.summary.histogram('matrix', p_inverse)

        with tf.name_scope('q_inverse'):
            q_inverse = weight_variable((28, 28))
            variable_summaries(q_inverse)
            tf.summary.histogram('matrix', q_inverse)

        decoded = permute(encoded, p_inverse, q_inverse)

        pictures = pickle.load(open('./static/pictures.pkl', 'rb'))
        for l, im in pictures.items():
            e = encode_with_p_q(im)
            d = permute(e, p_inverse, q_inverse)
            tf.summary.image('image_%d' % l, tf.reshape(d * 256, [-1, 28, 28, 1]), max_outputs=10)

        main_net = le_net(decoded, keep_prob, 'LeNet')
        train_step, accuracy = train_affair(y_, main_net, 'train_affair')

    merged = tf.summary.merge_all()

    sess = tf.Session()
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

    train_writer = tf.summary.FileWriter('./tensorboard/refine', sess.graph)
    saver = tf.train.Saver()
    saver_path = './checkpoints/refine.ckpt'
    if os.path.isfile(saver_path+'.meta'):
        saver.restore(sess, './checkpoints/normal.ckpt')
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
            # saver2.save(sess, './checkpoints/refine_%d.ckpt' % i)
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        train_writer.add_summary(summary, i)

    sess.run(tf.assign(nstep, n_max_step))
    saver.save(sess, saver_path)
    print("test accuracy %g" % sess.run(accuracy, feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    main()
