from basic_framework import *
from tensorflow.examples.tutorials.mnist import input_data
import os
import pickle


def main():
    with tf.name_scope('inverse_image'):
        y_ = tf.placeholder(tf.float32, [None, 784], name='y_')
        x = tf.placeholder(tf.float32, [None, 784], name='x')
        nstep = tf.Variable(0, trainable=False, name='step')
        tf.summary.scalar('step', nstep)
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        encoded = encode_with_p_q(x)
        p_inverse = weight_variable((28, 28), 'p_inverse')
        q_inverse = weight_variable((28, 28), 'q_inverse')
        decoded = permute(encoded, p_inverse, q_inverse)

        pictures = pickle.load(open('./static/pictures.pkl', 'rb'))
        for l, im in pictures.items():
            e = encode_with_p_q(im)
            d = permute(e, p_inverse, q_inverse)
            tf.summary.image('image_%d' % l, tf.reshape(d*256, [-1, 28, 28, 1]), max_outputs=10)
        loss = tf.losses.mean_squared_error(y_, decoded)
        tf.summary.scalar('loss', loss)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    merged = tf.summary.merge_all()

    sess = tf.Session()
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

    train_writer = tf.summary.FileWriter('./tensorboard/inverse_image' + '/train', sess.graph)
    saver = tf.train.Saver()
    saver_path = './checkpoints/inverse_image.ckpt'
    if os.path.isfile(saver_path+'.meta'):
        saver.restore(sess, saver_path)
    else:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

    start = sess.run(nstep)
    n_max_step = 20000
    for i in range(start, n_max_step):
        sess.run(tf.assign(nstep, i))
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_loss = sess.run(loss, feed_dict={
                x: batch[0], y_: batch[0], keep_prob: 1.0})
            print("step %d, training loss %g" % (i, train_loss))

        if i % 500 == 0:
            saver.save(sess, saver_path)
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[0], keep_prob: 0.5})
        train_writer.add_summary(summary, i)

    sess.run(tf.assign(nstep, n_max_step))
    saver.save(sess, saver_path)
    print("test loss %g" % sess.run(loss, feed_dict={
        x: mnist.test.images, y_: mnist.test.images, keep_prob: 1.0}))


if __name__ == '__main__':
    main()
