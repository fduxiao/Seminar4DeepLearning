import tensorflow as tf
import pickle


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, w, name=None):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', name=name)


def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


def cnn_layer(input_tensor, conv_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('conv_kernel'):
            weights = weight_variable(conv_dim)
            variable_summaries(weights)
        with tf.name_scope('conv_biases'):
            biases = bias_variable([conv_dim[-1]])
            variable_summaries(biases)
        with tf.name_scope('conv_Wx_plus_b'):
            conv_preactivate = tf.nn.relu(conv2d(input_tensor, weights) + biases)
            tf.summary.histogram('pre_activations', conv_preactivate)
        activations = act(conv_preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        pooled = max_pool_2x2(activations, name='pooled')
        tf.summary.histogram('pooled', pooled)
        return pooled


def le_net(input_tensor, keep_prob, net_name):
    with tf.name_scope(net_name):
        x_image = tf.reshape(input_tensor, [-1, 28, 28, 1])
        cnn1 = cnn_layer(x_image, [5, 5, 1, 32], "cnn1")
        cnn2 = cnn_layer(cnn1, [5, 5, 32, 64], "cnn2")
        cnn2_flat = tf.reshape(cnn2, [-1, 7 * 7 * 64])

        nn1 = nn_layer(cnn2_flat, 7 * 7 * 64, 1024, 'nn1')
        nn1_drop = tf.nn.dropout(nn1, keep_prob)

        nn2 = nn_layer(nn1_drop, 1024, 10, 'nn2')
        nn2_drop = tf.nn.dropout(nn2, keep_prob)
    return nn2_drop


def train_affair(labels, logits, name):
    with tf.name_scope(name):
        with tf.name_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    return train_step, accuracy


def permute(input_tensor, p, q, name='permutation'):
    with tf.name_scope(name):
        t = tf.reshape(input_tensor, [-1, 28])
        t = tf.matmul(t, q)
        t = tf.reshape(t, [-1, 28, 28])
        t = tf.map_fn(lambda x: tf.matmul(p, x), t)
        t = tf.reshape(t, [-1, 28 * 28])
        tf.summary.histogram('permuted', t)
    return t


matrix_p = None
matrix_q = None


def encode_with_p_q(input_tensor, name='encoded'):
    with tf.name_scope(name):
        global matrix_q
        global matrix_p
        if matrix_p is None or matrix_q is None:
            matrix_p, matrix_q = pickle.load(open('./static/matrix.pkl', 'rb'))
        p = tf.constant(matrix_p, dtype=tf.float32, shape=[28, 28])
        q = tf.constant(matrix_q, dtype=tf.float32, shape=[28, 28])
        permuted = permute(input_tensor, p, q)
    return permuted
