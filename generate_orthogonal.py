import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data


def rvs(dim=3):
    random_state = np.random
    h = np.eye(dim)
    d = np.ones((dim,))
    for nn in range(1, dim):
        x = random_state.normal(size=(dim - nn + 1,))
        d[nn - 1] = np.sign(x[0])
        x[0] -= d[nn - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        hs = (np.eye(dim - nn + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[nn - 1:, nn - 1:] = hs
        h = np.dot(h, mat)
        # Fix the last sign such that the determinant is 1
    d[-1] = (-1) ** (1 - (dim % 2)) * d.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    h = (d * h.T).T
    return h


matrix_p = rvs(28)
matrix_q = np.linalg.inv(matrix_p)

pickle.dump((matrix_p, matrix_q), open('./static/matrix2.pkl', 'wb'))


mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

pictures = {}
for i, l in zip(*mnist.test.next_batch(100)):
    n = l.argmax()
    if n in pictures:
        continue
    pictures[n] = i

assert len(pictures) == 10
pickle.dump(pictures, open('./static/pictures2.pkl', 'wb'))
