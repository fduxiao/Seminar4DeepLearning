import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data


matrix_p = np.random.normal(size=(28, 28))
matrix_q = np.random.normal(size=(28, 28))

assert 0 != np.linalg.det(matrix_p)
assert 0 != np.linalg.det(matrix_p)
pickle.dump((matrix_p, matrix_q), open('./static/matrix.pkl', 'wb'))


mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

pictures = {}
for i, l in zip(*mnist.test.next_batch(100)):
    n = l.argmax()
    if n in pictures:
        continue
    pictures[n] = i

assert len(pictures) == 10
pickle.dump(pictures, open('./static/pictures.pkl', 'wb'))
