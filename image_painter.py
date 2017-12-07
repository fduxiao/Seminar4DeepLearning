from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)


def draw(input_array):
    input_array = input_array.reshape([-1, 28, 28])
    return [Image.fromarray(a*256) for a in input_array]


def draw_permuted():
    pass
