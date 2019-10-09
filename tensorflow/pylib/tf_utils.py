#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from pylib import mnist_dataset

def _get_mnist(set_type):
    data = getattr(mnist_dataset, set_type)('/tmp/data{}'.format(set_type))
    iterator_full = data.batch(1000)
    x, y = zip(*[(x_t, y_t) for x_t, y_t in iterator_full])

    return np.vstack(x), np.hstack(y)

def mnist_train():
    return _get_mnist('train')

def mnist_test():
    return _get_mnist('test')
