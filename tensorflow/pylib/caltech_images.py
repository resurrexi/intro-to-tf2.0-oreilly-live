#!/usr/bin/env python3

"""Module for programmatic loading of the Caltech 101 image data set
from a local directory. Information about the data set available at
http://www.vision.caltech.edu/Image_Datasets/Caltech101/."""

import os
import tensorflow as tf

class Caltech101(object):
    def __init__(self, path, testing=False):
        """Uses path to root directory of data set to initialize object.

        Data set is expected to be stored as a collection of
        directories rooted at path. The name of each subdirectory is
        one of the 101 possible image categories. The names of the
        subdirectories are used to construct labels when images are
        loaded."""

        self.path = os.path.expanduser(path)
        self.categories = [category for category in os.listdir(self.path)]
        self.full_paths = [os.path.join(self.path, category, fn)
                           for category in self.categories
                           for fn in os.listdir(os.path.join(self.path, category))]
        self.N_IMAGES = len(self.full_paths)
        self.N_CLASSES = len(self.categories)
        self._testing = testing

    def __repr__(self):
        return 'caltech101(%s)' % self.path

    def _parse_image(self, filename, label, imsize=(128, 128)):
        image = tf.image.decode_jpeg(tf.io.read_file(filename))
        X = tf.image.resize_with_crop_or_pad(image,
                                             imsize[0],
                                             imsize[1])
        return tf.cast(X, tf.float32), label

    def generate_dataset(self):
        names = [i.split('/')[-2] for i in self.full_paths]
        un_names = list(set(names))
        mapping = {name:i for i,name in enumerate(sorted(un_names))}
        labels = [tf.one_hot(mapping[i], len(un_names)) for i in names]
        ds = (tf.data.Dataset.from_tensor_slices(
            (self.full_paths, labels))
                .shuffle(self.N_IMAGES//10)
                .map(self._parse_image)
                .filter(lambda x,y : tf.shape(x)[2] > 1)
               )
        return ds if not self._testing else ds.take(40)
