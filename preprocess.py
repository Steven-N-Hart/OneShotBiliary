import tensorflow as tf
from tensorflow.keras.utils import Sequence
import math
import random
import numpy as np
import os


class Preprocess(Sequence):

    def __init__(self, pos_set, neg_set, batch_size=25, img_size=256):
        """
        Required for Keras

        :param pos_set: The file path to the malignant images
        :param neg_set: The file path to the benign images
        :param batch_size: How large the batch size should be
        """
        self.pos_set, self.neg_set = pos_set, neg_set
        self.batch_size = batch_size
        self.img_size = img_size
        # Get a filename from each class path
        self.anchor_in = self._get_random_file_list(self.pos_set)
        self.pos_in = self._get_random_file_list(self.pos_set)
        self.neg_in = self._get_random_file_list(self.neg_set)

    def __len__(self):
        """
        Required for Keras
        :return:
        """
        return math.ceil(len(self.pos_set) / self.batch_size)

    @staticmethod
    def _get_random_file_list(directory):
        i = [os.path.join(directory, x) for x in os.listdir(directory)]
        random.shuffle(i)
        print('Number of images {}'.format(i.__len__()))
        return i

    def _format_example(self, image_path=None):
        """
        Apply any image preprocessing here

        :param image_path: file path of image that contains all the images
        :return:
        """
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image)
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, (self.img_size, self.img_size, 3))
        return image

    def __getitem__(self, idx):
        """
        Required for Keras
        :param idx:
        :return:
        """

        anchor_in = self.anchor_in[idx * self.batch_size:(idx + 1) *
                                                         self.batch_size]
        pos_in = self.pos_in[idx * self.batch_size:(idx + 1) *
                                                   self.batch_size]
        neg_in = self.neg_in[idx * self.batch_size:(idx + 1) *
                                                   self.batch_size]

        print('getting item number {} for {} entries'.format(idx, anchor_in.__len__()))

        a = [self._format_example(image_path=f) for f in anchor_in]
        print('Completed {} for anchor'.format(anchor_in.__len__()))
        b = [self._format_example(image_path=g) for g in pos_in]
        print('Completed {} for pos_in'.format(pos_in.__len__()))
        c = [self._format_example(image_path=h) for h in neg_in]
        print('Completed {} for neg_in'.format(neg_in.__len__()))
        d = [1, 1, 0] * anchor_in.__len__()
        return [[a, b, c], d]
