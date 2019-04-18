import os
from random import sample, choice

import tensorflow as tf


def format_example(image_path=None, image_name=None, img_size=256):
    """
    Apply any image preprocessing here
    :param image_path: file path of image that contains all the images
    :param image_name: the specific filename of the image
    :param img_size: size that images should be reshaped to
    :return:
    """
    image = tf.io.read_file(os.path.join(image_path, image_name))
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = (image / 255.0)
    image = tf.image.resize(image, (img_size, img_size))
    image = tf.reshape(image, (1, img_size, img_size, 3))
    return image


def get_epoch_size(base_directory, class_paths):
    """
    Given an array of directories, count the number of events in each one
    :param base_directory: root directory of images
    :param class_paths: list of directories that contain images
    :return: number of images
    """
    val = 0
    for i in class_paths:
        _target_dir = os.path.join(base_directory, i)
        val += os.listdir(_target_dir).__len__()
    return val

def generate_inputs(class_paths, img_size=256):
    """
    Randomly return inputs ready for classification
    :param class_paths: A list of file paths where each subfolder contains images of that class
    :param img_size: an integer to use for h & w of image
    :return: dict of inputs, targets
    """

    num_classes = class_paths.__len__()

    while 1:
        anchor_class = sample(range(num_classes), 1)[0]

        # select an example from another class
        other_class = anchor_class
        while other_class == anchor_class:
            other_class = sample(range(num_classes), 1)[0]

        # Get a filename from each class path
        anchor_in = choice(os.listdir(class_paths[anchor_class]))
        pos_in = anchor_in
        # Make sure you don't have the same image
        while anchor_in == pos_in:
            pos_in = choice(os.listdir(class_paths[anchor_class]))

        neg_in = choice(os.listdir(class_paths[other_class]))
        # Now read in the images
        anchor_in = format_example(image_path=class_paths[anchor_class], image_name=anchor_in, img_size=img_size)
        pos_in = format_example(image_path=class_paths[anchor_class], image_name=pos_in, img_size=img_size)
        neg_in = format_example(image_path=class_paths[other_class], image_name=neg_in, img_size=img_size)

        out = [anchor_class, anchor_class, other_class]
        yield ([anchor_in, pos_in, neg_in], out)
