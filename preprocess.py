import os
from random import sample, choice

import tensorflow as tf


def format_example(image_path=None, image_name=None, img_size=256):
    """
    Apply any image preprocessing here
    invoke as `train = imagelist.map(format_example)`
    :param image_path:
    :param image_name:
    :param img_size:
    :return:
    """
    image = tf.io.read_file(os.path.join(image_path, image_name))
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = (image / 255.0)
    image = tf.image.resize(image, (img_size, img_size))
    image = tf.reshape(image, (1, img_size, img_size, 3))
    return image


def get_epoch_size(class_paths):
    val = 0
    for i in class_paths:
        val += os.listdir(i).__len__()
    return val

def generate_inputs(class_paths, img_size=256):
    """
    Randomly return inputs ready for classification
    :param class_paths: A list of file paths where each subfolder contains pngs of that class
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
