from random import shuffle
import tensorflow as tf
import os


class GetFiles:

    def __init__(self, directory_path):
        """
        Return a randomized list of each directory's contents

        :param directory_path: a directory that contains sub-folders of images

        :returns class_files: a dict of each file in each folder
        """
        super().__init__()
        self.directory_path = directory_path
        self.class_files = self.__get_list()
        self.num_images = self.class_files['anchor'].__sizeof__()

    def __get_list(self):
        class_files = dict()
        classes = os.listdir(self.directory_path)

        for x in classes:
            class_files[x] = []
            for y in os.listdir(os.path.join(self.directory_path, x)):
                class_files[x].append(os.path.join(self.directory_path, x, y))

            i = class_files[x]
            shuffle(i)
            class_files[x] = i

            if x == 'positive' or x == 'pos':
                j = i.copy()
                shuffle(j)
                class_files['anchor'] = j

        return class_files


def format_example(image_name=None, img_size=256):
    """
    Apply any image preprocessing here
    :param image_name: the specific filename of the image
    :param img_size: size that images should be reshaped to
    :return:
    """
    image = tf.io.read_file(image_name)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (img_size, img_size))
    image = tf.reshape(image, (img_size, img_size, 3))
    return image

