import tensorflow as tf


def format_example(image, label, img_size = 256):
    """
    Apply any image preprocessing here
    invoke as `train = image_tensor.map(format_example)`
    :param image:
    :param label:
    :param img_size:
    :return:
    """
    image = tf.cast(image, tf.float32)
    image = (image / 255.0)
    image = tf.image.resize(image, (img_size, img_size))
    return image, label