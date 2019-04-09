import tensorflow as tf


def format_example(image_path, label, img_size = 256):
    """
    Apply any image preprocessing here
    invoke as `train = imagelist.map(format_example)`
    :param image_path:
    :param label:
    :param img_size:
    :return:
    """
    img_raw = tf.io.read_file(img_path)
    image = tf.cast(image, tf.float32)
    image = (image / 255.0)
    image = tf.image.resize(image, (img_size, img_size))
    return image, label