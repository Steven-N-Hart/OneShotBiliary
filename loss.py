import tensorflow as tf


def triplet_loss(y_true, y_pred):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, not used in this function.
    y_pred -- python list containing three objects:
            anchor:   the encodings for the anchor data
            positive: the encodings for the positive data (similar to anchor)
            negative: the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    alpha = 0.2

    anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)

    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))

    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha

    loss = tf.maximum(basic_loss, 0.0)

    return loss


def lossless_triplet_loss(y_true, y_pred, n=3, beta=3, epsilon=1e-8):
    """
    Implementation of the triplet loss function

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    N  --  The number of dimension
    beta -- The scaling factor, N is recommended
    epsilon -- The Epsilon value to prevent ln(0)


    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)
    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    #print('\n\tLinear distances for pos_dist and neg_dist are {} and {}'.format(pos_dist,neg_dist ))
    # Non Linear Values

    # -ln(-x/N+1)
    pos_dist = -tf.math.log(-tf.divide(pos_dist, beta) + 1 + epsilon)
    neg_dist = -tf.math.log(-tf.divide((n - neg_dist), beta) + 1 + epsilon)
    #print('\tNon-Linear distances for pos_dist and neg_dist are {} and {}\n'.format(pos_dist,neg_dist))

    # compute loss
    loss = neg_dist + pos_dist

    return loss
