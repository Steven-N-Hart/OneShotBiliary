def lossless_triplet_loss(y_true, y_pred, N=3, beta=N, epsilon=1e-8):
    """
    Implementation of the triplet loss function
    From https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
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
    anchor = tf.convert_to_tensor(y_pred[:, 0:N])
    positive = tf.convert_to_tensor(y_pred[:, N:N * 2])
    negative = tf.convert_to_tensor(y_pred[:, N * 2:N * 3])

    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    # Non Linear Values

    # -ln(-x/N+1)
    pos_dist = -tf.log(-tf.divide((pos_dist), beta) + 1 + epsilon)
    neg_dist = -tf.log(-tf.divide((N - neg_dist), beta) + 1 + epsilon)

    # compute loss
    loss = neg_dist + pos_dist

    return loss