import logging

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU, Dropout, GlobalMaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

from loss import triplet_loss


def convolutional_layer(filters, kernel_size):
    def _layer(x):
        x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.25)(x)
        return x

    return _layer


def build_embedding(shape, dimensions):
    inp = Input(shape=shape)
    x = inp

    # 3 Conv + MaxPooling + Relu w/ Dropout
    x = convolutional_layer(64, kernel_size=5)(x)
    x = convolutional_layer(128, kernel_size=3)(x)
    x = convolutional_layer(256, kernel_size=3)(x)

    # 1 Final Conv to get into 128 dim embedding
    x = Conv2D(dimensions, kernel_size=2, padding='same')(x)
    x = GlobalMaxPooling2D()(x)

    out = x
    model = Model(inputs=inp, outputs=out)

    return model


def build_network(img_size=256, out_dim=128):
    in_dims = (img_size, img_size, 3)

    # Create the 3 inputs
    anchor_in = tf.keras.Input(shape=in_dims, name='anchor')
    pos_in = tf.keras.Input(shape=in_dims, name='pos_img')
    neg_in = tf.keras.Input(shape=in_dims, name='neg_img')

    # Share base network with the 3 inputs
    base_network = build_embedding(in_dims, out_dim)
    anchor_out = base_network(anchor_in)
    pos_out = base_network(pos_in)
    neg_out = base_network(neg_in)

    y_pred = tf.stack([anchor_out, pos_out, neg_out], axis=1)

    # Define the trainable model
    model = tf.keras.Model(inputs=[anchor_in, pos_in, neg_in], outputs=y_pred)
    try:
        model = multi_gpu_model(model, cpu_merge=True)
        logging.info("Training using multiple GPUs..")
    except:
        logging.info("Training using single GPU or CPU..")
    model.compile(optimizer=Adam(),
                  loss=triplet_loss)

    return model
