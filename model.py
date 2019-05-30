import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU, Dropout, GlobalMaxPooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

from loss import lossless_triplet_loss as triplet_loss


def build_network(img_size=256, out_dim=128, lr=0.000001):
    in_dims = [img_size, img_size, 3]
    initial_input = Input(shape=in_dims)
    # ################################################################3
    # 3 Conv + MaxPooling + Relu w/ Dropout
    x = Conv2D(64, kernel_size=5, padding='same')(initial_input)
    x = MaxPooling2D(pool_size=2)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, kernel_size=3, padding='same')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, kernel_size=3, padding='same')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.25)(x)

    # 1 Final Conv to get into 128 dim embedding
    x = Conv2D(out_dim, kernel_size=2, padding='same')(x)
    x = GlobalMaxPooling2D()(x)

    # out = Flatten()(x)
    out = Dense(1, activation='sigmoid')(x)
    conv_model = Model(initial_input, out)
    # ################################################################3

    # Create the 3 inputs
    anchor_in = Input(shape=in_dims, name='anchor')
    pos_in = Input(shape=in_dims, name='pos_img')
    neg_in = Input(shape=in_dims, name='neg_img')

    # Share base network with the 3 inputs
    anchor_out = conv_model(anchor_in)
    pos_out = conv_model(pos_in)
    neg_out = conv_model(neg_in)

    y_pred = tf.keras.layers.concatenate([anchor_out, pos_out, neg_out])
    # Define the trainable model
    model = Model(inputs=[{'anchor': anchor_in,
                           'pos_img': pos_in,
                           'neg_img': neg_in}], outputs=y_pred)

    try:
        model = multi_gpu_model(model, gpus=3)
        print("Training using multiple GPUs..")
    except:
        print("Training using single GPU or CPU..")
    model.compile(optimizer=Adam(lr=lr),
                  loss=triplet_loss)

    return model
