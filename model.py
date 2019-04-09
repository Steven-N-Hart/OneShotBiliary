import tensorflow as tf
from loss import lossless_triplet_loss

def get_model(image_vector, label_vector, num_classes=3, base_learning_rate=0.0001):
    """

    :param image_vector:
    :param label_vector:
    :param num_classes:
    :param base_learning_rate:
    :return:
    """

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.DenseNet169(input_shape=image_vector.shape,
                                                   include_top=False,
                                                   weights='imagenet',
                                                   classes=num_classes)


    base_model.trainable = False

    # Add a classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    """"
    Apply a tf.keras.layers.Dense layer to convert these features into a set of predictions.  
    Ex. tumor, normal, and uninformative
    """
    prediction_layer = keras.layers.Dense(num_classes)

    # Now stack the feature extractor, and these two layers using a tf.keras.Sequential model:
    model = tf.keras.Sequential([
      base_model,
      global_average_layer,
      prediction_layer
    ])

    # set the loss function
    loss = lossless_triplet_loss(label_vector, image_vector, N=3, beta=N, epsilon=1e-8)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss=loss,
                  metrics=['accuracy'])

    return model