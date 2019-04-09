import tensorflow as tf



def get_model(IMG_SHAPE, base_learning_rate = 0.0001):

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    feature_batch = base_model(image_batch)

    base_model.trainable = False
    # Add a classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)

    """"
    Apply a tf.keras.layers.Dense layer to convert these features into a single prediction per image. You don't need an
    activation function here because this prediction will be treated as a logit, or a raw prediciton value. 
    Positive numbers predict class 1, negative numbers predict class 0.
    """
    prediction_layer = keras.layers.Dense(1)

    # Now stack the feature extractor, and these two layers using a tf.keras.Sequential model:
    model = tf.keras.Sequential([
      base_model,
      global_average_layer,
      prediction_layer
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model