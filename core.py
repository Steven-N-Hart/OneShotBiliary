import tensorflow as tf
from preprocess import format_example

anchor_path = ''
positive_path = ''
negative_path = ''

anchor,   aL = format_example(anchor_path,   label='anchor',   img_size=256)
positive, pL = format_example(positive_path, label='positive', img_size=256)
negative, nL = format_example(negative_path, label='negative', img_size=256)

image_vector = tf.concat(anchor, positive, negative)
label_vector = tf.concat(aL, pL, nL)


# Build the model
model = get_model(image_vector, label_vector, num_classes=3, base_learning_rate=0.0001)

# Training the model
model.fit(train_data, y_hat, batch_size=256, epochs=10)