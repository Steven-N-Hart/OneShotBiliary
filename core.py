import tensorflow as tf

from model import build_network
from preprocess import generate_inputs, get_epoch_size

class_paths = ['/people/m087494/OneShotBiliary/data/positive',
               '/people/m087494/OneShotBiliary/data/negative']
log_dir = "/people/m087494/OneShotBiliary/logs"
batch_size = 20
num_epochs = 5


steps_per_epoch = get_epoch_size(class_paths) / batch_size

# Build the model
model = build_network()

# tf.keras.utils.plot_model(
#    model,
#    to_file='/Users/m087494/Desktop/model.png',
#    show_shapes=True)
# model.summary()

# Write tensorboard callback function
tbCallback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                            histogram_freq=50,
                                            write_graph=True,
                                            write_grads=True,
                                            write_images=True)


# Training the model
def _sample_images():
    items = generate_inputs(class_paths, img_size=256)
    yield items


ds = tf.data.Dataset.from_generator(_sample_images(),
                                    (tf.float32, tf.float32))

model.fit(ds,
          batch_size=batch_size,
          num_epochs=num_epochs,
          verbose=1,
          shuffle=True)

