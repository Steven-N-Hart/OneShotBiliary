from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from model import build_network
from preprocess import generate_inputs, get_epoch_size

class_paths = ['/people/m087494/OneShotBiliary/data/positive',
               '/people/m087494/OneShotBiliary/data/negative']
log_dir = "/people/m087494/OneShotBiliary/logs"

num_epochs = 5
num_examples = get_epoch_size(class_paths)

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
                                            write_grads=False,
                                            write_images=True)

cpCallback = tf.keras.callbacks.ModelCheckpoint(filepath='mymodel_{epoch}.h5',
                                                save_best_only=True,
                                                verbose=1,
                                                monitor='mae'
)

# Training the model
model.fit_generator(generate_inputs(class_paths, img_size=256),
                    steps_per_epoch=num_examples,
                    epochs=num_epochs,
                    callbacks=[tbCallback, cpCallback],
                    validation_data=None,
                    validation_steps=None,
                    class_weight=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=True,
                    initial_epoch=0
                    )
