import tensorflow as tf

from model import build_network
from preprocess import generate_inputs

class_paths = ['/Users/m087494/Desktop/Example_Images/MELF/',
               '/Users/m087494/Desktop/Example_Images/Cytology/']
num_epochs = 5
log_dir = "/Users/m087494/Desktop/tb"

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
model.fit_generator(generate_inputs(class_paths, img_size=256),
                    steps_per_epoch=1,
                    epochs=num_epochs,
                    callbacks=[tbCallback],
                    validation_data=None,
                    validation_steps=None,
                    class_weight=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=True,
                    initial_epoch=0
                    )
