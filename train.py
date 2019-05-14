from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import logging, os, glob
from model import build_network
from preprocess import generate_inputs, get_epoch_size

import argparse

#class_paths = glob.glob("train/*")
#val_class_paths = glob.glob("val/*")
#tf.config.gpu.set_per_process_memory_fraction(0.95)
tf.config.gpu.set_per_process_memory_growth(True)
log_dir = "logs/"

num_epochs = 5

###############################################################################
# Input Arguments
###############################################################################

parser = argparse.ArgumentParser(description='Run a Siamese Network with a triplet loss on a folder of images.')
parser.add_argument("-t", "--image_dir_train",
                    dest='image_dir_train',
                    required=True,
                    help="File path ending in folders that are to be used for model training")

parser.add_argument("-v", "--image_dir_validation",
                    dest='image_dir_validation',
                    default=None,
                    help="File path ending in folders that are to be used for model validation")

parser.add_argument("-o", "--model_out", dest='model_out', default='model_out', help="Output file")

parser.add_argument("-p", "--patch_size",
                    dest='patch_size',
                    help="Patch size to use for training",
                    default=256, type=int)

parser.add_argument("-l", "--log_dir",
                    dest='log_dir',
                    default='log_dir',
                    help="Place to store the tensorboard logs")

parser.add_argument("-e", "--num_epocs",
                    dest='num_epochs',
                    help="Number of epochs to use for training",
                    default=10, type=int)

parser.add_argument("-V", "--verbose",
                    dest="logLevel",
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    default="INFO",
                    help="Set the logging level")

args = parser.parse_args()

if args.logLevel:
    #logging.basicConfig(level=getattr(logging, args.logLevel))
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
else:
    logging.basicConfig(level=getattr(logging, 'INFO'))


###############################################################################
# Begin actual work
###############################################################################

num_examples = get_epoch_size(args.image_dir_train, os.listdir(args.image_dir_train))

# Build the model
model = build_network()

# Write tensorboard callback function
tbCallback = tf.keras.callbacks.TensorBoard(log_dir=args.log_dir,
                                            histogram_freq=10000,
                                            write_graph=False,
                                            update_freq='batch',
                                            write_images=True)

cpCallback = tf.keras.callbacks.ModelCheckpoint(filepath='mymodel_{epoch}.h5',
                                   save_best_only=True)
# Training the model
model.fit_generator(generate_inputs(args.image_dir_train, img_size=256),
                    steps_per_epoch=num_examples,
                    epochs=args.num_epochs,
                    callbacks=[tbCallback, cpCallback],
                    validation_data=generate_inputs(args.image_dir_validation, img_size=256),
                    validation_steps=1000,
                    class_weight=None,
                    max_queue_size=5000,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=True,
                    initial_epoch=0
                    )

model.save(model_out)