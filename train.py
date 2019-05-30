from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging

import tensorflow as tf

from model import build_network
from preprocess import GetFiles, format_example

tf.config.gpu.set_per_process_memory_growth(True)

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

parser.add_argument("-L", "--learning-rate",
                    dest='lr',
                    help="Learning rate",
                    default=0.000001, type=float)

parser.add_argument("-e", "--num_epocs",
                    dest='num_epochs',
                    help="Number of epochs to use for training",
                    default=10, type=int)

parser.add_argument("-b", "--batch-size",
                    dest='BATCH_SIZE',
                    help="Number of batches to use for training",
                    default=1, type=int)

parser.add_argument("-w", "--num-workers",
                    dest='NUM_WORKERS',
                    help="Number of workers to use for training",
                    default=10, type=int)

parser.add_argument("-V", "--verbose",
                    dest="logLevel",
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    default="INFO",
                    help="Set the logging level")

args = parser.parse_args()

if args.logLevel:
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
else:
    logging.basicConfig(level=getattr(logging, 'INFO'))

###############################################################################
# Begin actual work
###############################################################################

BATCH_SIZE = args.BATCH_SIZE
NUM_WORKERS = args.NUM_WORKERS

# Build the model
model = build_network(lr=args.lr)


def generator():
    for s1, s2, s3 in zip(ds_a, ds_p, ds_n):
        yield {"anchor": format_example(s1, img_size=args.patch_size), "pos_img": format_example(s2, img_size=args.patch_size), "neg_img": format_example(s3, img_size=args.patch_size)}, [1, 1, 0]


def vgenerator():
    for s1, s2, s3 in zip(vds_a, vds_p, vds_n):
        yield {"anchor": format_example(s1, img_size=args.patch_size), "pos_img": format_example(s2, img_size=args.patch_size), "neg_img": format_example(s3, img_size=args.patch_size)}, [1, 1, 0]


# Prepare training dataset
file_list = GetFiles(args.image_dir_train)
ds_a = tf.data.Dataset.from_tensor_slices(file_list.class_files['anchor'])
ds_p = tf.data.Dataset.from_tensor_slices(file_list.class_files['positive'])
ds_n = tf.data.Dataset.from_tensor_slices(file_list.class_files['negative'])

train_dataset = tf.data.Dataset.from_generator(generator, output_types=(
    {"anchor": tf.float32, "pos_img": tf.float32, "neg_img": tf.float32}, tf.int64))
train_dataset = train_dataset.batch(args.BATCH_SIZE).repeat()

# Prepare validation dataset
if args.image_dir_validation is None:
    val_dataset = None
    validation_steps = None
else:
    val_list = GetFiles(args.image_dir_validation)
    vds_a = tf.data.Dataset.from_tensor_slices(val_list.class_files['anchor'])
    vds_p = tf.data.Dataset.from_tensor_slices(val_list.class_files['positive'])
    vds_n = tf.data.Dataset.from_tensor_slices(val_list.class_files['negative'])

    val_dataset = tf.data.Dataset.from_generator(vgenerator, output_types=(
        {"anchor": tf.float32, "pos_img": tf.float32, "neg_img": tf.float32}, tf.int64))
    val_dataset = val_dataset.batch(args.BATCH_SIZE).repeat()
    validation_steps = 1000

# Write tensorboard callback function
tbCallback = tf.keras.callbacks.TensorBoard(log_dir=args.log_dir + '_' + str(args.lr),
                                            histogram_freq=10000,
                                            write_graph=False,
                                            update_freq='batch',
                                            write_images=True)

cpCallback = tf.keras.callbacks.ModelCheckpoint(filepath='mymodel_{epoch}.h5',
                                                save_best_only=True)

esCallback = tf.keras.callbacks.EarlyStopping(monitor='batch_loss', patience=3)

# Training the model
model.fit(train_dataset,
          steps_per_epoch=file_list.num_images / args.BATCH_SIZE,
          epochs=args.num_epochs,
          callbacks=[tbCallback, cpCallback, esCallback],
          validation_data=val_dataset,
          validation_steps=1000,
          class_weight=None,
          max_queue_size=file_list.num_images,
          workers=NUM_WORKERS,
          use_multiprocessing=False,
          shuffle=True,
          initial_epoch=0
          )

model.save(args.model_out)
