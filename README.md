# One shot learning for Biliary Cytology

Works on TF2.0a

The objective is to write the simplest form of a Siamese network with triplet loss to minimize the difference between 
similar input images and maximize the difference between different images.

## How it should work
Assume that images are stored in a directory structure like this:
```
train/
    |-- Group1/
        |-- image1.jpg
        |-- image2.jpg
        |-- imageN.jpg
    |-- Group2/
        |-- image1.jpg
        |-- image2.jpg
        |-- imageN.jpg
    |-- GroupN/
        |-- image1.jpg
        |-- image2.jpg
        |-- imageN.jpg
val/
    |-- same as train
test/
    |-- same as train
```

Run `python train.py` with the following arguments:
  * `--image_dir_train=train` to know where the training directory containing the images is
  * `--image_dir_validation` to know where the validation directory containing the images is

The model should begin training, frequently reporting the loss function and providing examples in TensorBoard.

## Requirements include:
  * Working on TensorFlow 2.0alpha
  * Multiple GPU training
  * Multiple processors
  * Logging the loss to TensorBoard
  * Viewing example Anchor, positive, negative image patches and their losses in TensorBoard