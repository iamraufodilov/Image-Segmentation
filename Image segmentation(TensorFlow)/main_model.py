# load libraries
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_datasets as tfds

from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os

# load dataset
dataset = "G:/rauf/STEPBYSTEP/Data/oxford_pets"

image_path = "G:/rauf/STEPBYSTEP/Data/oxford_pets/images"
annotation_path = "G:/rauf/STEPBYSTEP/Data/oxford_pets/annotations/trimaps"

#
input_img_path = sorted([
    os.path.join(image_path, fname)
    for fname in os.listdir(image_path)
    if fname.endswith(".jpg")
    ])

input_annotation_path = ([
    os.path.join(annotation_path, fname)
    for fname in os.listdir(annotation_path)
    if fname.endswith(".png") and not fname.startswith(".")
    ])

# data preprocessing
def normalize(input_images, input_masks):
    input_images = tf.cast(input_images, tf.float32)/255.0
    input_masks -= 1
    return input_images, input_masks

def load_image(dadapoint):
    input_image = tf.image.resize(datapoint['images'], (128, 128))
    input_mask = tf.image.resize(datapoint['annotations/trimaps'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

#
img_train, annotation_train = input_img_path[:-1000], input_annotation_path[:-1000]
img_test, annotation_test = input_img_path[-1000:], input_annotation_path[-1000:]

train_ds = tf.data.Dataset.from_tensor_slices((img_train, annotation_train))
test_ds = tf.data.Dataset.from_tensor_slices((img_test, annotation_test))