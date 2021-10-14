# load necessary libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import cv2
import PIL
from IPython.display import clear_output


# download dataset
ds_path = "G:/rauf/STEPBYSTEP/Data2/oxford_pets/"

def normalize(input_img, input_maks):
    img = tf.cast(input_img, dtype=tf.float32) / 255.0
    input_maks -= 1
    return img, input_maks

def load_train_ds(data_path):
    img = tf.image.resize(data_path['images'], size=(224, 224))
    mask = tf.image.resize(data_path['masks'], size=(224, 224))

    img, mask = normalize(img, mask)

    return img, mask

#train = ds_path['train'].map(load_train_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)


# load dataset

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_img_path = "G:/rauf/STEPBYSTEP/Data2/oxford_pets/train/images"
ds_ann_path = "G:/rauf/STEPBYSTEP/Data2/oxford_pets/train/masks"

list_ds = tf.data.Dataset.list_files(str(ds_img_path))

def decode_img(input_img):
    image = tf.image.decode_jpeg(input_img, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return tf.image.resize(image, size=(IMG_HEIGHT, IMG_WIDTH))

def process_path(input_path):
    img = tf.io.read_file(input_path)
    img = decode_img(img)
    return img

dataset = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for i in dataset:
    print("first result", i)
    break
#for f in dataset.take(1):
    #print(f.numpy().shape())