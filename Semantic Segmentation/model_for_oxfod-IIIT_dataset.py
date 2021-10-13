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

train = ds_path['train'].map(load_train_ds, num_parallel_calls=tf.data.experimental.AUTOTUNE)