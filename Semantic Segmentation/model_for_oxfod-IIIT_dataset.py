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
ds_img_path = r'G:/rauf/STEPBYSTEP/Data2/oxford_pets/images/'
ds_ann_path = r'G:/rauf/STEPBYSTEP/Data/oxford_pets/annotations/trimaps/'

IMG_HEIGHT = 224
IMG_WIDTH = 224

def load_images(ds_path):
    images = list()
    for file in ds_path:
        data_path = os.path.join(ds_path, file)


    # convert to tensor
    images = tf.convert_to_tensor(images, dtype= tf.string)
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)

    image /=255

    return image

train = load_images(ds_img_path)
print(train)