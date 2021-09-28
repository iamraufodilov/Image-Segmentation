# import necessary libraries
import tensorflow as tf
import random
random.seed(0)
import warnings
import os
import cv2
import numpy as np
warnings.filterwarnings("ignore")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# load dataset
img_path = "G:/rauf/STEPBYSTEP/Data/oxford_pets/images"
ann_path = "G:/rauf/STEPBYSTEP/Data/oxford_pets/annotations/trimaps"

IMG_WIDTH=200
IMG_HEIGHT=200

def train_img_loader(img_path):
    image_array = []
    image_path = os.path.join(img_path)
    for img in image_path:
        image = cv2.imread(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
        image = np.array(image)
        image = image.astype('float32')
        image /=255
        image_array.append(image)
    return image_array

train_img_loader(img_path)