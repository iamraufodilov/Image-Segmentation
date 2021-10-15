#Load libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# lod dataset

# we create two instances with the same arguments

data_dir = "G:/rauf/STEPBYSTEP/Data2/oxford_pets/train/images"

list_ds = tf.data.Dataset.list_files(str(data_dir), shuffle=False)

for f in list_ds.take(5):
  print(f.numpy())