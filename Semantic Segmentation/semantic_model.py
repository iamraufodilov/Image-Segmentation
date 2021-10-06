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
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Concatenate, Conv2D, Reshape, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# load dataset
img_path = "G:/rauf/STEPBYSTEP/Data/oxford_pets/images"
ann_path = "G:/rauf/STEPBYSTEP/Data/oxford_pets/annotations/trimaps"

IMG_WIDTH=224
IMG_HEIGHT=224
BATCH_SIZE=32


input_img_path = sorted([
    os.path.join(img_path, fname)
    for fname in os.listdir(img_path)
    if fname.endswith(".jpg")
    ])

input_annotation_path = ([
    os.path.join(ann_path, fname)
    for fname in os.listdir(ann_path)
    if fname.endswith(".png") and not fname.startswith(".")
    ])

print(len(input_img_path), len(input_annotation_path)) #so we have 7390 photos and their annotation masks

# preprocess image dataset  
AUTOTUNE = tf.data.experimental.AUTOTUNE

def scale_down(image, mask):
    image = tf.cast(image, tf.float32)/255.0
    mask -= 1
    return image, mask

def load_and_preprocess(image_filepath, mask_filepath):
    img = tf.io.read_file(image_filepath)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    mask = tf.io.read_file(mask_filepath)
    mask = tf.io.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH])

    img, mask = scale_down(img, mask)

    return img, mask

input_img_path = tf.random.shuffle(input_img_path, seed=42)
input_annotation_path = tf.random.shuffle(input_annotation_path, seed=42)
input_img_path_train, input_annotation_path_train = input_img_path[:-1000], input_annotation_path[:-1000]
input_img_path_test, input_annotation_path_test = input_img_path[-1000:], input_annotation_path[-1000:]

trainloader = tf.data.Dataset.from_tensor_slices((input_img_path_train, input_annotation_path_train))
testloader = tf.data.Dataset.from_tensor_slices((input_img_path_test, input_annotation_path_test))

trainloader = (
    trainloader
    .shuffle(1024)
    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
    )

testloader = (
    testloader
    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
    )

print("Data prepared successfully")

HEIGHT_CELLS = 28
WIDTH_CELLS = 28

# next step is create model
def create_model(trainable=True):
    model = MobileNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights = 'imagenet')

    for layer in model.layers:
        layer.trainable = trainable

    block1 = model.get_layer("conv_pw_5_relu").output
    block2 = model.get_layer("conv_pw_11_relu").output
    block3 = model.get_layer("conv_pw_13_relu").output

    x = Concatenate()([UpSampling2D()(block3), block2])
    x = Concatenate()([UpSampling2D()(x), block1])

    x = Conv2D(1, kernel_size=1, activation="sigmoid")(x)
    x = Reshape((HEIGHT_CELLS, WIDTH_CELLS))(x)

    return Model(inputs=model.input, outputs=x)


def dice_coefficient(y_true, y_pred):
    numerator = 2 * tensorflow.reduce_sum(y_true * y_pred)
    denominator = tensorflow.reduce_sum(y_true + y_pred)

    return numerator / (denominator + tensorflow.keras.backend.epsilon())

def loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - tensorflow.keras.backend.log(dice_coefficient(y_true, y_pred) + tensorflow.keras.backend.epsilon())


model = create_model(False)
model.summary()

# compile the model
optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss=loss, optimizer=optimizer, metrics=[dice_coefficient])

#define callbacks
checkpoint = ModelCheckpoint("model-{val_loss:.2f}.h5", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)
stop = EarlyStopping(monitor = "val_loss", patience=5)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1)

# train the model
model.fit(trainloader, testloader, epochs=1, batch_size=BATCH_SIZE, verbose=1)