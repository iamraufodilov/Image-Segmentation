# load libraries
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_datasets as tfds

from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
from IPython import display
import os

# load dataset
# load dataset
image_path = "G:/rauf/STEPBYSTEP/Data/oxford_pets/images"
annotation_path = "G:/rauf/STEPBYSTEP/Data/oxford_pets/annotations/trimaps"

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

print(len(input_img_path), len(input_annotation_path)) #so we have 7390 photos and their annotation masks

# preprocess image dataset
IMG_SIZE = 128
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

def scale_down(image, mask):
    image = tf.cast(image, tf.float32)/255.0
    mask -= 1
    return image, mask

def load_and_preprocess(image_filepath, mask_filepath):
    img = tf.io.read_file(image_filepath)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

    mask = tf.io.read_file(mask_filepath)
    mask = tf.io.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE])

    img, mask = scale_down(img, mask)

    return img, mask

# shuffle and split data

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

# create model
base_model = tf.keras.applications.MobileNetV2(input_shape = [128, 128, 3], include_top=False)

# this part for downsampling
# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs = base_model.input, outputs = base_model_outputs)
down_stack.trainable = False

#this part is for upsampling
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # last layer of model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

OUTPUT_CHANNELS = 3

model = unet_model(output_channels=OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


EPOCHS = 5

model_history = model.fit(epochs=EPOCHS)

# cannot finish, because of some bugs