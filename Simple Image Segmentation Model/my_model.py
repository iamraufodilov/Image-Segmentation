# load libraries
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras import Model

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

# create model

class Segmentation_model:
    def prepare_model(self, OUTPUT_CHANNEL, input_size=(IMG_SIZE, IMG_SIZE,3)):
        inputs = tf.keras.layers.Input(input_size)

        # Encoder 
        conv1, pool1 = self.__ConvBlock(32, (3,3), (2,2), 'relu', 'same', inputs) 
        conv2, pool2 = self.__ConvBlock(64, (3,3), (2,2), 'relu', 'same', pool1)
        conv3, pool3 = self.__ConvBlock(128, (3,3), (2,2), 'relu', 'same', pool2) 
        conv4, pool4 = self.__ConvBlock(256, (3,3), (2,2), 'relu', 'same', pool3) 
    
        # Decoder
        conv5, up6 = self.__UpConvBlock(512, 256, (3,3), (2,2), (2,2), 'relu', 'same', pool4, conv4)
        conv6, up7 = self.__UpConvBlock(256, 128, (3,3), (2,2), (2,2), 'relu', 'same', up6, conv3)
        conv7, up8 = self.__UpConvBlock(128, 64, (3,3), (2,2), (2,2), 'relu', 'same', up7, conv2)
        conv8, up9 = self.__UpConvBlock(64, 32, (3,3), (2,2), (2,2), 'relu', 'same', up8, conv1)
    
        conv9 = self.__ConvBlock(32, (3,3), (2,2), 'relu', 'same', up9, False)

        outputs = Conv2D(OUTPUT_CHANNEL, (3, 3), activation='softmax', padding='same')(conv9)

        return Model(inputs=[inputs], outputs=[outputs]) 

    def __ConvBlock(self, filters, kernel_size, pool_size, activation, padding, connecting_layer, pool_layer=True):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(connecting_layer)
        conv = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
        if pool_layer:
            pool = MaxPooling2D(pool_size)(conv)
            return conv, pool
        else:
            return conv

    def __UpConvBlock(self, filters, up_filters, kernel_size, up_kernel, up_stride, activation, padding, connecting_layer, shared_layer):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(connecting_layer)
        conv = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
        up = Conv2DTranspose(filters=up_filters, kernel_size=up_kernel, strides=up_stride, padding=padding)(conv)
        up = tf.keras.layers.concatenate([up, shared_layer], axis=3)

        return conv, up


OUTPUT_CHANNELS = 3

model = Segmentation_model().prepare_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.summary()

segmentation_classes = ['pet', 'pet_outline', 'background']

def labels():
    l={}
    for i, label in enumerate(segmentation_classes):
        l[i] = label
    return l

def get_mask(back_img, pred_mask, true_mask):
    return 