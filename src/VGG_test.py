import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image
import numpy as np


def preprocess_dataset(example):
    image = example["image"]
    image = tf.image.resize(image, (224, 224))
    image = preprocess_input(image)
    return image


# Load the Stanford Dogs dataset
dogs_ds2, val_dogs_ds2, test_dogs_ds2 = tfds.load(
    "stanford_dogs",
    with_info=False,
    split=["train[:70%]", "train[70%:]", "test"],
)
dogs_ds2 = dogs_ds2.take(1000)
val_dogs_ds2 = val_dogs_ds2.take(1000)
test_dogs_ds2 = test_dogs_ds2.take(1000)

proc_dataset = dogs_ds2.map(preprocess_dataset)
non_proc_dataset = val_dogs_ds2.map(preprocess_dataset)

dogs_ds2 = dogs_ds2.map(lambda x: ((x, tf.constant(1))))
non_dogs_ds = val_dogs_ds2.map(lambda x: ((x, tf.constant(0))))

dataset2 = tf.data.Dataset.concatenate(proc_dataset, non_proc_dataset)

# Create target dataset
target_dataset = dogs_ds.map(lambda _, label: label).concatenate(
    non_dogs_ds.map(lambda _, label: label)
)

target_dataset = target_dataset.map(lambda x: tf.reshape(x, shape=(-1, 1)))

# Shuffle and batch the concatenated dataset
batch_size = 32
train_dataset = tf.data.Dataset.zip((dataset2, target_dataset))
train_dataset = train_dataset.shuffle(1000).batch(batch_size)


input_shape = (224, 224, 3)
pretrained_model = tf.keras.applications.VGG16(
    weights="imagenet", include_top=False, input_shape=input_shape
)
for layer in pretrained_model.layers:
    layer.trainable = False

print("Building model...")

dense_layer = tf.keras.layers.Dense(256, activation="relu")(pretrained_model.output)
output = tf.keras.layers.Dense(1, activation="sigmoid")(dense_layer)

model = tf.keras.Model(inputs=pretrained_model.input, outputs=output)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_dataset, epochs=1)
