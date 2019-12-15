import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

dataset_path = os.path.join("datasets")
train_dataset_path = os.path.join(dataset_path,"train")
test_dataset_path = os.path.join(dataset_path,"test")

height, width= 64, 64

train_gen = ImageDataGenerator(rescale=1/255,horizontal_flip= True, rotation_range=30)

train_dataset = train_gen.flow_from_directory(train_dataset_path,target_size=(width,height),color_mode="grayscale")
train_gen = ImageDataGenerator(rescale=1/255)
test_dataset = train_gen.flow_from_directory(test_dataset_path,target_size=(width,height),color_mode="grayscale")

model = keras.models.Sequential([
    keras.layers.Conv2D(20,(5,5),padding="same", activation=tf.nn.relu,input_shape=(height,width,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(50,(5,5),padding="same",activation=tf.nn.relu),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation=tf.nn.relu),
    keras.layers.Dense(2,activation=tf.nn.softmax)

])
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["acc"])
history = model.fit_generator(train_dataset,epochs=17,validation_data=test_dataset,class_weight=[2.7, 1.0])

model.save_weights("Smile_za_git_weightovi.h5")