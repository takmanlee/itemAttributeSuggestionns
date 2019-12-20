import pandas as pd
import os
import cv2
import pickle
import random
import argparse
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
import matplotlib


class nnet:
    @staticmethod
    def build(width, height, depth, classes, finalAct="softmax"):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        for index in range(3):
            model.add(Conv2D(32, (3, 3), padding="same",
                             input_shape=inputShape))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=chanDim))
            model.add(MaxPooling2D(pool_size=(3, 3)))
            model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation(finalAct))

        return model


matplotlib.use("Agg")

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
                help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=False, type=str, default="plot.png",
                help="path to output accuracy/loss plot")
ap.add_argument("-i", "--imagePath", required=False, type=str,
                default="plot.png", help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

epochs = 5
initial_learning_rate = 1e-3
batch_size = 16
IMAGE_DIMS = (96, 96, 3)

df = pd.read_csv(args["dataset"])
item_pk = df['item_pk']
item_attribute = df['attribute_values']

data = []
labels = []

for index in range(len(item_pk)):
    image = cv2.imread(args["imagePath"]+item_pk[index]+'.jpg')
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    label = item_attribute[index].split("|")
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("data matrix: {} images ({:.2f}MB)".format(
    len(item_pk), data.nbytes / (1024 * 1000.0)))

print("class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i + 1, label))

(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, random_state=42)

model = nnet.build(
    width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
    finalAct="sigmoid")

opt = Adam(lr=initial_learning_rate, decay=initial_learning_rate / epochs)

model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print("training network:")
H = model.fit(trainX, trainY, validation_data=(
    testX, testY), batch_size=batch_size, epochs=epochs)

print("serializing network:")
model.save(args["model"])

print("serializing label binarizer:")
f = open(args["items"], "wb")
f.write(pickle.dumps(mlb))
f.close()
