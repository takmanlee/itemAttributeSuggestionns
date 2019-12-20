from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False,
                help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=False,
                help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
output = imutils.resize(image, width=400)

image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

model = load_model(args["model"])
mlb = pickle.loads(open("labelbin", "rb").read())

proba = model.predict(image)[0]
idxs = np.argsort(proba)[::-1][:10]

for (i, j) in enumerate(idxs):
    label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
    k = cv2.putText(output, label, (10, (i * 30) + 25),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1)


print("=======The top 10 suggested attributes are:======== ")
for (i, j) in enumerate(idxs):
    label = "{} ({:.2f}%)".format(mlb.classes_[j], proba[j] * 100)
    print(label)
