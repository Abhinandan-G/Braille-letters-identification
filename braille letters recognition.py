import numpy as np
import pandas as pd

import os

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import PIL
import cv2

image_dir = Path('../input/braille-character-dataset/Braille Dataset/Braille Dataset')

image_count = len(dir_list)
print(image_count)
name_list = []
for i in dir_list:
    name_list.append(os.path.basename(i)[0])



images = []
for dir in dir_list:
    I = cv2.imread(str(dir))
    images.append(I)


images_list = np.array(images)
name_list = np.array(name_list).T
le = LabelEncoder()
name_list = le.fit_transform(name_list)
images_list = images_list / 255.0


print(images_list.shape)

plt.imshow(images_list[0])

print(images_list.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images_list, name_list, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(units=576, activation="relu"),
    keras.layers.Dense(units=288, activation="relu"),
    keras.layers.Dense(units=26, activation="softmax") #output layer
])

model.compile(optimizer="Adam", loss="SparseCategoricalCrossentropy", metrics=["sparse_categorical_accuracy"])

from keras.callbacks import EarlyStopping

es1 = EarlyStopping(patience=20, monitor="val_acc", mode="auto")
es2 = EarlyStopping(patience=20, monitor="val_loss", mode="auto")

history = model.fit(x=X_train,
                    y=y_train,
                    epochs=1000,
                    validation_split=0.3,
                    callbacks=[es1, es2])

print(model.summary())

time = np.arange(1, len(history.history['loss'])+1)

sns.lineplot(data=history.history, x=time, y='loss')
sns.lineplot(data=history.history, x=time, y='val_loss')
plt.title('Loss fitting history')
plt.legend(labels=['Loss', 'Validation loss'])

sns.lineplot(data=history.history, x=time, y='val_sparse_categorical_accuracy')
sns.lineplot(data=history.history, x=time, y='sparse_categorical_accuracy')
plt.title('Accuracy fitting history')
plt.legend(labels=['Accuracy', 'Valuation accuracy'])


print(model.evaluate(X_test, y_test))
