# Exercise 4.5
# Modified AlexNet program from Example 4.5 using a different number of dense layer units (2048 instead of 4096).

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()

# Convolutional layers
model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), activation='relu', strides=(4, 4), padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model.add(Conv2D(filters=256, kernel_size=(11, 11), activation='relu', strides=(1, 1), padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

model.add(Conv2D(filters=384, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid'))
model.add(Conv2D(filters=384, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Fully Connected layers with 2048 units instead of 4096
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(17, activation='softmax'))

print("Modified AlexNet Model Summary (2048 Dense units):")
model.summary()
