# Exercise 4.4
# Modified LeNet program from Example 4.4 to use a different number of filters (12 and 32 instead of 6 and 16).

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D

model = Sequential()
# Changed filters from 6 to 12
model.add(Conv2D(filters=12, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(AveragePooling2D())
# Changed filters from 16 to 32
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

print("Modified LeNet Model Summary (12 & 32 filters):")
model.summary()
