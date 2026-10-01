# Exercise 4.6
# Modified program from Example 4.7, adding another Conv2D, MaxPooling, and Dropout layer before Flatten.

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()

# First block (from Example 4.7)
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

# Added second block before Flatten
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Modified Model Summary (Additional Conv2D, MaxPooling, Dropout):")
print(model.summary())
