# Exercise 4.7
# Modified program from Example 4.9:
# 1. Changed kernel/filter count of 32 to 64 in Conv2D.
# 2. Changed filter size (kernel_size) from (5,5) to (3,3).

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# Create modified model: 64 filters of size (3,3)
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=200)

scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error (64 filters, (3,3) kernel): %.2f%%" % (100 - scores[1] * 100))
