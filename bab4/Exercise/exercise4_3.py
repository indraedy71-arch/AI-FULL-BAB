# Exercise 4.3
# Modified Python program from Example 4.3 using 5 neurons for dense layer instead of 10 neurons.

import time
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=7)

# Model with 5 neurons in dense layers
model = Sequential()
model.add(Dense(5, activation='relu', input_shape=[X_train.shape[1]]))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

print(model.summary())

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

start_time = time.time()
model.fit(X_train, y_train, batch_size=50, validation_split=0.2, epochs=100, verbose=1)
training_time = time.time() - start_time

results = model.evaluate(X_test, y_test)
print(f"Training Time (5 neurons): {training_time:.2f} seconds")
print('Loss: ', results[0])
print('MSE: ', results[1])
