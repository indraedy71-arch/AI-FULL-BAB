# Example 4.17 CNN Visualize Filters.ipynb
# Modified from:
# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/

from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from matplotlib import pyplot
from numpy import expand_dims

# Load VGG19 model and summarize
model = VGG19()
model.summary()

# Part 2: Print layer indices and names
n = 0
for layer in model.layers:
    print(n, layer.name)
    n += 1

# Part 3: Inspect convolutional filters and biases shape
n = 0
for layer in model.layers:
    if 'conv' in layer.name:
        filters, biases = layer.get_weights()
        print(n, layer.name, filters.shape, biases.shape)
    n += 1

# Part 4: Retrieve weights from the first hidden layer (n=1)
n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Color channels: ", s[0])
print("Filter size: ", s[1], s[2])
print("Total number of filters: ", s[3])

# Normalize filter values to 0-1 for visualization
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Plot first few filters
n_filters, ix = 4, 1
pyplot.figure(figsize=(10, 10))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(s[0]):
        ax = pyplot.subplot(n_filters, s[0], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[j, :, :], cmap='gray')
        ix += 1
pyplot.show()

# Part 5: Function to plot feature maps
def plot_feature_maps(feature_maps):
    col = 8
    row = int(feature_maps.shape[3] / col)
    ix = 1
    plt.figure(figsize=(20, 20))
    for _ in range(row):
        for _ in range(col):
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    plt.show()

# Part 6: Redefine model to output feature maps for the first hidden layer (n=1)
n = 1
model_hidden = Model(inputs=model.inputs, outputs=model.layers[n].output)
model_hidden.summary()

try:
    img = load_img('Elephant.jpg', target_size=(224, 224))
    img = img_to_array(img)
    img = expand_dims(img, axis=0)
    img = preprocess_input(img)

    feature_maps = model_hidden.predict(img)
    print("Feature maps shape: ", feature_maps.shape)
    plot_feature_maps(feature_maps)
except FileNotFoundError:
    print("Elephant.jpg not found for feature map generation.")
