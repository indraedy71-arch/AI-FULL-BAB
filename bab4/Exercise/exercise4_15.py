# Exercise 4.15
# Modified Example 4.17: Select the second convolutional layer (n = 2) to visualize.

from keras.applications.vgg19 import VGG19
from matplotlib import pyplot

model = VGG19()

# Selecting the second convolutional layer (n = 2: block1_conv2)
n = 2
filters, biases = model.layers[n].get_weights()
s = filters.shape

print(f"Layer {n} ({model.layers[n].name}) Filter Parameters:")
print("Filter shape:", s)

# Normalize filter values to 0-1
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters, ix = 4, 1
pyplot.figure(figsize=(10, 10))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(3):  # First 3 channels
        ax = pyplot.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[:, :, j], cmap='gray')
        ix += 1
pyplot.suptitle(f"Layer {n} ({model.layers[n].name}) First {n_filters} Filters")
pyplot.show()
