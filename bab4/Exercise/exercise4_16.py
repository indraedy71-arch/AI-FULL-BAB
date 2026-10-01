# Exercise 4.16
# Modified Example 4.17: Display ALL the filters by changing n_filters = s[3].

from keras.applications.vgg19 import VGG19
from matplotlib import pyplot
import math

model = VGG19()

n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape

print("Total number of filters in layer 1:", s[3])

# Normalize filter values to 0-1
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters = s[3]  # All 64 filters
cols = 8
rows = math.ceil(n_filters / cols)

pyplot.figure(figsize=(16, 16))
for i in range(n_filters):
    ax = pyplot.subplot(rows, cols, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    # Display average across channels
    f = filters[:, :, :, i].mean(axis=-1)
    pyplot.imshow(f, cmap='gray')

pyplot.suptitle("All 64 Filters of Layer 1 (VGG19)")
pyplot.show()
