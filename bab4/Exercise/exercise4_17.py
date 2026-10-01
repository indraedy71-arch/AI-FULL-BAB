# Exercise 4.17
# Modified Example 4.17: Select second (n=2), third (n=4), or fourth (n=5) conv layer for feature map visualization.

from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims

model = VGG19()

def plot_feature_maps(feature_maps, layer_name):
    col = 8
    num_maps = min(feature_maps.shape[3], 64)
    row = int(num_maps / col)
    ix = 1
    plt.figure(figsize=(16, 16))
    for _ in range(row):
        for _ in range(col):
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    plt.suptitle(f"Feature Maps for Layer: {layer_name}")
    plt.show()

# Select layer index (e.g. n=2 -> block1_conv2, n=4 -> block2_conv1, n=5 -> block2_conv2)
target_layer_idx = 4
layer_name = model.layers[target_layer_idx].name
print(f"Visualizing feature maps for layer index {target_layer_idx}: {layer_name}")

model_feat = Model(inputs=model.inputs, outputs=model.layers[target_layer_idx].output)

try:
    img = load_img('bab4/Elephant.jpg', target_size=(224, 224))
    img = img_to_array(img)
    img = expand_dims(img, axis=0)
    img = preprocess_input(img)

    feature_maps = model_feat.predict(img)
    print("Feature maps shape:", feature_maps.shape)
    plot_feature_maps(feature_maps, layer_name)
except FileNotFoundError:
    print("Elephant.jpg not found for feature map visualization.")
