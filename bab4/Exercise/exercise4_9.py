# Exercise 4.9
# Modified Python program from Example 4.10 so that it loads VGG19 model instead of VGG16 model.

from tensorflow.keras.applications.vgg19 import VGG19

model = VGG19(weights='imagenet')
print("VGG19 Model Summary:")
print(model.summary())
