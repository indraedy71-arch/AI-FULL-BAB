# Exercise 4.10
# Modified Python program from Example 4.11 to use VGG19 for classification and compare performance with VGG16.

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import numpy as np
import time

model = VGG19(weights='imagenet')

img_path = 'Elephant.jpg'
try:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    start_time = time.time()
    predictions = model.predict(x)
    inference_time = time.time() - start_time

    results = decode_predictions(predictions)
    print(f"VGG19 Inference Time: {inference_time:.4f} seconds")
    print("VGG19 Prediction Results:")
    print(results)
except FileNotFoundError:
    print(f"File '{img_path}' not found. Please provide an input image.")
