# Exercise 4.12
# Modified Example 4.14: Added DenseNet121 to the model to classify webcam images.

import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import densenet
import numpy as np

# Using DenseNet121
model = densenet.DenseNet121(weights='imagenet')
print("DenseNet121 Model Summary:")
print(model.summary())

camera = cv2.VideoCapture(0)
image_size = 224

while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break

    frame = cv2.resize(cam_frame, (image_size, image_size))
    numpy_image = img_to_array(frame)
    image_batch = np.expand_dims(numpy_image, axis=0)

    processed_image = densenet.preprocess_input(image_batch.copy())

    predictions = model.predict(processed_image)
    label = decode_predictions(predictions)

    cv2.putText(cam_frame, "DenseNet121: {}, {:.1f}".format(label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('video image', cam_frame)

    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()
