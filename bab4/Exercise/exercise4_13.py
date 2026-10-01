# Exercise 4.13
# Modified Example 4.15: Uncommented different models (ResNet50, MobileNetV2, DenseNet201) to compare performances.

from tensorflow.keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers
import numpy as np
import cv2

# Selecting MobileNetV2 / DenseNet201 / ResNet50
clf_name = 'mobilenetv2'  # Can be changed to 'resnet50', 'densenet201', 'vgg16'
clf, preprocess_input = Classifiers.get(clf_name)
sz = 224

model = clf(input_shape=(sz, sz, 3), weights='imagenet', classes=1000)
print(f"Model Summary for {clf_name}:")
model.summary()

camera = cv2.VideoCapture(0)
image_size = sz

while True:
    ret, cam_frame = camera.read()
    if not ret:
        break
    frame = cv2.resize(cam_frame, (image_size, image_size))
    image = np.asarray(frame)
    image = np.expand_dims(image, 0)
    image = preprocess_input(image)
    preds = model.predict(image)
    label = decode_predictions(preds)

    cv2.putText(cam_frame, f"{clf_name}: {label[0][0][1]}, {label[0][0][2]:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Classification", cam_frame)
    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()
