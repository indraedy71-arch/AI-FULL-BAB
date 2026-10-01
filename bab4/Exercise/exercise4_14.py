# Exercise 4.14
# Modified Example 4.15 to use an if-else statement to select the model for classification.

from tensorflow.keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers
import numpy as np
import cv2

model_choice = 'resnet50'  # Options: 'vgg16', 'resnet50', 'mobilenetv2', 'densenet201', 'inceptionv3'

if model_choice == 'vgg16':
    clf, preprocess_input = Classifiers.get('vgg16')
    sz = 224
elif model_choice == 'resnet50':
    clf, preprocess_input = Classifiers.get('resnet50')
    sz = 224
elif model_choice == 'mobilenetv2':
    clf, preprocess_input = Classifiers.get('mobilenetv2')
    sz = 224
elif model_choice == 'densenet201':
    clf, preprocess_input = Classifiers.get('densenet201')
    sz = 224
elif model_choice == 'inceptionv3':
    clf, preprocess_input = Classifiers.get('inceptionv3')
    sz = 299
else:
    raise ValueError(f"Unknown model choice: {model_choice}")

model = clf(input_shape=(sz, sz, 3), weights='imagenet', classes=1000)
print(f"Loaded model: {model_choice}")
model.summary()

camera = cv2.VideoCapture(0)

while True:
    ret, cam_frame = camera.read()
    if not ret:
        break
    frame = cv2.resize(cam_frame, (sz, sz))
    image = np.asarray(frame)
    image = np.expand_dims(image, 0)
    image = preprocess_input(image)
    preds = model.predict(image)
    label = decode_predictions(preds)

    cv2.putText(cam_frame, f"Model [{model_choice}]: {label[0][0][1]}, {label[0][0][2]:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Classification", cam_frame)
    key = cv2.waitKey(30)
    if key == 27:  # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()
