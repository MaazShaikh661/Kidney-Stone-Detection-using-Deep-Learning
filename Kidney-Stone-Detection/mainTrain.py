import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import normalize

from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

#optamizing our model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import TensorBoard
from tensorflow.keras.callbacks import TensorBoard

# evaluate model
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy, binary_crossentropy
from tensorflow.keras.models import load_model


img_directory = 'datasets/'

normal_image = os.listdir(img_directory+ 'Normal/')
stone_image = os.listdir(img_directory+ 'Stone/')
# cyst_image = os.listdir(img_directory+ 'Cyst/')
# tumor_image = os.listdir(img_directory+ 'Tumor/')


dataset = []
label = []

INPUT_SIZE = 64

#print(normal_image)
#path = "datasets/Normal/Normal- (1).jpg"
#print(path)

for i, image_name in enumerate(normal_image):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(img_directory+ 'Normal/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(stone_image):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(img_directory+ 'Stone/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

# for i, image_name in enumerate(cyst_image):
#     if(image_name.split('.')[1]=='jpg'):
#         image = cv2.imread(img_directory+ 'Cyst/'+image_name)
#         image = Image.fromarray(image, 'RGB')
#         image = image.resize((INPUT_SIZE,INPUT_SIZE))
#         dataset.append(np.array(image))
#         label.append(2)

# for i, image_name in enumerate(tumor_image):
#     if(image_name.split('.')[1]=='jpg'):
#         image = cv2.imread(img_directory+ 'Tumor/'+image_name)
#         image = Image.fromarray(image, 'RGB')
#         image = image.resize((INPUT_SIZE,INPUT_SIZE))
#         dataset.append(np.array(image))
#         label.append(3)


dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)

#Reshape = (n, image_width, image_height, n_channels)
x_train = x_train.reshape(x_train.shape[0], INPUT_SIZE, INPUT_SIZE, 3)
x_test = x_test.reshape(x_test.shape[0], INPUT_SIZE, INPUT_SIZE, 3)


#Normalize
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)


# building model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(INPUT_SIZE,INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Define Tensorboard as a keras callback
# tensorboard = TensorBoard(log_dir="logs/{}".format(1), histogram_freq=1, write_graph=True, write_images=True)
# this will save our best trained model
keras_callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min', min_delta=0.0001),
    ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True, verbose=1, mode='min'),
]

# Fit data to model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, verbose=1, epochs=300, validation_data=(x_test, y_test), callbacks=keras_callbacks)

# Plot history: MAE
plt.plot(model.history.history['loss'], label='training data')
plt.plot(model.history.history['val_loss'], label='validation data')
plt.title('MAE for kidney Stone')
plt.ylabel('MAE value')
plt.xlabel("No. of epoch")
plt.legend()
plt.show()

# saving the model to disk 
model.save('best_model.h5')

# evaluate model on test data and print accuracy 
model = load_model('best_model.h5')
scores = model.evaluate(x_test, y_test, verbose=1)
# print("Accuracy: %.2f%%" % (scores[1]*100))
print(f'Score: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')