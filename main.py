from gc import callbacks
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import scipy
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.callbacks import TensorBoard
import datetime

from gc import callbacks

#NAME = "Teeth-Cavity-Detection-{}".format(int(time.time()))

#tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

classifier = Sequential()
 
#CNN_Layer1

classifier.add(Convolution2D(32,(3,3), strides = 1 , padding = 'same', input_shape= (256,256,3), activation ='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

#CNN_Layer2

classifier.add(Convolution2D(64,(3,3), strides = 1 , padding = 'same', activation ='relu'))
classifier.add(Dropout(0.1))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

#CNN_layer3
classifier.add(Convolution2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

#CNN_Layer4
classifier.add(Convolution2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
classifier.add(Dropout(0.1))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

#CNN_Layer5
#classifier.add(Convolution2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
#classifier.add(Dropout(0.1))
#classifier.add(BatchNormalization())
#classifier.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))
#Flattening

classifier.add(Flatten())

#NN_Layer1

classifier.add(Dense(activation = 'relu',units = 128))

classifier.add(Dropout(0.1))
#NN_Layer2

classifier.add(Dense(activation = 'sigmoid',units = 1))

#Compile

classifier.compile(optimizer='adam',loss= 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=False,samplewise_center=False, featurewise_std_normalization=False,  
        samplewise_std_normalization=False, zca_whitening=False, rotation_range = 30,  
        zoom_range = 0.2,width_shift_range=0.1, height_shift_range=0.1,
        horizontal_flip = True,vertical_flip=False)
        

test_datagen = ImageDataGenerator(rescale=1./255, rotation_range = 30,  
        zoom_range = 0.2,width_shift_range=0.1, height_shift_range=0.1,
        horizontal_flip = True,vertical_flip=False)

training_set = train_datagen.flow_from_directory(
        '../environment/teeth_dataset/Trianing',
        target_size=(256,256),
        batch_size=16,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        '../environment/teeth_dataset/Test',
        target_size=(256,256),
        batch_size=16,
        class_mode='binary')


history = classifier.fit(x = training_set, validation_data = test_set, epochs = 25)


epochs = [i for i in range(25)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()

import numpy as np
from keras.preprocessing import image

test_image = image.load_img(r'C:\Users\HAFEEZ KHAN\Desktop\Cavity Dataset\cavity\10.jpg', target_size = (256, 256))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(training_set.class_indices)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img=mpimg.imread(r'C:\Users\HAFEEZ KHAN\Desktop\Cavity Dataset\cavity\10.jpg')
imgplot = plt.imshow(img)
plt=plt.title('Cavity Detected')

if result[0][0] == 1:
    prediction = 'healthy teeth'
else:
    prediction = 'cavity'
    
print("AI's prediction is: "+ prediction)

import matplotlib.pyplot as plt
def plot_acc_loss(results, epochs):
    acc = results.history['accuracy']
    loss = results.history['loss']
    val_acc = results.history['val_accuracy']
    val_loss = results.history['val_loss']
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(range(1,epochs), acc[1:], label='Train_acc')
    plt.plot(range(1,epochs), val_acc[1:], label='Test_acc')
    plt.title('Accuracy over' + str(epochs) + 'Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(1,epochs), loss[1:], label='Train_loss')
    plt.plot(range(1,epochs), val_loss[1:], label='Test_loss')
    plt.title('Loss over' + str(epochs) +  'Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()
plot_acc_loss(history, 25)
