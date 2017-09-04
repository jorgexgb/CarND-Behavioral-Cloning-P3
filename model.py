# importing base dependencies
import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# importing keras 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras import optimizers

# path to training images
data_path = 'data/IMG/'

def process_image(raw_image_path):
    '''
    raw_image_path (string): image path as recorded on the driving_log.csv file

    Method to extract the correct path for the training image and 
    return it after reading it from disk.

    '''
    filename = raw_image_path.split('/')[-1]
    current_path =  data_path + filename
    return cv2.imread(current_path)

# list to hold all lines from driving_log.csv
# read all lines
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# use scikit to split training set for training and validation
# 20% used for validation and 80% for training
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    '''
    samples (list): list of lines from driving_log.csv file
    batch_size (int): size for each batch the generator will create 

    Create the batch as needed when used by keras. This method
    helps to avoid loading all the dataset into memory at the 
    same time.

    '''

    # how manny 
    num_samples = len(samples)
    # forever loop
    while 1:
        for offset in range(0, num_samples, batch_size):
            # grab the samples for the batch
            batch_samples = samples[offset:offset+batch_size]
            # list to hold corresponding images and angles
            images = []
            angles = []
            for batch_sample in batch_samples:
                # create adjusted steering measurements for the side camera images
                correction = 0.2 
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                img_center = process_image(batch_sample[0])
                img_left = process_image(batch_sample[1])
                img_right = process_image(batch_sample[2])

                # add images and angles to data set
                images.extend([img_center, img_left, img_right])
                angles.extend([steering_center, steering_left, steering_right])

                # add flipped images for data augmentation
                images.extend([cv2.flip(img_center, 1), cv2.flip(img_left, 1), cv2.flip(img_right, 1)])
                angles.extend([steering_center*-1.0, steering_left*-1.0, steering_center*-1.0])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)            

# compile and train the model using the generator function
# get the train and validation datasets using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# input image data shape
row, col, ch = 160, 320, 3

# build the network
# using keras Sequential() API
model = Sequential()
# crop the image to include only most important data from image for the network
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(row,col,ch)))
# normalize the pixel data between -0.5 and 0.5
model.add(Lambda(lambda x: x/255.-0.5))

# using NVIDIA CNN Architecture for Self Driving Car
# slightly modified to use Spatial Dropouts for filters after each Convolution
# and using normal dropouts during the fully connected layers
# dropout helps the network generalize better and avoid overfitting
model.add(Convolution2D(24,5,5,border_mode="same",subsample=(2,2),activation='relu'))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(36,5,5,border_mode="same",subsample=(2,2),activation='relu'))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(48,5,5,border_mode="valid",subsample=(2,2),activation='relu'))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(64,3,3,border_mode="valid",activation='relu'))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(64,3,3,border_mode="valid",activation='relu'))
model.add(SpatialDropout2D(0.2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

# loss function = Mean Squared Error
model.compile(loss='mse', optimizer='adam')
# training for 5 epochs.
model.fit_generator(train_generator, samples_per_epoch=len(train_samples*6), 
                    validation_data=validation_generator, nb_val_samples=len(validation_samples), 
                    nb_epoch=5)

model.save('model.h5')
