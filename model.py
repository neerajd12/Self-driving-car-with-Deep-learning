import random
import numpy as np
import pandas
import json
import cv2

from sklearn.model_selection import train_test_split
from sklearn .utils import shuffle

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.core import Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D

from utils import imageUtils

#import image utils and set the image input shape
image_utils = imageUtils()
im_x = image_utils.im_x
im_y = image_utils.im_y
im_z = image_utils.im_z

def get_model():
    """
        Defines the CNN model architecture and returns the model.
        The architecture is the same as I developed for project 2
        https://github.com/neerajdixit/ND/tree/master/Deep%20Learning
        with an additional normalization layer in front and
        a final fully connected layer of size 1 since we need one output.
    """

    # Create a Keras sequential model
    model = Sequential()
    # Add a normalization layer to normalize between -0.5 and 0.5.
    model.add(Lambda(lambda x: x / 255. - .5,input_shape=(im_x,im_y,im_z), name='norm'))
    # Add a convolution layer with Input = 32x32x3. Output = 30x30x6. Strides 1 and VALID padding.
    # Perform RELU activation 
    model.add(Convolution2D(6, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu', name='conv1'))
    # Add a convolution layer with Input = 30x30x6. Output = 28x28x9. Strides 1 and VALID padding.
    # Perform RELU activation 
    model.add(Convolution2D(9, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu', name='conv2'))
    # Add Pooling layer with Input = 28x28x9. Output = 14x14x9. 2x2 kernel, Strides 2 and VALID padding
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid', name='pool1'))
    # Add a convolution layer with Input 14x14x9. Output = 12x12x12. Strides 1 and VALID padding.
    # Perform RELU activation 
    model.add(Convolution2D(12, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu', name='conv3'))
    # Add a convolution layer with Input = 30x30x6. Output = 28x28x9. Strides 1 and VALID padding.
    # Perform RELU activation 
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid", activation='relu', name='conv4'))
    # Add Pooling layer with Input = 10x10x16. Output = 5x5x16. 2x2 kernel, Strides 2 and VALID padding
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid', name='pool2'))
    # Flatten. Input = 5x5x16. Output = 400.
    model.add(Flatten(name='flat1'))
    # Add dropout layer with 0.2  
    model.add(Dropout(0.2, name='dropout1'))
    # Add Fully Connected layer. Input = 400. Output = 220
    # Perform RELU activation 
    model.add(Dense(220, activation='relu', name='fc1'))
    # Add Fully Connected layer. Input = 220. Output = 43
    # Perform RELU activation 
    model.add(Dense(43, activation='relu', name='fc2'))
    # Add Fully Connected layer. Input = 43. Output = 1
    # Perform RELU activation 
    model.add(Dense(1, name='fc3'))
    # Configure the model for training with Adam optimizer
    # "mean squared error" loss objective and accuracy metrics
    # Learning rate of 0.001 was chosen because this gave best performance after testing other values
    model.compile(optimizer=Adam(lr=0.001), loss="mse", metrics=['accuracy'])
    return model

def data_generator(data_path, images, steering_angles, batch_size):
    """
        Data generator for kera fit_generator
    """
    while True:
        # Select batch_size random indices from the image name array
        indices = np.random.randint(len(images),size=batch_size)
        # Get the corresponding steering angles
        y = steering_angles[indices]
        # Create empty numpy array of batch size for images
        x=np.zeros((batch_size, im_x, im_y, im_z))
        for i in range(batch_size):
            # Read the image from data path using open cv
            img = cv2.imread(data_path+images[indices[i]].strip())
            # pre process image and add to array
            x[i] = image_utils.pre_process_image(img)
        yield (x, y)

def setup_data(data_path):
    """
        Reads the log file from data_path and creates the data used by generators.
        Takes in the log file location as parameter
    """
  
    # Read the csv file using pandas
    test_data = pandas.read_csv(data_path+'driving_log.csv', header = None)

    # Create a numpy array consisting of images from center camera 
    # Append images from left camera to the above array followed by the images from right camera.
    # Shape [[center],[left],[right]]
    # This is the test data 
    a = np.concatenate((np.asarray(test_data[0].tolist()), np.asarray(test_data[1].tolist())), axis=0)
    images = np.concatenate((a, np.asarray(test_data[2].tolist())) , axis=0)

    # Create a numpy array consisting of steering angles. For center camera 
    # Create the steering angles for left camera images by adding 0.25 to the original and append it to above array
    # Create the steering angles for right camera images by subtracting 0.25 to the original and append it to above array
    # Shape [[steering_center],[steering_left],[steering_right]]
    # This is the Label data 
    a = np.concatenate((np.asarray(test_data[3].tolist()), np.asarray(test_data[3].tolist())+0.25), axis=0)
    steering_angles = np.concatenate((a, np.asarray(test_data[3].tolist())-0.25), axis=0)
    # Above strategy allows using left and right camera images to train for recovery as suggested in the project guidlines

    # Un reference the temp variable 
    a=None
    # Shuffle the data.
    # Important before dividing the test data into training and validation sets. To avoid same data on multiple runs.
    X_train, y_train = shuffle(images, steering_angles)
    # Split the test data in training and validation sets.
    # Validation set is chosen to be only 10 percent as the data set is small
    # and actual drive on tracks will be used for validation.
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train , test_size=0.1, random_state=0)
    print("Number of training examples =", X_train.size)
    print("Number of validation examples =", X_validation.size)
    return X_train, X_validation, y_train, y_validation


######## Processing ########
data_path = '/home/neeraj/Downloads/data/data/'
# Setup data from drive log csv
X_train, X_validation, y_train, y_validation = setup_data(data_path)

# Get model and print summary.
model = get_model()
print(model.summary())

# Set batch size and epocs
per_epoch_samples=8064
gen_batch_size=256
epochs=10

# Fit the data on model and validate using data generators. 
model.fit_generator(data_generator(data_path, 
                                    X_train, 
                                    y_train, 
                                    gen_batch_size),
                    samples_per_epoch=per_epoch_samples, 
                    nb_epoch=epochs,
                    validation_data=data_generator(data_path, 
                        X_validation, 
                        y_validation, 
                        gen_batch_size),
                    nb_val_samples=X_validation.size)

# Save the model and weights
print("Saving model weights and configuration file...")
model.save_weights("model.h5", True)
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
print("model saved...")