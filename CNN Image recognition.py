#Data set from https://drive.google.com/open?id=1H2ffQeyl5C5U1zhPbZl40tdH2bvWVdTR

# Convolutional Neural Network
#Installing the deep learning libraries 
# Installing Theano
# pip install Theano

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install  keras

# Installing Pillow
# pip install Pillow

# This all commands are for windows and above command are done on anaconda prompt.
# Tensorflow is working on backend 


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu')) # For latest version conv2D can be used

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
# Reason to use 2nd layer is to increase the performance and reducing overfit
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())


# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_data = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_data = ImageDataGenerator(rescale = 1./255)

training_set = train_data.flow_from_directory('C:/Users/singh/Desktop/Convolutional_Neural_Networks/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_data.flow_from_directory('C:/Users/singh/Desktop/Convolutional_Neural_Networks/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

cclassifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 24,
                         validation_data = test_set,
                         validation_steps = 2000)


