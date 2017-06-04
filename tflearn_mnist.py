# Import Numpy, TensorFlow, TFLearn, and MNIST data
import numpy as np
import tensorflow as tf
import tflearn
import tflearn.datasets.mnist as mnist

# Retrieve the training and test data
trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

# Visualizing the data
import matplotlib.pyplot as plt
%matplotlib inline

# Function for displaying a training image by it's index in the MNIST set
def show_digit(index):
    label = trainY[index].argmax(axis=0)
    # Reshape 784 array into 28x28 image
    image = trainX[index].reshape([28,28])
    plt.title('Training data, index: %d,  Label: %d' % (index, label))
    plt.imshow(image, cmap='gray_r')
    plt.show()
    
# Display the third (index 2) training image
show_digit(2)

# Define the neural network
def build_model():
    # This resets all parameters and variables
    tf.reset_default_graph()
    
    net = tflearn.input_data([None, trainX.shape[1]])
    
    # Include the input layer, hidden layer(s), and set how you want to train the model
    net = tflearn.fully_connected(net, 128, activation='ReLU')
    net = tflearn.fully_connected(net, 32, activation='ReLU')
    
    #Output layer and the model
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy')
    
    # This model assumes that your network is named "net"    
    model = tflearn.DNN(net)
    return model

# Build the model
model = build_model()

# Training
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=100, n_epoch=20)

#Calculate the accuracy of predictions
predictions = np.array(model.predict(testX)).argmax(axis=1)
actual = testY.argmax(axis=1)
test_accuracy = np.mean(predictions == actual, axis=0)