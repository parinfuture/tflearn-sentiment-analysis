#Sentiment analysis with TFlearn

#Import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical

#Read data from reivews.txt and labels.txt
reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)

#Count word frequency
from collections import Counter

total_counts = Counter()
for _, row in reviews.iterrows():
    total_counts.update(row[0].split(' '))

print("Total words in data set: ", len(total_counts))

#Build vocabulary and preview it
vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]
print(vocab[:60])

#Check the last word in our vocabulary
print(vocab[-1], ': ', total_counts[vocab[-1]])

#Creating a dictionary called word2idx and map each word in the vocabulary to an index
word2idx = {word: i for i, word in enumerate(vocab)}
#Check the dictionary
word2idx

#Create a text to vector function
def text_to_vector(text):
    word_vector = np.zeros(len(vocab), dtype=np.int_)
    for word in text.split(' '):
        idx = word2idx.get(word, None)
        if idx is None:
            continue
        else:
            word_vector[idx] += 1
    return np.array(word_vector)

#Check text_to_vector()
text_to_vector('My name is Parikshit and I am a robot'
               'While writing this code I ate three burgers')[:65]

#Let convert the entire dataset reviews to an individual word vector
word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
for ii, (_, text) in enumerate(reviews.iterrows()):
    word_vectors[ii] = text_to_vector(text[0])

#Preparing training and test data
Y = (labels=='positive').astype(np.int_)
records = len(labels)

shuffle = np.arange(records)
np.random.shuffle(shuffle)
test_fraction = 0.9

train_split, test_split = shuffle[:int(records*test_fraction)], shuffle[int(records*test_fraction):]
trainX, trainY = word_vectors[train_split,:], to_categorical(Y.values[train_split], 2)
testX, testY = word_vectors[test_split,:], to_categorical(Y.values[test_split], 2)

#Building the network
def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    
    #### Inputs ####
    net = tflearn.input_data([None, 10000])
    
    # Hidden layers
    net = tflearn.fully_connected(net, 200, activation = 'ReLU')
    net = tflearn.fully_connected(net, 25, activation = 'ReLU')
    
    #Ouput layers
    net = tflearn.fully_connected(net, 2, activation = 'softmax')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')
    
    model = tflearn.DNN(net)
    return model

#Initializing the model
model = build_model()

# Training
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=10)

#Testing model accuracy
predictions = (np.array(model.predict(testX))[:,0] >= 0.5).astype(np.int_)
test_accuracy = np.mean(predictions == testY[:,0], axis=0)
print("Test accuracy: ", test_accuracy)

#Check test accuracy and modify hyperparameters






