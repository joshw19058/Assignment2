import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import random
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import random

import matplotlib.pyplot as plt


tf.keras.utils.to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#read in the file data

from scipy.special import expit

def sigmoid(z):
    return expit(z)

def relu(Z):
    """
    ReLU activation function.

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of ReLU(Z), same shape as Z
    """
    A = np.maximum(0, Z)
    return A


def create_nn(layer_sizes, hidden_activation, output_activation, learning_algorithm):
    '''
    Create a fully connected feedforward neural network with arbitrary number of layers and neurons per layer.

    Parameters:
    layer_sizes (list): A list of integers specifying the number of neurons in each layer.
    hidden_activation (str): A string specifying the activation function to use in the hidden layers.
    output_activation (str): A string specifying the activation function to use in the output layer.

    Returns:
    nn (dict): A dictionary containing the weights and biases of the neural network.
    '''

    # Initialize weights and biases
    nn={}
    nn['num_layers'] = len(layer_sizes)
    nn['hidden_activation'] = hidden_activation
    nn['output_activation'] = output_activation
    nn['train_algorithm'] = learning_algorithm
    for i in range(1, len(layer_sizes)):
        #print(np.sqrt(2/layer_sizes[i-1]))
        nn['W' + str(i)] = np.random.randn(layer_sizes[i - 1], layer_sizes[i])* np.sqrt(2/layer_sizes[i-1])#*0.01#* np.sqrt(2/layer_sizes[i-1]) #*0.01
        nn['b' + str(i)] = np.zeros((1, layer_sizes[i]))
            # nn['prev_dW' + str(i)] = np.zeros((layer_sizes[i - 1], layer_sizes[i]))
            # nn['prev_db' + str(i)] = np.zeros((1, layer_sizes[i]))
        # Save number of layers and activation function
    return nn


def update_parameters(nn, grads, learning_rate, j):
    """
    Update the parameters of the neural network using the gradients and learning rate.

    Args:
    nn (dict): dictionary containing the parameters of the neural network
    grads (dict): dictionary containing the gradients of the parameters with respect to the cost function
    learning_rate (float): learning rate used to update the parameters

    Returns:
    nn (dict): updated dictionary containing the parameters of the neural network
    """

    # Update weights and biases for each layer
    #
    # if nn['train_algorithm'] == "RPROP":
    #     nn = rprop(grads, nn, learning_rate, j)
    #
    # elif nn['train_algorithm'] == "QProp":
    #     nn= quickprop(nn, grads, j, learning_rate, mu=1.75)
    # else:
    for i in range(1, nn['num_layers']):
        nn['W' + str(i)] -= learning_rate * grads['dW' + str(i)]
        nn['b' + str(i)] -= learning_rate * grads['db' + str(i)]

    return nn

def calculate_error(predictions, targets):
    # Calculate mean squared error
    mse = np.mean(np.square(predictions - targets)) # used in duda to plot the error. plots where correct by chance
    se = np.sum(np.square(predictions - targets))

    # Calculate squared error percentage
    omax = np.max(targets)
    omin = np.min(targets)
    #se_percent = 100 * np.sum(np.square(predictions - targets)) / (len(predictions) * len(targets[0]) * (omax - omin) ** 2)

    return mse

def train_nn(nn, X_train, X_validate, X_test, y_train, y_valid, y_test, num_epochs, learning_rate):
    '''
    Train a fully connected feedforward neural network.

    Parameters:
    X (array): Input data of shape (num_samples, input_dim).
    Y (array): Ground truth labels of shape (num_samples, num_classes).
    nn (dict): A dictionary containing the weights and biases of the neural network.
    learning_rate (float): Learning rate for gradient descent.
    num_iterations (int): Number of iterations to train the model.

    Returns:
    nn (dict): The updated neural network.
    '''
    train_error = []
    valid_error = []
    test_error = []
    epochs = []
    stop_train=0
    epoch=0
    print(num_epochs)
    while epoch<num_epochs:
        mse_erro_train=0
        for g in range(len(X_train)):
            # Forward propagation
            y_hat_train, cache = forward_propagation(X_train[g], nn)
            # print(y_hat_train)
            grads = backward_propagation(y_train[g], y_hat_train, X_train[g], cache, nn)
            nn = update_parameters(nn, grads, learning_rate, epoch)
            mse_erro_train += calculate_error(y_hat_train, y_train[g])
        train_error.append(mse_erro_train/(len(X_train)))

        y_hat_validate, cache_val = forward_propagation(X_validate, nn)

        y_hat_test, cache_test = forward_propagation(X_test, nn)

        mse_erro_val = calculate_error(y_hat_validate, y_valid)

        valid_error.append(mse_erro_val)

        mse_erro_test = calculate_error(y_hat_test, y_test)
        test_error.append(mse_erro_test)
        epochs.append(epoch)

        if epoch%5==0 and len(valid_error)>4 and valid_error[epoch]-valid_error[epoch-5]>=0:
            stop_train+=1
        if stop_train==5:
            break


        epoch+=1
        # Update weights and biases
        #print(i, " the vlsaue of the epoch")


    #print(np.argmin(valid_error))
    plt.figure()
    # Plot the first graph
    plt.plot(epochs, train_error, label='train error')

    # Plot the second graph on the same axis
    plt.plot(epochs, valid_error, label='valid error')

    plt.plot(epochs, test_error, label="test error")

    # Add legend and labels
    plt.legend()
    plt.xlabel('X-axis label')
    plt.ylabel('Y-axis label')
    plt.title('Multiple graphs on the same axis')

    # Display the plot
    plt.show()
    return nn


def get_data(filename):
    data=[]
    with open(filename, 'rb') as f:
        data = np.load(f)
    return data





#provide the activation functions
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def softmax(z):
    return tf.nn.softmax(z)


# Forward propagate through your own FFNN
def forward_propagation(X, nn):
    caches = []
    A_prev = X
    A_prev = np.array(A_prev)
    L = nn['num_layers']  # number of layers in the network
    for l in range(1, L):
        W = nn['W' + str(l)]
        b = nn['b' + str(l)]
        A = np.dot(A_prev, W) + b
        if l == L - 1:  # output layer
            if nn['output_activation'] == 'sigmoid':
                Z = sigmoid(A)
            elif nn['output_activation'] == 'softmax':
                Z = softmax(A)

            else:  # no activation
                Z = A
        else:  # hidden layers
            if nn['hidden_activation'] == 'sigmoid':
                Z = sigmoid(A)
            elif nn['hidden_activation'] == 'tanh': # tanh activation
                Z = tanh(A)
            elif nn['hidden_activation'] == 'relu':  # ReLU activation
                Z = relu(A)
            else:
                Z= softmax(A)
        caches.append({'A': A, 'Z': Z})
        A_prev = Z

    return Z, caches



#pefrom backporoagtion of the extracted output after the CNN has extracted features from the mnist data.
def backward_propagation(Y, Z, X, caches, nn):
    grads = {}
    L = len(caches)  # number of layers
    #m = len(Y)
    dZ =(Z - Y)

    for l in reversed(range(L)):
        current_cache = caches[l]
        if l >= 1:
            Z_prev = caches[l - 1]['Z']
        else:
            Z_prev = np.array(X)
        A = current_cache['A']
        if l == L - 1:  # output layer
            if nn['output_activation'] == 'sigmoid':
                dA = dZ * sigmoid(A)*(1-sigmoid(A))
            elif nn['output_activation'] == 'softmax':
                dA = dZ
            else:  # no activation
                dA = dZ
        else:  # hidden layers
            if nn['hidden_activation'] == 'sigmoid':
                dA = dZ * sigmoid(A)*(1-sigmoid(A))
            elif nn['hidden_activation'] == 'relu':  # ReLU activation
                dA = dZ * (A > 0)
            else:  # tanh activation
                dA = dZ * (1-np.square(tanh(A)))
        #print(np.shape(Z_prev[0])," ", np.shape(dA))
        dW = np.dot(Z_prev.T, dA)
        db = np.mean(dA, axis=0, keepdims=True)
        dZ = np.dot(dA, nn['W' + str(l + 1)].T)

        grads['dW' + str(l + 1)] = dW
        grads['db' + str(l + 1)] = db
    return grads


def main():


    train_images =get_data("image data 2828/images_train_28.npy")/255
    test_images = get_data("image data 2828/images_test_28.npy")/255
    train_labels = get_data("label data/labels_train.npy")
    test_labels_true = get_data("label data/labels_test.npy")
    new_train_images, val_images, new_train_labels, val_labels = train_test_split(train_images, train_labels,test_size=0.2)
    # random.seed(42)



    new_train_labels = keras.utils.to_categorical(new_train_labels)
    val_labels = keras.utils.to_categorical(val_labels)
    test_labels = keras.utils.to_categorical(test_labels_true)
    #print(test_labels[0:50], " test tables")



    #
    # # Extract the features from the CNN model

    #
    X_train = np.array([np.reshape(a, (1, 784)) for a in new_train_images])
    X_validate = np.array([np.reshape(a, (784)) for a in val_images])
    X_test = np.array([np.reshape(a, (784)) for a in test_images])
    print(np.shape(X_train))
    X_train= X_train
    X_validate = X_validate
    X_test = X_test


    # train_batch = []
    # train_batch_lables=[]
    # val_batch = []
    # val_batch_labels=[]
    #
    # for i in range(10):
    #     small_train_indices = random.sample(range(len(new_train_images)), int(len(new_train_labels) * 0.1))
    #     train_batch.append(X_train[small_train_indices])
    #     train_batch_lables.append(new_train_labels[small_train_indices])
    #
    # # Select 200 random images from the validation set
    #     val_indices = random.sample(range(len(val_images)), int(len(val_labels)))
    #     val_batch.append(X_validate[val_indices])
    #     val_batch_labels.append(val_labels[val_indices])

    #
    learning_rate=0.01
    layers = [784,64, 10]
    hidden_activation = "sigmoid"
    output_activation = "softmax"
    train_algorthim = ""
    use_simulated_annealing = False
    neural_network = create_nn(layers, hidden_activation, output_activation, train_algorthim)



    neural_network = train_nn(neural_network, X_train, X_validate, X_test, new_train_labels, val_labels, test_labels, 5 , learning_rate)
    predictions, _= forward_propagation(X_test, neural_network)
    #print(predictions[0:50])
    #print(small_test_labels[0:50])

    #print(len(test_pred))
    test_pred_labels= np.argmax(predictions, axis=1)
    for i in range(10):
        print(test_pred_labels[i], " ", test_labels_true[i])
    # print(test_pred_labels[0:10], test_labels_true[0:10])
    acc = accuracy_score(test_labels_true, test_pred_labels)
    print("acc: ", acc)
    cm = confusion_matrix(test_labels_true, test_pred_labels)

    cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = [0,1,2,3,4,5,6,7,8,9])
    cm_display.plot()
    plt.show()
    #accuracy_score_mine = accuracy_score(train_labels, predictions)
    # print(np.shape(np.argmax(predictions, axis =1 )))
    # # ffnn_output_test = forward_propagation(test_features, ffnn)

main()