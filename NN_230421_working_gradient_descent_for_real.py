import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)



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
            else:  # tanh activation
                Z = tanh(A)
        caches.append({'A': A, 'Z': Z})
        A_prev = Z
    return Z, caches


def backward_propagation(Y, Z, X, caches, nn):
    grads = {}
    L = len(caches)  # number of layers

    m = len(Y)
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
            else:  # tanh activation
                dA = dZ * (1-np.square(tanh(A)))
        dW = np.dot(Z_prev.T, dA)
        db = np.mean(dA, axis=0, keepdims=True)
        dZ = np.dot(dA, nn['W' + str(l + 1)].T)

        grads['dW' + str(l + 1)] = dW
        grads['db' + str(l + 1)] = db
    return grads

def quickprop(nn, grads,j, learning_rate, mu=1.75):
    if j == 0:
        for i in range(1, nn['num_layers']):
            nn['prev_dW' + str(i)] = np.ones_like(grads['dW' + str(i)]) * 1e-9
            nn['prev_db' + str(i)] = np.ones_like(grads['db' + str(i)]) * 1e-9

    for l in range(1, nn['num_layers']):
        prev_update = learning_rate * nn['prev_dW' + str(l)]

        delta_dW = grads['dW' + str(l)] / (nn['prev_dW' + str(l)] - grads['dW' + str(l)])

        update = np.clip(delta_dW * prev_update, -mu * prev_update, mu * prev_update)
        update_term = learning_rate * update

        nn['W' + str(l)] -= update_term
        nn['b' + str(l)] -= learning_rate * grads['db' + str(l)]
        nn['prev_dW' + str(l)] = grads['dW' + str(l)]
        nn['prev_db' + str(l)] = grads['db' + str(l)]
    return nn

def rprop(grads, nn, learning_rate,j,  eta_plus=1.2, eta_minus=0.5, delta_min=1e-6, delta_max=5):
    """
    Update the weights and biases of a neural network using the RPROP algorithm.

    Parameters:
    grads (dict): A dictionary containing the gradients of the loss function with respect to the weights and biases.
    nn (dict): A dictionary containing the weights and biases of the neural network.
    learning_rate (float): The learning rate for updating the weights.
    eta_plus (float): The scaling factor for increasing the step size.
    eta_minus (float): The scaling factor for decreasing the step size.
    delta_min (float): The minimum allowed step size.
    delta_max (float): The maximum allowed step size.

    Returns:
    nn (dict): A dictionary containing the updated weights and biases of the neural network.
    """
    if j==0:
       # print(j)
        for i in range(1, nn['num_layers']):
            nn['prev_dW' + str(i)] = 1e-9 * np.ones_like(grads['dW' + str(i)])
            nn['prev_db' + str(i)] = 1e-9 * np.ones_like(grads['db' + str(i)])

    for l in range(1, nn['num_layers']):
        dW = grads['dW' + str(l)]
        db = grads['db' + str(l)]
        prev_dW = nn['prev_dW' + str(l)]
        prev_db = nn['prev_db' + str(l)]
        delta_W = learning_rate * dW
        delta_b = learning_rate * db
        # Update weights
        mask = np.sign(dW * prev_dW)
        #print(mask[0])
        #print(mask)
        delta_W[mask > 0] *= eta_plus
        delta_W[mask < 0] *= eta_minus
        delta_W[mask == 0] *= 1.0
        delta_W = np.clip(delta_W, delta_min, delta_max)
        nn['W' + str(l)] -= np.sign(dW) * delta_W
        delta_W[mask < 0] *= 0
        # Update biases
        mask = np.sign(db * prev_db)

        delta_b[mask > 0] *= eta_plus
        delta_b[mask < 0] *= eta_minus
        delta_b[mask == 0] *= 1.0
        delta_b = np.clip(delta_b, delta_min, delta_max)
        nn['b' + str(l)] -= np.sign(db) * delta_b


        delta_b[mask<0]*=0

        # Update prev_dW and prev_db
        nn['prev_dW' + str(l)] = delta_W
        nn['prev_db' + str(l)] = delta_b

    return nn


def simulated_annealing(nn, X_train, y_train,layer_sizes,t_init=1, t_final=1e-8, num_iter=10000):
    '''
    Initialize the weights of a neural network using simulated annealing.

    Parameters:
    layer_sizes (list): A list of integers specifying the number of neurons in each layer.
    t_init (float): The initial temperature of the simulated annealing algorithm.
    t_final (float): The final temperature of the simulated annealing algorithm.
    num_iter (int): The number of iterations to run the simulated annealing algorithm.

    Returns:
    nn (dict): A dictionary containing the weights and biases of the neural network.
    '''

    # Initialize weights and biases with small random values

    for i in range(1, len(layer_sizes)):
        nn['W'+str(i)] = np.random.randn(layer_sizes[i-1], layer_sizes[i])*0.01
        nn['b'+str(i)] = np.zeros((1, layer_sizes[i]))
        nn['prev_dW' + str(i)] = np.zeros((layer_sizes[i - 1], layer_sizes[i]))
        nn['prev_db' + str(i)] = np.zeros((1, layer_sizes[i]))

    # Define the energy function as the mean squared error of the neural network
    def energy_function(nn):
        y_pred, caches = forward_propagation(X_train, nn)
        mse = np.mean((y_train - y_pred)**2)
        return mse

    # Define the neighbor function as a random perturbation of the weights
    def neighbor_function(nn, t):
        nn_new = nn.copy()
        #print(nn_new)
        for i in range(1, len(layer_sizes)):
            nn['W' + str(i)] += np.random.randn(*nn['W' + str(i)].shape) * t
            nn['b' + str(i)] += np.random.randn(*nn['b' + str(i)].shape) * t
        return nn_new

    # Run simulated annealing algorithm
    t = t_init
    for i in range(num_iter):
        nn_new = neighbor_function(nn, t)
        delta_e = energy_function(nn_new) - energy_function(nn)
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e/t):
            nn = nn_new
        t = t * (t_final/t_init)**(1/num_iter)

    return nn




def create_nn(layer_sizes, hidden_activation, output_activation, learning_algorithm, use_simulated_annealing, X_train, y_train):
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
    #print("train algorith", learning_algorithm, "\n\n")
    if use_simulated_annealing:
        print("gets here")

        nn= simulated_annealing(nn, X_train, y_train, layer_sizes)
    else:

        # Calculate the optimal range for the initial weights
        r = np.sqrt(6) / np.sqrt(layer_sizes[0] + layer_sizes[-1])

        # Initialize the weights with values drawn from a uniform distribution
        #W = np.random.uniform(low=-r, high=r, size=(layer_sizes[0], layer_sizes[-1]))
        for i in range(1, len(layer_sizes)):
            #print(np.sqrt(2/layer_sizes[i-1]))
            nn['W' + str(i)] = np.random.uniform(low = -r,high =  r, size=(layer_sizes[i - 1], layer_sizes[i]))#*0.01#* np.sqrt(2/layer_sizes[i-1]) #*0.01
            nn['b' + str(i)] = np.zeros((1, layer_sizes[i]))
            # nn['prev_dW' + str(i)] = np.zeros((layer_sizes[i - 1], layer_sizes[i]))
            # nn['prev_db' + str(i)] = np.zeros((1, layer_sizes[i]))


        # Save number of layers and activation function



    return nn

def momentum(nn, grads,j, learning_rate, beta=0.1):
    if j==0:
       # print(j)
        for i in range(1, nn['num_layers']):
            nn['prev_dW' + str(i)] = 1e-3 * np.ones_like(grads['dW' + str(i)])
            nn['prev_db' + str(i)] = 1e-3 * np.ones_like(grads['db' + str(i)])
    for i in range(1, nn['num_layers']):
        prev_dW = beta * nn['prev_dW' + str(i)] + (1 - beta) * grads['dW' + str(i)]
        prev_db = beta * nn['prev_db' + str(i)] + (1 - beta) * grads['db' + str(i)]
        nn['W' + str(i)] -= learning_rate * prev_dW
        nn['b' + str(i)] -= learning_rate * prev_db
        nn['prev_dW'] = prev_dW
        nn['prev_db'] = prev_db

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

    if nn['train_algorithm'] == "RPROP":
            nn = rprop(grads, nn, learning_rate, j)

    elif nn['train_algorithm'] == "QProp":

        nn = quickprop(nn, grads, j, learning_rate)
    elif nn == "momentum":
        nn = momentum(nn, grads,j, learning_rate, beta=0.9)
    else:
        #print("entered")
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
    se_percent = 100 * np.sum(np.square(predictions - targets)) / (len(predictions) * len(targets[0]) * (omax - omin) ** 2)

    return mse, se_percent


def train_nn(nn, X_train, X_validate, X_test, y_train, y_valid, y_test, num_epochs, learning_rate, d):
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
    best_nn={}

    for i in range(num_epochs):
        # Forward propagation
        y_hat_train, cache = forward_propagation(X_train, nn)
        y_hat_validate, cache_val = forward_propagation(X_validate, nn)
        y_hat_test, cache_test = forward_propagation(X_test, nn)
        # print(len(y_hat_train[0]), "len y_hat_train at first index")
        # print(y_hat_train[0], y_train[0])

        # Backward propagation
        # (Y, Z, X, caches, nn)
        grads = backward_propagation(y_train, y_hat_train, X_train, cache, nn)
        # print(grads)

        mse_erro_train, percent_error_train = calculate_error(y_hat_train, y_train)
        train_error.append(mse_erro_train)
        # print(train_error)
        mse_erro_val, percent_error_val = calculate_error(y_hat_validate, y_valid)
        valid_error.append(mse_erro_val)

        if np.argmin(valid_error) == i:
            best_nn = nn

        mse_erro_test, percent_error_test = calculate_error(y_hat_test, y_test)
        test_error.append(np.round(mse_erro_test, 9))

        epochs.append(i)

        # Update weights and biases
        nn = update_parameters(nn, grads, learning_rate, i)

    ff = np.argmin(valid_error)
    # print("min val error", (valid_error[ff]))
    # # aaa = np.argmin(train_error)
    # print("min train error", (train_error[ff]))
    test_results, _ = forward_propagation(X_test, best_nn)
    mse_error_test_results, _ = calculate_error(test_results, y_test)
    #print(test_results)

    #print(test_results)
    #print(test_results)
    test_results = np.round(test_results)
    #print(test_results[0:100])
    # print(y_test)
    #print(np.round(test_results))
    # count = 0
    # for i in range(len(test_results)):
    #     if np.array_equal(test_results[i], y_test[i]):
    #         count += 1
    # test_error_print = 1 - count / len(test_results)

    # print(test_results)
    # test_error_print = (1 - accuracy_score(np.argmax(y_test, axis=1), np.argmax(test_results, axis=1)))
    # print("test error: ", (test_error_print))

    print(d + 1, "\t", train_error[ff], "\t", valid_error[ff], "\t", mse_error_test_results)

        # with open('cancer_results_qprop.txt', 'a') as file:
        #     # Write new data to the file on a new line
        #     file.write(train_error[ff], valid_error[ff], test_error, "\n")
        # # with open("results.txt") as f:
        # #     write(val)

    # plt.figure()
    # # Plot the first graph
    # plt.plot(epochs, train_error, label='train error')
    #
    # # Plot the second graph on the same axis
    # plt.plot(epochs, valid_error, label='valid error')
    #
    # plt.plot(epochs, test_error, label="test error")
    #
    # # Add legend and labels
    # plt.legend()
    # plt.xlabel('X-axis label')
    # plt.ylabel('Y-axis label')
    # plt.title('Multiple graphs on the same axis')

    # Display the plot
    # plt.show()

    return nn


def read_in_data(filename):
    with open(filename, "r") as f:
        # read header lines
        bool_in = int(f.readline().strip().split("=")[1])
        real_in = int(f.readline().strip().split("=")[1])
        bool_out = int(f.readline().strip().split("=")[1])
        real_out = int(f.readline().strip().split("=")[1])

        num_inputs = bool_in + real_in
        num_outputs = bool_out + real_out
        # print(num_outputs, "outputs")

        training_examples = int(f.readline().strip().split("=")[1])
        validation_examples = int(f.readline().strip().split("=")[1])
        test_examples = int(f.readline().strip().split("=")[1])

        # read data lines
        # Read in the training set
        training_set_features = []
        training_set_labels = []
        for i in range(training_examples):
            line = f.readline().strip().split()
            example = [float(x) for x in line]
            features = example[:bool_in + real_in]
            label = example[bool_in + real_in:]
            training_set_features.append(features)
            training_set_labels.append(label)

        # Read in the validation set
        validation_set_features = []
        validation_set_labels = []
        for i in range(validation_examples):
            line = f.readline().strip().split()
            example = [float(x) for x in line]
            features = example[:bool_in + real_in]
            label = example[bool_in + real_in:]
            validation_set_features.append(features)
            validation_set_labels.append(label)

        # Read in the test set
        test_set_features = []
        test_set_labels = []
        for i in range(test_examples):
            line = f.readline().strip().split()
            example = [float(x) for x in line]  # reads in the data in the line and converts it to a float
            features = example[:bool_in + real_in]  # stores the feature data used as inputs to the nn
            label = example[bool_in + real_in:]  # stores the assocaited label data to compare to the output
            test_set_features.append(features)  # stores the test data and labels on the same line
            test_set_labels.append(label)
        # print(test_set_labels[0], "test labels 0")

        return num_inputs, num_outputs, training_set_features, training_set_labels, validation_set_features, validation_set_labels, test_set_features, test_set_labels


def main():
    learning_rate = 0.01
    iterations = 200

    num_inputs, num_outputs, training_set_features, training_set_labels, validation_set_features, validation_set_labels, test_set_features, test_set_labels = read_in_data(
       "proben1/card/card1.dt")
    # print(num_inputs, "in")
    # print(num_outputs, "out")
    # print(len(training_set_features[0]), "first set of training features")
    # print(len(training_set_features), "length of the whole set")
    hidden1 = int((num_inputs+num_outputs)/2)
    hidden2 = int(hidden1/2)
    layers = [num_inputs, hidden1, num_outputs]
    hidden_activation = "sigmoid"
    output_activation = "sigmoid"

    use_simulated_annealing = False



    # print(neural_network['W1'])
    #Z, caches = forward_propagation(training_set_features, neural_network)
    # print(A,"\n\n\n\n")
    # print(caches[-1])
    # print((caches[-1]['A']-caches[-1]['Z']))
    #grads = backward_propagation(training_set_labels, Z, training_set_features, caches, neural_network)
    #mse_erro_train, percent_error_train = calculate_error(Z, training_set_labels)
    #print(mse_erro_train)
    for i in range(10):
        train_algorthim1 = ""
        neural_network1 = create_nn(layers, hidden_activation, output_activation, train_algorthim1,
                                    use_simulated_annealing,
                                    training_set_features, training_set_labels)
        train_nn(neural_network1, training_set_features, validation_set_features, test_set_features, training_set_labels,
             validation_set_labels, test_set_labels, iterations, learning_rate, i)
    print("\n")
    for i in range(10):
        train_algorthim2 = "RPROP"

        neural_network2 = create_nn(layers, hidden_activation, output_activation, train_algorthim2,
                                    use_simulated_annealing,
                                    training_set_features, training_set_labels)
        train_nn(neural_network2, training_set_features, validation_set_features, test_set_features, training_set_labels,
             validation_set_labels, test_set_labels, iterations, learning_rate, i)
    print("\n")
    for i in range(10):
        train_algorthim = "Qprop"
        neural_network = create_nn(layers, hidden_activation, output_activation, train_algorthim,
                                   use_simulated_annealing, training_set_features, training_set_labels)
        train_nn(neural_network, training_set_features, validation_set_features, test_set_features, training_set_labels, validation_set_labels, test_set_labels, iterations, learning_rate, i)
    print("\n")
    for i in range(10):
        train_algorthim = "momentum"
        neural_network4 = create_nn(layers, hidden_activation, output_activation, train_algorthim,
                                   use_simulated_annealing, training_set_features, training_set_labels)
        train_nn(neural_network4, training_set_features, validation_set_features, test_set_features, training_set_labels, validation_set_labels, test_set_labels, iterations, learning_rate, i)

    # print(grads)
    # nn = update_parameters(neural_network, grads, 0.01)
    # forward_propagation()
    # train_nn(neural_network, training_set_features, validation_set_features, test_set_features, training_set_labels, validation_set_labels, test_set_labels, 5)


main()
