import numpy as np
import random
from scipy.special import expit
import matplotlib.pyplot as plt


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


    return num_inputs, num_outputs, training_set_features, training_set_labels, validation_set_features, validation_set_labels, test_set_features, test_set_labels

def xavier_init(n_in, n_out):
    xavier_stddev = np.sqrt(2 / (n_in + n_out))
    #print(xavier_stddev)
    return np.random.normal(0, xavier_stddev, (n_in, n_out))

def create_nn(layer_sizes, hidden_activation, output_activation):
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
    nn['PBest']=None


    for i in range(1, len(layer_sizes)):
        nn['W' + str(i)] = np.random.random((layer_sizes[i - 1], layer_sizes[i]))*4 # * np.sqrt(2/layer_sizes[i-1]) #*0.01
        nn['b' + str(i)] = np.random.random((1, layer_sizes[i]))*4
        nn['vW'+str(i)] = np.zeros((layer_sizes[i - 1], layer_sizes[i]))
        nn['vb'+str(i)] = np.zeros((1, layer_sizes[i]))

    return nn

def initialze_swarm(num_particles, layer_sizes, hidden_activation, output_activation):
    swarm=[]
    for i in range(num_particles):
        particle = create_nn(layer_sizes, hidden_activation, output_activation)
        swarm.append(particle)
    return swarm

def sigmoid(z):
    return expit(z)


def tanh(z):
    return np.tanh(z)


def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forward_propagation(X, nn):
    caches = []
    A_prev = X
    A_prev = np.array(A_prev)
    #print(nn['num_layers'])
    L = nn['num_layers']  # number of layers in the network
    #print(L)
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
            else:
                Z= softmax(A)
        A_prev = Z
    return Z

def evaluate(nn, X, y_true):
    '''
    Evaluate the performance of a neural network on a given dataset.

    Parameters:
    nn (dict): A dictionary containing the weights and biases of the neural network.
    X (ndarray): An input dataset of shape (num_features, num_examples).

    Returns:
    y_hat (ndarray): The predicted output of the neural network for the given input dataset.
    '''

    # Forward propagate through the neural network
    Z  =forward_propagation(X, nn)
    #print(Z)
    P = len(y_true)
    y_hat = Z
    #print(y_hat[0], y_true[0], " predicted vs actual")
    # y_hat = np.where(y_hat> 0.95,1, y_hat)
    #
    # y_hat = np.where(y_hat<1e-3, 0, y_hat)

    # Calculate the mean s


    mse = np.mean(np.square(np.array(y_hat)- np.array(y_true)), axis=0)



    omax = np.max(y_hat)
    omin = np.min(y_hat)

    fitness = 1/(1+mse)

    nn['fitness_score']=fitness
    return nn, y_hat


def find_Pbest(particle, PBest, counter):
    if particle['PBest'] is None:
        particle['PBest'] = particle['fitness_score']
        PBest[counter] = particle
    else:

        if np.mean(particle['PBest']) < np.mean(particle['fitness_score']):
            particle['PBest']=particle['fitness_score']
            PBest[counter]=particle


    return particle, PBest


def train(training_set_labels, training_set_features, swarm, num_particles, epoch, w, c1, c2, PBest, Gbest, swarm_train_mean_fitness):
    train_fit = 0


    counter = 0

    total_particle_fitness=0
    for particle in swarm:
        for g in range(len(training_set_features)):
            particle, Z = evaluate(particle, training_set_features[g], training_set_labels[g])
            particle, PBest = find_Pbest(particle, PBest, counter)

            train_fit += particle['fitness_score']
        counter+=1
        total_particle_fitness += train_fit/len(training_set_labels)
        train_fit=0
        if not Gbest:
            Gbest = particle
        else:
            #result = a > b
            if np.mean(particle['PBest']) > np.mean(Gbest['PBest']):
                # print(particle['PBest'], " PBEst")
                # print(np.mean(particle['PBest']), " ", np.mean(Gbest['PBest']))
                Gbest = particle
    swarm_train_mean_fitness.append(total_particle_fitness/len(swarm))


    count = 0
    #print(PBest['P_Num' + str(4)], "particle 5")
    for particle in swarm:
        for j in range(1, particle['num_layers']):
            r2 = np.random.random(particle['W' + str(j)].shape)

            r1 = np.random.random(particle['W' + str(j)].shape)
            #print(count)
            #print(PBest['P_Num' + str(0)]['W' + str(j)])
            velocity = w*particle['vW' + str(j)] + c1 * r1 * (PBest[count]['W' + str(j)] - particle['W'+ str(j)]) + c2 * r2 * (Gbest['W' + str(j)] - particle['W' + str(j)])
            particle['vW' + str(j)] = np.clip(w*particle['vW' + str(j)] + c1 * r1 * (PBest[count]['W' + str(j)] - particle['W' + str(j)]) + c2 * r2 * (Gbest['W' + str(j)] - particle['W' + str(j)]), -0.4, 0.4)
            r2 = np.random.random(particle['b' + str(j)].shape)*0.1
            r1 = np.random.random(particle['b' + str(j)].shape)*0.1

            particle['vb'+ str(j)] = np.clip(w * particle['vb' + str(j)] + c1 * r1 * (PBest[count]['b' + str(j)] - particle['b' + str(j)]) + c2 * r2 * (Gbest['b' + str(j)] - particle['b' + str(j)]),-0.4,0.4)
        count += 1


    # this updates the postion ie weights and biases of the each particle in the swarm
    for particle in swarm:
        for j in range(1, particle['num_layers']):
            particle['W' + str(j)] += particle['vW' + str(j)]
            particle['b' + str(j)] += particle['vb' + str(j)]

    return swarm, swarm_train_mean_fitness

def update_hyperparameters(mean_fitness,c1,c2,w,c1_initial,c2_initial,w_initial,stop_training,stop_training_2, epoch):
    if len(mean_fitness) > 1:
        difference = mean_fitness[-1] - mean_fitness[-2]
        if np.mean(mean_fitness[-1]) > 0.85:
            #c1 = c1 + 0.005
            #c2 = c2+ 0.005
            #w = w - 0.01
            if c1 > 2 * c1_initial:
                c1 = 2 * c1_initial
            if c2 > 2 * c2_initial:
                c2 = 2 * c2_initial
            if w < 0.5 * w_initial:
                w = 0.5 * w_initial
            if np.mean(mean_fitness[-1]) - np.mean(mean_fitness[-2]) < 0.01:
                stop_training += 1

        else:  # mean_fitness[-1] < 0.85
            # c1 = c1 -0.005
            # c2 = c2 - 0.005
            w = w - 0.05
            if c1 < 0.5 * c1_initial:
                c1 = 0.5 * c1_initial
            if c2 < 0.5 * c2_initial:
                c2 = 0.5 * c2_initial
            if w < 0.5 * w_initial:
                w = 0.3 * w_initial
            if np.mean(mean_fitness[-1]) -np.mean(mean_fitness[-2]) < 0.01:
                stop_training_2 += 1

    return c1, c2, w ,stop_training, stop_training_2, mean_fitness


def validate_swarm(swarm,validation_set_labels, validation_set_features, mean_fitness_val, Gbest_val, PBest_val):
    fitness_val = 0
    counter=0
    total_particle_fitness=0

    for particle in swarm:
        for v in range(len(validation_set_labels)):
            particle, predictions = evaluate(particle, validation_set_features[v], validation_set_labels[v])
            particle, PBest_val = find_Pbest(particle, PBest_val, counter)
            fitness_val += particle['fitness_score']
        total_particle_fitness+=(fitness_val/len(validation_set_labels))
        fitness_val = 0
        counter += 1

        if not Gbest_val:
            Gbest_val = particle
        else:
            #print(particle['PBest'], " PBeestst s ")
            if np.mean(particle['PBest']) > np.mean(Gbest_val['PBest']):
                Gbest_val = particle

    mean_fitness_val.append((total_particle_fitness) / (len(swarm)))
    return mean_fitness_val, Gbest_val

def main():

    #this method does not use the best model found during trainng but rather the model at the end of the training or when the stop training criterai is met
    num_inputs, num_outputs, training_set_features, training_set_labels, validation_set_features, validation_set_labels, test_set_features, test_set_labels = read_in_data("proben1/thyroid/thyroid2.dt")
    num_particles = 100
    layer_sizes = [num_inputs,5,num_outputs]
    hidden_activation = "sigmoid"
    output_activation = "sigmoid"

    PBest={}
    PBest_val={}
    Gbest = {}
    Gbest_val={}
    mean_fitness_val =[]

    epochs=200
    c1 = 2.05
    c2 = 2.05
    w = 0.9
    w_min = 0.4
    c1_initial = c1
    c2_initial = c2
    w_initial = w
    epoch=0
    batch_size = int(len(training_set_labels) * 0.1)


    training_set_features = np.split(training_set_features, range(batch_size, len(training_set_features), batch_size))
    training_set_labels = np.split(training_set_labels, range(batch_size, len(training_set_labels), batch_size))
    stop_train = 0
    stop_training_2 = 0
    batch_size_val = int(len(validation_set_labels) * 0.1)
    validation_set_features = np.split(validation_set_features, range(batch_size_val, len(validation_set_features), batch_size_val))
    validation_set_labels = np.split(validation_set_labels,range(batch_size_val, len(validation_set_labels), batch_size_val))

    for i in range(5):
        epoch_hold = []
        epoch = 0
        swarm = initialze_swarm(num_particles, layer_sizes, hidden_activation, output_activation)
        stop_train=0
        swarm_train_mean_fitness=[]
        swarm_validate_mean_fitness=[]
        for g in range(epochs):

            if stop_train == 10 or stop_training_2==10:
                print("entered")
                break
            epoch_hold.append(epoch)
            w = w_initial - ((w_initial - w_min) / epochs) * epoch

            swarm, swarm_train_mean_fitness = train(training_set_labels, training_set_features, swarm, num_particles,
                                                    epoch,
                                                    w, c1, c2, PBest, Gbest, swarm_train_mean_fitness)

            swarm_validate_mean_fitness, Gbest_val = validate_swarm(swarm, validation_set_labels,
                                                                    validation_set_features,
                                                                    swarm_validate_mean_fitness, Gbest_val, PBest_val)


            c1, c2, w, stop_train, stop_train_2, swarm_validate_mean_fitness = update_hyperparameters(swarm_validate_mean_fitness, c1, c2, w, c1_initial, c2_initial, w_initial, stop_train, stop_training_2,epoch)


            epoch += 1



        Gbest_test, predictions = evaluate(Gbest_val, test_set_features, test_set_labels)
        accruacy_score = 0


        for d in range(len(predictions)):
            if np.array_equal(np.round(np.array(predictions[d])), np.array(np.round(test_set_labels[d]))):
                accruacy_score += 1

        swarm_train_mean_fitness_2 =[]
        swarm_validate_mean_fitness_2=[]
        #print(len(swarm_validate_mean_fitness))
        for arr in range(len(swarm_validate_mean_fitness)):
            swarm_validate_mean_fitness_2.append(1/np.mean(swarm_validate_mean_fitness[arr])-1)
            swarm_train_mean_fitness_2.append(1/np.mean(swarm_train_mean_fitness[arr])-1)

        #print("exited loop", ccc)
        # all_fit = np.array(all_fit)
        # val_mean_fitness = np.array(val_mean_fitness)
        # plt.figure()
        # # Plot the first graph
        # plt.plot(epoch_hold, swarm_train_mean_fitness_2, label='train error')
        # plt.plot(epoch_hold, swarm_validate_mean_fitness_2, label='validate error')
        #
        # # Plot the second graph on the same axis
        #
        # # Add legend and labels
        # plt.legend()
        # plt.xlabel('iteration (epoch)')
        # plt.ylabel('MSE error')
        # plt.title('Train and validation MSE')
        #
        # plt.show()

        #print(swarm_validate_mean_fitness_2, "mean fit")
        #
        min_index = np.argmin(swarm_validate_mean_fitness_2)




        print(i+1, "\t", 1/np.mean(swarm_train_mean_fitness[min_index])-1, "\t", 1/np.mean(swarm_validate_mean_fitness[min_index])-1, "\t", 1-accruacy_score/len(predictions))


        #mean_fitness=[]
main()





