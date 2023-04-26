import random
import numpy as np
import matplotlib.pyplot as plt



# define the functions for creating and evaluating individuals
def create_nn(layer_sizes, hidden_activation, output_activation):
    nn = {}
    nn['num_layers'] = len(layer_sizes)
    nn['hidden_activation'] = hidden_activation
    nn['output_activation'] = output_activation
    for i in range(1, len(layer_sizes)):
        nn['W' + str(i)] = np.random.randn(layer_sizes[i - 1], layer_sizes[i]) *25
        nn['b' + str(i)] = np.random.random((1, layer_sizes[i]))*25
    return nn

def evaluate(nn, X, y_true):
    y_pred = forward_propagation(X, nn)
    mse = np.mean(np.square(np.array(y_pred) - np.array(y_true)))
    fitness = 1 / (1 + mse)
    nn['fitness'] = fitness
    return nn

def create_population(population_size, layer_sizes, hidden_activation, output_activation, X_train, y_train):
    population = []
    for i in range(population_size):
        nn = create_nn(layer_sizes, hidden_activation, output_activation)
        nn = evaluate(nn, X_train, y_train)
        population.append(nn)
    return population

def forward_propagation(X, nn):
    A = X
    for i in range(1, nn['num_layers']):
        W = nn['W' + str(i)]
        b = nn['b' + str(i)]
        Z = np.dot(A, W) + b
        if i == nn['num_layers'] - 1:
            if nn['output_activation'] == 'sigmoid':
                A = sigmoid(Z)
            elif nn['output_activation'] == 'softmax':
                A = softmax(Z)
            else:
                A = Z
        else:
            if nn['hidden_activation'] == 'sigmoid':
                A = sigmoid(Z)
            elif nn['hidden_activation'] == 'tanh':
                A = tanh(Z)
            else:
                A = relu(Z)
    return A

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

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

def tanh(Z):
    return np.tanh(Z)

def relu(Z):
    return np.maximum(0, Z)

def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def roulette_wheel_selection(population):
    '''
    Select two parents from the population using roulette wheel selection.

    Parameters:
    population (list): A list of dictionaries, where each dictionary contains the weights and biases of a neural network individual.

    Returns:
    parent1 (dict): The first parent selected from the population.
    parent2 (dict): The second parent selected from the population.
    '''

    # Calculate the sum of fitness scores in the population
    fitness_sum = sum([indiv['fitness'] for indiv in population])

    # Calculate the relative fitness of each individual in the population
    relative_fitness = [indiv['fitness'] / fitness_sum for indiv in population]

    # Choose two individuals from the population using roulette wheel selection
    parent1_idx = np.random.choice(len(population), p=relative_fitness)
    parent2_idx = np.random.choice(len(population), p=relative_fitness)

    parent1 = population[parent1_idx]
    parent2 = population[parent2_idx]

    return parent1, parent2


def crossover(parent1, parent2, crossover_rate):
    '''
    Perform crossover between two parent individuals to generate a new child individual.

    Parameters:
    parent1 (dict): The first parent individual.
    parent2 (dict): The second parent individual.
    crossover_rate (float): The probability of performing crossover between the parents.

    Returns:
    child (dict): The child individual generated by performing crossover between the parents.
    '''

    # Perform crossover with probability of crossover_rate
    x= np.random.random()
    if x < crossover_rate:
        # Choose a random crossover point
        crossover_point = np.random.randint(1, parent1['num_layers'])
        #print("crossed over")
        # Create the child individual
        child = {}
        child['num_layers'] = parent1['num_layers']
        child['hidden_activation'] = parent1['hidden_activation']
        child['output_activation'] = parent1['output_activation']

        #Copy the weights and biases of the parents up to the crossover point
        for i in range(1, crossover_point):
            child['W' + str(i)] = parent1['W' + str(i)].copy()
            child['b' + str(i)] = parent2['b' + str(i)].copy()

        # Copy the weights and biases of the other parent after the crossover point
        for i in range(crossover_point, parent2['num_layers']):
            child['W' + str(i)] = parent2['W' + str(i)].copy()
            child['b' + str(i)] = parent1['b' + str(i)].copy()


    # If no crossover is performed, simply copy one of the parents to create the child
    else:
        if np.random.random() < 0.5:
            child = parent1.copy()
        else:
            child = parent2.copy()

    return child


def mutation(child, mutation_rate, mutation_scale):
    '''
    Perform mutation on a child individual.

    Parameters:
    child (dict): The child individual to perform mutation on.
    mutation_rate (float): The probability of performing mutation on each weight or bias element.
    mutation_scale (float): The standard deviation of the Gaussian noise added to each weight or bias element during mutation.

    Returns:
    mutated_child (dict): The child individual after mutation has been performed.
    '''

    mutated_child = child.copy()
    x= np.random.random()
    if x < mutation_rate:

        for i in range(1, child['num_layers']):
            W_shape = child['W' + str(i)].shape
            b_shape = child['b' + str(i)].shape

        # Iterate over each weight element
        for j in range(W_shape[0]):
            for k in range(W_shape[1]):
                # Check if mutation should be performed
                if random.random() < mutation_rate:
                    # Add Gaussian noise to weight element
                    mutated_child['W' + str(i)][j, k] += random.gauss(0, mutation_scale)

        # Iterate over each bias element
        for j in range(b_shape[0]):
            # Check if mutation should be performed
            if random.random() < mutation_rate:
                # Add Gaussian noise to bias element
                mutated_child['b' + str(i)][j] += random.gauss(0, mutation_scale)
    return mutated_child

def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    winner = max(tournament, key=lambda individual: individual['fitness'])
    return winner



def evolve(population, X_train, y_train, X_val, y_val, elitism_rate, crossover_rate, mutation_rate, mutation_scale, num_epochs):
    '''
    Evolve a population of neural network individuals using the genetic algorithm.

    Parameters:
    population (list): A list of dictionaries, where each dictionary contains the weights and biases of a neural network individual.
    X_train (numpy.ndarray): An array of shape (num_samples, num_features) representing the input training data.
    y_train (numpy.ndarray): An array of shape (num_samples, num_outputs) representing the target training data.
    X_val (numpy.ndarray): An array of shape (num_samples, num_features) representing the input validation data.
    y_val (numpy.ndarray): An array of shape (num_samples, num_outputs) representing the target validation data.
    elitism_rate (float): The proportion of top individuals from the previous generation that are preserved for the next generation.
    crossover_rate (float): The probability of performing crossover between two parents when generating a new child.
    mutation_rate (float): The probability of performing mutation on an individual in the population.
    mutation_scale (float): The standard deviation of the normal distribution used for mutation.
    num_epochs (int): The number of epochs to train the neural networks for.

    Returns:
    population (list): The evolved population of neural network individuals.
    best_nn (dict): The neural network individual with the highest fitness score across all generations.
    '''
    all_fit=[]
    gener=[]
    mutation_scale=5
    stop_training = 0
    val_mean_fitness=[]
    # Train the initial population for num_epochs
    ccc=0
    for i in range(num_epochs):
        fit = 0
        for j in range(len(population)):
            nn = population[j]
            # Generate a minibatch from the training data
            # Perform forward propagation and backpropagation to update the weights and biases
            # Evaluate the fitness of the individual on the validation data
            population[j] = evaluate(nn, X_train, y_train)
            fit += population[j]['fitness']
        all_fit.append(fit/len(population))
        gener.append(i)
        # Sort the population by fitness score in descending order
        sort_population = sorted(population, key=lambda x: x['fitness'], reverse=True)
        best_old = sort_population[0]
        # Select the top individuals to be preserved for the next generation
        num_elites = int(len(population) * elitism_rate)
        elites = population[:num_elites]



        # Generate the next generation of individuals using selection, crossover, and mutation
        new_population = []

        while len(new_population) < len(population):
    # Select two parents using roulette wheel selection

            parent1 =tournament_selection(population, 2)

            parent2 = tournament_selection(population, 2)

            child = crossover(parent1, parent2, crossover_rate)
    # Generate a new child individual by performing crossover between the parents
            #parent1 = mutation(parent1, 1, mutation_scale)
            child = mutation(child, mutation_rate, mutation_scale)



    # Perform mutation on the child individual


    # Evaluate the fitness of the child individual on the validation data
            child = evaluate(child, X_val, y_val)

    # Add the child individual to the new population
            new_population.append(child)
        #print(len(new_population))
    # Preserve the elites in the new population
        #new_population[:num_elites] = elites
        n_fit=0
        val_fit=0
        sorted_new_pop = sorted(new_population, key=lambda x: x['fitness'], reverse=True)
        best = sorted_new_pop[0]
        #print(len(new_population))
        for f in new_population:
            n_fit += f['fitness']
            val_val = evaluate(f, X_val, y_val)
            val_fit+= val_val['fitness']

        n_fit = n_fit/(len(new_population))
        val_mean_fitness.append(val_fit/len(population))
        # print( n_fit, all_fit[-1])
        if n_fit>all_fit[-1]:
            pre_pop = population
            population = new_population

        mutation_rate = max(0.1, mutation_rate * (1 - i / num_epochs))
        crossover_rate = max(0.3, crossover_rate * (i / num_epochs))


        if n_fit<all_fit[-1]:
            stop_training+=1
        ccc=i
        if stop_training%5==0:
            mutation_rate = 0.9
            crossover_rate=0.3
        elif stop_training>30:
            break


    # print("exited loop", ccc)
    # all_fit = np.array(all_fit)
    # val_mean_fitness = np.array(val_mean_fitness)
    # plt.figure()
    # # Plot the first graph
    # plt.plot(gener, 1/all_fit-1, label='train error')
    # plt.plot(gener, 1/val_mean_fitness-1, label='validate error')
    #
    # # Plot the second graph on the same axis
    #
    #
    # # Add legend and labels
    # plt.legend()
    # plt.xlabel('iteration (epoch)')
    # plt.ylabel('MSE error')
    # plt.title('Train and validation MSE')
    #
    # plt.show()

    return new_population, all_fit, val_mean_fitness

def main():
    # define the Proben1 dataset
    num_inputs, num_outputs, X_train, y_train, X_val, y_val, X_test, y_test = read_in_data("proben1/heart/heart1.dt")
    #print(np.shape(X_train), np.shape(y_train))
    hidden1 = num_inputs+3
    hidden2 = int(hidden1/2)
    layers = [num_inputs, 5, num_outputs]
    population= create_population(50,[num_inputs, 5, num_outputs], "sigmoid","sigmoid", X_train, y_train) #80 orginally but 10 for large datasets

    # define the neural network architecture

    # define the genetic algorithm parameters
    elite_size = 1
    crossover_prob = 0.8 #set at 0.8 to indicate teh maximum corssoever rate. the algorthim iterativley increases it to 0.5 but can be changed to 0.8
    mutation_prob = 0.9
    num_epochs = 200

    for j in range(5):
        new_pop, train_mean, val_mean = evolve(population, X_train, y_train, X_val, y_val, elite_size, crossover_prob,
                                               mutation_prob, 5, num_epochs)
        for i in range(len(new_pop)):
            new_pop[i] = evaluate(new_pop[i], X_val, y_val)
        new_pop = sorted(new_pop, key=lambda x: x['fitness'], reverse=True)
        best_indivual = new_pop[0]

        pred = np.round(forward_propagation(X_test, best_indivual))
        test_fit = evaluate(best_indivual, X_test, y_test)
        count = 0

        index_min = np.argmin(val_mean)
        for i in range(len(pred)):
            if np.array_equal(pred[i], np.array(y_test[i])):
                count += 1
        # print(count / len(pred))

        print(j+1,"\t" ,1 / train_mean[-1] - 1,"\t", 1 / val_mean[-1] - 1,"\t" , 1- count / len(pred))

main()
