import random
import math
import csv

from NeuralNetwork import NeuralNetwork




# Activation function and its derivatives
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)






# Initialize weights and bias
def initialize_neural_network(structure: list[int],
                              activation_function,
                              activation_function_derivative):
    # a structure of type [a, b_1, ..., b_n, c] means:
    # a inputs, b_i neurons for each i hidden layer, and c outputs

    if (len(structure) <= 1):
        return "NN must more than one layer"

    weight_matrices = []
    biases = []

    # there are len(structure)-1 weight matrices
    for m in range(len(structure) - 1):

        # we will now construct the weight matrices by analysing all two consecutive layers
        # the first layer has structure[m] neurons
        # the second layer has structure[m+1] neurons
        # the weight matrix has a dimension of structure[m+1] x structure[m]
        # for example, if the structure is [2, 3, 4], then the weight matrices are in order, 3x2 and 4x3
        weight_matrices.append([[random.uniform(-1, 1) for _ in range(structure[m+1])] for _ in range(structure[m])])

        # finally there are as many biases as neurons on the second layer
        # biases.append([random.uniform(-1, 1) for _ in range(structure[m+1])])
        biases.append([random.uniform(-1, 1) for _ in range(structure[m+1])])

    # record the weights and biases in a csv file
    with open('MultilayerPercepton/Storage/example.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        i = 0
        for matrix in weight_matrices:
            for row in matrix:
                writer.writerow(row)
            writer.writerow([])
            writer.writerow(biases[i])
            writer.writerow([])
            i+=1

    # return these matrices inside a custum class object
    nn = NeuralNetwork(weight_matrices, biases, activation_function, activation_function_derivative)
    return nn
        

def main():
    nn_structure = [2,1]
    activation_function = sigmoid
    activation_function_derivative = sigmoid_derivative

    neuralNetwork = initialize_neural_network(nn_structure, activation_function, activation_function_derivative)

    data = [[[0, 0], 0], [[0, 1], 1], [[1, 0], 1], [[1, 1], 0]]

    # hyper parameters


    # Training loop
    while True:
        total_error = 0

        for input, output in data:

            prediction = neuralNetwork.feed_forward(input)
            print(prediction)

            break

        break








main()


# # Dataset for training (XOR problem)
# inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
# outputs = [0, 1, 1, 0]

# # Hyperparameters
# learning_rate = 0.1
# epochs = 10000

# # Training loop
# for epoch in range(epochs):
#     total_error = 0
#     for x, y in zip(inputs, outputs):
#         # Forward propagation
#         weighted_sum = sum(w * xi for w, xi in zip(weights, x)) + bias
#         prediction = sigmoid(weighted_sum)

#         # Compute error
#         error = y - prediction
#         total_error += error ** 2

#         # Backward propagation
#         gradient = error * sigmoid_derivative(prediction)
#         weights = [w + learning_rate * gradient * xi for w, xi in zip(weights, x)]
#         bias += learning_rate * gradient

#     if epoch % 1000 == 0:
#         print(f"Epoch {epoch}, Error: {total_error:.4f}")

# # Test
# for x in inputs:
#     weighted_sum = sum(w * xi for w, xi in zip(weights, x)) + bias
#     prediction = sigmoid(weighted_sum)
#     print(f"Input: {x}, Prediction: {round(prediction)}")
    