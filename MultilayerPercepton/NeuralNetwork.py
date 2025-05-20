



class NeuralNetwork():
    def __init__(self, 
                 weight_matrices: list[list[list[float]]], 
                 biases: list[list[float]], 
                 activation_function: function, 
                 activation_function_derivative: function):
        
        self.weight_matrices = weight_matrices
        self.biases = biases
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
    
    def feed_foward(self, inputs):
        return self.biases
