



class NeuralNetwork():
    def __init__(self, 
                 weight_matrices,           # list of matrices
                 biases,                    # list of vectors
                 activation_fun,  
                 activation_fun_derivative):
        
        self._weight_matrices = weight_matrices
        self._biases = biases
        self._activation_fun = activation_fun
        self._activation_fun_derivative = activation_fun_derivative

    def _vector_activation_fun(self, input):
        return [self._activation_fun(element) for element in input]
    
    def feed_forward(self, input):

        for n, matrix in enumerate(self._weight_matrices):
            
            current_sum = [0 for _ in range(len(matrix[0]))]

            # matrix multiplication
            for i, row in enumerate(matrix):
                for j in range(len(row)):
                    current_sum[j] += row[j]*input[j]
           
            # vector addition
            current_sum = [current_sum[k] + self._biases[n][k] for k in range(len(current_sum))]

            # new input is now the old sum (after activation function) and the cycle repeats
            input = self._vector_activation_fun(current_sum)

        prediction = self._vector_activation_fun(current_sum)

        return prediction
