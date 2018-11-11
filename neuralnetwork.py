import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        #first layer assumed input so no bias applied
        self.num_layers = len(sizes)
        self.sizes = sizes
        #biases and weights initiated randomly w/ gaussian dist.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    #return output using a as input
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    #Stochastic Gradient Descent
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):

        #data as list of tuples (x, y) --> (training inputs, desired output)
        training_data = list(training_data)
        n = len(training_data)
        #if test_data provided evaluate after each epoch and print progress
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        #running through epochs and mini_batches
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(j))

    #apply gradient descent with backprop to mini_batch
    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(learning_rate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, a, y):
        #tuples for cost gradient, layer-by-layer arrays
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = a
        # list to store all the activations, layer-by-layer
        activations = [a]
        # list to store all the z vectors (weighted input), layer-by-layer
        zs = []
        #forward pass
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l = layers where 1 is last layer, 2 previous, etc.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    #return num of correct predictions (neuron with highest activation value = desired output)
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    #vector of cost derivatives
    def cost_derivative(self, a, y):
        return (a - y)

#sigmoid functions
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))