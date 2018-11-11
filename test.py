import neuralnetwork as net
import mnist_loader

#loading MNIST handwritten number image data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

#Pre-training visual example of data
mnist_loader.show_data()

#Creating network with 784 input sigmoid neurons, 30 hidden, 10 output
net = net.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)