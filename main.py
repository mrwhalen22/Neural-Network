import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()

def create_data(points, classes):
    # creates data points for N amount of classes and points
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        #initializes weights and biases
        self.biases = np.zeros((1, n_neurons))
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01

    def forward(self, inputs):
        # calculates each neurons output based on weights and losses
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        # sets the output to 0 if inputs are 0 or less
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        # used to get all positive outputs normalized to a probability dist.
        self.output = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output /= np.sum(self.output, axis=1, keepdims=True)

class Loss:
    def calculate(self, output, y):
        # returns the mean loss of the input batch
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_Entropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # clip range to prevent calculating log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # check for 1-hot-encoding vs target classes exact references
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        # actual Loss entropy functionality
        neg_log_prob = -np.log(correct_confidences)
        return neg_log_prob

X, y = spiral_data(100,3)

layer1 = Layer_Dense(2,3)
activation_relu = Activation_ReLU()

layer2 = Layer_Dense(3, 3)
activation_soft = Activation_Softmax()

layer1.forward(X)
activation_relu.forward(layer1.output)

layer2.forward(activation_relu.output)
activation_soft.forward(layer2.output)

loss_function = Loss_Entropy()
loss = loss_function.calculate(activation_soft.output, y)
print(loss)



