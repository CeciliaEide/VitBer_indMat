import numpy as np

from layers import Attention, Softmax, CrossEntropy, LinearLayer, Relu, EmbedPosition, FeedForward, UnembedPosition
from neural_network import NeuralNetwork

def TrainingAlgorithm(n_iter, B):
    NN = NeuralNetwork()
    alpha = 0.01 #Endre senere
    D = None


    for j in range(1, n_iter+1):
        for k in range(1, B+1):
            x, y = D[k]
            NN.forward(x)
            NN.backward(y)
            NN.step_adam(alpha)