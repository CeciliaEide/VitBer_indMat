'''
import numpy as np

from layers import Attention, Softmax, CrossEntropy, LinearLayer, Relu, EmbedPosition, FeedForward, UnembedPosition
from neural_network import NeuralNetwork
from data_generators import get_xy_sort

def TrainingAlgorithm(n_iter, B, alpha):
    NN = NeuralNetwork()
    D = np.zeros(B)
    

    for j in range(1, n_iter+1):
        for k in range(1, B+1):
            x, y = D[k]
            NN.forward(x)
            NN.backward(y)
            NN.step_adam(alpha)

    return NN #Får i teorien et neural network som er ferdigtrent
'''


from data_generators import get_train_test_addition

#dimensjoner og størrelser til x og y
n_digits = 2
n_max = 3*n_digits
m = 10

#definerer størrelsen på parametermatrisene
d = 15
k = 5
p = 20

#henter treningsdata
data = get_train_test_addition(n_digits,samples_per_batch=100,n_batches_train=4)

from utils import onehot

x = data['x_train'][0]
X = onehot(x,m)
y = data['y_train'][0]

from layers import LinearLayer,EmbedPosition,Attention,Softmax,CrossEntropy,FeedForward

embed = EmbedPosition(n_max,m,d)
att1 = Attention(d,k)
ff1 = FeedForward(d,p)
un_embed = LinearLayer(d,m)
softmax = Softmax()
loss = CrossEntropy()