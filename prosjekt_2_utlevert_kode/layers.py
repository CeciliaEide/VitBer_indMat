import numpy as np
from utils import onehot

class Layer:

    """
    Base class for layers in the neural network with forward and backward pass.
    """
    def __init__(self):
        
        return

    def forward(self,inputs):
        raise NotImplementedError

    def backward(self,grad):
        raise NotImplementedError
    
    def step_gd(self,alpha):
        """
        Performs a gradient descent step given learning rate.
        Assumes that the layer has a parameter dictionary "params" on the form

        params = {
            'w1': {         
                'w': w,         The parameter matrix
                'd': d,         The gradient of loss wrt the parameter matrix
                },
            'w2': {....},
            
        }
        where each parameter has a key 'w' for weights and 'd' for gradients.
        """
        for param in self.params:
            self.params[param]['w'] -= alpha*self.params[param]['d']
        return #lagt til, bør den ikke returnere noe her?

    def adam(self,alpha = 0.01, param):
        b_1 = 0.9
        b_2 = 0.999
        eps = 10**-8
        
        #Make V_0 and M_0 in same dimensions as W_i
        dimensions = param.shape
        rows, columns = dimensions
        V_0 = np.zeros([rows, columns])
        M_0 = np.zeros([rows, columns])

        W_j = param #setting init value for j-iteration

        for j in range()
            G_j = dLdW_j #legg inn riktig parameterliste med dens deriverte her
            M_j = b_1*M_j + (1-b_1)*G_j
            V_j = b_2*V_j + (1-b_2)*np.multiply(G_j,G_j)
            M_j_hat = (1/(1-(b_1)**j))*M_j
            V_j_hat = (1/(1-(b_2)**j))*V_j

            W_j = W_j - alpha(np.multiply(M_j_hat,(np.sqrt(V_j_hat)+eps)))
        
        return W_j

class Attention(Layer):

    def __init__(self,your_arguments_here):
        """
        Your code here
        """
        return

        

    def forward(self,z):

        #definisjon av D, D er en nxn matrise
        D = np.zeros((n,n))
        i1,i2 = np.tril_indices(n,-1)
        D[i1,i2] -= np.inf

        A = Softmax.forward((np.transpose(z)@np.transpose(W_Q)@W_K@x)+D) #definer parameterene. Hvor? Definer ogs z
        z_l = z + np.transpose(W_O)@W_V@z@A
        return z_l


    def backward(self,grad):
        gOV = np.transpose(W_v) @ W_o @ grad
        g_s = Softmax.backward(np.transpose(z) * gOV)
        dLdz = grad + gOV@np.transpose(A) + np.transpose(W_k)@W_q@z@g_s
        return dLdZ
    
    


class Softmax(Layer):
    """
    For å unngå numerisk ustabilitet (overflow i exp), kan du bruke følgende triks
    når du beregner P, Q i softmax over aksen axis:

    P = np.exp(x - x.max(axis=axis,keepdims=True))
    Q = np.sum(P,axis=axis,keepdims=True)
    """

    def __init__(self,z): #sjekk x, viser her til matrisen, hva gjør denne?
        """
        Your code here
        """
        self.z = z
        return

    
    def forward(self,z):
        P = np.exp(z - z.max(axis=axis,keepdims=True))
        Q = np.sum(P,axis=axis,keepdims=True)
        eps = 10**-8 #legges til for å unngå divisjon med null

        z_l = np.multiply(P,(Q+eps)**(-1))

        return z_l


    def backward(self,grad): #midlertidig
        P = np.exp(z - z.max(axis=axis,keepdims=True))
        S = np.multiply(P,((np.multiply(Q,Q)+eps)**-1))
        eps = 10**-8 #legges til for å unngå divisjon med null

        dLdZ = np.multiply(grad,forward(z))-np.multiply((np.multiply(grad,S)).sum(axis=0),P)
        return dLdZ



class CrossEntropy(Layer):

    def __init__(self,your_arguments_here):
        """
        Your code here
        """
        return

        

    def forward(self,y,Y_hat):
        Y = onehot(y)
        one = np.ones(m)
        p = one*np.multiply(Y_hat,Y) 
        q = -np.log(p) #naturlig eller tier?

        L = (1/n)*((q).sum(axis=0))

        return L


    def backward(self):
        eps = 10**-8
        dLdY = (1/n)*(np.multiply(Y,Y_hat+eps))
        return dLdY
    

l1 = LinearLayer(t,r)


layers = [l1,l2,....,l_U,l_S]

class LinearLayer(Layer):

    """
    Linear Layer
    """
    def __init__(self,input_size, output_size,init_scale = 0.1):
        """
        Constructor takes input size and output size of layer 
        and scale for the weights
        """

        #Initialize weights using a sample from the normal distribution
        #scaled with the init_scale
        self.w = np.random.randn(output_size,input_size)*init_scale
        self.params = {"w":{'w':self.w,
                            'd':np.zeros_like(self.w)}}
        

    def forward(self,x):
        """
        Computes the affine transformation of the forward pass
        Stores input for backwards pass and returns output y = Wx.

        x: input, array of shape (batch_size, input_size, n) = (b,d,n)
        y: output, array of shape (batch_size, output_size, n) = (b,o,n)
        """

        self.x = x
        
        #Return output of layer
        #y = w@x
        y = np.einsum('od,bdn->bon',self.params['w']['w'],x)
        return y
        
    def backward(self,grad):
        """
        Performs backward pass.

        grad: gradient of loss wrt output of layer, shape (batch_size, output_size, n) = (b,o,n)
        """

        b = grad.shape[0]

        #Compute gradient (average over B batches) of loss wrt weight w: 
        #dL/dw = (1/B)*sum_b^B (grad_b@x_b^T)
        self.params['w']['d'] = np.einsum('bon,bdn->od',grad,self.x)/b

        #Return gradient of loss wrt input of layer
        #dL/dw = w@grad.T
        return np.einsum('od,bon->bdn',self.params['w']['w'],grad)
    

class Relu(Layer):
    """
    Relu activation function
    """

    def __init__(self):
        return

    def relu(self,x):
        #relu(x) = max(0,x)
        return np.maximum(np.zeros(x.shape), x)

    def forward(self,x):
        
        #Store input for backwards pass
        self.x = x
        return self.relu(x)

    def backward(self,grad):

        #dL/dx = grad * relu'(x)
        return grad * np.where(self.x > 0, np.ones_like(self.x), np.zeros_like(self.x))



class EmbedPosition(Layer):
    def __init__(self,n_max,m,d,init_scale=1e-1):   

        """
        n_max: maximum length of input sequence
        m: number of items in the vocabulary / number of integers
        d: embedding dimension
        """

        #Initialize a linear layer for the embedding
        self.embed = LinearLayer(m,d,init_scale)
        #Initialize the position embedding matrix
        self.w = np.random.randn(d,n_max)*init_scale

        #Initialize the parameter dictionary for weight with key "Wp"
        self.params = {"Wp":{'w':self.w,'d':None}}

    def forward(self,X):

        """
        Input:
            X: one-hot encoded array of shape (b,m,n).

        Output:
            z_0: array of shape (b,d,n)

        embed.forward(X) maps (b,m,n) to (b,d,n). 
        Assigns a column of size d to each integer in the sequence
        and add positional embedding matrix (params['Wp']['w'][:,:n]) (b,d,n).

        Equivalent to 

        z_0 = W_E@X + W_P[:,:n]

        """

        #We assume that n < n_max
        n = X.shape[-1]
        z_0 = self.embed.forward(X) + self.params['Wp']['w'][:,:n]
        return z_0
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - None
        """

        
        b = grad.shape[0]

        #Compute gradient (average over B batches) of loss wrt positional embedding w:
        self.params['Wp']['d'] = np.zeros_like(self.w)
        self.params['Wp']['d'] += np.sum(grad,axis=0)/b

        #Use backwards pass of the linear layer
        self.embed.backward(grad)

        #This is always the final layer, so we return None
        return None
    
    def step_gd(self,step_size):

        #We need to call the step_gd method of the linear layer
        self.embed.step_gd(step_size)

        #And since we override step_gd(), we use super 
        #which calls the step_gd() of the base class
        #and does gd for the paramters in the params dict
        super().step_gd(step_size)




class FeedForward(Layer):


    def __init__(self,d, p,init_scale = 0.1):
        """
        Input:
            d: input dimension of first layer and output of second
            p: output dimension of first and input of second.

        """

        #first linear layer with input size d and output size p
        self.l1 = LinearLayer(d,p,init_scale)

        #We use the Relu activation function
        self.activation = Relu()

        #second linear layer with input size p and output size d
        self.l2 = LinearLayer(p,d,init_scale)


    def forward(self,x):
        """
        Input:
            - x of shape (b,d,n)
        Output:
            - shape (b,d,n)

        This is equivalent to
        y = x + W2.T@Relu(W1@x)

         (W1,W2 are p x d)
        """

        self.x = x

        return x + self.l2.forward(self.activation.forward(self.l1.forward(x)))
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - derivative of loss wrt input x. Shape (b,d,n)
        
        """

        #We use backward pass of the linear layers and activation.
        #Recall that the backward pass reverse the order of the layers. 
        grad_feed_forward = self.l1.backward(self.activation.backward(self.l2.backward(grad)))

        #Since forward pass is x + W2.T@Relu(W1@x)
        return grad + grad_feed_forward


    def step_gd(self,step_size):

        #Call the step_gd method of the linear layers
        self.l1.step_gd(step_size)
        self.l2.step_gd(step_size)