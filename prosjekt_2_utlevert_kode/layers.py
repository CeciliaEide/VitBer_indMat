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
    
    def step_adam(self, alpha):
        self.j +=1
        
        #Variable initialization 
        b_1 = 0.9 #first decaying average with proposed default value of 0.9
        b_2 = 0.999 #second decaying average with proposed default value of 0.999
        eps = 10**(-8) #variable for numerical stability during division

        for param in self.params:
            G_j = self.params[param]['d']
            
            M_j = b_1*self.params[param]['M'] + (1-b_1)*G_j
            V_j = b_2*self.params[param]['V'] + (1-b_2)*np.multiply(G_j,G_j)

            self.params[param]['M'] = M_j #Update M
            self.params[param]['V'] = V_j #Update V
            
            M_j_hat = (1/(1-(b_1)**self.j))*M_j
            V_j_hat = (1/(1-(b_2)**self.j))*V_j

            self.params[param]['w'] -= alpha*(M_j_hat/(np.sqrt(V_j_hat)+eps))




class Attention(Layer):

    def __init__(self,d,k,init_scale=0.1):
        #Initialize the parameter dictionary for weight with key "W_.."
        self.params = {"W_k":{'w':np.random.randn(k,d)*init_scale,'d':None, 'M':np.zeros((k,d)), 'V':np.zeros((k,d))},
                       "W_q":{'w':np.random.randn(k,d)*init_scale,'d':None, 'M':np.zeros((k,d)), 'V':np.zeros((k,d))},
                       "W_o":{'w':np.random.randn(k,d)*init_scale,'d':None, 'M':np.zeros((k,d)), 'V':np.zeros((k,d))},
                       "W_v":{'w':np.random.randn(k,d)*init_scale,'d':None, 'M':np.zeros((k,d)), 'V':np.zeros((k,d))}}
        
        self.softmax = Softmax()
        self.j = 0 
        return

        
    def forward(self,z):
        self.z = z
        n = z.shape[-1] 

        D = np.zeros((n,n))
        i1,i2 = np.tril_indices(n,-1)
        D[i1,i2] -= np.inf

        self.A = self.softmax.forward(np.einsum('bdN,kd,kD,bDn->bNn', z, self.params['W_q']['w'],self.params['W_k']['w'],z, optimize = True) + D)
    
        z_l = z + np.einsum('kd,kD,bDn,bNn->bdn', self.params['W_o']['w'], self.params['W_v']['w'], z, self.A, optimize=True)
        return z_l


    def backward(self,grad):
        b = grad.shape[0]
        
        gOV = np.einsum('kd,kD,bDn->bdn', self.params['W_v']['w'], self.params['W_o']['w'], grad, optimize = True)
        g_s = self.softmax.backward(np.einsum('bdn,bdN->bnN',self.z,gOV,optimize=True))
        h1 = np.einsum('bdn,BnN->bdN',gOV,np.transpose(self.A,(0,2,1)), optimize=True)
        h2 = np.einsum('kd,kD,bDn,bnN->bdN',self.params['W_k']['w'],self.params['W_q']['w'],self.z,g_s, optimize=True)
        h3 = np.einsum('kd,kD,bDn,bnN->bdN',self.params['W_q']['w'],self.params['W_k']['w'],self.z,np.transpose(g_s, (0, 2, 1)), optimize=True)
        dLdz = grad + h1 + h2 + h3

        #Compute gradient (average over B batches) of loss wrt weight w: #Gjøre til 2D array i einsum
        
        self.params['W_o']['d'] = np.einsum('kd,bdN,bNn,bdn->kd',self.params['W_v']['w'],self.z,self.A,grad, optimize=True)/b
        self.params['W_v']['d'] = np.einsum('kd,bdN,bNn,bdn->kd',self.params['W_o']['w'],grad,np.transpose(self.A, (0, 2, 1)),self.z, optimize=True)/b
        self.params['W_k']['d'] = np.einsum('kd,bdN,bNn,bdn->kd',self.params['W_q']['w'],self.z,g_s,self.z, optimize=True)/b
        self.params['W_q']['d'] = np.einsum('kd,bdN,bNn,bdn->kd',self.params['W_k']['w'],self.z,np.transpose(g_s, (0, 2, 1)),self.z, optimize=True)/b

        return dLdz
        



class Softmax(Layer):
    """
    For å unngå numerisk ustabilitet (overflow i exp), kan du bruke følgende triks
    når du beregner P, Q i softmax over aksen axis:

    P = np.exp(x - x.max(axis=axis,keepdims=True))
    Q = np.sum(P,axis=axis,keepdims=True)
    """

    def __init__(self):
        
        return
        

    def forward(self,z):
        self.z = z
        self.P = np.exp(z - z.max(axis=1,keepdims=True)) #Lagrer her for å kunne bruke i backward
        self.Q = np.sum(self.P,axis=1,keepdims=True)
        eps = 10**-8 #legges til for å unngå divisjon med null

        self.z_l = self.P/(self.Q+eps)

        return self.z_l


    def backward(self,grad): 
        eps = 10**-8 #legges til for å unngå divisjon med null
        S = self.P/(np.multiply(self.Q,self.Q)+eps)
        
        k1 = np.multiply(grad, self.z_l) 
        k2 = np.multiply(np.sum(np.multiply(grad,S),axis=1,keepdims=True),  self.P)
        #dLdz = np.multiply(grad, self.z_l) - np.multiply((np.multiply(grad,S)).sum(axis=1), self.P) #Se videre på
        return k1-k2




class CrossEntropy(Layer):

    def __init__(self):

        return

        
    def forward(self,Z,y):
        b, m, self.n = Z.shape
        self.Z = Z
        self.y = y
        self.n = Z.shape[-1]
        r = y.shape[1]

        self.Y = onehot(y,m) 
        self.Y_hat = Z[:,:,-r:]

        p = np.sum(np.multiply(self.Y_hat,self.Y),axis=1)
        q = -np.log(p) 

        L = np.mean(q)

        return L


    def backward(self):
        eps = 10**-8
        
        padded_Y = np.zeros_like(self.Z)
        padded_Y[:,:,-self.y.shape[-1]:] = self.Y 

        dLdY = -(1/self.n)*(padded_Y/(self.Z+eps)) 
        return dLdY




class LinearLayer(Layer):

    """
    Linear Layer
    """
    def __init__(self,input_size,output_size,init_scale = 0.1):
        """
        Constructor takes input size and output size of layer 
        and scale for the weights
        """

        #Initialize weights using a sample from the normal distribution
        #scaled with the init_scale
        self.w = np.random.randn(output_size,input_size)*init_scale
        self.params = {"w":{'w':self.w,
                            'd':np.zeros_like(self.w),
                            'M':np.zeros_like(self.w),
                            'V':np.zeros_like(self.w)}}
        
        self.j = 0
        

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
        self.params = {"Wp":{'w':self.w,'d':None, 'M':np.zeros_like(self.w), 'V': np.zeros_like(self.w)}}

        self.j = 0

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
    
    def step_adam(self,step_size):
        
        self.embed.step_adam(step_size)
        super().step_adam(step_size)




class FeedForward(Layer):


    def __init__(self,d,p,init_scale = 0.1):
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

    def step_adam(self,step_size):
        
        self.l1.step_adam(step_size)
        self.l2.step_adam(step_size)