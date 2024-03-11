#evt. lage en felles med å endre hva man henter av treningsdata med en if-løkke

def TrainingAlgorithmAdding():

    import numpy as np
    from data_generators import get_train_test_addition
    from utils import onehot
    from layers import LinearLayer,EmbedPosition,Attention,Softmax,CrossEntropy,FeedForward
    from neural_network import NeuralNetwork

    #dimensjoner og størrelser til x og y
    n_digits = 2
    n_max = 3*n_digits
    m = 10

    #definerer størrelsen på parametermatrisene
    d = 15
    k = 5
    p = 20

    #henter treningsdata - kan ta inn argumenter her avhengig av 
    data = get_train_test_addition(n_digits,samples_per_batch=250,n_batches_train=10)

    x = data['x_train'][0]
    X = onehot(x,m)
    y = data['y_train'][0]

    embed = EmbedPosition(n_max,m,d)
    att1 = Attention(d,k)
    ff1 = FeedForward(d,p)
    un_embed = LinearLayer(d,m)
    softmax = Softmax()
    loss = CrossEntropy()

    layers = [embed,att1,ff1,un_embed,softmax]
    nn = NeuralNetwork(layers)

    xs = data['x_train']
    ys = data['y_train']

    n_batches = xs.shape[0]
    n_iters = 100
    step_size = 0.1

    #treningsløkke tilsvarende algoritme 4 (med gradient descent)
    mean_losses = np.zeros(n_iters)
    for j in range(n_iters):
        losses = []
        for i in range(n_batches):
            x = xs[i]
            y = ys[i]

            X = onehot(x,m)
            Z = nn.forward(X)

            losses.append(loss.forward(Z,y))
            dLdZ = loss.backward()
            nn.backward(dLdZ)
            nn.step_adam(step_size)
        mean_loss = np.mean(losses)
        mean_losses[j] = mean_loss
        print("Iterasjon ", str(j), " L = ",mean_loss, "") #Kan fjernes senere, bruker bare mean_losses

    return nn, mean_losses

def prosentSortetRight(nn):
    from data_generators import get_xy_sort #Skal kanskje bare gjøre for sort
    
    x = None #hente data fra funk over, vil klassen neural network være med videre? Evt. sette rett inn
    y = None

    y_hat = nn.forward(x)
    like = y==y_hat
    prosent = np.sum(like)/len(y)

    return prosent