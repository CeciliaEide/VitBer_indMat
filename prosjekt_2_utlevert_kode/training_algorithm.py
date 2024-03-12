import numpy as np

def TrainingAlgorithm(problem):

    import numpy as np
    from data_generators import get_train_test_addition, get_train_test_sorting
    from utils import onehot
    from layers import LinearLayer,EmbedPosition,Attention,Softmax,CrossEntropy,FeedForward
    from neural_network import NeuralNetwork

    #definerer størrelsen på parametermatrisene
    d = 15
    k = 5
    p = 20

    """Opprette treningsdata for addisjonsproblem (problem = 0)"""
    if problem == 0:
        #dimensjoner og størrelser til x og y
        n_digits = 2
        n_max = 3*n_digits
        m = 10

        #henter treningsdata - kan ta inn argumenter her avhengig av 
        data = get_train_test_addition(n_digits,samples_per_batch=250,n_batches_train=10)

    """Opprette treningsdata for sorteringsproblem (problem = 1)"""
    if problem == 1:
        #dimensjoner og størrelser til x og y
        length = 4 #r = antall sifre i sekvensen
        m = 9 #M = eks heltall mellom 0 og 9, mulige siffre

        #henter treningsdata - kan ta inn argumenter her avhengig av 
        data = get_train_test_sorting(length, m, samples_per_batch = 250,n_batches_train = 10, n_batches_test = 1)
        n_max = length*2 - 1

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
    step_size = 0.01

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


def prosentSortetRight(nn, length, antall_forsok):
    from data_generators import get_xy_sort
    from utils import onehot
    
    velykkede_forsok = 0
    
    for i in range(antall_forsok):
        x, y = data = get_xy_sort(length)
        X = onehot(x)
        for i in range(length):
            Z = nn.forward(X)
            x.append(Z[-1:])
            X = onehot(x)

        y_hat = x[-length:]
        if y_hat == y:
            velykkede_forsok += 1

    return velykkede_forsok/antall_forsok


def prosentAddedRight(nn, antall_forsok):
    from data_generators import get_xy_sort #Skal kanskje bare gjøre for sort
    from utils import onehot
    
    length = 4 #Summere to to-sifrede tall
    
    velykkede_forsok = 0
    
    for i in range(antall_forsok):
        x = np.random.randint(0, 10, size=4)
        tall1 = x[0] * 10 + x[1]
        tall2 = x[2] * 10 + x[3]
        summen = tall1 + tall2
        y = np.array([summen%100, (summen%10)//10, summen//100])

        X = onehot(x)
        for i in range(3):
            Z = nn.forward(X)
            x.append(Z[-1:])
            X = onehot(x)
        y_hat = x[-3:]
        if y_hat == y:
            velykket_forsok += 1

    return velykket_forsok/antall_forsok
