def TrainingAlgorithmAdding(n_iter):

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

    #henter treningsdata
    data = get_train_test_addition(n_digits,samples_per_batch=100,n_batches_train=4)

    x = data['x_train'][0]
    X = onehot(x,m)
    y = data['y_train'][0]

    
    embed = EmbedPosition(n_max,m,d)
    att1 = Attention(d,k)
    ff1 = FeedForward(d,p)
    un_embed = LinearLayer(d,m)
    softmax = Softmax()
    loss = CrossEntropy()

    #"manuelt" forward pass (tilsvarende algoritme 1)
    z0 = embed.forward(X)
    z1 = att1.forward(z0)
    z2 = ff1.forward(z1)
    z = un_embed.forward(z2)
    Z = softmax.forward(z)

    layers = [embed,att1,ff1,un_embed,softmax]
    nn = NeuralNetwork(layers)

    #forward pass tilsvarende algoritme 1
    Z = nn.forward(X)

    #beregner loss med CrossEntropy
    L = loss.forward(Z,y)
    print(L)

    #backward pass tilsvarende algoritme 2
    dLdZ = loss.backward()
    nn.backward(dLdZ)