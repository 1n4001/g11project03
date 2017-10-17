import numpy as np

# Training a two layered neural network to approximate the XOR function

# sigmoid function
def sigmoid( x , deriv = False ):
    if( deriv == True ):
        return x * ( 1 - x )
    return 1 / ( 1 + np.exp( -x ) )

# input dataset, [p, q, bias=1]
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
# output dataset, p XOR q
y = np.array([[0,1,1,0]]).T
print("X = \n", X)
print("y = \n", y)

# seed random numbers to make calculation deterministic
np.random.seed(42)
# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3,4)) - 1
syn1 = 2 * np.random.random((4,1)) - 1
print("Initial random weights: ")
print("synapse0 = \n", syn0)
print("synapse1 = \n", syn1)

for iter in range(60000):

    # forward propagation
    layer0 = X
    layer1 = sigmoid(np.dot( layer0 , syn0) )
    layer2 = sigmoid(np.dot( layer1 , syn1) )

    # back propagation
    layer2_error = y - layer2
    layer2_delta = layer2_error * sigmoid( layer2 , True )

    layer1_error = layer2_delta.dot(syn1.T)
    layer1_delta = layer1_error * sigmoid( layer1 , True )

    # update weights
    syn1 += layer1.T.dot(layer2_delta)
    syn0 += np.dot( layer0.T , layer1_delta )

print("\nOutput After Training:")
print("layer1 = \n", layer1)
print("layer2 = \n", layer2)
print("\nWeights:")
print("synapse0 = \n", syn0)
print("synapse1 = \n", syn1)
