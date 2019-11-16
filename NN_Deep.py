import numpy as np
import matplotlib.pyplot as plt

import h5py

"""
    - 4 Hidden layers
    - No of neurons are specified by layers_dims variable
    - Activations are as follows:
        [Relu] -> [Relu] -> [Relu]-> [Sigmoid] 
    - The responsibility of the NN is to classify cat images
    - This is done as a part of Prof Andrew Ng's Neural Network course
"""


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def get_no_of_train_test_samples(train_set_x_orig, train_set_y, test_set_x_origm, test_set_y):
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_origm.shape[0]
    tot_pixel_size_per_img = train_set_x_orig.shape[1] * train_set_x_orig.shape[2] * \
                             train_set_x_orig.shape[3]

    return m_train, m_test, tot_pixel_size_per_img


def flatten_train_test_set(train_set_x_orig, test_set_x_orig):
    """
    Flatten the RGB image to a single column vector
    :param train_set_x_orig:
    :param test_set_x_orig:
    :return:
    """
    train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_x_flatten = train_x_flatten / 255
    test_x_flatten = test_x_flatten / 255
    return train_x_flatten, test_x_flatten


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    A = np.maximum(0, Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(
            layer_dims[l - 1])  # *0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    Z = W.dot(A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A_prev = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation="relu")
        A_prev = A
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  activation="sigmoid")

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers in the neural network
    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def predict(X, y, parameters):

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p


def model(X_train, Y_train, layers_dims, num_iterations=10000, learning_rate=0.5, print_cost=False):
    # layer sizes
    n_x = X_train.shape[0]
    n_y = Y_train.shape[0]

    # initialize parameters
    parameters = initialize_parameters_deep(layers_dims)
    costs = []
    for i in range(0, num_iterations):
        # forward prop
        AL, caches = L_model_forward(X_train, parameters)

        # cost function
        cost = compute_cost(AL, Y_train)
        costs.append(cost)
        # back prop
        grads = L_model_backward(AL, Y_train, caches)
        # update param
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration {0}: {1}".format(i, cost))

    return parameters, costs


def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probs, caches = L_model_forward(X, parameters)
    for i in range(0, probs.shape[1]):
        if probs[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p


def plot(cost, learning_rate):
    costs = np.squeeze(cost)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def exec_main():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    m_train, m_test, tot_pixel_size_per_img = get_no_of_train_test_samples(train_set_x_orig, train_set_y,
                                                                           test_set_x_orig,
                                                                           test_set_y)
    print("Train Set Size: " + str(m_train))
    print("Test Set Size: " + str(m_test))
    print("Pixel Size: " + str(tot_pixel_size_per_img))

    train_x_flatten, test_x_flatten = flatten_train_test_set(train_set_x_orig, test_set_x_orig)
    print("Flatten Train X Shape: " + str(train_x_flatten.shape))
    print("Flatten Train Y Shape: " + str(train_set_y.shape))
    print("Flatten Test X Shape: " + str(test_x_flatten.shape))
    print("Flatten Test Y Shape: " + str(test_set_y.shape))

    learning_rate = 0.0075
    layers_dims = [12288, 20, 7, 5, 1]  # 4-layer model
    parameters, costs = model(train_x_flatten, train_set_y, layers_dims,
                              num_iterations=2501,
                              learning_rate=learning_rate,
                              print_cost=True)

    y_preds_train = predict(train_x_flatten, train_set_y, parameters)
    y_preds_test = predict(test_x_flatten, test_set_y, parameters)

    plot(costs, learning_rate)


if __name__ == '__main__':
    exec_main()
