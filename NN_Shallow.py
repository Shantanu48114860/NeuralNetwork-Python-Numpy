import numpy as np
import matplotlib.pyplot as plt

import h5py

"""
    - Two layer NN with one hidden layer with 4 neurons
    - Hidden layer activation function - tanh
    - Output layer activation function - sigmoid
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


def initialize(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x) * 0.01
    w2 = np.random.randn(n_y, n_h) * 0.01

    b1 = np.zeros((n_h, 1))
    b2 = np.zeros((n_y, 1))

    assert (w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (w2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {
        "W1": w1,
        "W2": w2,
        "b1": b1,
        "b2": b2
    }
    return parameters


def sigmoid(z):
    e = 1 + np.exp(-z)
    s = 1 / e
    return s


def forward_prop(X_train, parameters):
    w1 = parameters["W1"]
    w2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    A0 = X_train
    Z1 = w1 @ A0 + b1
    A1 = np.tanh(Z1)
    Z2 = w2 @ A1 + b2
    A2 = sigmoid(Z2)
    cache = {
        "A1": A1,
        "A2": A2,
        "Z1": Z1,
        "Z2": Z2
    }
    assert (A2.shape[1] == X_train.shape[1])
    return cache


def back_prop(cache, parameters, X, Y):
    A1 = cache["A1"]
    A2 = cache["A2"]
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    m = X.shape[1]

    dZ2 = A2 - Y
    dW2 = (1 / m) * dZ2 @ A1.T
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    g_prime = 1 - np.power(A1, 2)
    dZ1 = (W2.T @ dZ2) * g_prime

    dW1 = (1 / m) * dZ1 @ X.T
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    assert dW2.shape == W2.shape
    assert dW1.shape == W1.shape

    grads = {
        "dW1": dW1,
        "dW2": dW2,
        "db1": db1,
        "db2": db2
    }
    return grads


def calculate_cost(A2, Y):
    m = Y.shape[1]
    cost = ((-1) / m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))

    return cost


def update_parameters(grads, learning_rate, parameters):
    dW1 = grads["dW1"]
    dW2 = grads["dW2"]
    db1 = grads["db1"]
    db2 = grads["db2"]

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    W1 = W1 - (learning_rate * dW1)
    b1 = b1 - (learning_rate * db1)
    W2 = W2 - (learning_rate * dW2)
    b2 = b2 - (learning_rate * db2)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def model(X_train, Y_train, n_h, num_iterations=10000, learning_rate=0.5, print_cost=False):
    # layer sizes
    n_x = X_train.shape[0]
    n_y = Y_train.shape[0]

    # initialize parameters
    parameters = initialize(n_x, n_h, n_y)
    costs = []
    for i in range(0, num_iterations):
        # forward prop
        cache = forward_prop(X_train, parameters)

        # cost function
        cost = calculate_cost(cache["A2"], Y_train)
        costs.append(cost)

        # back prop
        grads = back_prop(cache, parameters, X_train, Y_train)

        # update param
        parameters = update_parameters(grads, learning_rate, parameters)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration {0}: {1}".format(i, cost))

    return parameters, costs


def predict(X, parameters):
    cache = forward_prop(X, parameters)
    y_hat = cache["A2"]
    y_pred = []
    for i in range(y_hat.shape[1]):
        if y_hat[0, i] > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)

    y_preds = np.array([y_pred])
    return y_preds


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

    learning_rate = 0.005

    parameters, costs = model(train_x_flatten, train_set_y, 4, num_iterations=10001,
                              learning_rate=learning_rate,
                              print_cost=True)

    y_preds_train = predict(train_x_flatten, parameters)
    y_preds_test = predict(test_x_flatten, parameters)
    train_accuracy = 100 - np.mean((y_preds_train - train_set_y)) * 100
    test_accuracy = 100 - np.mean(np.abs(y_preds_test - test_set_y)) * 100

    print("Train Accuracy: " + str(train_accuracy))
    print("Test Accuracy: " + str(test_accuracy))
    plot(costs, learning_rate)


if __name__ == '__main__':
    exec_main()
