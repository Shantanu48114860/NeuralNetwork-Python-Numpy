import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import math

import h5py


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
    train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_x_flatten = train_x_flatten / 255
    test_x_flatten = test_x_flatten / 255
    return train_x_flatten, test_x_flatten


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def sigmoid(z):
    e = 1 + np.exp(-z)
    s = 1 / e
    return s


def propagate(w, b, X, Y):
    Z = (w.T @ X) + b
    A = sigmoid(Z)
    m = X.shape[1]
    dw = (1 / m) * (X @ (A - Y).T)
    db = (1 / m) * np.sum((A - Y))
    cost = ((-1) / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    grads = {"dw": dw, "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration {0}: {1}".format(i, cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):
    y_hat = sigmoid((w.T @ X) + b)
    y_pred = []
    for i in range(y_hat.shape[1]):
        if y_hat[0, i] > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)

    y_preds = np.array([y_pred])
    return y_preds


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    # initialize parameters with zeros (≈ 1 line of code)
    print(X_train.shape[0])
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train,
                                        num_iterations=num_iterations,
                                        learning_rate=learning_rate,
                                        print_cost=True)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    print(w.shape)
    print(b)
    print(costs[-1])
    y_preds_train = predict(w, b, X_train)
    y_preds_test = predict(w, b, X_test)

    train_accuracy = 100 - np.mean((y_preds_train - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(y_preds_test - Y_test)) * 100
    return {"costs": costs,
            "Y_prediction_test": test_accuracy,
            "Y_prediction_train": train_accuracy,
            "w": w,
            "b": b,
            "learning_rate": learning_rate,
            "num_iterations": num_iterations}


def plot(cost):
    costs = np.squeeze(cost)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train, m_test, tot_pixel_size_per_img = get_no_of_train_test_samples(train_set_x_orig, train_set_y, test_set_x_orig,
                                                                       test_set_y)
print("Train Set Size: " + str(m_train))
print("Test Set Size: " + str(m_test))
print("Pixel Size: " + str(tot_pixel_size_per_img))

train_x_flatten, test_x_flatten = flatten_train_test_set(train_set_x_orig, test_set_x_orig)
print("Flatten Train X Shape: " + str(train_x_flatten.shape))
print("Flatten Train Y Shape: " + str(train_set_y.shape))
print("Flatten Test X Shape: " + str(test_x_flatten.shape))
print("Flatten Test Y Shape: " + str(test_set_y.shape))

d = model(train_x_flatten, train_set_y, test_x_flatten, test_set_y, num_iterations=2000, learning_rate=0.005,
          print_cost=True)
plot(d["costs"])
