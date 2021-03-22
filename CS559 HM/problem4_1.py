# -*- coding: utf-8 -*-
"""

@author: Jiang
"""

import sklearn.datasets as db
import numpy as np
import matplotlib.pyplot as plt


def change_one_hot_label(X):
   
    T = np.zeros((X.size, 3))  
    for idx, row in enumerate(T):
        # idx return the row number，row return the value of row，like[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        row[X[idx]] = 1  
    return T

def load_iris(t_sample=130):
    
    iris_set = db.load_iris()
    train_mask = np.random.choice(150, 130,replace=False) 
    base_array = np.array([i for i in range(0, 150)]) 
    test_mask = np.delete(base_array, train_mask, axis=0)  
    x_train = iris_set.data[train_mask]  
    t_train = iris_set.target[train_mask]  
    x_test = iris_set.data[test_mask] 
    t_test = iris_set.target[test_mask] 

    # translate into one-hot label
    t_train = change_one_hot_label(t_train)
    t_test = change_one_hot_label(t_test)

    return (x_train, t_train), (x_test, t_test)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    # sigmoid grad is (1 - y)*y,which y = sigmoid(x)
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    # Softmax exponentially normalizes x, and the output is similar to "probability", with a total of 1.0
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)

        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
      
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # A1 = X*W1 + B1
        # Z1 = sigmoid(A1)
        # A2 = Z1*W2 + B2
        # Y = softmax(A2)
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)  # activate
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)  # normalization processing

        return y

    def loss(self, x, t):
        y = self.predict(x)
        # construct the loss function based on cross entropy product 
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1) 
        t = np.argmax(t, axis=1)  
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

#if __name__ == '__main__':
(x_train, t_train), (x_test, t_test) = load_iris(t_sample=130)
network = TwoLayerNet(input_size=4, hidden_size=10, output_size=3)

iters_num = 1000  
train_size = x_train.shape[0]  # train_size = 130
batch_size = 10  # Number of imported data per batch
learning_rate = 0.05

train_loss_list = []  
train_acc_list = []
test_acc_list = []


iter_per_epoch = max(train_size / batch_size, 1)  

for i in range(iters_num):
    
    batch_mask = np.random.choice(train_size, batch_size)
  
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # Calculate the gradient
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)  

    # update parameters
    for key in ('W1', 'b1', 'W2', 'b2'):
        # W <-- W - lr*Grad
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)


    if i % iter_per_epoch == 0:
        # Calculate the recognition accuracy of the train set
        train_acc = network.accuracy(x_train, t_train)

        # Calculate the recognition accuracy of the test set
        test_acc = network.accuracy(x_test, t_test)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# Draw the picture
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()



x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.xlim(0, len(train_loss_list))
plt.show()