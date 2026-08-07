import numpy as np
from data_loader import load_data

encoded_labels, features = load_data()

#print(features.shape)
#print(encoded_labels.shape)

w1 = np.random.randn(784,30)
b1 = np.zeros(30)
print(b1.shape)

w2 = np.random.randn(30,10)
b2 = np.zeros(10)

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x,axis = 1, keepdims = True)

#forward pass
z1 = np.dot(features,w1) + b1
a1 = sigmoid(z1)

#2nd pass
z2 = np.dot(a1,w2) + b2
a2 = softmax(z2)

dz2 = a2 - encoded_labels
