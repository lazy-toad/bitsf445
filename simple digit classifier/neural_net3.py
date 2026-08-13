from socket import AF_INET

import numpy as np
from pandas.compat import pa_version_under10p1


#affine fully connected layer

def affine_forward(a_prev,w,b):

    #flatten to row for any input
    # x-> (no. of examples, components of features, 784 here)
    # W-> inverted rule of what I did in derivation(transpose) -> weight of a single layer only
    a_flat = a_prev.reshape(a_prev.shape[0],-1)
    z = np.dot(a_flat, w) + b
    cache = (a_prev,w,b)
    return z, cache


def affine_backward(delta, cache):
    #?
    a_prev, w, b = cache
    a_flat = a_prev.reshape(a_prev.shape[0],-1)

    delta_prev_layer = np.dot(delta, w.T).reshape(a_prev.shape)

    #gradient should match the shape of input
    dw = np.dot(a_flat.T, delta)
    db = np.sum(delta, axis=0, keepdims=True)

    return delta_prev_layer, dw, db


# activation (ReLu here)
def relu_forward(z):
    a = np.maximum(0,z)
    cache = z
    return a, cache


def relu_backward(delta, cache):
    z = cache
    delta_prev = delta * (z>0)
    return delta_prev


#composite layers
def affine_relu_forward(x,w,b):
    z, affine_forward_cache = affine_forward(x,w,b)
    a, relu_cache = relu_forward(z)
    return a, (affine_forward_cache, relu_cache)

def affine_relu_backward(delta, cache):
    affine_forward_cache,  relu_cache = cache

    # dl/da to dl/dz
    delta_prev = relu_backward(delta, relu_cache)
    #dl/dz to dl/da_prev
    delta_prev, dw, db = affine_backward(delta_prev,affine_forward_cache)

    return delta_prev, dw, db


#inverted dropout
def dropout_forward(a_prev, dropout_parameters):
    p, mode = dropout_parameters["p"], dropout_parameters["mode"]

    if "seed" in dropout_parameters:
        np.random.seed(dropout_parameters["seed"])

    if mode == "train":
        mask = (np.random.rand(*a_prev.shape) < p) / p
        a_dropped = a_prev * mask
    else:
        mask = None
        a_dropped = a_prev

    return a_dropped, (dropout_parameters,mask)


def dropout_backward(delta, cache):
    dropout_parameters, mask = cache
    if dropout_parameters["mode"] == "train":
        return delta*mask

    return delta


#softmax and cross entropy
def softmax_loss(z_final, y):

    z_shifted = z_final - z_final.max(axis=1, keepdims = True)
    exp_z = np.exp(z_shifted)
    probab_distri = exp_z / np.sum(exp_z,axis=1,keepdims=True)

    m = z_final.shape[0]
    loss = -np.log()
