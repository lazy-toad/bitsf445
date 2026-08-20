import numpy as np

# affine fully connected layer


def affine_forward(a_prev, w, b):

    # flatten to row for any input
    # x-> (no. of examples, components of features, 784 here)
    # W-> inverted rule of what I did in derivation(transpose) -> weight of a single layer only
    a_flat = a_prev.reshape(a_prev.shape[0], -1)
    z = np.dot(a_flat, w) + b
    cache = (a_prev, w, b)
    return z, cache


def affine_backward(delta, cache):
    # ?
    a_prev, w, b = cache
    a_flat = a_prev.reshape(a_prev.shape[0], -1)

    delta_prev_layer = np.dot(delta, w.T).reshape(a_prev.shape)

    # gradient should match the shape of input
    dw = np.dot(a_flat.T, delta)
    db = np.sum(delta, axis=0, keepdims=True)

    return delta_prev_layer, dw, db


# activation (ReLu here)
def relu_forward(z):
    a = np.maximum(0, z)
    cache = z
    return a, cache


def relu_backward(delta, cache):
    z = cache
    delta_prev = delta * (z > 0)
    return delta_prev


# composite layers
def affine_relu_forward(x, w, b):
    z, affine_forward_cache = affine_forward(x, w, b)
    a, relu_cache = relu_forward(z)
    return a, (affine_forward_cache, relu_cache)


def affine_relu_backward(delta, cache):
    affine_forward_cache, relu_cache = cache

    # dl/da to dl/dz
    delta_prev = relu_backward(delta, relu_cache)
    # dl/dz to dl/da_prev
    delta_prev, dw, db = affine_backward(delta_prev, affine_forward_cache)

    return delta_prev, dw, db


# inverted dropout
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

    return a_dropped, (dropout_parameters, mask)


def dropout_backward(delta, cache):
    dropout_parameters, mask = cache
    if dropout_parameters["mode"] == "train":
        return delta * mask

    return delta


# softmax and cross entropy
def softmax_loss(z_final, y):

    z_shifted = z_final - z_final.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    probab_distri = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    m = z_final.shape[0]
    # probs[[0, 1, 2, ..., m-1], [y[0], y[1], y[2], ..., y[m-1]]]
    loss = -np.log(probab_distri[np.arange(m), y] + 1e-12).mean()

    # the delta at last layer
    delta = probab_distri.copy()
    delta[np.arange(m), y] -= 1
    delta /= m

    return loss, delta


"""
FOR SOFTMAX+ENTROPY:

--> Setup: 2 examples, 3 classes

Say our network's final affine layer just output raw logits (`z_final`)

z_final = np.array([
    [ 2.0,  1.0,  0.1],   # example 0's raw scores for classes 0, 1, 2
    [ 0.5, -1.0,  3.2],   # example 1's raw scores for classes 0, 1, 2
])

`shifted = z_final - z_final.max(axis=1, keepdims=True)`

z_final.max(axis=1, keepdims=True)
# [[2.0],
#  [3.2]]

Subtracting that from each row shifts every row so its largest entry becomes exactly `0`:
shifted = [
    [ 2.0-2.0,  1.0-2.0,  0.1-2.0 ]   =  [ 0.0, -1.0, -1.9 ]
    [ 0.5-3.2, -1.0-3.2,  3.2-3.2 ]   =  [-2.7, -4.2,  0.0 ]
]

exponentiate every entry elementwise:
exp_scores = [
    [ e^0.0,  e^-1.0,  e^-1.9 ]   =  [ 1.000,  0.368,  0.150 ]
    [ e^-2.7, e^-4.2,  e^0.0  ]   =  [ 0.067,  0.015,  1.000 ]
]

probs = [
    [ 1.000/1.518,  0.368/1.518,  0.150/1.518 ]   ≈  [ 0.659,  0.242,  0.099 ]
    [ 0.067/1.082,  0.015/1.082,  1.000/1.082 ]   ≈  [ 0.062,  0.014,  0.924 ]
]
These are ig predicted probabilities of each class per example

the network is saying "example 0 is 65.9% class 0, 24.2% class 1, 9.9% class 2" and
"example 1 is 6.2% class 0, 1.4% class 1, 92.4% class 2."

--> Now tie it back: suppose `y = [0, 2]`

Meaning example 0's true label is class 0, and example 1's true label is class 2.

Loss: `probs[np.arange(2), y]` grabs `probs[0,0]=0.659` [that is example, prob of our class corresponding real]
and `probs[1,2]=0.924` — the probability each example assigned to its *own* correct answer.
Loss is `-log(0.659)` and `-log(0.924)`, averaged: the model was very confident and correct on example 1 (loss ≈ 0.079, tiny),
and only moderately confident on example 0 (loss ≈ 0.417, bigger) — so example 0 contributes more to the total loss.

Delta (the starting gradient): `delta = probs.copy()` then subtract 1 at the true-label positions:

delta = [
    [ 0.659-1,  0.242,     0.099   ]   =  [-0.341,  0.242,  0.099 ]
    [ 0.062,    0.014,     0.924-1 ]   =  [ 0.062,  0.014, -0.076 ]
]

DELTA UPDATE:

What the derived formula says, class by class

derived:
dL/dz_j = p_j - 1[j = y]

On applying to classes sep:

True class: y = 1
Predicted probabilities: p_0 = 0.7, p_1 = 0.2, p_2 = 0.1

---------------------------------------------------------
Class 0:
- Is j = 0 the true label? No (0 != 1), so 1[j = y] = 0
- dL/dz_0 = p_0 - 0 = 0.7 - 0 = 0.7

Class 1:
- Is j = 1 the true label? Yes (1 == 1), so 1[j = y] = 1
- dL/dz_1 = p_1 - 1 = 0.2 - 1 = -0.8

Class 2:
- Is j = 2 the true label? No (2 != 1), so 1[j = y] = 0
- dL/dz_2 = p_2 - 0 = 0.1 - 0 = 0.1
---------------------------------------------------------

As wrong, we wanna subtract the whole thing so in future 0.7-0.7 = 0 and the true on prob increae
"""
