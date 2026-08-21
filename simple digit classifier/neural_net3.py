import numpy as np
import pandas as pd

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


class FullyConnectedNet:
    def __init__(
        self,
        hidden_dims,
        input_dim,
        num_classes,
        dropout_keep_ratio=1.0,
        l2_lmbda=0.0,
        seed=None,
    ):
        self.l2_lmbda = l2_lmbda
        self.num_layers = len(hidden_dims) + 1
        self.params = {}

        # build full chain
        dims = [input_dim] + list(hidden_dims) + [num_classes]

        for i in range(self.num_layers):
            fan_in, fan_out = dims[i], dims[i + 1]
            # he init
            self.params[f"w{i + 1}"] = (np.random.randn(fan_in, fan_out)) * np.sqrt(
                2.0 / fan_in
            )
            self.params[f"b{i + 1}"] = np.zeros((1, fan_out))

        self.use_dropout = dropout_keep_ratio != 1.0
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

    def loss(self, a_in, y=None):
        mode = "test" if y is None else "train"
        if self.use_dropout:
            self.dropout_param["mode"] = mode

        # forward pass
        a = a_in
        caches = []

        for i in range(1, self.num_layers):
            w, b = self.params[f"w{i}"], self.params[f"b{i}"]
            a, ar_cache = affine_relu_forward(a, w, b)

            drop_cache = None
            if self.use_dropout:
                a, drop_cache = dropout_forward(a, self.dropout_param)

            caches.append((ar_cache, drop_cache))

        # final_layer
        w_L, b_L = (
            self.params[f"w{self.num_layers}"],
            self.params[f"b{self.num_layers}"],
        )
        z_final, out_cache = affine_forward(a, w_L, b_L)

        if mode == "test":
            return z_final

        # loss
        loss, delta = softmax_loss(z_final, y)

        # l2 regularization
        loss += (
            0.5
            * self.l2_lmbda
            * sum(
                np.sum(self.params[f"w{i}"] ** 2) for i in range(1, self.num_layers + 1)
            )
        )

        # backward
        grads = {}

        delta, dw, db = affine_backward(delta, out_cache)
        grads[f"w{self.num_layers}"] = dw + self.l2_lmbda * w_L
        grads[f"b{self.num_layers}"] = db

        # backpropogating starting from last 2nd layer
        for i in range(self.num_layers - 1, 0, -1):
            ar_cache, drop_cache = caches[i - 1]
            # at each step gotta undo dropout first (as it was applied last)
            if self.use_dropout:
                delta = dropout_backward(delta, drop_cache)

            delta, dw, db = affine_relu_backward(delta, ar_cache)

            grads[f"w{i}"] = dw + self.l2_lmbda * self.params[f"w{i}"]
            grads[f"b{i}"] = db

        return loss, grads

    def predict(self, a_in):
        z_final = self.loss(a_in)
        z_final_array = np.array(z_final)
        return np.argmax(z_final_array, axis=1)


# SGD with momentum
def sgd_momentum(w, dw, config):
    config.setdefault("eta", 1e-12)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    v_next = config["momentum"] * v - config["eta"] * dw
    w_next = w + v_next

    return w_next, {**config, "velocity": v_next}


# trainer
class Trainer:
    def __init__(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        mini_batch_size=128,
        epochs=10,
        eta=1e-12,
        eta_decay=1.0,
        momentum=0.9,
        verbose=True,
        print_every=100,
    ):

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.eta_decay = eta_decay
        self.verbose = verbose
        self.print_every = print_every

        self.optim_configs = {
            p: {"eta": eta, "momentum": momentum} for p in model.params
        }

        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        self.best_val_acc = 0.0
        self.best_params = {}

    def _step(self):
        m = self.X_train.shape[0]
        idx = np.random.choice(m, self.mini_batch_size, replace=False)
        X_b = self.X_train[idx]
        y_b = self.y_train[idx]

        loss, grads = self.model.loss(X_b, y_b)
        self.loss_history.append(loss)

        for p in self.model.params:
            w_new, cfg_new = sgd_momentum(
                self.model.params[p], grads[p], self.optim_configs[p]
            )
            self.model.params[p] = w_new
            self.optim_configs[p] = cfg_new

    def check_accuracy(self, X, y, num_samples=1000):
        m = X.shape[0]
        if num_samples is not None and m > num_samples:
            idx = np.random.choice(m, num_samples, replace=False)
            X, y = X[idx], y[idx]
        return float(np.mean(self.model.predict(X) == y))

    def train(self):
        m = self.X_train.shape[0]
        iters_per_epoch = max(m // self.mini_batch_size, 1)
        total_iters = self.epochs * iters_per_epoch

        for t in range(total_iters):
            self._step()

            if self.verbose and t % self.print_every == 0:
                print(f"  iter {t:5d}/{total_iters}  loss={self.loss_history[-1]:.4f}")

            epoch_end = (t + 1) % iters_per_epoch == 0
            if epoch_end:
                epoch = (t + 1) // iters_per_epoch

                for cfg in self.optim_configs.values():
                    cfg["eta"] *= self.eta_decay

                train_acc = self.check_accuracy(self.X_train, self.y_train)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    eta_now = next(iter(self.optim_configs.values()))["eta"]
                    print(
                        f"Epoch {epoch:2d}/{self.epochs}  "
                        f"eta={eta_now:.6f}  "
                        f"train_acc={train_acc:.3f}  "
                        f"val_acc={val_acc:.3f}"
                    )

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {
                        k: v.copy() for k, v in self.model.params.items()
                    }

        self.model.params = self.best_params
        print(f"\n✓ Done.  Best val_acc = {self.best_val_acc:.4f}")


# final testing
def load_kaggle_mnist():
    print("Loading Kaggle CSVs... (this might take a few seconds)")

    # 1. Load Train Data
    train_data = pd.read_csv("./archive/mnist_train.csv").values
    y_tr = train_data[:, 0].astype(int)  # Integer labels
    X_tr = train_data[:, 1:].astype(np.float64)  # Pixel features
    X_tr /= 255.0  # Normalize to [0, 1]

    # 2. Load Test Data
    test_data = pd.read_csv("./archive/mnist_test.csv").values
    y_te = test_data[:, 0].astype(int)  # Integer labels
    X_te = test_data[:, 1:].astype(np.float64)  # Pixel features
    X_te /= 255.0  # Normalize to [0, 1]

    # 3. Mean Centering (to prevent lopsided gradients)
    mean = X_tr.mean(axis=0)
    X_tr -= mean
    X_te -= mean

    return X_tr, y_tr, X_te, y_te


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = load_kaggle_mnist()
    print(f"Train: {X_tr.shape}   Test: {X_te.shape}\n")

    # The rest of the setup is identical!
    net = FullyConnectedNet(
        input_dim=784,
        hidden_dims=[256, 128],
        num_classes=10,
        dropout_keep_ratio=0.8,
        l2_lmbda=1e-3,
    )

    trainer = Trainer(
        model=net,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_te,
        y_val=y_te,
        mini_batch_size=128,
        epochs=20,
        eta=1e-2,
        eta_decay=0.95,
        momentum=0.9,
        print_every=300,
    )

    trainer.train()

    preds = net.predict(X_te)
    correct = int(np.sum(preds == y_te))
    print(
        f"\nFinal test accuracy: {correct}/{len(y_te)} "
        f"= {100 * correct / len(y_te):.2f}%"
    )


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
