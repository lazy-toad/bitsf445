import numpy as np


def sigmoid(z):
    return 1 / (1+ np.exp(-z))

def softmax(x):
    #to prevent NaN explosion
    x_safe = x - np.max(x,axis=1,keepdims=True)
    exp_x = np.exp(x_safe)
    return exp_x / np.sum(exp_x, axis = 1, keepdims= True)

class NeuralNetwork:
    def __init__(self, layers_sizes, l2_lmbda = 0.0,dropout_rate = 0.0):
        self.num_layers = len(layers_sizes)
        self.layers_sizes = layers_sizes

        self.l2_lmbda = l2_lmbda
        self.dropout_rate  =dropout_rate

        self.weights = [np.random.randn(x,y)/np.sqrt(x) for x,y in zip(layers_sizes[:-1],layers_sizes[1:])]

        self.bias = [np.random.randn(1,y) for y in layers_sizes[1:]]



    def forward(self, a, training=True):

        caches = [{"a":a}]


        for w,b in zip(self.weights[:-1],self.bias[:-1]):
            a_prev = caches[-1]["a"]
            z = np.dot(a_prev,w)+b
            s = sigmoid(z)

            if training and self.dropout_rate > 0.0:
                mask = (np.random.rand(*s.shape) > self.dropout_rate)
                a = (s*mask) / (1.0-self.dropout_rate)
            else:
                mask = None
                a = s


            caches.append({"a_prev":a_prev, "z":z,"s":s,"mask":mask,"a":a})

        a_prev = caches[-1]["a"]
        final_z = np.dot(a_prev, self.weights[-1]) + self.bias[-1]
        final_a = softmax(final_z)

        caches.append({"a_prev":a_prev,"z":final_z,"a":final_a})

        return caches


    def backprop(self, caches, batch_labels):

        m = batch_labels.shape[0]

        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.bias]

        #softmax and cross entropy
        delta = caches[-1]["a"] - batch_labels
        dw[-1] = np.dot(caches[-1]["a_prev"].T,delta) / m
        db[-1] = np.sum(delta,axis=0, keepdims=True) / m


        for l in range(self.num_layers-2,0,-1):

            cache = caches[l]
            delta = np.dot(delta,self.weights[l].T)

            if cache["mask"] is not None:
                delta = delta* cache["mask"] / (1-self.dropout_rate)
            delta = delta * cache["s"] * (1-cache["s"])

            dw[l-1] = np.dot(cache["a_prev"].T, delta) / m
            db[l-1] = np.sum(delta, axis=0, keepdims=True)

        return dw, db



    def update_mini_batch(self, batch_features, batch_labels, eta, n):
        caches = self.forward(batch_features, training= True)
        dw, db = self.backprop(caches, batch_labels)

        for i in range(self.num_layers-1):
            weight_decay = 1.0 - (eta*(self.l2_lmbda/n))

            self.weights[i] = (self.weights[i]*weight_decay) - (eta*dw[i])

            self.bias[i] = self.bias[i] - (eta*db[i])




    def train(self, training_data, epochs, batch_size, eta, test_data=None):

        n = len(training_data)
        n_test = len(test_data) if test_data is not None else None

        for j in range(epochs):

            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size] for k in range(0,n,batch_size)]

            for mini_batch in mini_batches:

                batch_features, batch_labels = zip(*mini_batch)

                batch_features = np.array(batch_features)
                batch_labels = np.array(batch_labels)

                self.update_mini_batch(batch_features, batch_labels,eta, n)


            if test_data is not None:
                correct_predictions = self.evaluate(test_data)
                print(f"Epoch {j}: {correct_predictions} / {n_test} | Accuracy: {100*correct_predictions/n_test:.2f}%")
            else:
                print(f"Epoch {j} complete")

    def evaluate(self, test_data):
        test_features, test_labels = zip(*test_data)
        test_features = np.array(test_features)
        test_labels = np.array(test_labels)

        caches = self.forward(test_features, training=False)
        predictions = np.argmax(caches[-1]["a"], axis=1)
        return int(np.sum(predictions == test_labels))
