from data_loader import load_data, load_test_data
from neural_net2 import NeuralNetwork
import numpy as np

print("loading training data..")
train_labels, train_features = load_data()
print("loading test data...")
test_labels, test_features = load_test_data()

print("zipping..")
training_data = list(zip(train_features,train_labels))
test_data = list(zip(test_features, test_labels))

print("building network..")
net = NeuralNetwork([784,30,10],l2_lmbda=5.0,dropout_rate=0.2 )

print("start training...")
net.train(training_data, epochs=30, batch_size=10, eta = 0.1, test_data=test_data)

np.savez("net2_weights.npy", *net.weights)
np.savez("net2_bias.npy", *net.bias)
print("done")
