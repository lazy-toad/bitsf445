import numpy as np
from data_loader import load_test_data

def sigmoid(z):
    return 1/(1+np.exp(-z))

def softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x, axis=1, keepdims=True)

print("\n--- Starting Final Evaluation on Test Data ---")

test_encoded_labels, test_features = load_test_data()

print("Loading trained weights...")
weights = np.load("trained_weights.npz")
w1 = weights['w1']
b1 = weights['b1']
w2 = weights['w2']
b2 = weights['b2']

print("final")
final_z1 = np.dot(test_features,w1)+b1
final_a1 = sigmoid(final_z1)

final_z2 = np.dot(final_a1,w2)+b2
final_a2 = sigmoid(final_z2)

test_predictions = np.argmax(final_a2,axis=1)
test_true_labels = np.argmax(test_encoded_labels,axis=1)


test_accuracy = np.mean(test_predictions == test_true_labels) * 100
print(f"\n================================")
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
print(f"================================\n")
