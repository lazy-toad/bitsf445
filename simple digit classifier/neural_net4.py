import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=784, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))

        z_final = self.fc3(x)
        return z_final


# data loading
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # std dev used and not variance
    ]
)

print("loading data...")
train_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_data = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)


if __name__ == "__main__":
    model = NeuralNet()

    criterion = nn.CrossEntropyLoss()  # applies stable softmax

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

    print("starting training...")
    epochs = 30

    for epoch in range(epochs):
        model.train()  # dropout on (training mode)

        for X_batch, y_batch in train_loader:
            # forward pass
            z_final = model(X_batch)
            loss = criterion(z_final, y_batch)

            # backprop
            # gotta zero out the grads so that we dont add it to prev ones cuz diff examples, diff grad and adjustment needed, can't mix them up
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

        scheduler.step()  # lr decay

        # evaluation
        model.eval()
        correct = 0
        # temporarily disable grad calc
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                z_final = model(
                    X_batch
                )  # invokes __call__ from nn which after doing sm req things, calls custom forward
                predictions = torch.argmax(z_final, dim=1)
                correct += (predictions == y_batch).sum().item()

            acc = correct / len(test_data)

            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:2d}/{epochs} | lr: {current_lr:.6f} | acc: {acc:.4f}"
            )

    print("done")
