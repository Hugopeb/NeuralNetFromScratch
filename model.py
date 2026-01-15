import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchvision as tv

########### Auxiliary functions ############

def get_batches(X, y, batch_size = 64, shuffle = True):
    indices = np.arange(X.shape[0])

    if shuffle: 
        np.random.shuffle(indices)

    for i in range(0, X.shape[0], batch_size): 
        batch_idx = indices[i:i+batch_size]
        yield X[batch_idx], y[batch_idx]


def train_test_split(data, labels, test_ratio = 0.3, shuffle=True):
    number_of_samples = data.shape[0]
    indices = np.arange(number_of_samples)
    
    if shuffle:
        np.random.shuffle(indices)
        X = data[indices]
        y = labels[indices]

    test_size = round(number_of_samples * test_ratio)
 
    #Split
    X_train = X[:-test_size] 
    y_train = y[:-test_size] 
    X_test = X[-test_size:] 
    y_test = y[-test_size:]
    return X_train, y_train, X_test, y_test


########### Activation functions ############

class ReLU():
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask
    def backwards(self, grad_loss_output):
        return grad_loss_output * self.mask

class softmax():
    def forward(self, X):
        return torch.softmax(X, dim = 1)

############## Neural Network ##############

class layer():
    def __init__(self, output_size, input_size):
        self.W = torch.randn(output_size, input_size) * 0.01 # Instance attributes
        self.b = torch.randn(output_size,) * 0.01 # Python automatic broadcast
        
    def forward(self, X_input):
        self.X_input = X_input
        Z = X_input @ self.W.T + self.b
        return Z
        
    def backwards(self, grad_output):
        batch_size = self.X_input.shape[0]
        self.dW = grad_output.T @ self.X_input / batch_size # Instance attributes
        self.db = grad_output.sum(dim=0) / batch_size
        self.dX = grad_output @ self.W
        return self.dX
    
class NN():
    def __init__(self, layers_sizes, activations):
        self.layers = []
        self.activations = activations

        for i in range(len(layers_sizes)-1):
            input_size = layers_sizes[i]
            output_size = layers_sizes[i+1]
            self.layers.append(layer(output_size, input_size))

    def forward(self, X):
        out = X
        for layer, activation in zip(self.layers, self.activations):
            out = activation.forward(layer.forward(out))
        return out
        
    def backwards(self, y_pred, y_true):
        batch_size = y_pred.shape[0]

        grad = y_pred.clone()
        grad[torch.arange(batch_size), y_true] -= 1

        grad = self.layers[-1].backwards(grad)

        for layer, activation in reversed(list(zip(self.layers[1:-1], self.activations[:-1]))):
            grad = activation.backwards(layer.backwards(grad))

        grad = self.layers[0].backwards(grad)
            
        CE = -torch.log(y_pred[torch.arange(batch_size), y_true] + 1e-9).mean()
        return CE

    def update_parameters(self, lr=0.01):
        for layer in self.layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db    

########### Load the DATA ##############

import torchvision.transforms as transforms

train_data = tv.datasets.MNIST(root = './train_MNIST', train = True, 
                               transform = transforms.ToTensor(), download = True)

test_data = tv.datasets.MNIST(root = './test_MNIST', train = False, 
                               transform = transforms.ToTensor(), download = True)

images = train_data.data
labels = train_data.targets

images = images.reshape(images.shape[0], 28*28) / 255

########### Training loop ##############

num_epochs = 100

x_train, y_train, x_test, y_test = train_test_split(images, labels)

# Initialize the model
model = NN(
    layers_sizes = [784, 256, 10],
    activations = [ReLU(), softmax()]
)

batch_size = 64
train_samples = x_train.shape[0]
number_batches = int(train_samples / batch_size)
avg_CE_losses = []

for epoch in range(num_epochs):
    total_loss = 0
    for x_batch, y_batch in get_batches(x_train, y_train):
        output = model.forward(x_batch)
        batch_loss = backwards = model.backwards(output, y_batch)
        total_loss += batch_loss
        model.update_parameters()

    avg_CE_losses.append(total_loss/number_batches)
    
    if epoch%10 == 0:
        print("Average CE per epoch (nats): %.5f" %(total_loss/number_batches))

plt.figure(figsize=(10,8))
plt.title("Average CE per epoch")
plt.plot(np.arange(num_epochs), avg_CE_losses, 'r-')
plt.ylabel("TCE(epoch) / N_batches")
plt.xlabel("Epoch")
plt.grid()

print("done")