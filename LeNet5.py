import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
"""
    LeNet5 is a CNN, described in the LeCun paper, 1998 (http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf).
    Was applied back in the day to recognise handwritten digits on checks that had been digitized to 32x32x1 images (greyscale)

    General architecture:   Layer:      Output Shape: (h x w x channels)
                            input       (32x32x1)
                            conv2d      (28x28x6)
                            avg pooling (14x14x6)
                            conv2d      (10x10x16)
                            avg pooling (5x5x16)
                            conv2d      (1x1x120)
                            linear      120
                            linear      84
                            linear      10 (for 10 digits)

    A tanh squashing function is used after each conv2d and linear layer (except the last layer) 

    Calculate Number of Output Channels found using:    n_out = floor((n_in + 2p - k) / (s)) + 1
                                                            n_out = number of output features
                                                            n_in = number of input features
                                                            p = convolution padding size
                                                            k = convolution kernel size
                                                            s = convolution stride size
    
    Calculate Number of Learning Parameters using:      n = (i * f^2 * b) + b
                                                            n = number of parameters
                                                            i = number of input channels in conv layer
                                                            f = filter size
                                                            b = number of biases

    Calculations: (S=1 and P=0)
        1. Input Layer: 32x32x1 (given)
        
        2. Applying conv2d (5x5)@6:
            Output Channels = floor((32 + 0 - 5) / 1) + 1 = 28
            Learning Params = (1 * 5^2 * 1) + 1 = 26, since 6 filters, total no. of learning params = 26*6 = 156
        
        3. Applying avg pooling (2x2):
            Output Channels = floor((28 + 0–2) / 2) + 1 = 14
            Learning Params = 0 (since avg pooling layer)
        
        4. Applying conv2d (5x5)@16:
            Output Channels = floor((14 + 0 – 5) / 1) + 1 = 10
            Learning Params = (6 * 5^2 * 1) + 1 = 151, since 6 filters, total no. of learning params = 151 * 16 = 2416
            
        5. Applying avg pooling (2x2):
            Output Channels = floor((10 + 0 – 2) / 2) + 1 = 5
            Learning Params = 0 (since avg pooling layer)
        
        6. Applying conv2d (5x5)@150:
            Output Channels = floor((5 + 0–5) / 1) + 1 = 1
            Learning Params = (16 * 5^2 * 1) + 1 = 401, since 120 filters, total np. of learning params = 401 * 120 = 48120
        
        7. Applying linear layer with 84 neurons:
            Learning Params = (120 * 84 + 84) = 10164
        
        8. Applying linear layer with 10 neurons:
            Learning Params = (84 * 10 + 10) = 850
        
        Total Params: 61,706
"""

# Defining Model:
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        ## Create Layers:
        # Conv2d layers:
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2d_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv2d_3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        
        # Linear Layers:
        self.linear_1 = nn.Linear(in_features=120, out_features=84)
        self.linear_2 = nn.Linear(in_features=84, out_features=10)
        
        # Reusable layers:
        self.tanh = nn.Tanh()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Conv2d layer
        output = self.conv2d_1(x)
        output = self.tanh(output)
        
        # Avg pooling layer:
        output = self.avg_pool(output)
        
        # Conv2d layer:
        output = self.conv2d_2(output)
        output = self.tanh(output)
        
        # Avg pooling layer:
        output = self.avg_pool(output)
        
        # Conv2d layer:
        output = self.conv2d_3(output)
        output = self.tanh(output)
        
        # Reshape:
        output = output.reshape(output.shape[0], -1)
        
        # Linear Layer:
        output = self.linear_1(output)
        output = self.tanh(output)
        
        # Linear Layer:
        output = self.linear_2(output)
        
        return output


## Hyper parameters:
learning_rate = 0.005
num_epochs = 10
batch_size = 64

if __name__ == "__main__":
    
    # Load in train and test data:
    # Define custom transform since MNIST is 28x28 per img and LeNet takes input of size 32x32:
    transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])  
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Define model, loss, and optimiser:
    model = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(params=model.parameters(), lr=learning_rate)


    # Training the model:

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            
            # Running forward pass:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backwards pass:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            # printing live training updates:
            if (i+1)%100 == 0:
                print(f"Epoch: [{epoch+1}/{num_epochs}], Step[{i+1}/{len(train_loader)}], Loss: {loss.item()}")
            

    # Evaluating the model's performance on the training and the test data set:
    with torch.no_grad():  # no grad to avoid unnecessary computations
        samples = 0
        correct = 0
        
        for images, labels in train_loader:
            outputs = model(images)
            
            _, pred = torch.max(outputs.data, dim=1)
            samples += labels.size(0)
            correct += (pred == labels).sum().item()
        
        print(f"Train accuracy is: {100*correct/samples}") 
        
        samples = 0
        correct = 0
        
        for images, labels in test_loader:
            outputs = model(images)
            
            _, pred = torch.max(outputs.data, dim=1)
            samples += labels.size(0)
            correct += (pred == labels).sum().item()
        
        print(f"Test accuracy is: {100*correct/samples}") 

"""
Using the hyper params:
    learning_rate = 0.005
    num_epochs = 10
    batch_size = 64

The accuracies were approx:
    Train accuracy is: 98.21166666666667
    Test accuracy is: 97.59
"""