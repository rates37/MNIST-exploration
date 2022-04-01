import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchsummary import summary
from torch.utils.data import DataLoader
"""
    VGG16 is a CNN architecture, described in the Simonyan and Zisserman paper (https://arxiv.org/abs/1409.1556).
    Its main unique feature is the 3x3 conv layers with stride fixed to 1.
    
    General architecture:   ##TODO
"""

class VGG16(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG16, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Making conv layers:
        layer_sizes = [64, 64, "maxPooling", 
                       128, 128, "maxPooling", 
                       256, 256, 256, "maxPooling", 
                       512, 512, 512, "maxPooling", 
                       512, 512, 512, "maxPooling"]
        layers = []
        
        for layer in layer_sizes:
            if type(layer) == int:
                out_channels = layer
                
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
                layers.append(nn.ReLU())
                
                in_channels = layer
            elif layer == "maxPooling":
                layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
                
        self.conv2d_layers = nn.Sequential(*layers)
        
        # Making fully connected layers:
        """
        TODO: Look into why this raises errors
        self.FC = nn.Sequential(
            nn.Linear(in_features=512 * 7*7, out_features=4096),  # 7 comes from (224/(2**5))**2
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )"""
        self.flatten = nn.Flatten(start_dim=1)
        self.FC = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Conv layers:
        output = self.conv2d_layers(x)
        # Flatten:
        output = self.flatten(output)
        # Fully connected linear layers:
        output = self.FC(output)
        return output
    

## Hyper parameters:
learning_rate = 0.005
num_epochs = 1
batch_size = 64


if __name__ == "__main__":
    # Load in MNIST data:
    # Define custom transform:
    transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()]) 
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    
        # Define model, loss, and optimiser:
    model = VGG16(in_channels=1, num_classes=10)
    # using pytorch VGG16: extraLayer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
    # using pytorch VGG16: model = VGG16(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    


    # Training the model:

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Running forward pass:
            # using pytorch VGG16: outputs = model(extraLayer(images))
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backwards pass:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            print(f"Epoch: [{epoch+1}/{num_epochs}], Step[{i+1}/{len(train_loader)}], Loss: {loss.item()}")
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