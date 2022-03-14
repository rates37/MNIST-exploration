import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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
        self.FC = nn.Sequential(
            nn.Linear(in_features=512 * 7*7, out_features=4096),  # 7 comes from (224/(2**5))**2
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )
    
    def forward(self, x):
        # Conv layers:
        output = self.conv2d_layers(x)
        
        # Flatten:
        output = output.reshape(output.shape[0], -1)
        
        # Fully connected linear layers:
        output = self.FC(output)
        return output
    

## Hyper parameters:
learning_rate = 0.005
num_epochs = 10
batch_size = 64

## TODO: test architecture on mnist