"""
A simple CNN with 2 convolutional layers and a fully-connected layer.
"""
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes=10, linear_dim=7):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.linear_dim = linear_dim
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=self.in_channels,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,
                bias=False
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2, bias=False),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # Fully connected layer, output 10 classes
        self.out = nn.Linear(32 * self.linear_dim * self.linear_dim, self.num_classes, bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.reshape(x.size(0), -1)       
        output = self.out(x)
        return output    # return x for visualization