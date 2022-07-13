"""
A 8-layer CNN model in PyTorch.
"""
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels, out_dim=1, linear_dim=7):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
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
            nn.Conv2d(16, 64, 5, 1, 2, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 256, 5, 1, 2, bias=False),     
            nn.ReLU(),
            nn.Conv2d(256, 256, 5, 1, 2, bias=False),     
            nn.ReLU(),
            nn.Conv2d(256, 64, 5, 1, 2, bias=False),     
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, 1, 2, bias=False),     
            nn.ReLU(),
            nn.MaxPool2d(2),                
        )
        # Fully connected layer, output 10 classes
        self.out = nn.Linear(32 * self.linear_dim * self.linear_dim, 10, bias=False)
        self.out2 = nn.Linear(10, self.out_dim, bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.reshape(x.size(0), -1)       
        output = self.out(x)
        output = self.out2(output)
        return output
    
    def get_activation_before_last_layer(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)       
        output = self.out(x)
        return output