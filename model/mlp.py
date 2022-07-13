"""
A 10-layer MLP model in PyTorch.
"""
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, 10, bias=False),
            nn.ReLU(),
        )
        self.out = nn.Linear(10, self.out_dim, bias=False)
                
    def forward(self, x):
        output = self.linear(x.flatten(start_dim=1))
        output = self.out(output)
        return output
    
    def get_activation_before_last_layer(self, x):
        output = self.linear(x.flatten(start_dim=1))
        return output