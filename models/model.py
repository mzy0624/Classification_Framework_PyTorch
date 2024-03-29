import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Do initialization
    
    def forward(self, x):
        # Forward propagation
        return x
    
    def __str__(self):
        return 'Model Name'