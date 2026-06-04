import torch
import torch.nn as nn
import torch.nn.functional as F

class NetworkLinear(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.fc1 = nn.Linear(3*224*224, 100) 
        self.fc2 = nn.Linear(100, 3)
        self.sm = nn.Softmax(dim=1)
 

    def forward(self, x): 
        x = x.view(-1, 3*224*224)
        x = F.relu(self.fc1(x))
        x = self.sm(self.fc2(x)) # softmax activation on final layer 
        return x
