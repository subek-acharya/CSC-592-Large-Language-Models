import sys
import torch
import numpy as np


class NNLN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.LN = torch.nn.LayerNorm(4)
        
    def forward(self, x):
        out = self.LN(x)
        return out
    
def main():
    x1 = np.arange(4)
    print(x1)
    d = torch.arange(4).float()
    print(d)
    x = d.view(1,-1)
    net = NNLN()
    z = net(x)
    print(z)
    #TODO: Write a simple code that computes the layer normalization
    x1_tensor = torch.from_numpy(x1).float()
    # print(x1_tensor)
    # Calculate the mean and variance
    x1_mean = x1_tensor.mean()
    x1_var = x1_tensor.var(unbiased=False)
    eps = 1e-10

    # Normalize the tensor
    x1_norm = (x1_tensor - x1_mean) / torch.sqrt(x1_var + eps)
    print("x1_norm: ", x1_norm)
    

if __name__ == "__main__":
    sys.exit(int(main() or 0))
