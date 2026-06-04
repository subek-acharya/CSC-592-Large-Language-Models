import sys
import numpy as np
import torch

def main():  
    # assume 3 outputs and batch size of 2, so logits = 2x3 tensor
    logits=torch.tensor([[1,3.0,5],[2,4.0,1]]) 
    # above indicates predicted output is 2 and 1 (index of highest value)
    print('-----logits--------')
    print(logits)
    
    targets=torch.tensor([2,1]) 
    print("targets: ", targets)
    # targets are specified as long, i.e.,
    # index of which output is to be recognized, try with [2,1] to see if loss decreases
    # pytorch's cross entropy loss, operates on logits (not on softmax layer)
    loss = torch.nn.functional.cross_entropy(logits,targets)
    print('\ncross entropy loss by pytorch=', loss)
    
    # pytorch's nll_loss (negative log likelihood loss) is similar to cross entropy
    # it operates on log_softmax, rather than raw logits
    outs = torch.softmax(logits,dim=1)
    # print("outs[0,targets[0]: ", (outs[0,targets[0] )
    print('-----softmax------')
    print("outs: ", outs)
    outs2 = torch.nn.functional.log_softmax(logits, dim=1)
    print("outs2: ", outs2)
    loss_nll = torch.nn.functional.nll_loss(outs2,targets)
    print('nll loss by pytorch =',loss_nll)
    
    
    # # compute cross entropy ourselves
    z = (np.log(outs[0,targets[0]]) + np.log(outs[1,targets[1]]))/2
    print("\ncross entropy by our calculation=",-z)
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))
