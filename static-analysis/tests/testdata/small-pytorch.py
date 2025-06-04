# this small example uses pytorch but in a very simple way and should be clássified as CPU execution mode

import torch
import numpy

def add_tensors():
    a = torch.tensor([1.0])
    b = torch.tensor([2.0])
    print(a + b)
    return a + b

if __name__ == "__main__":
    add_tensors()