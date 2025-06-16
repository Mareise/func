import torch
import numpy

def add_tensors():
    a = torch.tensor([1.0])
    b = torch.tensor([2.0])
    c = torch.randn(512, 1024)
    print(a + b)
    return a + b

if __name__ == "__main__":
    add_tensors()