import torch

if torch.cuda.is_available():
    model.to("cuda")
