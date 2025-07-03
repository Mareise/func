import torch


def test_cuda_availability():
    """Test if CUDA is available on the system."""
    is_cuda_available = torch.cuda.is_available()
    return is_cuda_available
