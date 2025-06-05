import os
import pytest
import ast

from analyze_file import analyze_file
from tensor_estimation import estimate_tensor_size

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def test_cuda_classification():
    test_file = os.path.join(TESTDATA_DIR, "gpu.py")
    result = analyze_file(test_file)
    assert result["execution_mode"] == "GPU"
    assert result["details"]["uses_cuda"] == True

def test_cpu_classification():
    test_file = os.path.join(TESTDATA_DIR, "cpu.py")
    result = analyze_file(test_file)
    assert result["execution_mode"] == "CPU"
    assert result["details"]["uses_cuda"] == False
    assert len(result["details"]["big_calls"]) == 0
    assert len(result["details"]["small_calls"]) == 0

def test_cpu_classification_with_small_pytorch_tensor_size():
    test_file = os.path.join(TESTDATA_DIR, "small-pytorch.py")
    result = analyze_file(test_file)
    assert result["execution_mode"] == "CPU"
    assert result["details"]["uses_cuda"] == False
    assert len(result["details"]["small_calls"]) == 2
    assert len(result["details"]["big_calls"]) == 0

def test_cpu_classification_with_big_pytorch_tensor_size():
    test_file = os.path.join(TESTDATA_DIR, "big-pytorch.py")
    result = analyze_file(test_file)
    assert result["execution_mode"] == "GPU"
    assert result["details"]["uses_cuda"] == False
    assert len(result["details"]["small_calls"]) == 2
    assert len(result["details"]["big_calls"]) == 1

# Test the estimation of pytorch tensor size
@pytest.mark.parametrize("code,expected", [
    ("torch.zeros(3, 4)", 12),
    ("torch.zeros((2, 5))", 10),
    ("torch.ones(16)", 16),
    ("torch.randn(128, 256)", 128 * 256),
    ("torch.empty()", None),
    ("torch.zeros(a, b)", None),
    ("torch.tensor([2.0])", 1),
    ("torch.tensor([1, 2, 3])", 3),
    ("torch.tensor([[1, 2], [3, 4]])", 4),
    ("torch.tensor(" + str([[i for i in range(100)] for _ in range(100)]) + ")", 100 * 100),
])
def test_estimate_tensor_size(code, expected):
    node = ast.parse(code).body[0].value
    assert estimate_tensor_size(node) == expected