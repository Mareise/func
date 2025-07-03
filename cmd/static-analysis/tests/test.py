import os
import pytest
import ast

from analyze_file import analyze_file
from tensor_estimation import estimate_pytorch_tensor_size, estimate_tensorflow_tensor_size

GPU_TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "testdata/gpu")
CPU_TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "testdata/cpu")

class TestCPUClassification:
    def test_basic_cpu_classification(self):
        test_file = os.path.join(CPU_TESTDATA_DIR, "cpu.py")
        result = analyze_file(test_file)
        assert result["execution_mode"] == "cpu"
        assert result["details"]["uses_cuda"] is False
        assert len(result["details"]["big_calls"]) == 0
        assert len(result["details"]["small_calls"]) == 0

    def test_with_small_pytorch_tensor(self):
        test_file = os.path.join(CPU_TESTDATA_DIR, "small-pytorch.py")
        result = analyze_file(test_file)
        assert result["execution_mode"] == "cpu_preferred"
        assert result["details"]["explicit_gpu_calls"] is False
        assert len(result["details"]["small_calls"]) == 2
        assert len(result["details"]["big_calls"]) == 0

    def test_with_device_request(self):
        test_file = os.path.join(CPU_TESTDATA_DIR, "cpu-with-device-request.py")
        result = analyze_file(test_file)
        assert result["execution_mode"] == "cpu_preferred"
        assert result["details"]["explicit_gpu_calls"] is False


class TestGPUClassification:
    def test_with_cuda_usage(self):
        test_file = os.path.join(GPU_TESTDATA_DIR, "gpu.py")
        result = analyze_file(test_file)
        assert result["execution_mode"] == "gpu"
        assert result["details"]["explicit_gpu_calls"] is True

    def test_with_cuda_usage_1(self):
        test_file = os.path.join(GPU_TESTDATA_DIR, "gpu_1.py")
        result = analyze_file(test_file)
        assert result["execution_mode"] == "gpu"
        assert result["details"]["explicit_gpu_calls"] is True

    def test_with_big_pytorch_tensor(self):
        test_file = os.path.join(GPU_TESTDATA_DIR, "big-pytorch.py")
        result = analyze_file(test_file)
        assert result["execution_mode"] == "gpu_preferred"
        assert result["details"]["explicit_gpu_calls"] is False
        assert len(result["details"]["small_calls"]) == 2
        assert len(result["details"]["big_calls"]) == 1

    def test_with_big_tensorflow_tensor(self):
        test_file = os.path.join(GPU_TESTDATA_DIR, "big-tensorflow.py")
        result = analyze_file(test_file)
        print(result)
        assert result["execution_mode"] == "gpu_preferred"
        assert result["details"]["explicit_gpu_calls"] is False
        assert len(result["details"]["small_calls"]) == 2
        assert len(result["details"]["big_calls"]) == 1

class TestTensorFlowTensorSizeEstimation:
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
    def test_estimate_pytorch_tensor_size(self, code, expected):
        node = ast.parse(code).body[0].value
        assert estimate_pytorch_tensor_size(node) == expected

    @pytest.mark.parametrize("code,expected", [
        ("tf.zeros([3, 4])", 12),
        ("tf.zeros((2, 5))", 10),
        ("tf.ones([16])", 16),
        ("tf.random.uniform([128, 256])", 128 * 256),
        ("tf.zeros([])", None),
        ("tf.zeros(shape)", None),
        ("tf.constant([2.0])", 1),
        ("tf.constant([1, 2, 3])", 3),
        ("tf.constant([[1, 2], [3, 4]])", 4),
        ("tf.constant(" + str([[i for i in range(100)] for _ in range(100)]) + ")", 100 * 100),
    ])
    def test_estimate_tensorflow_tensor_size(self, code, expected):
        node = ast.parse(code).body[0].value
        assert estimate_tensorflow_tensor_size(node) == expected