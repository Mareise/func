import os

from analyze_file import analyze_file

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

def test_cpu_classification_with_pytorch():
    test_file = os.path.join(TESTDATA_DIR, "small-pytorch.py")
    result = analyze_file(test_file)
    assert result["execution_mode"] == "CPU"
    assert result["details"]["uses_cuda"] == False
    assert len(result["details"]["small_calls"]) == 2
