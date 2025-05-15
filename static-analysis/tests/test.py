import os
from analysis import is_gpu_related, analyze_directory_for_gpu_code
from io import StringIO
import sys

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")

def test_is_gpu_related_true():
    code = 'import torch\nmodel.to("cuda")'
    assert is_gpu_related(code) is True

def test_is_gpu_related_false():
    code = 'def foo():\n    return "CPU only"'
    assert is_gpu_related(code) is False

def test_is_gpu_related_on_static_files():
    gpu_file = os.path.join(TESTDATA_DIR, "gpu.py")
    cpu_file = os.path.join(TESTDATA_DIR, "cpu.py")

    with open(gpu_file, 'r') as f:
        assert is_gpu_related(f.read()) is True

    with open(cpu_file, 'r') as f:
        assert is_gpu_related(f.read()) is False

def test_analyze_directory_for_gpu_code_static(monkeypatch):
    monkeypatch.chdir(TESTDATA_DIR)

    captured_output = StringIO()
    sys.stdout = captured_output

    analyze_directory_for_gpu_code()

    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()

    assert "gpu.py" in output
    assert "cpu.py" not in output
