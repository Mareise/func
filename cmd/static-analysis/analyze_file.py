import ast

from constants import ExecutionModes, GPU_IMPORTS, CUDA_KEYWORDS, TENSOR_SIZE_THRESHOLD_TENSORFLOW, \
    TENSOR_SIZE_THRESHOLD_PYTORCH, PYTORCH_TENSOR_OPS, TENSORFLOW_TENSOR_OPS
from tensor_estimation import estimate_pytorch_tensor_size, estimate_tensorflow_tensor_size
from util import get_full_attr_name


def analyze_file(filename):
    result = {
        "execution_mode": ExecutionModes.CPU,
        "reason": "",
        "confidence": 0.0,
        "details": {
            "imports": [],
            "uses_cuda": False,
            "cuda_calls": [],
            "lines_considered": []
        }
    }

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)
        analyzer = GPUCodeAnalyzer()
        analyzer.visit(tree)

        cuda_calls = analyzer.cuda_calls
        imports_found = analyzer.imports
        lines_considered = analyzer.lines_considered
        small_calls = analyzer.small_calls
        big_calls = analyzer.big_calls

        result["details"]["cuda_calls"] = list(set(cuda_calls))
        result["details"]["imports"] = list(imports_found)
        result["details"]["uses_cuda"] = bool(cuda_calls)
        result["details"]["lines_considered"] = lines_considered
        result["details"]["small_calls"] = small_calls
        result["details"]["big_calls"] = big_calls

        # TODO rework
        if cuda_calls:
            result["execution_mode"] = ExecutionModes.GPU
            result["reason"] = (
                f"Detected {len(cuda_calls)} cuda call(s)"
            )
            result["confidence"] = round(0.5 + 0.1 * len(cuda_calls) + 0.1 * len(imports_found), 2)
        elif imports_found and small_calls and not big_calls:
            result["execution_mode"] = ExecutionModes.CPU_PREFERRED
            result["reason"] = (
                f"Detected {len(small_calls)} small pytorch call(s) and {len(imports_found)} relevant import(s)."
            )
            result["confidence"] = round(0.5 + 0.1 * len(cuda_calls) + 0.1 * len(imports_found), 2)
        elif imports_found and big_calls:
            result["execution_mode"] = ExecutionModes.GPU_PREFERRED
            result["reason"] = (
                f"Detected {len(big_calls)} big pytorch call(s) and {len(imports_found)} relevant import(s)."
            )
            result["confidence"] = round(0.5 + 0.1 * len(cuda_calls) + 0.1 * len(imports_found), 2)
        else:
            result["reason"] = "No GPU-related calls or imports detected."
            result["confidence"] = 0.1

    except Exception as e:
        result["reason"] = f"Failed to analyze {filename}: {e}"
        result["confidence"] = 0.0

    # print(result)
    return result


class GPUCodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.imports = set()
        self.cuda_calls = []
        self.lines_considered = []
        self.small_calls = []
        self.big_calls = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name in GPU_IMPORTS:
                self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module in GPU_IMPORTS:
            self.imports.add(node.module)
        self.generic_visit(node)

    def visit_Call(self, node):
        full_name = get_full_attr_name(node.func)

        # Track GPU-related function calls
        if any(keyword in full_name for keyword in CUDA_KEYWORDS):
            self.cuda_calls.append(full_name)
            self.lines_considered.append(node.lineno)

        # TODO check for numpy, pandas, https://github.com/rapidsai/cuml
        # todo maybe check if function uses a AI model
        # Track pytorch function calls
        if is_pytorch_tensor_op(full_name):
            size = estimate_pytorch_tensor_size(node)
            if size is not None:
                if size < TENSOR_SIZE_THRESHOLD_PYTORCH:
                    self.small_calls.append(("pytorch", full_name, size, node.lineno))
                else:
                    self.big_calls.append(("pytorch", full_name, size, node.lineno))

        # Track tensorflow function calls
        elif is_tensorflow_tensor_op(full_name):
            size = estimate_tensorflow_tensor_size(node)
            if size is not None:
                if size < TENSOR_SIZE_THRESHOLD_TENSORFLOW:
                    self.small_calls.append(("tensorflow", full_name, size, node.lineno))
                else:
                    self.big_calls.append(("tensorflow", full_name, size, node.lineno))

        self.generic_visit(node)

def is_pytorch_tensor_op(full_name):
    return full_name.startswith("torch.") and any(full_name.endswith(f".{op}") for op in PYTORCH_TENSOR_OPS)

def is_tensorflow_tensor_op(full_name):
    return full_name.startswith("tf.") and any(full_name.endswith(f".{op}") for op in TENSORFLOW_TENSOR_OPS)

