import ast

from constants import ExecutionModes, GPU_IMPORTS, TENSOR_SIZE_THRESHOLD_TENSORFLOW, \
    TENSOR_SIZE_THRESHOLD_PYTORCH, PYTORCH_TENSOR_OPS, TENSORFLOW_TENSOR_OPS
from tensor_estimation import estimate_pytorch_tensor_size, estimate_tensorflow_tensor_size
from util import get_full_attr_name


def analyze_file(filename):
    result = {
        "execution_mode": ExecutionModes.CPU,
        "reason": "",
        "details": {
            "imports": [],
            "uses_cuda": False,
            "explicit_gpu_calls": [],
            "lines_considered": []
        }
    }

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)
        analyzer = GPUCodeAnalyzer()
        analyzer.visit(tree)

        explicit_gpu_calls = analyzer.explicit_gpu_calls
        imports_found = analyzer.imports
        lines_considered = analyzer.lines_considered
        small_calls = analyzer.small_calls
        big_calls = analyzer.big_calls

        result["details"]["explicit_gpu_calls"] = list(set(explicit_gpu_calls))
        result["details"]["imports"] = list(imports_found)
        result["details"]["has_explicit_gpu_calls"] = bool(explicit_gpu_calls)
        result["details"]["lines_considered"] = lines_considered
        result["details"]["small_calls"] = small_calls
        result["details"]["big_calls"] = big_calls

        # TODO rework
        if explicit_gpu_calls:
            result["execution_mode"] = ExecutionModes.GPU
            result["reason"] = (
                f"Detected {len(explicit_gpu_calls)} explicit gpu calls"
            )
        elif imports_found and small_calls and not big_calls:
            result["execution_mode"] = ExecutionModes.CPU_PREFERRED
            result["reason"] = (
                f"Detected {len(small_calls)} small pytorch/tensorflow call(s) and {len(imports_found)} relevant imports."
            )
        elif imports_found and big_calls:
            result["execution_mode"] = ExecutionModes.GPU_PREFERRED
            result["reason"] = (
                f"Detected {len(big_calls)} big pytorch/tensorflow call(s) and {len(imports_found)} relevant imports."
            )
        elif imports_found:
            result["execution_mode"] = ExecutionModes.CPU_PREFERRED
            result["reason"] = (
                f"Detected {len(imports_found)} relevant imports."
            )
        else:
            result["reason"] = "No GPU-related calls or imports detected."

    except Exception as e:
        result["reason"] = f"Failed to analyze {filename}: {e}"

    # print(result)
    return result


class GPUCodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.imports = set()
        self.explicit_gpu_calls = []
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

        self.explicit_gpu_calls, self.lines_considered = explicit_gpu_calls_check(node)

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

    # Track tensorflow with blocks
    # I am not sure if this is necessary because we scan for tensors anyway and tensorflow opts always for cpu if no gpu is available
    # def visit_With(self, node):
    #     for item in node.items:
    #         if isinstance(item.context_expr, ast.Call):
    #             func = item.context_expr.func
    #             if isinstance(func, ast.Attribute) and func.attr == 'device':
    #                 for arg in item.context_expr.args:
    #                     if isinstance(arg, ast.Constant) and 'GPU' in str(arg.value).upper():
    #                         self.explicit_gpu_calls.append('tf.device')
    #                         self.lines_considered.append(node.lineno)
    #     self.generic_visit(node)


def is_pytorch_tensor_op(full_name):
    return full_name.startswith("torch.") and any(full_name.endswith(f".{op}") for op in PYTORCH_TENSOR_OPS)


def is_tensorflow_tensor_op(full_name):
    return full_name.startswith("tf.") and any(full_name.endswith(f".{op}") for op in TENSORFLOW_TENSOR_OPS)


def explicit_gpu_calls_check(node):
    explicit_gpu_calls = []
    explicit_gpu_calls_lines = []
    if (isinstance(node.func, ast.Attribute) and node.func.attr == 'device') or \
            (isinstance(node.func, ast.Name) and node.func.id == 'device'):
        # Check args
        if node.args:
            first_arg = node.args[0]

            # Case 1: The arg is a constant string "cuda" => GPU only
            if isinstance(first_arg, ast.Constant) and first_arg.value == "cuda":
                explicit_gpu_calls_lines.append(node.lineno)
                explicit_gpu_calls.append(get_full_attr_name(node.func))

            # # Case 2: Check for something like: torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # elif isinstance(first_arg, ast.IfExp):
            #     # Check if the test calls torch.cuda.is_available()
            #     test = first_arg.test
            #
            #     # If test is a call to torch.cuda.is_available()
            #     if not is_cuda_is_available(test):
            #         # We don't mark it as an explicit gpu call
            #
            #     # Otherwise, conservative fallback:
            #     else:
            #         self.gpu_only_lines.append(node.lineno)
            # else:
            #     # For other cases, you can add more rules or ignore
            #     pass

    # Detect model.to("cuda")
    elif is_attr_call(node.func, 'to'):
        if node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and first_arg.value == "cuda":
                explicit_gpu_calls_lines.append(node.lineno)
                explicit_gpu_calls.append(get_full_attr_name(node.func))

    # Detect model.cuda()
    elif is_attr_call(node.func, 'cuda'):
        # cuda() typically has no args, but if it has args you can extend this logic
        explicit_gpu_calls_lines.append(node.lineno)
        explicit_gpu_calls.append(get_full_attr_name(node.func))

    return explicit_gpu_calls, explicit_gpu_calls_lines


# A helper function to check if this is a call to torch.cuda.is_available()
def is_cuda_is_available(call_node):
    # call_node should be an ast.Call with func torch.cuda.is_available
    if not isinstance(call_node, ast.Call):
        return False
    func = call_node.func
    # Check func is Attribute 'is_available' of Attribute 'cuda' of Name 'torch'
    if isinstance(func, ast.Attribute) and func.attr == 'is_available':
        value = func.value
        if isinstance(value, ast.Attribute) and value.attr == 'cuda':
            if isinstance(value.value, ast.Name) and value.value.id == 'torch':
                return True
    return False


def is_attr_call(node, attr_name):
    return isinstance(node, ast.Attribute) and node.attr == attr_name