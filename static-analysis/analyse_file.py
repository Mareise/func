import ast

GPU_IMPORTS = {'torch', 'tensorflow', 'cupy', 'jax'}
CUDA_KEYWORDS = {'cuda', 'device', 'is_available'}
TENSOR_OPS = {'tensor', 'randn', 'zeros', 'ones', 'empty'}


def analyze_file(filename):
    result = {
        "execution_mode": "CPU",
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

        if cuda_calls:
            result["execution_mode"] = "GPU"
            result["reason"] = (
                f"Detected {len(cuda_calls)} cuda call(s)"
            )
            result["confidence"] = round(0.5 + 0.1 * len(cuda_calls) + 0.1 * len(imports_found), 2)
        elif imports_found and small_calls and not big_calls:
            result["execution_mode"] = "CPU"
            result["reason"] = (
                f"Detected {len(small_calls)} small pytorch call(s) and {len(imports_found)} relevant import(s)."
            )
            result["confidence"] = round(0.5 + 0.1 * len(cuda_calls) + 0.1 * len(imports_found), 2)
        elif imports_found and big_calls:
            result["execution_mode"] = "GPU"
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

    return result


class GPUCodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.imports = set()
        self.cuda_calls = []
        self.lines_considered = []
        self.small_calls = []
        self.big_calls = []

    def visit_Import(self, node):
        print("visit_ImportFrom")
        for alias in node.names:
            if alias.name in GPU_IMPORTS:
                self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        print("visit_ImportFrom")
        if node.module in GPU_IMPORTS:
            self.imports.add(node.module)
        self.generic_visit(node)

    def visit_Call(self, node):
        print("visit_Call")
        full_name = get_full_attr_name(node.func)

        # Track GPU-related function calls
        if any(keyword in full_name for keyword in CUDA_KEYWORDS):
            self.cuda_calls.append(full_name)
            self.lines_considered.append(node.lineno)

        # TODO - add support for other tensor libraries
        # Check for "small" tensor operations
        if is_tensor_op(full_name):
            size = estimate_tensor_size(node)
            if size < 1000:
                self.small_calls.append((full_name, size, node.lineno))
            else:
                self.big_calls.append((full_name, size, node.lineno))
        self.generic_visit(node)


def get_full_attr_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts)) if parts else ""

def is_tensor_op(full_name):
    return any(full_name.endswith(f".{op}") for op in TENSOR_OPS)

def estimate_tensor_size(call_node):
    size = 1
    if isinstance(call_node, ast.Call):
        for arg in call_node.args:
            if isinstance(arg, ast.Tuple):
                for elt in arg.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                        size *= elt.value
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                size *= arg.value
    return size
