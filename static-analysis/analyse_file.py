import ast

GPU_IMPORTS = {'torch', 'tensorflow', 'cupy', 'jax'}
GPU_FUNCTION_KEYWORDS = {'cuda', 'to', 'device', 'is_available'}

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

        cuda_calls = analyzer.gpu_calls
        imports_found = analyzer.imports
        lines_considered = analyzer.lines_considered

        if cuda_calls or imports_found:
            result["execution_mode"] = "GPU"
            result["reason"] = (
                f"Detected {len(cuda_calls)} GPU-related call(s) and {len(imports_found)} relevant import(s)."
            )
            result["confidence"] = round(0.5 + 0.1 * len(cuda_calls) + 0.1 * len(imports_found), 2)
            result["details"]["uses_cuda"] = True
            result["details"]["cuda_calls"] = list(set(cuda_calls))
            result["details"]["lines_considered"] = lines_considered
            result["details"]["imports"] = list(imports_found)
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
        self.gpu_calls = []
        self.lines_considered = []

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
        if any(keyword in full_name for keyword in GPU_FUNCTION_KEYWORDS):
            self.gpu_calls.append(full_name)
            self.lines_considered.append(node.lineno)
        self.generic_visit(node)

def get_full_attr_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts)) if parts else ""
