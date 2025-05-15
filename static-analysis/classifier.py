import json
import os
import ast

# from graphviz import Digraph

from analyse_file import analyze_file

GPU_KEYWORDS = [
    'torch.cuda', 'tensorflow', 'cupy', 'cuda',
    'torch.Tensor.to("cuda")', 'model.to("cuda")',
    'jax', 'tensorflow.device', 'cuda.is_available',
]

# dot = Digraph()

def is_gpu_related(code):
    """Simple string matching to detect GPU-related usage."""
    for keyword in GPU_KEYWORDS:
        if keyword in code:
            return True
    return False

# def print_ast(code):
#     try:
#         tree = ast.parse(code)
#
#         add_node(tree)
#         dot.format = 'png'
#         dot.render('my_ast', view=True)
#
#         print(ast.dump(tree, indent=2))
#     except SyntaxError as e:
#         print(f"Syntax error in code: {e}")

# def add_node(node, parent=None):
#     node_name = str(node.__class__.__name__)
#     dot.node(str(id(node)), node_name)
#     if parent:
#         dot.edge(str(id(parent)), str(id(node)))
#     for child in ast.iter_child_nodes(node):
#         add_node(child, node)
#
# def analyze_file_for_gpu_usage(filename):
#     """Analyze a single file for GPU-related patterns."""
#     try:
#         with open(filename, 'r', encoding='utf-8') as file:
#             code = file.read()
#             if is_gpu_related(code):
#                 print_ast(code)
#                 return True
#     except Exception as e:
#         print(f"Error reading {filename}: {e}")
#     return False

def analyze_directory_for_gpu_code():
    """Analyze all non-test .py files in the directory."""
    current_dir = os.getcwd()
    analysis_results = {}

    for root, _, files in os.walk(current_dir):
        for filename in files:
            if (
                    filename.endswith('.py')
                    and 'test' not in filename.lower()
                    and 'classifier' not in filename.lower() # not needed when executing executable
                    and 'analyse_file' not in filename.lower() # not needed when executing executable
                    and 'venv' not in root # not needed when executing executable
            ):
                filepath = os.path.join(root, filename)
                # print(f"Analyzing {filepath}...")  # only for testing
                analysis_results[filename] = analyze_file(filepath)

    print(json.dumps(analysis_results, indent=4))

if __name__ == "__main__":
    analyze_directory_for_gpu_code()

