import re

# Define GPU-related patterns
GPU_IMPORTS = ['torch', 'tensorflow', 'cupy', 'jax']
GPU_FUNCTIONS = [
    'torch.cuda', 'cuda()', '.to("cuda")', ".to('cuda')",
    'model.to("cuda")', 'model.to(\'cuda\')',
    'cuda.is_available', 'tensorflow.device'
]

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
            lines = f.readlines()

        imports_found = set()
        cuda_calls = []
        lines_considered = []

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            # Match GPU-related imports
            for lib in GPU_IMPORTS:
                if re.match(rf'(from\s+{lib}\b|import\s+{lib}\b)', stripped):
                    imports_found.add(lib)

            # Match GPU-related function calls
            for pattern in GPU_FUNCTIONS:
                if pattern in line:
                    cuda_calls.append(pattern)
                    lines_considered.append(i)

        # If any GPU calls found, build the JSON result
        if cuda_calls:
            result["execution_mode"] = "GPU"
            result["reason"] = f"Found {len(cuda_calls)} GPU-related call(s): " + ", ".join(set(cuda_calls)) + "."
            result["confidence"] = round(0.8 + 0.02 * len(cuda_calls), 2)
            result["details"]["uses_cuda"] = True
            result["details"]["cuda_calls"] = list(set(cuda_calls))
            result["details"]["lines_considered"] = lines_considered
        result["details"]["imports"] = list(imports_found)

    except Exception as e:
        result["reason"] = f"Failed to analyze {filename}: {e}"
        result["confidence"] = 0.0

    return result
