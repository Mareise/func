# processor.py
import os

GPU_KEYWORDS = [
    'torch.cuda', 'tensorflow', 'cupy', 'cuda', 
    'torch.Tensor.to("cuda")', 'model.to("cuda")',
    'jax', 'tensorflow.device', 'cuda.is_available',
]

def is_gpu_related(code):
    """Simple string matching to detect GPU-related usage."""
    for keyword in GPU_KEYWORDS:
        if keyword in code:
            return True
    return False

def analyze_file_for_gpu_usage(filename):
    """Analyze a single file for GPU-related patterns."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            code = file.read()
            if is_gpu_related(code):
                return True
    except Exception as e:
        print(f"Error reading {filename}: {e}")
    return False

def analyze_directory_for_gpu_code():
    """Analyze all non-test .py files in the directory."""
    current_dir = os.getcwd()
    found_gpu_usage = []

    for filename in os.listdir(current_dir):
        if filename.endswith('.py') and 'test' not in filename.lower():
            if analyze_file_for_gpu_usage(filename):
                found_gpu_usage.append(filename)

    if found_gpu_usage:
        print("GPU-related code detected in the following files:")
        for f in found_gpu_usage:
            print(f" - {f}")
    else:
        print("No GPU-related code detected.")

if __name__ == "__main__":
    print("This is the static analysis script.")
    # print the current working directory
    print("Current working directory:", os.getcwd())
    # print every file in the current directory
    print("Files in the current directory:")
    for file in os.listdir(os.getcwd()):
        print(file)

    analyze_directory_for_gpu_code()

