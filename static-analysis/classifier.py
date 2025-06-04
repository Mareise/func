import json
import os
from analyse_file import analyze_file

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
                print(f"Analyzing {filepath}...")  # only for testing
                analysis_results[filename] = analyze_file(filepath)

    print(json.dumps(analysis_results, indent=4))

if __name__ == "__main__":
    analyze_directory_for_gpu_code()

