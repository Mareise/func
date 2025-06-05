class ExecutionModes:
    CPU = 'cpu'
    GPU = 'gpu'
    GPU_PREFERRED = 'gpu_preferred'
    CPU_PREFERRED = 'cpu_preferred'


GPU_IMPORTS = {'torch', 'tensorflow', 'cupy', 'jax'}
CUDA_KEYWORDS = {'cuda', 'device', 'is_available'}
PYTORCH_TENSOR_OPS = {'tensor', 'randn', 'zeros', 'ones', 'empty'}
TENSORFLOW_TENSOR_OPS = {'constant', 'zeros', 'ones', 'fill', 'random.uniform', 'random.normal'}

TENSOR_SIZE_THRESHOLD_TENSORFLOW = 1000
TENSOR_SIZE_THRESHOLD_PYTORCH = 1000