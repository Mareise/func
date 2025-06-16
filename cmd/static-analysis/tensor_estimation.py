import ast

from util import get_full_attr_name


def estimate_pytorch_tensor_size(call_node):
    size = 1
    found_shape = False

    if not isinstance(call_node, ast.Call):
        return None

    func_name = get_full_attr_name(call_node.func)

    for arg in call_node.args:
        # Case 1: torch.tensor([...]) — count elements in nested list
        if func_name.endswith("tensor") and isinstance(arg, ast.List):
            size = count_elements(arg)
            return size if size > 0 else None

        # Case 2: Shape as Tuple or List
        if isinstance(arg, (ast.Tuple, ast.List)):
            dims = []
            for elt in arg.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    dims.append(elt.value)
                else:
                    return None
            if dims:
                found_shape = True
                for dim in dims:
                    size *= dim

        # Case 3: Positional int args
        elif isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            found_shape = True
            size *= arg.value

        elif isinstance(arg, (ast.Name, ast.Call, ast.Starred)):
            return None

    return size if found_shape else None

def estimate_tensorflow_tensor_size(call_node):
    if not isinstance(call_node, ast.Call):
        return None

    full_name = get_full_attr_name(call_node.func)
    if full_name == "tf.constant":
        if call_node.args:
            arg = call_node.args[0]
            return count_elements(arg)
        return None

    for arg in call_node.args:
        if isinstance(arg, (ast.List, ast.Tuple)):
            dims = []
            for elt in arg.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    dims.append(elt.value)
                else:
                    return None
            if dims:
                size = 1
                for dim in dims:
                    size *= dim
                return size

        elif isinstance(arg, ast.ListComp) or isinstance(arg, ast.Call):
            return None

    return None


def count_elements(node):
    """
    Recursively count total number of constants in a nested ast.List
    """
    if not isinstance(node, ast.List):
        return 0
    total = 0
    for elt in node.elts:
        if isinstance(elt, ast.Constant):
            total += 1
        elif isinstance(elt, ast.List):
            total += count_elements(elt)
    return total