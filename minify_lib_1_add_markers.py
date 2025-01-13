import ast
import os
import astor

class MarkerInjector(ast.NodeTransformer):
    function_idx = 0  # Initialize a global counter for function index

    def __init__(self, filename, log_file):
        self.filename = filename
        self.log_file = log_file

    def visit_FunctionDef(self, node):
        MarkerInjector.function_idx += 1
        message = f"Filip YuNet Minify: Function fidx={MarkerInjector.function_idx} {node.name} called in {self.filename}:L{node.lineno}"
        print_stmt = ast.Expr(value=ast.Call(
            func=ast.Name(id='print', ctx=ast.Load()),
            args=[ast.Str(s=message)],
            keywords=[]
        ))
        ast.copy_location(print_stmt, node)
        node.body.insert(0, print_stmt)
        # Log the function index and details to the log file
        with open(self.log_file, 'a') as log:
            log.write(f"{MarkerInjector.function_idx},{self.filename},{node.lineno},{node.name}\n")
        return node

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

def inject_markers(file_path, log_file):
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source)
    transformer = MarkerInjector(file_path, log_file)
    new_tree = transformer.visit(tree)
    new_source = astor.to_source(new_tree)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_source)

def process_directory(directory, log_file):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                inject_markers(file_path, log_file)

if __name__ == "__main__":
    mmdet_path = 'mmdet'
    log_file = 'function_calls.log'
    process_directory(mmdet_path, log_file)