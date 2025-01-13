import ast
import os
import astor

def remove_injected_code(file_path, used_fidx_details):
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source)

    class CodeRemover(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            print(f"Processing function: {node.name} in file: {self.filename}, line: {node.lineno}")
            # Check if this function's start line and file path match any in the used_fidx_details
            if (self.filename, node.lineno) not in used_fidx_details:
                print(f"Removing function: {node.name} in file: {self.filename}, line: {node.lineno}")
                # Assuming the first statement is a print and the last is a return
                if isinstance(node.body[0], ast.Expr) and isinstance(node.body[-1], ast.Return):
                    node.body = node.body[1:-1]  # Remove the first and last statements
            return node

    transformer = CodeRemover()
    transformer.filename = file_path  # Pass the filename to the transformer
    new_tree = transformer.visit(tree)
    new_source = astor.to_source(new_tree)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_source)

def process_directory(directory, log_file):
    # Read the log file to determine which functions were used
    used_fidx_details = set()
    with open(log_file, 'r') as log:
        for line in log:
            fidx, filename, lineno, _ = line.strip().split(',')
            used_fidx_details.add((filename, int(lineno)))

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                remove_injected_code(file_path, used_fidx_details)

if __name__ == "__main__":
    mmdet_path = 'mmdet'
    log_file = 'function_calls.log'
    process_directory(mmdet_path, log_file)