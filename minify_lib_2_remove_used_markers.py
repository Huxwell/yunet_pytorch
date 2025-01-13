import ast
import os
import astor

def parse_log_line(line):
    # Format: "Filip YuNet Minify: Function fidx=X name called in path:LY"
    if "Filip YuNet Minify" not in line:
        return None
    try:
        # Extract filename and line number
        file_part = line.split(" called in ")[1]
        filename = file_part.split(":L")[0]
        lineno = int(file_part.split(":L")[1])
        return (filename, lineno)
    except:
        return None

def remove_injected_code(file_path, used_fidx_details):
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source)

    class CodeRemover(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            print(f"Processing function: {node.name} in file: {self.filename}, line: {node.lineno}")
            if (self.filename, node.lineno) not in used_fidx_details:
                print(f"Removing marker from: {node.name} in file: {self.filename}, line: {node.lineno}")
                if isinstance(node.body[0], ast.Expr):
                    node.body = node.body[1:]  # Remove just the first statement (print)
            return node

    transformer = CodeRemover()
    transformer.filename = file_path
    new_tree = transformer.visit(tree)
    new_source = astor.to_source(new_tree)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_source)

def process_directory(directory, log_file):
    # Read the log file to determine which functions were used
    used_fidx_details = set()
    with open(log_file, 'r') as log:
        for line in log:
            result = parse_log_line(line.strip())
            if result:
                used_fidx_details.add(result)

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                remove_injected_code(file_path, used_fidx_details)

if __name__ == "__main__":
    mmdet_path = 'mmdet'
    log_file = 'function_calls.log'
    process_directory(mmdet_path, log_file)