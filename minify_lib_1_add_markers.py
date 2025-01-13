import ast
import os
import astor

class MarkerInjector(ast.NodeTransformer):
    function_idx = 0  # Initialize a global counter for function index

    def visit_FunctionDef(self, node):
        # Create a single-line print statement
        print_stmt = ast.Expr(
            value=ast.Call(
                func=ast.Name(id='print', ctx=ast.Load()),
                args=[ast.Constant(value=f'Filip YuNet Minify: Function fidx={self.function_idx} {node.name} called in {self.filename}:L{node.lineno}')],
                keywords=[]
            )
        )
        self.function_idx += 1
        node.body.insert(0, print_stmt)
        return node

def inject_markers(file_path):
    print(f"Processing file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        tree = ast.parse(source)
        transformer = MarkerInjector(file_path)
        new_tree = transformer.visit(tree)
        new_source = astor.to_source(new_tree)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_source)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                inject_markers(file_path)

if __name__ == "__main__":
    mmdet_path = 'mmdet'
    process_directory(mmdet_path)