import ast
import os
import astor

class FunctionRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.body and isinstance(node.body[0], ast.Expr):
            # Check if first statement is a print with our marker
            if isinstance(node.body[0].value, ast.Call) and isinstance(node.body[0].value.func, ast.Name):
                print_source = astor.to_source(node.body[0]).strip()
                if 'Filip YuNet Minify' in print_source:
                    print(f"Removing function '{node.name}' at line {node.lineno} in {self.filename}")
                    # Create function signature for the comment
                    params = ', '.join(astor.to_source(arg).rstrip('\n') for arg in node.args.args)
                    comment = ast.Expr(value=ast.Constant(value=f"#Auto-removed {node.name}({params})"))
                    return comment
        print(f"Keeping function '{node.name}' at line {node.lineno} in {self.filename}")
        return node

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

def remove_unused_functions(file_path):
    print(f"\nProcessing file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source)
    remover = FunctionRemover()
    remover.filename = file_path
    new_tree = remover.visit(tree)
    new_source = astor.to_source(new_tree)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_source)
    print(f"Finished processing: {file_path}\n")

def process_directory(directory):
    print(f"Starting to process directory: {directory}")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    remove_unused_functions(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    mmdet_path = 'mmdet'
    print(f"Starting unused function removal in {mmdet_path}")
    process_directory(mmdet_path)
    print("Finished processing all files")