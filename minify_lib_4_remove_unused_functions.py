import os
import ast
import astor

class FunctionRemover(ast.NodeTransformer):
    """
    AST NodeTransformer that removes function definitions containing the
    'Filip YuNet Minify' print markers.
    """
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.functions_removed = []

    def visit_FunctionDef(self, node):
        """
        Visit each function definition and remove it if it contains the marker print.
        """
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if (isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == 'print'):
                    if (stmt.value.args and isinstance(stmt.value.args[0], ast.Constant) and
                            'Filip YuNet Minify' in stmt.value.args[0].value):
                        func_name = node.name
                        print(f"Removing function '{func_name}' from {self.filename} at line {node.lineno}")
                        self.functions_removed.append(func_name)
                        return None  # Remove this function definition
        return node  # Keep the function definition if no marker is found

    def visit_AsyncFunctionDef(self, node):
        """
        Handle async function definitions similarly.
        """
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if (isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == 'print'):
                    if (stmt.value.args and isinstance(stmt.value.args[0], ast.Constant) and
                            'Filip YuNet Minify' in stmt.value.args[0].value):
                        func_name = node.name
                        print(f"Removing async function '{func_name}' from {self.filename} at line {node.lineno}")
                        self.functions_removed.append(func_name)
                        return None  # Remove this async function definition
        return node  # Keep the async function definition if no marker is found

def remove_unused_functions(file_path):
    """
    Remove functions containing the 'Filip YuNet Minify' marker from the given file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        remover = FunctionRemover(filename=file_path)
        modified_tree = remover.visit(tree)
        
        if remover.functions_removed:
            # Convert the modified AST back to source code
            new_source = astor.to_source(modified_tree)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_source)
            print(f"Updated {file_path}: Removed {len(remover.functions_removed)} function(s).")
        else:
            print(f"No unused functions found in {file_path}.")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_directory(directory):
    """
    Traverse the directory and process all Python files.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                remove_unused_functions(file_path)

def main():
    """
    Main function to execute the script.
    """
    # Specify the root directory of your codebase
    codebase_path = 'mmdet'  # Change this path if necessary
    print(f"Starting removal of unused functions in directory: {codebase_path}")
    process_directory(codebase_path)
    print("Finished processing all files.")

if __name__ == "__main__":
    main()