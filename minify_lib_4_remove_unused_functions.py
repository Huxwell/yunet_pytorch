import os
import ast
import astor
import json  # Add import for JSON handling

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

def remove_unused_functions(file_path, removed_functions_record):
    """
    Remove functions containing the 'Filip YuNet Minify' marker from the given file
    and log the removed functions.
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
            # Log removed functions
            relative_path = os.path.relpath(file_path)
            removed_functions_record[relative_path] = remover.functions_removed
        else:
            print(f"No unused functions found in {file_path}.")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_directory(directory, removed_functions_record):
    """
    Traverse the directory and process all Python files.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                remove_unused_functions(file_path, removed_functions_record)

def get_used_functions():
    """Parse the log file to get used functions."""
    used = {}  # file -> set of line numbers

    print("Reading function_calls.log...")
    try:
        with open('function_calls.log') as f:
            for line in f:
                if "Filip YuNet Minify" in line and "called in" in line:
                    try:
                        # Split on "called in" and take the second part
                        location = line.split("called in ")[1].strip()
                        # Split on ":L" to separate file path and line number
                        file_path, rest = location.split(":L")
                        # Take only the numeric part for line number
                        line_num = int(''.join(c for c in rest if c.isdigit()))

                        # Add to our dict
                        if file_path not in used:
                            used[file_path] = set()
                        used[file_path].add(line_num)
                        print(f"Found: {file_path} -> L{line_num}")
                    except Exception as e:
                        print(f"Skipping malformed line: {line.strip()}")
                        print(f"Error: {e}")
                        continue

    except Exception as e:
        print(f"Error parsing log file: {e}")
        return {}

    print(f"\nSummary: Found {sum(len(lines) for lines in used.values())} functions in {len(used)} files")
    return used

def main():
    # Get mapping of files to their used function lines
    used_functions = get_used_functions()
    total_functions = sum(len(lines) for lines in used_functions.values())
    print(f"Found {total_functions} used functions across {len(used_functions)} files")

    # Initialize a record for removed functions
    removed_functions_record = {}

    # Process each file for removal
    for file_path, line_numbers in used_functions.items():
        if os.path.exists(file_path):
            remove_unused_functions(file_path, removed_functions_record)
        else:
            print(f"Warning: File not found: {file_path}")

    # Save the removed functions to a JSON file for later use
    with open('removed_functions.json', 'w') as json_file:
        json.dump(removed_functions_record, json_file, indent=4)
    print("Logged removed functions to 'removed_functions.json'.")

if __name__ == "__main__":
    main()