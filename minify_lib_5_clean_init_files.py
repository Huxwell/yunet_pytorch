import os
import ast
import astor
import json

def load_removed_functions(record_file='removed_functions.json'):
    """Load the removed functions record from a JSON file."""
    try:
        with open(record_file, 'r') as f:
            removed = json.load(f)
        return removed
    except Exception as e:
        print(f"Error loading removed functions record: {e}")
        return {}

def clean_init_file(init_file_path, removed_functions):
    """Remove imports of removed functions from an __init__.py file."""
    try:
        with open(init_file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        modified = False

        class ImportCleaner(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                # node.module could be something like '.module'
                module = node.module.lstrip('.')  # Remove leading dots
                if not module:
                    return node  # Skip if module is empty

                # Get the relative path
                module_path = module.replace('.', os.sep) + '.py'
                # Construct absolute path based on init_file_path
                dir_path = os.path.dirname(init_file_path)
                module_file = os.path.join(dir_path, module_path)
                
                # Get the relative path from root
                if os.path.exists(module_file):
                    relative_module = os.path.relpath(module_file, start=os.getcwd()).replace('\\', '/')
                else:
                    relative_module = module
                
                # Get removed functions for this module
                removed_funcs = removed_functions.get(relative_module, [])
                
                # Filter out the names to be imported
                original_names = [alias.name for alias in node.names]
                filtered_names = [alias for alias in node.names if alias.name not in removed_funcs]
                
                if len(filtered_names) < len(node.names):
                    node.names = filtered_names
                    modified_nonlocal[0] = True
                    print(f"Updated import in {init_file_path}: Removed {len(node.names) - len(filtered_names)} import(s) from '{module}'.")
                
                return node

        modified_nonlocal = [False]  # To track modifications within the nested class

        cleaner = ImportCleaner()
        cleaner.visit(tree)

        if modified_nonlocal[0]:
            # Reconstruct the code from the modified AST
            new_source = astor.to_source(tree)
            with open(init_file_path, 'w', encoding='utf-8') as f:
                f.write(new_source)
            print(f"Cleaned imports in {init_file_path}")
        else:
            print(f"No changes needed for {init_file_path}")

    except Exception as e:
        print(f"Error processing {init_file_path}: {e}")

def process_init_files(removed_functions, directory):
    """Traverse the directory and clean all __init__.py files."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file == '__init__.py':
                init_file_path = os.path.join(root, file)
                clean_init_file(init_file_path, removed_functions)

def main():
    # Load the record of removed functions
    removed_functions = load_removed_functions()
    if not removed_functions:
        print("No removed functions to process.")
        return

    # Normalize file paths to use forward slashes
    normalized_removed = {k.replace('\\', '/'): v for k, v in removed_functions.items()}

    # Process all __init__.py files in the codebase
    codebase_path = 'mmdet'  # Change if your codebase root is different
    print(f"Starting cleanup of __init__.py files in directory: {codebase_path}")
    process_init_files(normalized_removed, codebase_path)
    print("Finished cleaning __init__.py files.")

if __name__ == "__main__":
    main()