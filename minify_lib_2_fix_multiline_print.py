import os

def fix_multiline_prints(file_path):
    """Convert multi-line Filip YuNet Minify prints to single line."""
    try:
        print(f"\nProcessing {file_path}")
        
        # Read all lines
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Track if we made any changes
        modified = False
        
        # Process each line that needs modification
        i = 0
        new_lines = []
        while i < len(lines):
            # Get the line and its original indentation
            original_line = lines[i]
            indentation = original_line[:len(original_line) - len(original_line.lstrip())]
            current_line = original_line.strip()
            
            if current_line == 'print(':
                # Look ahead for Filip YuNet Minify marker
                if i + 1 < len(lines) and 'Filip YuNet Minify' in lines[i + 1]:
                    # Collect all lines until closing parenthesis
                    print_content = []
                    j = i + 1
                    while j < len(lines) and ')' not in lines[j]:
                        print_content.append(lines[j].strip().strip("'").strip())
                        j += 1
                    if j < len(lines):
                        print_content.append(lines[j].strip().strip("'").strip().rstrip(')'))
                    
                    # Create single-line print with original indentation
                    new_lines.append(f"{indentation}print('{' '.join(print_content)}')\n")
                    modified = True
                    i = j + 1
                    continue
            
            new_lines.append(original_line)
            i += 1
        
        # Write back only if modified
        if modified:
            print("Making changes...")
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            print("Done!")
        else:
            print("No changes needed")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                fix_multiline_prints(file_path)

if __name__ == "__main__":
    mmdet_path = 'mmdet'
    process_directory(mmdet_path) 