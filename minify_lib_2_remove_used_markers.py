import os

def remove_markers(file_path, line_numbers):
    """Remove markers from specific line numbers in a file."""
    try:
        # Read all lines
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Track if we made any changes
        modified = False
        
        # Process each line that needs modification
        i = 0
        while i < len(lines):
            if (i + 1) in line_numbers and 'Filip YuNet Minify' in lines[i]:
                # Check if this is a 3-line print statement
                if i + 2 < len(lines) and lines[i].strip().startswith('print('):
                    lines[i] = ''  # Remove first line
                    lines[i + 1] = ''  # Remove second line
                    lines[i + 2] = ''  # Remove third line
                    modified = True
                    print(f"Removed 3-line marker at {file_path}:{i+1}")
                    i += 3
                    continue
            i += 1
        
        # Write back only if modified
        if modified:
            # Remove any empty lines we created
            lines = [line for line in lines if line.strip()]
            print(f"Writing changes to {file_path}")
            with open(file_path, 'w') as f:
                f.write(''.join(lines))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def get_used_functions():
    """Parse the log file to get used functions."""
    used = {}  # file -> set of line numbers
    
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
                        print(f"Found used function at {file_path}:{line_num}")
                    except Exception as e:
                        # Skip malformed lines
                        print(f"Skipping malformed line: {line.strip()}")
                        continue
                        
    except Exception as e:
        print(f"Error parsing log file: {e}")
        return {}
        
    return used

def main():
    # Get mapping of files to their used function lines
    used_functions = get_used_functions()
    total_functions = sum(len(lines) for lines in used_functions.values())
    print(f"Found {total_functions} used functions across {len(used_functions)} files")
    
    # Process each file
    for file_path, line_numbers in used_functions.items():
        if os.path.exists(file_path):
            remove_markers(file_path, line_numbers)
        else:
            print(f"Warning: File not found: {file_path}")

if __name__ == "__main__":
    main()