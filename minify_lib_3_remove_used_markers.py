import os

def remove_markers(file_path, line_numbers):
    """Remove markers from specific line numbers in a file."""
    try:
        print(f"\nProcessing {file_path}")
        print(f"Looking for line numbers: {sorted(line_numbers)}")
        
        # Read all lines
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        print(f"File has {len(lines)} lines")
        
        # Track if we made any changes
        modified = False
        
        # Process each line that needs modification
        i = 0
        while i < len(lines):
            current_line = lines[i].strip()
            if current_line.startswith("print('Filip YuNet Minify:") or current_line.startswith('print("Filip YuNet Minify:'):
                try:
                    # Extract the line number, being careful with spaces
                    marker_text = current_line.split('called in')[1].strip()
                    # Remove trailing quote, parenthesis and any spaces
                    marker_text = marker_text.rstrip("')").rstrip("'").strip()
                    reported_line = int(''.join(c for c in marker_text.split(':L')[1] if c.isdigit()))
                    print(f"Found marker at line {i+1} reporting line {reported_line}")
                    
                    if reported_line in line_numbers:
                        print(f"This is a match! Removing line {i+1}")
                        lines[i] = ''
                        modified = True
                except Exception as e:
                    print(f"Error parsing line: {e}")
            i += 1
        
        # Write back only if modified
        if modified:
            print("\nMaking changes...")
            # Remove any empty lines we created
            original_length = len(lines)
            lines = [line for line in lines if line.strip()]
            new_length = len(lines)
            print(f"Removed {original_length - new_length} lines")
            print(f"Writing changes to {file_path}")
            with open(file_path, 'w') as f:
                f.write(''.join(lines))
        else:
            print("\nNo changes needed")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

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
    
    # Process each file
    for file_path, line_numbers in used_functions.items():
        if os.path.exists(file_path):
            remove_markers(file_path, line_numbers)
        else:
            print(f"Warning: File not found: {file_path}")

if __name__ == "__main__":
    main()