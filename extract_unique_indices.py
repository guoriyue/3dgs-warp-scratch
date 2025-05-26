#!/usr/bin/env python3
"""
Script to extract unique line indices from output.txt file.
Reads all lines, filters for numerical values, removes duplicates,
and saves the unique indices as a sorted list.
"""

import re
import json

def extract_unique_indices(input_file='output.txt', output_file='unique_indices.json'):
    """
    Extract unique numerical indices from the input file.
    
    Args:
        input_file (str): Path to the input file containing indices
        output_file (str): Path to save the unique indices as JSON
    
    Returns:
        list: Sorted list of unique indices
    """
    unique_indices = set()
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Check if the line contains only digits (is a numerical index)
                if line.isdigit():
                    unique_indices.add(int(line))
                # Also check for lines that might have numbers with some whitespace
                elif re.match(r'^\s*\d+\s*$', line):
                    unique_indices.add(int(line.strip()))
    
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    # Convert to sorted list
    unique_indices_list = sorted(list(unique_indices))
    
    # Save to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unique_indices_list, f, indent=2)
        print(f"Unique indices saved to '{output_file}'")
    except Exception as e:
        print(f"Error saving file: {e}")
    
    # Also save as a simple text file (one index per line)
    text_output_file = output_file.replace('.json', '.txt')
    try:
        with open(text_output_file, 'w', encoding='utf-8') as f:
            for index in unique_indices_list:
                f.write(f"{index}\n")
        print(f"Unique indices also saved to '{text_output_file}'")
    except Exception as e:
        print(f"Error saving text file: {e}")
    
    return unique_indices_list

def main():
    """Main function to run the extraction process."""
    print("Extracting unique indices from output.txt...")
    
    unique_indices = extract_unique_indices()
    
    if unique_indices:
        print(f"\nFound {len(unique_indices)} unique indices:")
        print(f"Range: {min(unique_indices)} to {max(unique_indices)}")
        print(unique_indices)
        print(f"First 10 indices: {unique_indices[:10]}")
        if len(unique_indices) > 10:
            print(f"Last 10 indices: {unique_indices[-10:]}")
    else:
        print("No indices found or error occurred.")

if __name__ == "__main__":
    main() 