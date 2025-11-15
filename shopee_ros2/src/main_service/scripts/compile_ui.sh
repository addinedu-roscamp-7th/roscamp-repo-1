#!/bin/bash

# This script finds all .ui files in main_service package and compiles them into .py files
# using pyuic6, placing them in a corresponding 'ui_gen' directory.

set -e # Exit immediately if a command exits with a non-zero status.

# Find all .ui files within the main_service package only
# Using find is more robust for paths with spaces.
find shopee_ros2/src/main_service -name "*.ui" | while read ui_file; do
    # Construct the output path
    # 1. Get the directory of the .ui file
    ui_dir=$(dirname "$ui_file")
    # 2. Get the base name of the .ui file (e.g., "main_window")
    base_name=$(basename "$ui_file" .ui)
    # 3. Construct the output directory by replacing 'ui' with 'ui_gen'
    out_dir=$(echo "$ui_dir" | sed 's|/ui$|/ui_gen|')
    # 4. Construct the output file path
    out_file="$out_dir/${base_name}_ui.py"

    # Check if the directory was correctly substituted
    if [ "$ui_dir" == "$out_dir" ]; then
        echo "Skipping $ui_file: Could not determine output directory 'ui_gen'. Make sure it's under a 'ui' directory."
        continue
    fi

    echo "Compiling $ui_file -> $out_file"

    # Create the output directory if it doesn't exist
    mkdir -p "$out_dir"

    # Run the pyuic6 compiler
    pyuic6 -x "$ui_file" -o "$out_file"
done

echo "UI compilation complete."