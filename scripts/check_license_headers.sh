#!/bin/bash

# Define the directory containing the Python files
SRC_DIR="src"

# Define the required heading
REQUIRED_HEADING=$(printf "# Copyright Contributors to the Opensynth-energy Project.\n# SPDX-License-Identifier: Apache-2.0")

# Initialize a list to store files missing the required heading
missing_files=()

# Loop through all .py files in the src directory
while IFS= read -r file; do
  echo "Collecting file: $file"
  # Read the first two lines of the file
  first_two_lines=$(head -n 2 "$file")

  # Check if the first two lines match the required heading
  if [[ "$first_two_lines" != "$REQUIRED_HEADING" ]]; then
    missing_files+=("$file")
  fi
done < <(find "$SRC_DIR" -name '*.py')

# Check if any files were missing the heading
if [ ${#missing_files[@]} -ne 0 ]; then
  echo "The following files are missing the required heading:"
  for file in "${missing_files[@]}"; do
    echo "$file"
  done
  exit 1
else
  echo "All files contain the required heading."
  exit 0
fi
