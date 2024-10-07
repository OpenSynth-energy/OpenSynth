#!/bin/bash

# Path to the JSON file
JSON_FILE="scan_results.json"

# List of GPL licenses to search for
GPL_LICENSES=("GPL-1.0-only" "GPL-1.0-or-later" "GPL-2.0-only" "GPL-2.0-or-later" "GPL-3.0-only" "GPL-3.0-or-later")

# Directory to exclude from the scan
EXCLUDED_DIR="project/notebooks"

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "jq is not installed. Please install jq to run this script."
    exit 1
fi

# Function to check for GPL licenses
check_for_gpl_licenses() {
    local file=$1
    local found=0

    # Exclude files in the "notebooks" directory
    if [[ "$file" == *"$EXCLUDED_DIR"* ]]; then
        echo "Skipping file in excluded directory: $file"
        return
    fi

    for license in "${GPL_LICENSES[@]}"; do
        # Find all matching detections for the GPL licenses
        local matches=$(jq -c ".license_detections[] | select(.license_expression_spdx == \"$license\")" "$file")

        if [ -n "$matches" ]; then
            found=1
            echo "GPL license found: $license"
            # Extract and print file locations for each match
            echo "$matches" | jq -r '.reference_matches[] | "\(.license_expression_spdx) found in file: \(.from_file), lines \(.start_line)-\(.end_line)"'
            # Append file locations to the array
            while IFS= read -r line; do
                file_locations+=("$line")
            done < <(echo "$matches" | jq -r '.reference_matches[] | "\(.license_expression_spdx) found in file: \(.from_file), lines \(.start_line)-\(.end_line)"')
        fi
    done

    if [ $found -eq 0 ]; then
        echo "No GPL license found."
    fi

    # Set the GitHub Actions outputs
    echo "::set-output name=found::$found"
    echo "::set-output name=file_locations::${file_locations[*]}"
}

# Run the function with the JSON file
check_for_gpl_licenses "$JSON_FILE"
