#!/bin/bash

# Set the directory path
DIR="A_Test_Collection_for_Matching_Patient_to_Clinical_Trials_small/data/clinicaltrials.gov-16_dec_2015/clinicaltrials.gov-16_dec_2015"

# Count the number of .xml files
total_files=$(find "$DIR" -maxdepth 1 -name "*.xml" | wc -l)

# Calculate how many files to delete
files_to_delete=$((total_files - 5000))

if [ $files_to_delete -le 0 ]; then
    echo "There are 5000 or fewer .xml files. No deletion needed."
    exit 0
fi

echo "Deleting $files_to_delete .xml files..."

# Find .xml files, sort them by modification time (oldest first),
# and delete the excess files
find "$DIR" -maxdepth 1 -name "*.xml" -print0 | \
    sort -z -t. -k2 | \
    head -z -n "$files_to_delete" | \
    xargs -0 rm -f

echo "Deletion complete. 5000 .xml files should remain."
