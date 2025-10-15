#!/bin/bash

# find all tv*.bin files
files=(tv*.bin)

if [ ${#files[@]} -eq 0 ]; then
    echo "no tv*.bin files found"
    exit 1
fi

# get the size of the first file (assuming all files are same size)
size=$(stat -c%s "${files[0]}")

# temp output file
out="majority.bin"
: > "$out"

# loop through each byte position
for ((i=0; i<size; i++)); do
    declare -A counts
    for f in "${files[@]}"; do
        # extract the ith byte as decimal
        byte=$(xxd -p -c1 -s $i "$f" | tr 'a-f' 'A-F')
        ((counts[$byte]++))
    done

    # find the byte with max count
    max_byte=""
    max_count=0
    for b in "${!counts[@]}"; do
        if (( counts[$b] > max_count )); then
            max_count=${counts[$b]}
            max_byte=$b
        fi
    done

    # append the majority byte to output
    printf "\\x$max_byte" >> "$out"

    unset counts
done

echo "majority consensus written to $out"

