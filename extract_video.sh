#!/bin/bash

# Function to show usage information
usage() {
    echo "Usage: $0 -f <input_directory> -s <start_frame> -e <end_frame> -d <output_directory>"
    echo "Example: $0 -f input_folder -s 100 -e 200 -d output_folder"
    exit 1
}

# Parse command line options
while getopts ":f:s:e:d:" opt; do
    case $opt in
        f)
            input_dir="$OPTARG"
            ;;
        s)
            start_frame=$OPTARG
            ;;
        e)
            end_frame=$OPTARG
            ;;
        d)
            output_dir=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

# Check if required options are provided
if [[ -z $input_dir || -z $start_frame || -z $end_frame || -z $output_dir ]]; then
    echo "Error: Missing required option(s)." >&2
    usage
fi

# Create the output directory if it doesn't exist
mkdir -p "${output_dir}_${start_frame}_${end_frame}/videos/"

# Loop through input directory
for file in "$input_dir"/*; do
    if [[ -f "$file" ]]; then
        filename=$(basename -- "$file")
        extension="${filename##*.}"
        filename="${filename%.*}"
        output_file="${output_dir}_${start_frame}_${end_frame}/videos/${filename}.mp4"
        # Extract frames between start_frame and end_frame
        ffmpeg -i "$file" -vf "select=between(n\,$start_frame\,$end_frame),setpts=PTS-STARTPTS" -an "$output_file" || { echo "Error extracting frames from $file"; exit 1; }
        # Check the size of the output video
        output_size=$(du -sh "$output_file" | cut -f1)
        echo "Output video size: $output_size"
    fi
done
