#!/bin/bash

# Function to delete and recreate the tracks folder
create_tracks_folder() {
    folder=$1
    tracks_folder="../physpose/dataset/$folder/tracks"
    if [ -d "$tracks_folder" ]; then
        rm -rf "$tracks_folder"
    fi
    mkdir -p "$tracks_folder"
}

# Loop through each folder
for folder in "$@"
do
    create_tracks_folder "$folder"
    for file in {1..4}  
    do
        file_path="../physpose/dataset/$folder/videos/$file.mp4"
        if [ -f "$file_path" ]; then  # Check if file exists
            python tracking.py --video_path "$file_path" --output_video_path="../physpose/dataset/$folder/vis_$file.mp4" --output_track="../physpose/dataset/$folder/tracks/$file.xml"
        else
            echo "File $file_path does not exist."
        fi
    done
done

