#!/bin/bash
folders=("$@") 
# Loop through each folder
for folder in "${folders[@]}"
    do
        for file in  1 2 3 4
        do 

            python tracking.py  --video_path "Dataset/new/$folder/0000_1000/videos/$file.mp4" --output_video_path="Dataset/new/$folder/0000_1000/vis_$file.mp4"
        done
    
done
