param (
    [string[]]$folderNames
)

# Define the base directory containing all the folders
$baseDir = "C:\Users\AR13620\Desktop\physpose\revision"

foreach ($folderName in $folderNames) {
    $folderPath = Join-Path -Path $baseDir -ChildPath $folderName
    
    if (-not (Test-Path -Path $folderPath -PathType Container)) {
        Write-Host "The path $folderPath does not exist or is not a directory."
        continue
    }

    # Iterate through video numbers 1 to 4
    for ($i = 1; $i -le 4; $i++) {
        # Define the video file path
        $videoPath = Join-Path -Path $folderPath -ChildPath "videos\$i.mp4"

        if (-not (Test-Path -Path $videoPath -PathType Leaf)) {
            Write-Host "The video file $videoPath does not exist."
            continue
        }

        # Define the output video file path
        $outputVideoPath = Join-Path -Path $folderPath -ChildPath "vis_$i.mp4"

        # Define the output track file path
        $outputTrackPath = Join-Path -Path $folderPath -ChildPath "tracks\$i.xml"

        # Run the Python script with the specified arguments
        Write-Host "Processing $videoPath..."
        python tracking.py --video_path $videoPath --output_video_path=$outputVideoPath --output_track=$outputTrackPath
    }
}
