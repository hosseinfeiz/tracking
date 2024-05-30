@echo off
setlocal enabledelayedexpansion

REM Function to delete and recreate the tracks folder
:CREATE_TRACKS_FOLDER
set folder=%1
set tracks_folder=C:\Users\AR13620\Desktop\physpose\dataset\%folder%\tracks
if exist "%tracks_folder%" (
    rmdir /s /q "%tracks_folder%"
)
mkdir "%tracks_folder%"
goto :eof

REM Loop through each folder passed as arguments
for %%F in (%*) do (
    call :CREATE_TRACKS_FOLDER %%F
    for %%I in (1 2 3 4) do (
        set file_path=C:\Users\AR13620\Desktop\physpose\dataset\%%F\videos\%%I.mp4
        if exist "!file_path!" (
            python tracking.py --video_path "!file_path!" --output_video_path="C:\Users\AR13620\Desktop\physpose\dataset\%%F\vis_%%I.mp4" --output_track="C:\Users\AR13620\Desktop\physpose\dataset\%%F\tracks\%%I.xml"
        ) else (
            echo File !file_path! does not exist.
        )
    )
)

endlocal
