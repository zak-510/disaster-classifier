@echo off
REM data_finalize.bat
REM Sets up SpaceNet model directory structure, creates train/val splits, and runs compute_mean.py
REM Usage: data_finalize.bat -i <input_dir> -x <repo_dir> -s <split_ratio>

setlocal EnableDelayedExpansion

:parse_args
if "%1"=="" goto :done_parsing
if "%1"=="-i" set "INPUT_DIR=%2" & shift & shift & goto :parse_args
if "%1"=="-x" set "REPO_DIR=%2" & shift & shift & goto :parse_args
if "%1"=="-s" set "SPLIT_RATIO=%2" & shift & shift & goto :parse_args
shift
goto :parse_args
:done_parsing

if "%INPUT_DIR%"=="" (
    echo Error: Input directory not specified
    exit /b 1
)
if "%REPO_DIR%"=="" (
    echo Error: Repository directory not specified
    exit /b 1
)
if "%SPLIT_RATIO%"=="" set SPLIT_RATIO=0.75

set "OUTPUT_DIR=%INPUT_DIR%\spacenet_gt"

REM Create directory structure
mkdir "%OUTPUT_DIR%\dataSet" 2>nul
mkdir "%OUTPUT_DIR%\images" 2>nul
mkdir "%OUTPUT_DIR%\labels" 2>nul

set count=0
set total=0

REM Count total pre-disaster images across all disasters
for /f "delims=" %%I in ('dir /b /s "%INPUT_DIR%\*_pre_disaster.png"') do (
    set /a total+=1
)

REM Calculate split point
set /a split_point=total * %SPLIT_RATIO:~-2% / 100

REM Create train.txt and val.txt
type nul > "%OUTPUT_DIR%\dataSet\train.txt"
type nul > "%OUTPUT_DIR%\dataSet\val.txt"

REM Process each disaster directory
for /d %%D in ("%INPUT_DIR%\*") do (
    if not "%%~nxD"=="spacenet_gt" (
        for /f "delims=" %%I in ('dir /b "%%D\images\*_pre_disaster.png"') do (
            set /a count+=1
            set "basename=%%~nI"
            set "basename=!basename:_pre_disaster=!"
            
            REM Copy images and labels
            copy "%%D\images\!basename!_pre_disaster.png" "%OUTPUT_DIR%\images\" >nul
            copy "%%D\images\!basename!_post_disaster.png" "%OUTPUT_DIR%\images\" >nul
            copy "%%D\labels\!basename!_pre_disaster.json" "%OUTPUT_DIR%\labels\" >nul 2>nul
            copy "%%D\labels\!basename!_post_disaster.json" "%OUTPUT_DIR%\labels\" >nul 2>nul
            
            REM Add to train.txt or val.txt
            if !count! LEQ !split_point! (
                echo !basename!_pre_disaster.png >> "%OUTPUT_DIR%\dataSet\train.txt"
                echo !basename!_post_disaster.png >> "%OUTPUT_DIR%\dataSet\train.txt"
            ) else (
                echo !basename!_pre_disaster.png >> "%OUTPUT_DIR%\dataSet\val.txt"
                echo !basename!_post_disaster.png >> "%OUTPUT_DIR%\dataSet\val.txt"
            )
        )
    )
)

REM Run compute_mean.py (placeholder, to be implemented)
echo [Info] Running compute_mean.py (not implemented in this script)
REM python compute_mean.py --data_dir "%OUTPUT_DIR%" --output "%OUTPUT_DIR%\mean.npy"

echo [Done] SpaceNet structure ready. 