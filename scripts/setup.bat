@echo off
REM FlightPath Setup Script for Windows
REM Helps verify dependencies and setup

echo ========================================
echo FlightPath Setup Verification
echo ========================================
echo.

REM Check CMake
echo Checking CMake...
cmake --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [X] CMake not found!
    echo     Download from: https://cmake.org/download/
    set CMAKE_OK=0
) else (
    cmake --version | findstr /C:"version"
    echo [OK] CMake found
    set CMAKE_OK=1
)
echo.

REM Check Visual Studio
echo Checking Visual Studio...
if exist "C:\Program Files\Microsoft Visual Studio\2022" (
    echo [OK] Visual Studio 2022 found
    set VS_OK=1
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019" (
    echo [OK] Visual Studio 2019 found
    set VS_OK=1
) else (
    echo [X] Visual Studio not found!
    echo     Download from: https://visualstudio.microsoft.com/
    set VS_OK=0
)
echo.

REM Check OpenCV
echo Checking OpenCV...
if defined OpenCV_DIR (
    echo [OK] OpenCV_DIR environment variable set
    echo     Path: %OpenCV_DIR%
    set OPENCV_OK=1
) else (
    echo [X] OpenCV_DIR environment variable not set!
    echo     1. Download OpenCV from: https://opencv.org/releases/
    echo     2. Extract to C:\opencv
    echo     3. Set OpenCV_DIR=C:\opencv\build
    echo     4. Add C:\opencv\build\x64\vc16\bin to PATH
    set OPENCV_OK=0
)
echo.

REM Check if models exist
echo Checking YOLO models...
if exist "models\yolov4.weights" (
    echo [OK] yolov4.weights found
    set MODEL_OK=1
) else (
    echo [!] yolov4.weights not found
    echo     See models\download_models.md for instructions
    set MODEL_OK=0
)
echo.

REM Summary
echo ========================================
echo Setup Summary
echo ========================================
if %CMAKE_OK%==1 if %VS_OK%==1 if %OPENCV_OK%==1 (
    echo [OK] All required dependencies are installed!
    echo.
    echo You can now build the project:
    echo     scripts\build.bat
    echo.
    if %MODEL_OK%==0 (
        echo Don't forget to download YOLO models before running!
    )
) else (
    echo [X] Some dependencies are missing. Please install them first.
)
echo.

pause
