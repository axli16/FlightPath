@echo off
REM FlightPath Build Script for Windows
REM Requires: CMake, Visual Studio, OpenCV

echo ========================================
echo FlightPath Build Script
echo ========================================
echo.

REM Check if build directory exists
if exist build (
    echo Build directory already exists.
    set /p REBUILD="Rebuild from scratch? (y/n): "
    if /i "%REBUILD%"=="y" (
        echo Cleaning build directory...
        rmdir /s /q build
    )
)

REM Create build directory
if not exist build (
    echo Creating build directory...
    mkdir build
)

cd build

REM Configure with CMake
echo.
echo Configuring project with CMake...
cmake .. -G "Visual Studio 17 2022" -A x64

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: CMake configuration failed!
    echo.
    echo Possible issues:
    echo - OpenCV not found. Set OpenCV_DIR environment variable
    echo - CMake not in PATH
    echo - Visual Studio 2022 not installed
    echo.
    pause
    exit /b 1
)

REM Build the project
echo.
echo Building project (Release configuration)...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Executable location: build\bin\Release\FlightPath.exe
echo.
echo Next steps:
echo 1. Download YOLO models (see models\download_models.md)
echo 2. Place your dashcam videos in the data\ directory
echo 3. Run: build\bin\Release\FlightPath.exe data\your_video.mp4
echo.

cd ..
pause
