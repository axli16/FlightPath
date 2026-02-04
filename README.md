# FlightPath - Computer Vision Path Detection System

A C++ application that processes dashcam footage to detect objects and identify navigable paths, similar to autonomous driving systems used by Tesla and other self-driving car companies.

## Features

- **Real-time Object Detection**: Uses YOLOv4 to detect vehicles, pedestrians, and obstacles
- **Path Planning**: Identifies navigable gaps and calculates safe driving paths
- **Visual Overlay**: Draws arrows and indicators showing potential paths
- **Video Processing**: Supports pre-recorded dashcam footage with output video generation

## Prerequisites

### Required
- **CMake** (3.15 or higher)
- **C++ Compiler** with C++17 support (MSVC 2019+, GCC 7+, or Clang 5+)
- **OpenCV** (4.x recommended)

### Installation on Windows

1. **Install CMake**
   - Download from [cmake.org](https://cmake.org/download/)
   - Add to PATH during installation

2. **Install OpenCV**
   - Download pre-built binaries from [opencv.org](https://opencv.org/releases/)
   - Extract to `C:\opencv`
   - Add `C:\opencv\build\x64\vc16\bin` to system PATH
   - Set environment variable: `OpenCV_DIR=C:\opencv\build`

3. **Install Visual Studio 2019 or 2022**
   - Include "Desktop development with C++" workload

## Setup

1. **Clone or navigate to the project**
   ```bash
   cd f:\FlightPath
   ```

2. **Download YOLO model files**
   - See [models/download_models.md](models/download_models.md) for instructions
   - Required files:
     - `yolov4.weights`
     - `yolov4.cfg`
     - `coco.names`

3. **Place your dashcam videos**
   - Put test videos in the `data/` directory

## Building

### Windows (Visual Studio)

```bash
# Create build directory
mkdir build
cd build

# Generate Visual Studio project
cmake ..

# Build the project
cmake --build . --config Release

# Executable will be in build/bin/Release/FlightPath.exe
```

### Alternative: Using Visual Studio directly

```bash
# Generate solution
cmake -B build -G "Visual Studio 17 2022"

# Open the generated solution
start build/FlightPath.sln
```

## Usage

```bash
# Basic usage
.\build\bin\Release\FlightPath.exe <video_path>

# Example
.\build\bin\Release\FlightPath.exe data\dashcam_highway.mp4

# With custom model path
.\build\bin\Release\FlightPath.exe data\dashcam.mp4 --model models\yolov4.weights --config models\yolov4.cfg

# Save output video
.\build\bin\Release\FlightPath.exe data\dashcam.mp4 --output output\result.mp4
```

### Controls (during playback)
- **Space**: Pause/Resume
- **ESC**: Exit
- **S**: Save current frame
- **+/-**: Adjust detection confidence threshold

## Project Structure

```
FlightPath/
├── src/
│   ├── main.cpp              # Entry point
│   ├── VideoProcessor.h/cpp  # Video I/O handling
│   ├── ObjectDetector.h/cpp  # YOLO object detection
│   ├── PathPlanner.h/cpp     # Path planning algorithm
│   ├── Visualizer.h/cpp      # Overlay rendering
│   └── Config.h              # Configuration constants
├── models/
│   └── download_models.md    # Model download instructions
├── data/                     # Input videos (gitignored)
├── output/                   # Output videos (gitignored)
├── CMakeLists.txt            # Build configuration
└── README.md                 # This file
```

## How It Works

1. **Video Processing**: Reads dashcam footage frame by frame
2. **Object Detection**: YOLOv4 detects vehicles and obstacles in each frame
3. **Path Planning**: Analyzes detected objects to find navigable gaps
4. **Visualization**: Draws arrows and overlays showing potential paths
5. **Output**: Displays processed video and optionally saves to file

## Learning Resources

This project demonstrates key computer vision concepts:
- Deep learning-based object detection
- Video processing pipelines
- Spatial reasoning and path planning
- Real-time visualization techniques

## Troubleshooting

### OpenCV not found
- Ensure `OpenCV_DIR` environment variable is set
- Verify OpenCV bin directory is in PATH
- Try specifying manually: `cmake -DOpenCV_DIR=C:\opencv\build ..`

### Model files missing
- Download from the links in `models/download_models.md`
- Ensure files are in the `models/` directory

### Low FPS / Performance issues
- Try reducing video resolution
- Adjust detection confidence threshold
- Use GPU-accelerated OpenCV build (CUDA)

## Future Enhancements

- [ ] Multi-threaded processing for better performance
- [ ] GPU acceleration with CUDA
- [ ] Lane detection integration
- [ ] Distance estimation using monocular depth
- [ ] Upgrade to YOLOv8 for better accuracy
- [ ] Live camera feed support

## License

This is an educational project for learning computer vision and autonomous driving concepts.

## Acknowledgments

- YOLOv4 by Alexey Bochkovskiy
- OpenCV community
- Inspired by Tesla's Autopilot visualization
