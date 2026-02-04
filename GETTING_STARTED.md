# FlightPath - Getting Started Guide

Welcome to FlightPath! This guide will help you get the project up and running.

## Prerequisites Checklist

Before you begin, make sure you have:

- [ ] **CMake** (3.15+) - [Download](https://cmake.org/download/)
- [ ] **Visual Studio 2019 or 2022** with C++ support
- [ ] **OpenCV 4.x** - [Download](https://opencv.org/releases/)
- [ ] **Dashcam video files** for testing

## Step-by-Step Setup

### 1. Verify Dependencies

Run the setup verification script:

```bash
cd f:\FlightPath
scripts\setup.bat
```

This will check if all required dependencies are installed.

### 2. Install OpenCV (if needed)

1. Download OpenCV from https://opencv.org/releases/
2. Extract to `C:\opencv`
3. Add to system environment variables:
   - `OpenCV_DIR = C:\opencv\build`
   - Add `C:\opencv\build\x64\vc16\bin` to PATH
4. Restart your terminal

### 3. Download YOLO Models

```powershell
cd f:\FlightPath\models

# Download YOLOv4 weights (~250 MB)
Invoke-WebRequest -Uri "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights" -OutFile "yolov4.weights"

# Download config
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg" -OutFile "yolov4.cfg"

# Download class names
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names" -OutFile "coco.names"
```

Or see [models/download_models.md](models/download_models.md) for detailed instructions.

### 4. Get Test Videos

Place dashcam videos in the `data/` directory. See [data/README.md](data/README.md) for sources of free dashcam footage.

### 5. Build the Project

```bash
cd f:\FlightPath
scripts\build.bat
```

This will:
- Create a `build` directory
- Configure the project with CMake
- Build the Release executable

The executable will be at: `build\bin\Release\FlightPath.exe`

### 6. Run FlightPath

```bash
# Basic usage
.\build\bin\Release\FlightPath.exe data\your_video.mp4

# Save output video
.\build\bin\Release\FlightPath.exe data\your_video.mp4 --output output\result.mp4

# Adjust detection threshold
.\build\bin\Release\FlightPath.exe data\your_video.mp4 --confidence 0.6
```

## Keyboard Controls

While the video is playing:

- **SPACE** - Pause/Resume
- **ESC** - Exit
- **S** - Save current frame as JPG

## Understanding the Output

FlightPath displays:

1. **Red Bounding Boxes** - Detected vehicles and obstacles
2. **Green Arrows** - Safe, wide paths
3. **Yellow Arrows** - Tight but passable paths
4. **Info Panel** - FPS, detection count, path statistics

## Customization

Edit `src/Config.h` to adjust:

- Detection confidence threshold
- Vehicle dimensions
- Path planning parameters
- Visualization colors
- Target object classes

After changes, rebuild:

```bash
cd build
cmake --build . --config Release
```

## Troubleshooting

### "OpenCV not found"
- Set `OpenCV_DIR` environment variable
- Verify OpenCV bin directory is in PATH
- Restart terminal after setting variables

### "Model not loading"
- Check that model files are in `models/` directory
- Verify file sizes (yolov4.weights should be ~250 MB)
- Try re-downloading the models

### Low FPS
- Try lower resolution videos (720p instead of 1080p)
- Reduce detection confidence threshold
- Consider GPU acceleration (requires CUDA-enabled OpenCV)

### No paths detected
- Adjust `minGapWidth` in `src/Config.h`
- Lower `confidenceThreshold` to detect more objects
- Try videos with clearer lane separation

## Next Steps

1. **Experiment** with different dashcam videos
2. **Tune parameters** in `Config.h` for better results
3. **Learn** by reading the source code:
   - Start with `main.cpp` for the overall flow
   - `ObjectDetector.cpp` for YOLO integration
   - `PathPlanner.cpp` for the path finding algorithm
4. **Extend** the project:
   - Add lane detection
   - Implement distance estimation
   - Try different object detection models

## Learning Resources

- **OpenCV Documentation**: https://docs.opencv.org/
- **YOLO Paper**: https://arxiv.org/abs/2004.10934
- **Autonomous Driving**: Tesla AI Day presentations
- **Computer Vision**: Stanford CS231n course

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review the README.md
3. Verify all dependencies are correctly installed
4. Ensure model files are downloaded

Happy learning! 🚗🛣️
