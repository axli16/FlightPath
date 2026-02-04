# YOLO Model Files

This directory contains the YOLO model files needed for object detection.

## Required Files

You need to download the following files:

### YOLOv4 (Recommended)

1. **yolov4.weights** (~250 MB)
   - Download from: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
   - Direct link: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

2. **yolov4.cfg**
   - Download from: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
   - Save as: `yolov4.cfg`

3. **coco.names**
   - Download from: https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
   - Save as: `coco.names`

## Quick Download (Windows PowerShell)

Run these commands from the `models` directory:

```powershell
# Download YOLOv4 weights
Invoke-WebRequest -Uri "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights" -OutFile "yolov4.weights"

# Download YOLOv4 config
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg" -OutFile "yolov4.cfg"

# Download COCO class names
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names" -OutFile "coco.names"
```

## Alternative: YOLOv3 (Smaller, Faster)

If you want a lighter model:

1. **yolov3.weights** (~240 MB)
   - https://pjreddie.com/media/files/yolov3.weights

2. **yolov3.cfg**
   - https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg

3. **coco.names** (same as above)

## Verify Downloads

After downloading, your `models/` directory should contain:

```
models/
├── yolov4.weights  (~250 MB)
├── yolov4.cfg      (~12 KB)
└── coco.names      (~1 KB)
```

## COCO Classes

The COCO dataset includes 80 classes. FlightPath focuses on:
- **Class 0**: person
- **Class 2**: car
- **Class 3**: motorcycle
- **Class 5**: bus
- **Class 7**: truck

You can modify the `targetClasses` in `src/Config.h` to detect different objects.

## Troubleshooting

**Download fails?**
- Try downloading manually from the links above
- Use a download manager for large files
- Check your internet connection

**Model not loading?**
- Verify file sizes match expected values
- Ensure files are in the correct directory
- Check file permissions

## GPU Acceleration (Optional)

For faster processing, you can use GPU acceleration:
1. Install CUDA Toolkit (11.x or compatible)
2. Build OpenCV with CUDA support
3. Run FlightPath with `--gpu` flag

This can improve FPS from ~5-10 to ~30-60 depending on your GPU.
