# Sample Dashcam Videos

This directory is for your dashcam video files.

## Recommended Video Sources

### Free Dashcam Footage

1. **YouTube Dashcam Channels**
   - Search for "dashcam highway" or "dashcam freeway"
   - Use a YouTube downloader to save videos
   - Recommended channels:
     - Dashcam Lessons
     - Road Incidents
     - Highway Patrol

2. **Pexels (Free Stock Videos)**
   - https://www.pexels.com/search/videos/dashcam/
   - https://www.pexels.com/search/videos/highway%20driving/
   - High quality, royalty-free

3. **Pixabay**
   - https://pixabay.com/videos/search/dashcam/
   - https://pixabay.com/videos/search/highway/

### Video Requirements

For best results, use videos with:
- **Resolution**: 720p or 1080p
- **Format**: MP4, AVI, MOV
- **Content**: Highway/freeway driving with multiple vehicles
- **Quality**: Clear visibility, good lighting
- **Duration**: 30 seconds to 5 minutes (for testing)

### Example Search Terms

- "dashcam highway traffic"
- "freeway driving pov"
- "highway dashcam footage"
- "car driving highway"

## Using Your Own Videos

If you have your own dashcam:
1. Copy videos to this directory
2. Supported formats: MP4, AVI, MOV, MKV
3. No special preprocessing needed

## File Organization

```
data/
├── highway_traffic.mp4
├── city_driving.mp4
├── freeway_rush_hour.mp4
└── test_video.mp4
```

## Testing Tips

1. **Start with short videos** (30-60 seconds) to test quickly
2. **Use highway footage** for best path detection results
3. **Avoid night videos** initially (harder to detect)
4. **Multiple lanes** work better for path planning demonstration

## Sample Command

```bash
# Run FlightPath on a video
.\build\bin\Release\FlightPath.exe data\highway_traffic.mp4

# Save output
.\build\bin\Release\FlightPath.exe data\highway_traffic.mp4 --output output\result.mp4
```

## Note

Video files are gitignored (too large for version control). Each user should download their own test footage.
