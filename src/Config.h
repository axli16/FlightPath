#ifndef CONFIG_H
#define CONFIG_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace FlightPath {

/**
 * @brief Represents a detected object
 */
struct Detection {
  int classId;           // Class ID from COCO dataset
  std::string className; // Human-readable class name
  float confidence;      // Detection confidence [0, 1]
  cv::Rect boundingBox;  // Bounding box in image coordinates
  cv::Point center;      // Center point of bounding box
};

/**
 * @brief Represents a navigable path
 */
struct Path {
  cv::Point start; // Starting point of the path
  cv::Point end;   // End point of the path
  float width;     // Width of the gap (in pixels)
  float score;     // Path quality score [0, 1]
  bool isSafe;     // Is the path wide enough?

  // Multi-point path for perspective-correct drawing
  std::vector<cv::Point> waypoints;

  // Path classification
  enum class Type {
    SAFE,   // Wide, safe path
    TIGHT,  // Narrow but passable
    BLOCKED // Too narrow or blocked
  };
  Type type;
};

// Detection Configuration
struct DetectionConfig {
  float confidenceThreshold = 0.5f; // Minimum confidence for detections
  float nmsThreshold = 0.4f;        // Non-maximum suppression threshold
  int inputWidth = 256;  // YOLO input width (reduced for performance)
  int inputHeight = 256; // YOLO input height (reduced for performance)
  int frameSkip =
      5; // Process every Nth frame (1 = all frames, 5 = every 5th frame)
  bool usingCuda = false;
  int batchSize =
      4; // Number of frames to process in a batch (for CUDA optimization)

  // Region of Interest (ROI) - only process this region of the frame
  // Values are in normalized coordinates [0.0, 1.0] relative to frame size
  // Set all to 0.0 to disable ROI (process entire frame)
  bool useROI = true;     // Enable/disable ROI processing
  float roiX = 0.3f;      // ROI left edge (0.0 = left side of frame)
  float roiY = 0.0f;      // ROI top edge (0.3 = 30% down from top)
  float roiWidth = 0.5f;  // ROI width (1.0 = full width)
  float roiHeight = 0.7f; // ROI height (0.7 = 70% of frame height)

  // Classes to detect (COCO dataset indices)
  // 2: car, 3: motorcycle, 5: bus, 7: truck, 0: person
  std::vector<int> targetClasses = {0, 2, 3, 5, 7};
};

// Path Planning Configuration
struct PathConfig {
  float vehicleWidth = 2.0f;     // Vehicle width in meters
  float safetyMargin = 0.5f;     // Additional clearance in meters
  float minGapWidth = 2.5f;      // Minimum gap to consider (meters)
  int gridResolution = 50;       // Grid cells for path planning
  float maxPathDistance = 50.0f; // Maximum path distance in meters

  // Perspective model (no camera calibration needed)
  float horizonRatio =
      0.35f; // Vanishing point as fraction of frame height from top
  float laneWidthAtBottom =
      0.45f; // Lane width as fraction of frame width at bottom edge
};

// Visualization Configuration
struct VisualConfig {
  // Colors (BGR format for OpenCV)
  cv::Scalar colorDetection = cv::Scalar(0, 0, 255);   // Red for detections
  cv::Scalar colorPathSafe = cv::Scalar(0, 255, 0);    // Green for safe paths
  cv::Scalar colorPathTight = cv::Scalar(0, 255, 255); // Yellow for tight paths
  cv::Scalar colorPathBlocked = cv::Scalar(0, 0, 255); // Red for blocked
  cv::Scalar colorGrid = cv::Scalar(100, 100, 100);    // Gray for grid

  // Drawing parameters
  int boxThickness = 2;
  int arrowThickness = 3;
  float arrowTipLength = 0.2f;
  int fontSize = 1;
  int fontThickness = 2;

  // Display options
  bool showBoundingBoxes = true;
  bool showConfidence = true;
  bool showGrid = false; // Debug: show planning grid
  bool showFPS = false;
  bool showROI = true; // Show ROI rectangle
};

// Video Processing Configuration
struct VideoConfig {
  std::string inputPath;
  std::string outputPath = ""; // Empty = no output file
  bool displayWindow = true;
  int displayWidth = 1280;
  int displayHeight = 720;
  bool saveOutput = false;

  // Auto-crop settings
  bool autoCrop = false;    // Enable auto-cropping for large frames
  int maxCropWidth = 1920;  // Maximum width before cropping
  int maxCropHeight = 1080; // Maximum height before cropping

  // Auto-scale resolution
  bool autoScale = true;
  int targetWidth = 1920;
  int targetHeight = 1080;
};

// Model Configuration
struct ModelConfig {
  std::string weightsPath = "models/yolov4-tiny.weights";
  std::string configPath = "models/yolov4-tiny.cfg";
  std::string classNamesPath = "models/coco.names";
  bool useGPU = false; // Set to true if OpenCV built with CUDA
};

// Main application configuration
struct AppConfig {
  DetectionConfig detection;
  PathConfig path;
  VisualConfig visual;
  VideoConfig video;
  ModelConfig model;

  // Runtime state
  bool isPaused = false;
  bool shouldExit = false;
};

struct preProcessFrameData {
  cv::Mat frame;
  int frameNumber;
};
struct FrameData {
  cv::Mat frame;
  std::vector<Detection> detections;
  std::vector<Path> paths;
  int frameNumber;
};

} // namespace FlightPath

#endif // CONFIG_H
