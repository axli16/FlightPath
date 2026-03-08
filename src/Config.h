#ifndef CONFIG_H
#define CONFIG_H

#include <array>
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

/**
 * @brief Trapezoidal ROI defined by 4 vertices
 * Order: bottom-left, top-left, top-right, bottom-right
 */
struct TrapezoidROI {
  std::array<cv::Point, 4> vertices; // BL, TL, TR, BR
  bool valid = false;                // False = no lanes detected, use fallback

  /// Bounding rectangle that encloses the trapezoid
  cv::Rect boundingRect() const {
    if (!valid)
      return cv::Rect();
    std::vector<cv::Point> pts(vertices.begin(), vertices.end());
    return cv::boundingRect(pts);
  }

  /// Get vertices as a vector (convenient for OpenCV polygon ops)
  std::vector<cv::Point> asVector() const {
    return std::vector<cv::Point>(vertices.begin(), vertices.end());
  }
};

// Detection Configuration
struct DetectionConfig {
  float confidenceThreshold = 0.5f; // Minimum confidence for detections
  float nmsThreshold = 0.4f;        // Non-maximum suppression threshold
  int inputWidth = 256;  // YOLO input width (reduced for performance)
  int inputHeight = 256; // YOLO input height (reduced for performance)
  int frameSkip =
      2; // Process every Nth frame (1 = all frames, 5 = every 5th frame)
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

// Lane Detection Configuration
// Tuned for curb/barrier detection (structural edges, not painted lines)
struct LaneConfig {
  bool enabled = true; // Enable dynamic lane-based ROI

  // --- Color filtering (HSV) ---
  bool useColorFilter = false; // Disabled by default for curb/barrier detection
  // Yellow line range
  cv::Scalar yellowLow = cv::Scalar(15, 80, 120);
  cv::Scalar yellowHigh = cv::Scalar(35, 255, 255);
  // White line range
  cv::Scalar whiteLow = cv::Scalar(0, 0, 200);
  cv::Scalar whiteHigh = cv::Scalar(180, 30, 255);

  // --- Edge detection ---
  int gaussianKernel = 5;   // Gaussian blur kernel size (must be odd)
  double cannyLow = 40.0;   // Canny low threshold (lower catches curb edges)
  double cannyHigh = 120.0; // Canny high threshold

  // --- Hough transform ---
  double houghRho = 1.0; // Distance resolution in pixels
  double houghTheta = 3.14159265358979323846 / 180.0; // ~1 degree in radians
  int houghThreshold = 30;                            // Accumulator threshold
  double houghMinLineLen =
      40.0; // Min line length (lower to catch shorter curb edges)
  double houghMaxLineGap =
      15.0; // Max gap to bridge (higher for barrier gaps/posts)

  // --- Line classification ---
  float minSlopeThreshold = 0.3f; // Accept more angles for barriers
  float roiTopRatio = 0.40f;      // Top of search region (fraction from top)
  float roiBottomRatio = 0.95f;   // Bottom of search region
  float minSolidCoverage =
      0.35f; // Lower for barriers (gaps between guardrail posts)

  // --- Temporal smoothing ---
  float smoothingAlpha = 0.3f; // EMA alpha (0 = no update, 1 = no smoothing)
};

// Road Detection Configuration (UFLDv2 ONNX)
struct RoadConfig {
  std::string modelPath = "models/ufldv2_culane_res34_320x1600.onnx";
  bool enabled = true;
  bool useGPU = false;

  // UFLDv2 CULane model parameters
  int inputHeight = 320;
  int inputWidth = 1600;
  int numLanes = 4; // CULane: 4 lanes max
  int numRows = 72; // CULane row anchors count
  int numCols = 81; // CULane column classification bins
  float confThreshold = 0.5f;

  // CULane row anchors (y-positions in 320-px height image)
  std::vector<int> rowAnchors = {
      121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241, 251,
      261, 271, 281, 291, 301, 311, 241, 251, 261, 271, 281, 291, 301, 311,
      // Filled to 72 below in constructor — these are the standard CULane
      // anchors
  };
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
  bool showROI = false; // Show ROI rectangle

  // --- HUD / Iron Man style settings ---
  // Primary HUD color (cyan)
  cv::Scalar hudColorPrimary = cv::Scalar(255, 255, 0);   // Cyan in BGR
  cv::Scalar hudColorSecondary = cv::Scalar(255, 102, 0); // Electric blue BGR
  cv::Scalar hudColorWarning = cv::Scalar(0, 170, 255);   // Amber BGR
  cv::Scalar hudColorDanger = cv::Scalar(0, 0, 255);      // Red BGR

  float hudGlowIntensity = 0.6f;   // Glow alpha multiplier
  int hudChevronCount = 8;         // Number of chevron arrows on path
  float hudAnimationSpeed = 2.0f;  // Chevron scroll speed (chevrons/sec)
  float hudPathAlpha = 0.25f;      // Path polygon fill opacity
  int hudScanlineSpacing = 12;     // Pixels between scanlines
  bool hudShowDirectionArc = true; // Show directional arc at bottom
  bool hudShowScanlines = true;    // Holographic scanline effect
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
  std::string weightsPath = "models/yolov4.weights";
  std::string configPath = "models/yolov4.cfg";
  std::string classNamesPath = "models/coco.names";
  bool useGPU = false; // Set to true if OpenCV built with CUDA
};

// Main application configuration
struct AppConfig {
  DetectionConfig detection;
  PathConfig path;
  LaneConfig lane;
  RoadConfig road;
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
  TrapezoidROI trapezoidROI;
  cv::Mat roadMask; // Binary mask of drivable road area
  std::vector<std::vector<cv::Point>> laneLines; // Detected lane polylines
  int frameNumber;
};

} // namespace FlightPath

#endif // CONFIG_H
