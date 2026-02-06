#ifndef CONFIG_H
#define CONFIG_H

#include "ObjectDetector.h"
#include "PathPlanner.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


namespace FlightPath {

// Detection Configuration
struct DetectionConfig {
    float confidenceThreshold = 0.5f;    // Minimum confidence for detections
    float nmsThreshold = 0.4f;           // Non-maximum suppression threshold
    int inputWidth = 416;                // YOLO input width
    int inputHeight = 416;               // YOLO input height
    
    // Classes to detect (COCO dataset indices)
    // 2: car, 3: motorcycle, 5: bus, 7: truck, 0: person
    std::vector<int> targetClasses = {0, 2, 3, 5, 7};
};

// Path Planning Configuration
struct PathConfig {
    float vehicleWidth = 2.0f;           // Vehicle width in meters
    float safetyMargin = 0.5f;           // Additional clearance in meters
    float minGapWidth = 2.5f;            // Minimum gap to consider (meters)
    int gridResolution = 50;             // Grid cells for path planning
    float maxPathDistance = 50.0f;       // Maximum path distance in meters
    
    // Pixel to meter conversion (approximate, depends on camera)
    float pixelsPerMeter = 20.0f;        // Calibrate based on your camera
};

// Visualization Configuration
struct VisualConfig {
    // Colors (BGR format for OpenCV)
    cv::Scalar colorDetection = cv::Scalar(0, 0, 255);      // Red for detections
    cv::Scalar colorPathSafe = cv::Scalar(0, 255, 0);       // Green for safe paths
    cv::Scalar colorPathTight = cv::Scalar(0, 255, 255);    // Yellow for tight paths
    cv::Scalar colorPathBlocked = cv::Scalar(0, 0, 255);    // Red for blocked
    cv::Scalar colorGrid = cv::Scalar(100, 100, 100);       // Gray for grid
    
    // Drawing parameters
    int boxThickness = 2;
    int arrowThickness = 3;
    float arrowTipLength = 0.2f;
    int fontSize = 1;
    int fontThickness = 2;
    
    // Display options
    bool showBoundingBoxes = true;
    bool showConfidence = true;
    bool showGrid = false;               // Debug: show planning grid
    bool showFPS = false;
};

// Video Processing Configuration
struct VideoConfig {
    std::string inputPath;
    std::string outputPath = "";         // Empty = no output file
    bool displayWindow = true;
    int displayWidth = 1280;
    int displayHeight = 720;
    bool saveOutput = false;
};

// Model Configuration
struct ModelConfig {
    std::string weightsPath = "models/yolov4.weights";
    std::string configPath = "models/yolov4.cfg";
    std::string classNamesPath = "models/coco.names";
    bool useGPU = false;                 // Set to true if OpenCV built with CUDA
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

struct FrameData {
    cv::Mat frame;
    std::vector<Detection> detections;
    std::vector<Path> paths;
};

} // namespace FlightPath

#endif // CONFIG_H
