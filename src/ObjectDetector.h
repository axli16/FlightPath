#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include "Config.h"

namespace FlightPath {

/**
 * @brief Represents a detected object
 */
struct Detection {
    int classId;                // Class ID from COCO dataset
    std::string className;      // Human-readable class name
    float confidence;           // Detection confidence [0, 1]
    cv::Rect boundingBox;       // Bounding box in image coordinates
    cv::Point center;           // Center point of bounding box
};

/**
 * @brief Handles object detection using YOLO
 * 
 * Loads YOLO model and performs inference on video frames
 * to detect vehicles and obstacles.
 */
class ObjectDetector {
public:
    ObjectDetector();
    ~ObjectDetector();
    
    /**
     * @brief Load YOLO model and class names
     * @param config Model configuration
     * @return true if successful, false otherwise
     */
    bool loadModel(const ModelConfig& config);
    
    /**
     * @brief Detect objects in a frame
     * @param frame Input frame
     * @param detectionConfig Detection parameters
     * @return Vector of detected objects
     */
    std::vector<Detection> detect(const cv::Mat& frame, const DetectionConfig& detectionConfig);
    
    /**
     * @brief Check if model is loaded
     */
    bool isLoaded() const { return modelLoaded_; }
    
private:
    /**
     * @brief Load class names from file
     * @param classNamesPath Path to class names file
     * @return true if successful
     */
    bool loadClassNames(const std::string& classNamesPath);
    
    /**
     * @brief Get output layer names from the network
     * @return Vector of output layer names
     */
    std::vector<std::string> getOutputLayerNames();
    
    /**
     * @brief Post-process network outputs to extract detections
     * @param outputs Network outputs
     * @param frame Original frame (for dimensions)
     * @param config Detection configuration
     * @return Vector of detections
     */
    std::vector<Detection> postProcess(const std::vector<cv::Mat>& outputs, 
                                       const cv::Mat& frame,
                                       const DetectionConfig& config);
    
    cv::dnn::Net network_;
    std::vector<std::string> classNames_;
    bool modelLoaded_;
    
    // Cache output layer names
    std::vector<std::string> outputLayerNames_;
};

} // namespace FlightPath

#endif // OBJECT_DETECTOR_H
