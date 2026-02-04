#include "ObjectDetector.h"
#include <fstream>
#include <iostream>

namespace FlightPath {

ObjectDetector::ObjectDetector() : modelLoaded_(false) {
}

ObjectDetector::~ObjectDetector() {
}

bool ObjectDetector::loadClassNames(const std::string& classNamesPath) {
    std::ifstream file(classNamesPath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open class names file: " << classNamesPath << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        classNames_.push_back(line);
    }
    
    std::cout << "Loaded " << classNames_.size() << " class names" << std::endl;
    return true;
}

bool ObjectDetector::loadModel(const ModelConfig& config) {
    try {
        // Load class names
        if (!loadClassNames(config.classNamesPath)) {
            return false;
        }
        
        // Load YOLO network
        std::cout << "Loading YOLO model..." << std::endl;
        std::cout << "  Config: " << config.configPath << std::endl;
        std::cout << "  Weights: " << config.weightsPath << std::endl;
        
        network_ = cv::dnn::readNetFromDarknet(config.configPath, config.weightsPath);
        
        if (network_.empty()) {
            std::cerr << "Error: Failed to load YOLO network" << std::endl;
            return false;
        }
        
        // Set backend and target
        if (config.useGPU) {
            std::cout << "Using CUDA backend" << std::endl;
            network_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            network_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } else {
            std::cout << "Using CPU backend" << std::endl;
            network_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            network_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
        
        // Cache output layer names
        outputLayerNames_ = getOutputLayerNames();
        
        modelLoaded_ = true;
        std::cout << "YOLO model loaded successfully" << std::endl;
        
        return true;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception while loading model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::string> ObjectDetector::getOutputLayerNames() {
    std::vector<std::string> names;
    std::vector<int> outLayers = network_.getUnconnectedOutLayers();
    std::vector<std::string> layersNames = network_.getLayerNames();
    
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) {
        names[i] = layersNames[outLayers[i] - 1];
    }
    
    return names;
}

std::vector<Detection> ObjectDetector::detect(const cv::Mat& frame, const DetectionConfig& detectionConfig) {
    if (!modelLoaded_ || frame.empty()) {
        return std::vector<Detection>();
    }
    
    try {
        // Create blob from frame
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, 
                              cv::Size(detectionConfig.inputWidth, detectionConfig.inputHeight),
                              cv::Scalar(0, 0, 0), true, false);
        
        // Set input to network
        network_.setInput(blob);
        
        // Forward pass
        std::vector<cv::Mat> outputs;
        network_.forward(outputs, outputLayerNames_);
        
        // Post-process outputs
        return postProcess(outputs, frame, detectionConfig);
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception during detection: " << e.what() << std::endl;
        return std::vector<Detection>();
    }
}

std::vector<Detection> ObjectDetector::postProcess(const std::vector<cv::Mat>& outputs,
                                                   const cv::Mat& frame,
                                                   const DetectionConfig& config) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    // Parse YOLO outputs
    for (const auto& output : outputs) {
        for (int i = 0; i < output.rows; ++i) {
            const float* data = output.ptr<float>(i);
            
            // Get class scores (skip first 5 values: x, y, w, h, objectness)
            cv::Mat scores = output.row(i).colRange(5, output.cols);
            cv::Point classIdPoint;
            double confidence;
            
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
            
            if (confidence > config.confidenceThreshold) {
                int classId = classIdPoint.x;
                
                // Check if this class is in our target classes
                bool isTargetClass = std::find(config.targetClasses.begin(),
                                              config.targetClasses.end(),
                                              classId) != config.targetClasses.end();
                
                if (isTargetClass) {
                    // Extract bounding box
                    int centerX = static_cast<int>(data[0] * frame.cols);
                    int centerY = static_cast<int>(data[1] * frame.rows);
                    int width = static_cast<int>(data[2] * frame.cols);
                    int height = static_cast<int>(data[3] * frame.rows);
                    
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    
                    classIds.push_back(classId);
                    confidences.push_back(static_cast<float>(confidence));
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
    }
    
    // Apply Non-Maximum Suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, config.confidenceThreshold, 
                     config.nmsThreshold, indices);
    
    // Create Detection objects
    std::vector<Detection> detections;
    for (int idx : indices) {
        Detection det;
        det.classId = classIds[idx];
        det.className = classNames_[classIds[idx]];
        det.confidence = confidences[idx];
        det.boundingBox = boxes[idx];
        det.center = cv::Point(boxes[idx].x + boxes[idx].width / 2,
                              boxes[idx].y + boxes[idx].height / 2);
        
        detections.push_back(det);
    }
    
    return detections;
}

} // namespace FlightPath
