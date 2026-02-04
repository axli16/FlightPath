#include "Visualizer.h"
#include <sstream>
#include <iomanip>

namespace FlightPath {

Visualizer::Visualizer() {
}

Visualizer::~Visualizer() {
}

cv::Scalar Visualizer::getPathColor(const Path& path, const VisualConfig& config) {
    switch (path.type) {
        case Path::Type::SAFE:
            return config.colorPathSafe;
        case Path::Type::TIGHT:
            return config.colorPathTight;
        case Path::Type::BLOCKED:
            return config.colorPathBlocked;
        default:
            return config.colorPathSafe;
    }
}

void Visualizer::drawDetections(cv::Mat& frame,
                                const std::vector<Detection>& detections,
                                const VisualConfig& config) {
    if (!config.showBoundingBoxes) {
        return;
    }
    
    for (const auto& det : detections) {
        // Draw bounding box
        cv::rectangle(frame, det.boundingBox, config.colorDetection, config.boxThickness);
        
        // Prepare label
        std::stringstream label;
        label << det.className;
        
        if (config.showConfidence) {
            label << " " << std::fixed << std::setprecision(2) << (det.confidence * 100) << "%";
        }
        
        // Draw label background
        int baseline = 0;
        cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 
                                             0.5, config.fontThickness, &baseline);
        
        cv::Point labelPos(det.boundingBox.x, det.boundingBox.y - 10);
        if (labelPos.y < 0) labelPos.y = det.boundingBox.y + labelSize.height + 10;
        
        cv::rectangle(frame,
                     cv::Point(labelPos.x, labelPos.y - labelSize.height - 5),
                     cv::Point(labelPos.x + labelSize.width, labelPos.y + baseline),
                     config.colorDetection,
                     cv::FILLED);
        
        // Draw label text
        cv::putText(frame, label.str(), labelPos,
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255),
                   config.fontThickness);
    }
}

void Visualizer::drawPaths(cv::Mat& frame,
                           const std::vector<Path>& paths,
                           const VisualConfig& config) {
    // Draw paths from lowest score to highest (so best path is on top)
    for (auto it = paths.rbegin(); it != paths.rend(); ++it) {
        const Path& path = *it;
        
        // Only draw safe and tight paths
        if (path.type == Path::Type::BLOCKED) {
            continue;
        }
        
        cv::Scalar color = getPathColor(path, config);
        
        // Draw arrow from start to end
        cv::arrowedLine(frame, path.start, path.end, color,
                       config.arrowThickness, cv::LINE_AA, 0, config.arrowTipLength);
        
        // Draw circle at end point
        cv::circle(frame, path.end, 8, color, -1);
        
        // Draw path score
        std::stringstream scoreText;
        scoreText << std::fixed << std::setprecision(2) << (path.score * 100) << "%";
        
        cv::Point textPos(path.end.x + 15, path.end.y - 5);
        cv::putText(frame, scoreText.str(), textPos,
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
}

void Visualizer::drawInfoPanel(cv::Mat& frame,
                               const std::vector<Detection>& detections,
                               const std::vector<Path>& paths,
                               const VisualConfig& config,
                               double fps) {
    if (!config.showFPS && detections.empty() && paths.empty()) {
        return;
    }
    
    // Create semi-transparent panel
    int panelHeight = 120;
    int panelWidth = 250;
    cv::Mat panel(panelHeight, panelWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    
    int y = 25;
    int lineSpacing = 25;
    
    // FPS
    if (config.showFPS && fps > 0) {
        std::stringstream fpsText;
        fpsText << "FPS: " << std::fixed << std::setprecision(1) << fps;
        cv::putText(panel, fpsText.str(), cv::Point(10, y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        y += lineSpacing;
    }
    
    // Detection count
    std::stringstream detText;
    detText << "Detections: " << detections.size();
    cv::putText(panel, detText.str(), cv::Point(10, y),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    y += lineSpacing;
    
    // Path count
    int safePaths = 0;
    for (const auto& path : paths) {
        if (path.isSafe) safePaths++;
    }
    
    std::stringstream pathText;
    pathText << "Safe Paths: " << safePaths;
    cv::putText(panel, pathText.str(), cv::Point(10, y),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    y += lineSpacing;
    
    // Best path score
    if (!paths.empty() && paths[0].isSafe) {
        std::stringstream scoreText;
        scoreText << "Best Score: " << std::fixed << std::setprecision(1) 
                 << (paths[0].score * 100) << "%";
        cv::putText(panel, scoreText.str(), cv::Point(10, y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }
    
    // Blend panel onto frame
    cv::Mat roi = frame(cv::Rect(10, 10, panelWidth, panelHeight));
    cv::addWeighted(roi, 0.3, panel, 0.7, 0, roi);
}

void Visualizer::draw(cv::Mat& frame,
                     const std::vector<Detection>& detections,
                     const std::vector<Path>& paths,
                     const VisualConfig& config,
                     double fps) {
    if (frame.empty()) {
        return;
    }
    
    // Draw in order: detections, paths, info panel
    drawDetections(frame, detections, config);
    drawPaths(frame, paths, config);
    drawInfoPanel(frame, detections, paths, config, fps);
}

} // namespace FlightPath
