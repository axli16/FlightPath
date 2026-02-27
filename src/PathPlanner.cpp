#include "PathPlanner.h"
#include <algorithm>
#include <iostream>

namespace FlightPath {

PathPlanner::PathPlanner() {
}

PathPlanner::~PathPlanner() {
}

const cv::Mat& PathPlanner::createOccupancyGrid(const std::vector<Detection>& detections,
                                         const cv::Size& frameSize,
                                         int gridSize) {
    // Optimization: Reuse buffer to avoid reallocation
    if (occupancyGrid_.size() != cv::Size(gridSize, gridSize) || occupancyGrid_.type() != CV_8UC1) {
        occupancyGrid_.create(gridSize, gridSize, CV_8UC1);
    }
    occupancyGrid_.setTo(0); // Reset grid
    
    if (detections.empty()) {
        return occupancyGrid_;
    }
    
    float cellWidth = static_cast<float>(frameSize.width) / gridSize;
    float cellHeight = static_cast<float>(frameSize.height) / gridSize;
    
    // Mark occupied cells based on detections
    for (const auto& det : detections) {
        // Convert bounding box to grid coordinates
        int gridLeft = static_cast<int>(det.boundingBox.x / cellWidth);
        int gridTop = static_cast<int>(det.boundingBox.y / cellHeight);
        int gridRight = static_cast<int>((det.boundingBox.x + det.boundingBox.width) / cellWidth);
        int gridBottom = static_cast<int>((det.boundingBox.y + det.boundingBox.height) / cellHeight);
        
        // Clamp to grid bounds
        gridLeft = std::max(0, std::min(gridLeft, gridSize - 1));
        gridTop = std::max(0, std::min(gridTop, gridSize - 1));
        gridRight = std::max(0, std::min(gridRight, gridSize - 1));
        gridBottom = std::max(0, std::min(gridBottom, gridSize - 1));
        
        // Mark cells as occupied using optimized OpenCV primitive
        cv::rectangle(occupancyGrid_, cv::Point(gridLeft, gridTop), cv::Point(gridRight + 1, gridBottom + 1),
                      cv::Scalar(255), cv::FILLED);
    }
    
    return occupancyGrid_;
}

std::vector<Path> PathPlanner::findGaps(const cv::Mat& occupancyGrid,
                                        const cv::Size& frameSize,
                                        const PathConfig& config) {
    std::vector<Path> paths;
    
    int gridSize = occupancyGrid.rows;
    float cellWidth = static_cast<float>(frameSize.width) / gridSize;
    float cellHeight = static_cast<float>(frameSize.height) / gridSize;
    
    // Scan horizontal slices at different depths
    // Focus on the middle-to-far region (40% to 80% of frame height)
    int startRow = static_cast<int>(gridSize * 0.4);
    int endRow = static_cast<int>(gridSize * 0.8);
    int rowStep = std::max(1, gridSize / 10);
    
    for (int row = startRow; row < endRow; row += rowStep) {
        // Find gaps in this row
        bool inGap = false;
        int gapStart = 0;
        
        // Optimization: Get row pointer to avoid repeated .at() calls
        const uchar* rowPtr = occupancyGrid.ptr<uchar>(row);

        for (int col = 0; col < gridSize; ++col) {
            bool isFree = (rowPtr[col] == 0);
            
            if (isFree && !inGap) {
                // Start of a gap
                inGap = true;
                gapStart = col;
            } else if (!isFree && inGap) {
                // End of a gap
                inGap = false;
                int gapEnd = col - 1;
                int gapWidth = gapEnd - gapStart + 1;
                
                // Convert to pixel coordinates
                float gapWidthPixels = gapWidth * cellWidth;
                float gapWidthMeters = gapWidthPixels / config.pixelsPerMeter;
                
                // Check if gap is wide enough to consider
                if (gapWidthMeters >= config.minGapWidth) {
                    Path path;
                    
                    // Start point at bottom center of frame
                    path.start = cv::Point(frameSize.width / 2, frameSize.height - 10);
                    
                    // End point at center of gap
                    int gapCenterCol = (gapStart + gapEnd) / 2;
                    path.end = cv::Point(
                        static_cast<int>(gapCenterCol * cellWidth + cellWidth / 2),
                        static_cast<int>(row * cellHeight + cellHeight / 2)
                    );
                    
                    path.width = gapWidthPixels;
                    
                    paths.push_back(path);
                }
            }
        }
        
        // Handle gap extending to edge
        if (inGap) {
            int gapEnd = gridSize - 1;
            int gapWidth = gapEnd - gapStart + 1;
            float gapWidthPixels = gapWidth * cellWidth;
            float gapWidthMeters = gapWidthPixels / config.pixelsPerMeter;
            
            if (gapWidthMeters >= config.minGapWidth) {
                Path path;
                path.start = cv::Point(frameSize.width / 2, frameSize.height - 10);
                
                int gapCenterCol = (gapStart + gapEnd) / 2;
                path.end = cv::Point(
                    static_cast<int>(gapCenterCol * cellWidth + cellWidth / 2),
                    static_cast<int>(row * cellHeight + cellHeight / 2)
                );
                
                path.width = gapWidthPixels;
                paths.push_back(path);
            }
        }
    }
    
    return paths;
}

float PathPlanner::scorePath(const Path& path, const cv::Size& frameSize, const PathConfig& config) {
    float score = 0.0f;
    
    // Width score (wider is better)
    float widthMeters = path.width / config.pixelsPerMeter;
    float minWidth = config.vehicleWidth + config.safetyMargin;
    float widthScore = std::min(1.0f, widthMeters / (minWidth * 2.0f));
    
    // Distance score (closer paths preferred for immediate decisions)
    float distance = cv::norm(path.end - path.start);
    float maxDistance = config.maxPathDistance * config.pixelsPerMeter;
    float distanceScore = 1.0f - std::min(1.0f, distance / maxDistance);
    
    // Alignment score (prefer paths closer to center)
    float centerX = frameSize.width / 2.0f;
    float deviation = std::abs(path.end.x - centerX);
    float maxDeviation = frameSize.width / 2.0f;
    float alignmentScore = 1.0f - (deviation / maxDeviation);
    
    // Weighted combination
    score = widthScore * 0.5f + distanceScore * 0.2f + alignmentScore * 0.3f;
    
    return score;
}

void PathPlanner::classifyPath(Path& path, const PathConfig& config) {
    float widthMeters = path.width / config.pixelsPerMeter;
    float minWidth = config.vehicleWidth + config.safetyMargin;
    float comfortableWidth = minWidth * 1.5f;
    
    if (widthMeters >= comfortableWidth) {
        path.type = Path::Type::SAFE;
        path.isSafe = true;
    } else if (widthMeters >= minWidth) {
        path.type = Path::Type::TIGHT;
        path.isSafe = true;
    } else {
        path.type = Path::Type::BLOCKED;
        path.isSafe = false;
    }
}

std::vector<Path> PathPlanner::findPaths(const std::vector<Detection>& detections,
                                         const cv::Size& frameSize,
                                         const PathConfig& config) {
    // Create occupancy grid (returns reference to member buffer)
    const cv::Mat& occupancyGrid = createOccupancyGrid(detections, frameSize, config.gridResolution);
    
    // Find potential paths
    std::vector<Path> paths = findGaps(occupancyGrid, frameSize, config);
    
    // Score and classify each path
    for (auto& path : paths) {
        path.score = scorePath(path, frameSize, config);
        classifyPath(path, config);
    }
    
    // Sort by score (highest first)
    std::sort(paths.begin(), paths.end(), 
              [](const Path& a, const Path& b) { return a.score > b.score; });
    
    // Keep only top paths (limit to 5 for clarity)
    if (paths.size() > 5) {
        paths.resize(5);
    }
    
    return paths;
}

} // namespace FlightPath
