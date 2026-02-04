#ifndef PATH_PLANNER_H
#define PATH_PLANNER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "Config.h"
#include "ObjectDetector.h"

namespace FlightPath {

/**
 * @brief Represents a navigable path
 */
struct Path {
    cv::Point start;            // Starting point of the path
    cv::Point end;              // End point of the path
    float width;                // Width of the gap (in pixels)
    float score;                // Path quality score [0, 1]
    bool isSafe;                // Is the path wide enough?
    
    // Path classification
    enum class Type {
        SAFE,                   // Wide, safe path
        TIGHT,                  // Narrow but passable
        BLOCKED                 // Too narrow or blocked
    };
    Type type;
};

/**
 * @brief Plans navigable paths based on detected objects
 * 
 * Analyzes the scene to find gaps between obstacles and
 * calculates potential driving paths.
 */
class PathPlanner {
public:
    PathPlanner();
    ~PathPlanner();
    
    /**
     * @brief Find navigable paths in the scene
     * @param detections Detected objects in the frame
     * @param frameSize Size of the video frame
     * @param config Path planning configuration
     * @return Vector of potential paths
     */
    std::vector<Path> findPaths(const std::vector<Detection>& detections,
                                const cv::Size& frameSize,
                                const PathConfig& config);
    
private:
    /**
     * @brief Create occupancy grid from detections
     * @param detections Detected objects
     * @param frameSize Frame dimensions
     * @param gridSize Grid resolution
     * @return Occupancy grid (1 = occupied, 0 = free)
     */
    cv::Mat createOccupancyGrid(const std::vector<Detection>& detections,
                                const cv::Size& frameSize,
                                int gridSize);
    
    /**
     * @brief Find gaps in the occupancy grid
     * @param occupancyGrid Grid showing occupied/free space
     * @param frameSize Original frame size
     * @param config Path configuration
     * @return Vector of potential paths
     */
    std::vector<Path> findGaps(const cv::Mat& occupancyGrid,
                              const cv::Size& frameSize,
                              const PathConfig& config);
    
    /**
     * @brief Score a path based on width, distance, and alignment
     * @param path Path to score
     * @param frameSize Frame dimensions
     * @param config Path configuration
     * @return Score [0, 1]
     */
    float scorePath(const Path& path, const cv::Size& frameSize, const PathConfig& config);
    
    /**
     * @brief Classify path type based on width
     * @param path Path to classify
     * @param config Path configuration
     */
    void classifyPath(Path& path, const PathConfig& config);
};

} // namespace FlightPath

#endif // PATH_PLANNER_H
