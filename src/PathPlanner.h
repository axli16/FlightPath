#ifndef PATH_PLANNER_H
#define PATH_PLANNER_H

#include "Config.h"
#include "ObjectDetector.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace FlightPath {

/**
 * @brief Plans navigable paths based on detected objects
 *
 * Analyzes the scene to find gaps between obstacles and
 * calculates the single best driving path using perspective
 * heuristics to estimate 3D road geometry from the 2D frame.
 */
class PathPlanner {
public:
  PathPlanner();
  ~PathPlanner();

  /**
   * @brief Find the single best navigable path in the scene
   * @param detections Detected objects in the frame
   * @param frameSize Size of the video frame
   * @param config Path planning configuration
   * @return Vector containing 0 or 1 path (the most confident)
   */
  std::vector<Path> findPaths(const std::vector<Detection> &detections,
                              const cv::Size &frameSize,
                              const PathConfig &config);

private:
  /**
   * @brief Create occupancy grid from detections
   */
  cv::Mat createOccupancyGrid(const std::vector<Detection> &detections,
                              const cv::Size &frameSize, int gridSize);

  /**
   * @brief Find gaps in the occupancy grid
   */
  std::vector<Path> findGaps(const cv::Mat &occupancyGrid,
                             const cv::Size &frameSize,
                             const PathConfig &config);

  /**
   * @brief Score a path based on width, distance, and alignment
   */
  float scorePath(const Path &path, const cv::Size &frameSize,
                  const PathConfig &config);

  /**
   * @brief Classify path type based on width
   */
  void classifyPath(Path &path, const cv::Size &frameSize,
                    const PathConfig &config);

  /**
   * @brief Compute perspective scale factor at a given y coordinate
   * @param y Pixel row in the frame
   * @param frameHeight Total frame height
   * @param horizonY Horizon y coordinate in pixels
   * @return Scale factor (higher = each pixel represents more real-world
   * distance)
   */
  float perspectiveScale(float y, float frameHeight, float horizonY);

  /**
   * @brief Convert pixel gap width to approximate meters using perspective
   */
  float pixelWidthToMeters(float pixelWidth, float y, const cv::Size &frameSize,
                           const PathConfig &config);

  /**
   * @brief Build a multi-waypoint path from bottom-center to target
   *        with perspective-correct tapering
   */
  Path buildPerspectivePath(cv::Point target, const cv::Size &frameSize,
                            const PathConfig &config);
};

} // namespace FlightPath

#endif // PATH_PLANNER_H
