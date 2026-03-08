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
   * @param detectionConfig Detection config (for ROI bounds)
   * @return Vector containing 0 or 1 path (the most confident)
   */
  std::vector<Path>
  findPaths(const std::vector<Detection> &detections, const cv::Size &frameSize,
            const PathConfig &config, const DetectionConfig &detectionConfig,
            const TrapezoidROI &trapezoidROI = TrapezoidROI());

  /**
   * @brief Find paths using ML road mask + vehicle detections
   * @param detections Detected vehicles
   * @param frameSize Frame dimensions
   * @param config Path planning configuration
   * @param detectionConfig Detection config (for ROI bounds)
   * @param roadMask Binary mask of drivable road (CV_8UC1)
   * @param trapezoidROI Optional trapezoid ROI fallback
   * @return Vector containing best path
   */
  std::vector<Path>
  findPaths(const std::vector<Detection> &detections, const cv::Size &frameSize,
            const PathConfig &config, const DetectionConfig &detectionConfig,
            const cv::Mat &roadMask,
            const TrapezoidROI &trapezoidROI = TrapezoidROI());

private:
  /**
   * @brief Compute pixel ROI rectangle from normalized DetectionConfig
   */
  cv::Rect computeROI(const cv::Size &frameSize,
                      const DetectionConfig &detectionConfig);

  /**
   * @brief Create occupancy grid from detections within the ROI
   */
  cv::Mat createOccupancyGrid(const std::vector<Detection> &detections,
                              const cv::Rect &roi, int gridSize);

  /**
   * @brief Mask occupancy grid cells outside the trapezoid polygon
   */
  void maskOccupancyGrid(cv::Mat &grid, const cv::Rect &roi,
                         const TrapezoidROI &trapezoid);

  /**
   * @brief Find gaps in the occupancy grid, preferring paths near center
   */
  std::vector<Path> findGaps(const cv::Mat &occupancyGrid, const cv::Rect &roi,
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
   * @brief Build a multi-waypoint path from ROI bottom-center to target
   *        with perspective-correct tapering, clamped inside ROI
   */
  Path buildPerspectivePath(cv::Point target, const cv::Rect &roi,
                            const cv::Size &frameSize,
                            const PathConfig &config);

  /**
   * @brief Create occupancy grid from detections + road mask
   */
  cv::Mat createOccupancyGridWithRoad(const std::vector<Detection> &detections,
                                      const cv::Mat &roadMask,
                                      const cv::Rect &roi, int gridSize);

  /**
   * @brief Smooth path waypoints between frames using EMA
   */
  void smoothPath(Path &path);

  // Previous frame's best path for temporal smoothing
  Path prevPath_;
  bool hasPrevPath_ = false;
  float pathSmoothingAlpha_ = 0.5f;
};

} // namespace FlightPath

#endif // PATH_PLANNER_H
