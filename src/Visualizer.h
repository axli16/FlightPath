#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "Config.h"
#include "ObjectDetector.h"
#include "PathPlanner.h"
#include <opencv2/opencv.hpp>
#include <vector>


namespace FlightPath {

/**
 * @brief Handles visualization of detections and paths
 *
 * Draws bounding boxes, labels, arrows, and overlays
 * on video frames to show detection and path planning results.
 */
class Visualizer {
public:
  Visualizer();
  ~Visualizer();

  /**
   * @brief Draw all visualizations on a frame
   * @param frame Frame to draw on (modified in place)
   * @param detections Detected objects
   * @param paths Planned paths
   * @param config Visualization configuration
   */
  void draw(cv::Mat &frame, const std::vector<Detection> &detections,
            const std::vector<Path> &paths, const VisualConfig &config);

private:
  /**
   * @brief Draw bounding boxes for detections
   */
  void drawDetections(cv::Mat &frame, const std::vector<Detection> &detections,
                      const VisualConfig &config);

  /**
   * @brief Draw path arrows
   */
  void drawPaths(cv::Mat &frame, const std::vector<Path> &paths,
                 const VisualConfig &config);

  /**
   * @brief Draw info panel with stats
   */
  void drawInfoPanel(cv::Mat &frame, const std::vector<Detection> &detections,
                     const std::vector<Path> &paths,
                     const VisualConfig &config);

  /**
   * @brief Get color for path based on type
   */
  cv::Scalar getPathColor(const Path &path, const VisualConfig &config);
};

} // namespace FlightPath

#endif // VISUALIZER_H
