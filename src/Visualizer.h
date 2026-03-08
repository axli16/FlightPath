#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "Config.h"
#include "ObjectDetector.h"
#include "PathPlanner.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace FlightPath {

/**
 * @brief Iron Man HUD-style visualization of detections and paths
 *
 * Renders futuristic holographic overlays: animated chevron arrows,
 * neon glow path boundaries, scanline grid, directional arc,
 * and vehicle detection boxes.
 */
class Visualizer {
public:
  Visualizer();
  ~Visualizer();

  /**
   * @brief Draw all visualizations on a frame
   */
  void draw(cv::Mat &frame, const std::vector<Detection> &detections,
            const std::vector<Path> &paths, const VisualConfig &visualConfig,
            const DetectionConfig &detectionConfig,
            const TrapezoidROI &trapezoidROI = TrapezoidROI());

private:
  // --- Detection rendering ---
  void drawDetections(cv::Mat &frame, const std::vector<Detection> &detections,
                      const VisualConfig &config);

  // --- HUD Path rendering ---
  /**
   * @brief Main HUD path renderer (replaces old drawPaths)
   * Draws the full Iron Man-style navigation overlay
   */
  void drawHUDPath(cv::Mat &frame, const std::vector<Path> &paths,
                   const VisualConfig &config);

  /**
   * @brief Draw animated V-shaped chevrons along the path
   */
  void drawChevrons(cv::Mat &frame, const std::vector<cv::Point> &waypoints,
                    const VisualConfig &config);

  /**
   * @brief Draw neon glow effect along path edges
   */
  void drawGlowEffect(cv::Mat &frame, const std::vector<cv::Point> &polygon,
                      const cv::Scalar &color, const VisualConfig &config);

  /**
   * @brief Draw holographic scanline grid across the path polygon
   */
  void drawScanlines(cv::Mat &frame, const std::vector<cv::Point> &polygon,
                     const VisualConfig &config);

  /**
   * @brief Draw semicircular directional arc at bottom of screen
   */
  void drawDirectionArc(cv::Mat &frame, const std::vector<Path> &paths,
                        const VisualConfig &config);

  /**
   * @brief Draw modernized HUD info panel
   */
  void drawHUDInfoPanel(cv::Mat &frame,
                        const std::vector<Detection> &detections,
                        const std::vector<Path> &paths,
                        const VisualConfig &config);

  /**
   * @brief Draw ROI overlay
   */
  void drawROI(cv::Mat &frame, const DetectionConfig &detectionConfig,
               const VisualConfig &visualConfig,
               const TrapezoidROI &trapezoidROI = TrapezoidROI());

  // --- Utility ---
  cv::Scalar getPathColor(const Path &path, const VisualConfig &config);

  // Animation frame counter (increments each draw call)
  int frameCounter_ = 0;
};

} // namespace FlightPath

#endif // VISUALIZER_H
