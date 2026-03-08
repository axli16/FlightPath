#ifndef LANE_DETECTOR_H
#define LANE_DETECTOR_H

#include "Config.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace FlightPath {

/**
 * @brief Detects solid lane lines and builds a trapezoidal ROI
 *
 * Uses classical CV: HSV color filtering → Canny → HoughLinesP
 * to find yellow and white solid lines, then extrapolates them
 * into a trapezoidal region of interest that narrows toward
 * the vanishing point.
 */
class LaneDetector {
public:
  LaneDetector();
  ~LaneDetector();

  /**
   * @brief Detect lane lines and produce a trapezoidal ROI
   * @param frame Input BGR frame
   * @param config Lane detection parameters
   * @return TrapezoidROI with valid==true if two lines found
   */
  TrapezoidROI detectLanes(const cv::Mat &frame, const LaneConfig &config);

private:
  /**
   * @brief Create a combined color mask for yellow and white lines
   */
  cv::Mat createColorMask(const cv::Mat &hsvFrame, const LaneConfig &config);

  /**
   * @brief Classify line segments into left and right groups by slope
   * @param lines HoughLinesP output
   * @param frameMidX Horizontal midpoint of the frame
   * @param config Lane config for slope thresholds
   * @param leftLines Output left-side line segments
   * @param rightLines Output right-side line segments
   */
  void classifyLines(const std::vector<cv::Vec4i> &lines, int frameMidX,
                     const LaneConfig &config,
                     std::vector<cv::Vec4i> &leftLines,
                     std::vector<cv::Vec4i> &rightLines);

  /**
   * @brief Fit a group of line segments to a single averaged line
   * @param lines Group of line segments
   * @param yBottom Bottom Y coordinate to extrapolate to
   * @param yTop Top Y coordinate to extrapolate to
   * @param x1 Output: X at yBottom
   * @param x2 Output: X at yTop
   * @return true if fitting succeeded
   */
  bool fitLine(const std::vector<cv::Vec4i> &lines, int yBottom, int yTop,
               int &x1, int &x2);

  /**
   * @brief Check if a group of segments forms a solid (continuous) line
   * @param segments Line segments to evaluate
   * @param minCoverage Minimum fraction of Y-range covered to be "solid"
   * @return true if coverage >= minCoverage (i.e. solid line, not dashed)
   */
  bool isSolidLine(const std::vector<cv::Vec4i> &segments, float minCoverage);

  /// Temporally-smoothed ROI from previous frames
  TrapezoidROI smoothedROI_;
  bool hasSmoothedROI_ = false;

  /// Apply exponential moving average to stabilize the ROI
  TrapezoidROI smoothROI(const TrapezoidROI &current, float alpha);

  /// Build a default converging perspective trapezoid (straight road)
  TrapezoidROI buildDefaultROI(int frameW, int frameH);
};

} // namespace FlightPath

#endif // LANE_DETECTOR_H
