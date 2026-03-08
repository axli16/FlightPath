#ifndef ROAD_DETECTOR_H
#define ROAD_DETECTOR_H

#include "Config.h"
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace FlightPath {

/**
 * @brief ML-based road/lane detector using UFLDv2 ONNX model
 *
 * Runs the Ultra-Fast-Lane-Detection-v2 model to detect lane lines,
 * then derives a drivable road polygon and binary road mask.
 * Outputs: per-lane polylines + binary road mask.
 */
class RoadDetector {
public:
  RoadDetector();
  ~RoadDetector();

  /**
   * @brief Load the UFLDv2 ONNX model
   * @param config Road detection configuration
   * @return true if model loaded successfully
   */
  bool loadModel(const RoadConfig &config);

  /**
   * @brief Detect road lanes and produce a drivable-area mask
   * @param frame Input BGR frame (original resolution)
   * @param config Road detection parameters
   * @param outLaneLines Output: detected lane polylines (in frame coords)
   * @return Binary mask (CV_8UC1) of drivable road area at frame resolution
   */
  cv::Mat detectRoad(const cv::Mat &frame, const RoadConfig &config,
                     std::vector<std::vector<cv::Point>> &outLaneLines);

  bool isLoaded() const { return modelLoaded_; }

private:
  /**
   * @brief Post-process UFLDv2 outputs into lane point coordinates
   *
   * Decodes loc_row[1,200,72,4] + exist_row[1,2,72,4] +
   *         loc_col[1,100,81,4] + exist_col[1,2,81,4]
   * into per-lane polylines in the original image coordinate space.
   */
  std::vector<std::vector<cv::Point>>
  decodeOutputs(const std::vector<cv::Mat> &outputs, int origWidth,
                int origHeight, const RoadConfig &config);

  /**
   * @brief Build a binary road mask from the outermost detected lanes
   */
  cv::Mat buildRoadMask(const std::vector<std::vector<cv::Point>> &lanes,
                        int width, int height);

  cv::dnn::Net network_;
  bool modelLoaded_ = false;

  // CULane row anchor positions (normalized to [0,1] range of crop height)
  std::vector<float> rowAnchors_;

  // Temporal smoothing
  std::vector<std::vector<cv::Point>> prevLanes_;
  bool hasPrevLanes_ = false;
  float smoothingAlpha_ = 0.4f;
};

} // namespace FlightPath

#endif // ROAD_DETECTOR_H
