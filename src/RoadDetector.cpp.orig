#include "RoadDetector.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace FlightPath {

// CULane row anchors: 18 predefined Y-positions in the 320-pixel-tall crop
// These are the standard CULane anchors from the UFLDv2 paper.
// Normalized later to [0,1] for scaling to any frame size.
static const int CULANE_ROW_ANCHORS[] = {
    121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241, 251, 261,
    271, 281, 291, 301, 311, 241, 251, 261, 271, 281, 291, 301, 311, 121, 131,
    141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241, 251, 261, 271, 281,
    291, 301, 311, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231,
    241, 251, 261, 271, 281, 291, 301, 311, 121, 131, 141, 151};
static const int NUM_CULANE_ANCHORS = 72;

RoadDetector::RoadDetector() {}

RoadDetector::~RoadDetector() {}

bool RoadDetector::loadModel(const RoadConfig &config) {
  try {
    std::cout << "Loading UFLDv2 road detection model..." << std::endl;
    std::cout << "  Model: " << config.modelPath << std::endl;

    network_ = cv::dnn::readNetFromONNX(config.modelPath);

    if (network_.empty()) {
      std::cerr << "Error: Failed to load UFLDv2 ONNX model" << std::endl;
      return false;
    }

    // Set backend
    if (config.useGPU) {
      std::cout << "  Using CUDA backend for road detection" << std::endl;
      network_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
      network_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
      network_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      network_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    // Initialize row anchors (normalized to [0,1] in the crop height)
    rowAnchors_.clear();
    for (int i = 0; i < config.numRows && i < NUM_CULANE_ANCHORS; ++i) {
      rowAnchors_.push_back(static_cast<float>(CULANE_ROW_ANCHORS[i]) /
                            static_cast<float>(config.inputHeight));
    }

    modelLoaded_ = true;
    std::cout << "UFLDv2 road model loaded successfully (" << config.inputWidth
              << "x" << config.inputHeight << ", " << config.numLanes
              << " lanes, " << config.numRows << " row anchors)" << std::endl;
    return true;

  } catch (const cv::Exception &e) {
    std::cerr << "OpenCV exception loading UFLDv2 model: " << e.what()
              << std::endl;
    return false;
  }
}

// ---------------------------------------------------------------------------
// Decode UFLDv2 ONNX outputs into lane polylines
//
// Output tensor layout (CULane res34):
//   loc_row:   [1, 200, 72, 4]  — 200 position bins, 72 row anchors, 4 lanes
//   loc_col:   [1, 100, 81, 4]  — 100 position bins, 81 col anchors, 4 lanes
//   exist_row: [1, 2, 72, 4]    — binary existence per row anchor per lane
//   exist_col: [1, 2, 81, 4]    — binary existence per col anchor per lane
//
// We only use the ROW anchors (loc_row + exist_row) since they give us
// the x-coordinate at each predefined y-row — perfect for lane polylines.
// ---------------------------------------------------------------------------
std::vector<std::vector<cv::Point>>
RoadDetector::decodeOutputs(const std::vector<cv::Mat> &outputs, int origWidth,
                            int origHeight, const RoadConfig &config) {
  std::vector<std::vector<cv::Point>> lanes;

  if (outputs.size() < 4) {
    std::cerr << "UFLDv2: Expected 4 output tensors, got " << outputs.size()
              << std::endl;
    return lanes;
  }

  // Identify outputs by size heuristic:
  // loc_row should have dim with 200, exist_row has dim with 2 and 72
  const cv::Mat *locRow = nullptr;
  const cv::Mat *existRow = nullptr;

  for (const auto &out : outputs) {
    // loc_row: total elements = 1 * 200 * 72 * 4 = 57600
    // exist_row: total elements = 1 * 2 * 72 * 4 = 576
    // loc_col: total elements = 1 * 100 * 81 * 4 = 32400
    // exist_col: total elements = 1 * 2 * 81 * 4 = 648
    int total = out.total();
    if (total == 200 * config.numRows * config.numLanes) {
      locRow = &out;
    } else if (total == 2 * config.numRows * config.numLanes) {
      existRow = &out;
    }
  }

  if (!locRow || !existRow) {
    std::cerr << "UFLDv2: Could not identify loc_row/exist_row tensors"
              << std::endl;
    // Print actual sizes for debugging
    for (size_t i = 0; i < outputs.size(); ++i) {
      std::cerr << "  Output[" << i << "] total=" << outputs[i].total()
                << " dims=" << outputs[i].dims;
      for (int d = 0; d < outputs[i].dims; ++d)
        std::cerr << " size[" << d << "]=" << outputs[i].size[d];
      std::cerr << std::endl;
    }
    return lanes;
  }

  const int numBins = 200; // Number of position classification bins
  const int numRows = config.numRows;
  const int numLanes = config.numLanes;

  // Access raw data pointers
  const float *locData = locRow->ptr<float>();
  const float *existData = existRow->ptr<float>();

  // For each lane
  for (int laneIdx = 0; laneIdx < numLanes; ++laneIdx) {
    std::vector<cv::Point> lanePoints;

    for (int rowIdx = 0; rowIdx < numRows; ++rowIdx) {
      // Check existence: exist_row is [1, 2, numRows, numLanes]
      // Index: [0, :, rowIdx, laneIdx]
      // exist[0] = no lane, exist[1] = lane present
      int existOffset0 = 0 * numRows * numLanes + rowIdx * numLanes + laneIdx;
      int existOffset1 = 1 * numRows * numLanes + rowIdx * numLanes + laneIdx;

      float noLane = existData[existOffset0];
      float hasLane = existData[existOffset1];

      // Softmax check: if hasLane > noLane, the lane exists at this row
      if (hasLane <= noLane)
        continue;

      // Get location: loc_row is [1, numBins, numRows, numLanes]
      // For this row+lane, we have numBins classification scores
      // Find argmax over the bins dimension
      float maxVal = -1e30f;
      int maxBin = 0;

      // Also compute softmax-weighted expectation for sub-pixel precision
      float sumExp = 0.0f;
      float sumExpIdx = 0.0f;
      std::vector<float> binScores(numBins);

      for (int b = 0; b < numBins; ++b) {
        int locOffset = b * numRows * numLanes + rowIdx * numLanes + laneIdx;
        float val = locData[locOffset];
        binScores[b] = val;
        if (val > maxVal) {
          maxVal = val;
          maxBin = b;
        }
      }

      // Compute softmax expectation for the position
      // This is more precise than raw argmax
      float maxScore = *std::max_element(binScores.begin(), binScores.end());
      for (int b = 0; b < numBins; ++b) {
        float expVal = std::exp(binScores[b] - maxScore);
        sumExp += expVal;
        sumExpIdx += expVal * static_cast<float>(b);
      }

      float expectedBin = sumExpIdx / (sumExp + 1e-9f);

      // Convert bin position to x-coordinate in original image
      // expectedBin is in [0, numBins-1], map to [0, origWidth]
      float xNorm = (expectedBin + 0.5f) / static_cast<float>(numBins);
      int x = static_cast<int>(xNorm * origWidth);

      // Convert row anchor to y-coordinate in original image
      float yNorm = (rowIdx < static_cast<int>(rowAnchors_.size()))
                        ? rowAnchors_[rowIdx]
                        : static_cast<float>(rowIdx) / numRows;
      int y = static_cast<int>(yNorm * origHeight);

      // Clamp to frame
      x = std::max(0, std::min(origWidth - 1, x));
      y = std::max(0, std::min(origHeight - 1, y));

      lanePoints.push_back(cv::Point(x, y));
    }

    // Only keep lanes with a minimum number of points
    if (lanePoints.size() >= 4) {
      // Sort points by y (top to bottom)
      std::sort(
          lanePoints.begin(), lanePoints.end(),
          [](const cv::Point &a, const cv::Point &b) { return a.y < b.y; });
      lanes.push_back(lanePoints);
    }
  }

  return lanes;
}

// ---------------------------------------------------------------------------
// Build road mask from detected lane lines
// Fills the region between the leftmost and rightmost detected lanes
// ---------------------------------------------------------------------------
cv::Mat
RoadDetector::buildRoadMask(const std::vector<std::vector<cv::Point>> &lanes,
                            int width, int height) {
  cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);

  if (lanes.size() < 2) {
    // If fewer than 2 lanes, create a default road mask (center road area)
    if (lanes.size() == 1) {
      // Single lane detected — create a road region around it
      std::vector<cv::Point> roadPoly;
      const auto &lane = lanes[0];

      for (size_t i = 0; i < lane.size(); ++i) {
        roadPoly.push_back(
            cv::Point(std::max(0, lane[i].x - width / 6), lane[i].y));
      }
      for (int i = static_cast<int>(lane.size()) - 1; i >= 0; --i) {
        roadPoly.push_back(
            cv::Point(std::min(width - 1, lane[i].x + width / 6), lane[i].y));
      }

      std::vector<std::vector<cv::Point>> polys = {roadPoly};
      cv::fillPoly(mask, polys, cv::Scalar(255));
    } else {
      // No lanes at all — fill center ~60% as road
      int margin = width / 5;
      cv::rectangle(mask, cv::Point(margin, 0),
                    cv::Point(width - margin, height), cv::Scalar(255),
                    cv::FILLED);
    }
    return mask;
  }

  // Find the leftmost and rightmost lanes
  // "Leftmost" = lane with smallest average x, "rightmost" = largest avg x
  int leftIdx = 0, rightIdx = 0;
  float minAvgX = 1e9f, maxAvgX = -1e9f;

  for (size_t i = 0; i < lanes.size(); ++i) {
    float avgX = 0;
    for (const auto &pt : lanes[i])
      avgX += pt.x;
    avgX /= lanes[i].size();

    if (avgX < minAvgX) {
      minAvgX = avgX;
      leftIdx = static_cast<int>(i);
    }
    if (avgX > maxAvgX) {
      maxAvgX = avgX;
      rightIdx = static_cast<int>(i);
    }
  }

  if (leftIdx == rightIdx && lanes.size() > 1) {
    // Fallback: use first and last lane
    leftIdx = 0;
    rightIdx = static_cast<int>(lanes.size()) - 1;
  }

  // Build polygon: left lane top-to-bottom, then right lane bottom-to-top
  std::vector<cv::Point> roadPoly;

  // Left lane (top to bottom)
  for (const auto &pt : lanes[leftIdx]) {
    roadPoly.push_back(pt);
  }

  // Right lane (bottom to top — reversed)
  const auto &rightLane = lanes[rightIdx];
  for (int i = static_cast<int>(rightLane.size()) - 1; i >= 0; --i) {
    roadPoly.push_back(rightLane[i]);
  }

  // Fill the road polygon
  if (roadPoly.size() >= 3) {
    std::vector<std::vector<cv::Point>> polys = {roadPoly};
    cv::fillPoly(mask, polys, cv::Scalar(255));
  }

  return mask;
}

// ---------------------------------------------------------------------------
// Main entry point: preprocess → infer → decode → mask
// ---------------------------------------------------------------------------
cv::Mat
RoadDetector::detectRoad(const cv::Mat &frame, const RoadConfig &config,
                         std::vector<std::vector<cv::Point>> &outLaneLines) {
  outLaneLines.clear();

  if (!modelLoaded_ || frame.empty() || !config.enabled) {
    return cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
  }

  try {
    // --- Preprocessing ---
    // UFLDv2 expects: [1, 3, inputHeight, inputWidth] normalized with
    // ImageNet mean/std
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(config.inputWidth, config.inputHeight));

    // Convert to float and normalize with ImageNet stats
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32F, 1.0 / 255.0);

    // ImageNet normalization: (pixel - mean) / std
    cv::Scalar mean(0.485, 0.456, 0.406); // RGB order
    cv::Scalar stddev(0.229, 0.224, 0.225);

    // Split channels, normalize, merge back
    std::vector<cv::Mat> channels;
    cv::split(floatImg, channels);

    // OpenCV is BGR, ImageNet mean is RGB → swap
    channels[0] = (channels[0] - mean[2]) / stddev[2]; // B channel
    channels[1] = (channels[1] - mean[1]) / stddev[1]; // G channel
    channels[2] = (channels[2] - mean[0]) / stddev[0]; // R channel

    cv::Mat normalized;
    cv::merge(channels, normalized);

    // Create blob (NCHW format, already normalized — don't re-scale)
    cv::Mat blob = cv::dnn::blobFromImage(normalized, 1.0, cv::Size(),
                                          cv::Scalar(), true, false);

    // --- Inference ---
    network_.setInput(blob);

    std::vector<std::string> outNames = network_.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outputs;
    network_.forward(outputs, outNames);

    // --- Post-processing ---
    std::vector<std::vector<cv::Point>> lanes =
        decodeOutputs(outputs, frame.cols, frame.rows, config);

    // Temporal smoothing
    if (hasPrevLanes_ && !prevLanes_.empty() && !lanes.empty()) {
      // Match lanes by average x position and smooth
      for (size_t i = 0; i < lanes.size() && i < prevLanes_.size(); ++i) {
        size_t numPts = std::min(lanes[i].size(), prevLanes_[i].size());
        for (size_t j = 0; j < numPts; ++j) {
          lanes[i][j].x =
              static_cast<int>(smoothingAlpha_ * lanes[i][j].x +
                               (1.0f - smoothingAlpha_) * prevLanes_[i][j].x);
          lanes[i][j].y =
              static_cast<int>(smoothingAlpha_ * lanes[i][j].y +
                               (1.0f - smoothingAlpha_) * prevLanes_[i][j].y);
        }
      }
    }

    prevLanes_ = lanes;
    hasPrevLanes_ = true;

    outLaneLines = lanes;

    // Build road mask
    return buildRoadMask(lanes, frame.cols, frame.rows);

  } catch (const cv::Exception &e) {
    std::cerr << "UFLDv2 inference error: " << e.what() << std::endl;
    return cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
  }
}

} // namespace FlightPath
