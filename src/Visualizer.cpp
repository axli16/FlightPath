#include "Visualizer.h"
#include <iomanip>
#include <sstream>

namespace FlightPath {

Visualizer::Visualizer() {}

Visualizer::~Visualizer() {}

cv::Scalar Visualizer::getPathColor(const Path &path,
                                    const VisualConfig &config) {
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

void Visualizer::drawDetections(cv::Mat &frame,
                                const std::vector<Detection> &detections,
                                const VisualConfig &config) {
  if (!config.showBoundingBoxes) {
    return;
  }

  for (const auto &det : detections) {
    // Draw bounding box
    cv::rectangle(frame, det.boundingBox, config.colorDetection,
                  config.boxThickness);

    // Prepare label
    std::stringstream label;
    label << det.className;

    if (config.showConfidence) {
      label << " " << std::fixed << std::setprecision(2)
            << (det.confidence * 100) << "%";
    }

    // Draw label background
    int baseline = 0;
    cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX,
                                         0.5, config.fontThickness, &baseline);

    cv::Point labelPos(det.boundingBox.x, det.boundingBox.y - 10);
    if (labelPos.y < 0)
      labelPos.y = det.boundingBox.y + labelSize.height + 10;

    cv::rectangle(
        frame, cv::Point(labelPos.x, labelPos.y - labelSize.height - 5),
        cv::Point(labelPos.x + labelSize.width, labelPos.y + baseline),
        config.colorDetection, cv::FILLED);

    // Draw label text
    cv::putText(frame, label.str(), labelPos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), config.fontThickness);
  }
}

void Visualizer::drawPaths(cv::Mat &frame, const std::vector<Path> &paths,
                           const VisualConfig &config) {
  if (paths.empty())
    return;

  // We only draw the single best path
  const Path &path = paths[0];

  if (path.type == Path::Type::BLOCKED)
    return;
  if (path.waypoints.size() < 2)
    return;

  cv::Scalar color = getPathColor(path, config);
  float frameHeight = static_cast<float>(frame.rows);

  // --- Build a filled polygon that tapers with perspective ---
  // For each waypoint, compute a left and right offset that shrinks as
  // the point moves up the frame (farther from the camera).
  std::vector<cv::Point> leftEdge, rightEdge;

  // Base width at the bottom of the frame (in pixels)
  float baseHalfWidth = frame.cols * 0.04f; // ~4% of frame width on each side
  float minHalfWidth = 4.0f;                // Don't go thinner than 4 pixels

  for (size_t i = 0; i < path.waypoints.size(); ++i) {
    const cv::Point &wp = path.waypoints[i];

    // Perspective taper: width shrinks toward the top of the frame
    float t = static_cast<float>(frame.rows - wp.y) / frameHeight;
    t = std::max(0.0f, std::min(1.0f, t));

    // Cubic falloff for a more realistic perspective taper
    float taperedWidth = baseHalfWidth * (1.0f - t * t * t);
    taperedWidth = std::max(minHalfWidth, taperedWidth);

    leftEdge.push_back(cv::Point(wp.x - static_cast<int>(taperedWidth), wp.y));
    rightEdge.push_back(cv::Point(wp.x + static_cast<int>(taperedWidth), wp.y));
  }

  // Combine into a single polygon (left edge forward, right edge reversed)
  std::vector<cv::Point> polygon;
  polygon.insert(polygon.end(), leftEdge.begin(), leftEdge.end());
  polygon.insert(polygon.end(), rightEdge.rbegin(), rightEdge.rend());

  // Draw semi-transparent filled polygon (the "road arrow")
  cv::Mat overlay = frame.clone();
  std::vector<std::vector<cv::Point>> polys = {polygon};
  cv::fillPoly(overlay, polys, color, cv::LINE_AA);
  cv::addWeighted(overlay, 0.30, frame, 0.70, 0, frame);

  // --- Draw the outline for clarity ---
  cv::polylines(frame, polys, true, color, 1, cv::LINE_AA);

  // --- Draw thin centerline along waypoints ---
  for (size_t i = 0; i + 1 < path.waypoints.size(); ++i) {
    // Fade the line as it goes further
    float t = static_cast<float>(i) / path.waypoints.size();
    int alpha = static_cast<int>(255 * (1.0f - t * 0.6f));
    cv::Scalar lineColor(color[0] * alpha / 255, color[1] * alpha / 255,
                         color[2] * alpha / 255);
    cv::line(frame, path.waypoints[i], path.waypoints[i + 1], lineColor, 2,
             cv::LINE_AA);
  }

  // --- Draw arrowhead at the target end ---
  if (path.waypoints.size() >= 2) {
    cv::Point tip = path.waypoints.back();
    cv::Point prev = path.waypoints[path.waypoints.size() - 2];

    // Direction vector from prev to tip
    float dx = static_cast<float>(tip.x - prev.x);
    float dy = static_cast<float>(tip.y - prev.y);
    float len = std::sqrt(dx * dx + dy * dy);
    if (len > 0.0f) {
      dx /= len;
      dy /= len;

      // Arrowhead size
      float arrowLen = 18.0f;
      float arrowSpread = 10.0f;

      // Perpendicular direction
      float px = -dy;
      float py = dx;

      cv::Point left(tip.x - static_cast<int>(dx * arrowLen + px * arrowSpread),
                     tip.y -
                         static_cast<int>(dy * arrowLen + py * arrowSpread));
      cv::Point right(
          tip.x - static_cast<int>(dx * arrowLen - px * arrowSpread),
          tip.y - static_cast<int>(dy * arrowLen - py * arrowSpread));

      std::vector<cv::Point> arrowHead = {tip, left, right};
      std::vector<std::vector<cv::Point>> arrowPolys = {arrowHead};
      cv::fillPoly(frame, arrowPolys, color, cv::LINE_AA);
    }
  }

  // --- Draw score label near the target ---
  std::stringstream scoreText;
  scoreText << std::fixed << std::setprecision(0) << (path.score * 100) << "%";

  cv::Point textPos(path.end.x + 15, path.end.y - 10);
  // Background for readability
  int baseline = 0;
  cv::Size textSize = cv::getTextSize(scoreText.str(), cv::FONT_HERSHEY_SIMPLEX,
                                      0.55, 2, &baseline);
  cv::rectangle(
      frame, cv::Point(textPos.x - 2, textPos.y - textSize.height - 4),
      cv::Point(textPos.x + textSize.width + 4, textPos.y + baseline + 2),
      cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(frame, scoreText.str(), textPos, cv::FONT_HERSHEY_SIMPLEX, 0.55,
              color, 2, cv::LINE_AA);
}

void Visualizer::drawInfoPanel(cv::Mat &frame,
                               const std::vector<Detection> &detections,
                               const std::vector<Path> &paths,
                               const VisualConfig &config) {
  if (!config.showFPS && detections.empty() && paths.empty()) {
    return;
  }

  // Create semi-transparent panel
  int panelHeight = 120;
  int panelWidth = 250;
  cv::Mat panel(panelHeight, panelWidth, CV_8UC3, cv::Scalar(0, 0, 0));

  int y = 25;
  int lineSpacing = 25;

  // Detection count
  std::stringstream detText;
  detText << "Detections: " << detections.size();
  cv::putText(panel, detText.str(), cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
              0.6, cv::Scalar(255, 255, 255), 2);
  y += lineSpacing;

  // Path info
  if (!paths.empty() && paths[0].isSafe) {
    std::stringstream pathText;
    pathText << "Path: CLEAR";
    cv::putText(panel, pathText.str(), cv::Point(10, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    y += lineSpacing;

    std::stringstream scoreText;
    scoreText << "Confidence: " << std::fixed << std::setprecision(0)
              << (paths[0].score * 100) << "%";
    cv::putText(panel, scoreText.str(), cv::Point(10, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
  } else if (!paths.empty()) {
    cv::putText(panel, "Path: TIGHT", cv::Point(10, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
  } else {
    cv::putText(panel, "Path: NONE", cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                0.6, cv::Scalar(0, 0, 255), 2);
  }

  // Blend panel onto frame
  if (frame.cols >= panelWidth + 10 && frame.rows >= panelHeight + 10) {
    cv::Mat roi = frame(cv::Rect(10, 10, panelWidth, panelHeight));
    cv::addWeighted(roi, 0.3, panel, 0.7, 0, roi);
  }
}

void Visualizer::drawROI(cv::Mat &frame, const DetectionConfig &detectionConfig,
                         const VisualConfig &visualConfig) {
  if (!visualConfig.showROI || !detectionConfig.useROI) {
    return;
  }

  // Calculate ROI rectangle in pixel coordinates
  int roiX = static_cast<int>(detectionConfig.roiX * frame.cols);
  int roiY = static_cast<int>(detectionConfig.roiY * frame.rows);
  int roiWidth = static_cast<int>(detectionConfig.roiWidth * frame.cols);
  int roiHeight = static_cast<int>(detectionConfig.roiHeight * frame.rows);

  cv::Rect roi(roiX, roiY, roiWidth, roiHeight);
  // Ensure ROI is within frame bounds
  roi &= cv::Rect(0, 0, frame.cols, frame.rows);

  // Draw ROI rectangle (simplified for performance)
  cv::Scalar roiColor(0, 255, 0);         // Green
  cv::rectangle(frame, roi, roiColor, 2); // No anti-aliasing for speed

  // Add simple label without background
  std::string label = "ROI";
  cv::Point labelPos(roi.x + 5, roi.y + 20);
  cv::putText(frame, label, labelPos, cv::FONT_HERSHEY_SIMPLEX, 0.6, roiColor,
              2);
}

void Visualizer::draw(cv::Mat &frame, const std::vector<Detection> &detections,
                      const std::vector<Path> &paths,
                      const VisualConfig &visualConfig,
                      const DetectionConfig &detectionConfig) {
  if (frame.empty()) {
    return;
  }

  // Draw in order: ROI, detections, paths, info panel
  drawROI(frame, detectionConfig, visualConfig);
  drawDetections(frame, detections, visualConfig);
  drawPaths(frame, paths, visualConfig);
  drawInfoPanel(frame, detections, paths, visualConfig);
}

} // namespace FlightPath
