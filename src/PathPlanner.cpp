#include "PathPlanner.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace FlightPath {

PathPlanner::PathPlanner() {}

PathPlanner::~PathPlanner() {}

// --- Perspective helpers ---

float PathPlanner::perspectiveScale(float y, float frameHeight,
                                    float horizonY) {
  // Objects near the bottom of the frame are close (large scale).
  // Objects near the horizon are far away (small scale).
  // Returns a factor in (0, 1] where 1 = bottom of frame (closest).
  float range = frameHeight - horizonY;
  if (range <= 0.0f)
    return 1.0f;

  float depth = (y - horizonY) / range; // 0 at horizon, 1 at bottom
  depth = std::max(0.0f, std::min(1.0f, depth));

  // Quadratic falloff gives a more realistic perspective feel
  return depth * depth;
}

float PathPlanner::pixelWidthToMeters(float pixelWidth, float y,
                                      const cv::Size &frameSize,
                                      const PathConfig &config) {
  // At the bottom of the frame, laneWidthAtBottom * frameWidth pixels ≈ typical
  // lane (3.7 m). Scale that ratio by the perspective factor at row y.
  float horizonY = config.horizonRatio * frameSize.height;
  float scale =
      perspectiveScale(y, static_cast<float>(frameSize.height), horizonY);
  if (scale < 0.001f)
    scale = 0.001f; // avoid division by zero

  float pixelsPerMeterAtBottom =
      (config.laneWidthAtBottom * frameSize.width) / 3.7f;
  float pixelsPerMeterAtY = pixelsPerMeterAtBottom * scale;

  return pixelWidth / pixelsPerMeterAtY;
}

// --- Occupancy grid ---

cv::Mat
PathPlanner::createOccupancyGrid(const std::vector<Detection> &detections,
                                 const cv::Size &frameSize, int gridSize) {
  cv::Mat grid = cv::Mat::zeros(gridSize, gridSize, CV_8UC1);

  if (detections.empty()) {
    return grid;
  }

  float cellWidth = static_cast<float>(frameSize.width) / gridSize;
  float cellHeight = static_cast<float>(frameSize.height) / gridSize;

  for (const auto &det : detections) {
    int gridLeft = static_cast<int>(det.boundingBox.x / cellWidth);
    int gridTop = static_cast<int>(det.boundingBox.y / cellHeight);
    int gridRight = static_cast<int>(
        (det.boundingBox.x + det.boundingBox.width) / cellWidth);
    int gridBottom = static_cast<int>(
        (det.boundingBox.y + det.boundingBox.height) / cellHeight);

    gridLeft = std::max(0, std::min(gridLeft, gridSize - 1));
    gridTop = std::max(0, std::min(gridTop, gridSize - 1));
    gridRight = std::max(0, std::min(gridRight, gridSize - 1));
    gridBottom = std::max(0, std::min(gridBottom, gridSize - 1));

    // Expand obstacle footprint slightly for safety buffer
    int expand = std::max(1, gridSize / 25);
    gridLeft = std::max(0, gridLeft - expand);
    gridTop = std::max(0, gridTop - expand);
    gridRight = std::min(gridSize - 1, gridRight + expand);
    gridBottom = std::min(gridSize - 1, gridBottom + expand);

    for (int y = gridTop; y <= gridBottom; ++y) {
      for (int x = gridLeft; x <= gridRight; ++x) {
        grid.at<uchar>(y, x) = 255;
      }
    }
  }

  return grid;
}

// --- Gap finding ---

std::vector<Path> PathPlanner::findGaps(const cv::Mat &occupancyGrid,
                                        const cv::Size &frameSize,
                                        const PathConfig &config) {
  std::vector<Path> paths;

  int gridSize = occupancyGrid.rows;
  float cellWidth = static_cast<float>(frameSize.width) / gridSize;
  float cellHeight = static_cast<float>(frameSize.height) / gridSize;

  // Scan from 35% to 85% of the frame height (road region)
  int startRow = static_cast<int>(gridSize * 0.35);
  int endRow = static_cast<int>(gridSize * 0.85);
  int rowStep = std::max(1, gridSize / 12);

  for (int row = startRow; row < endRow; row += rowStep) {
    bool inGap = false;
    int gapStart = 0;

    for (int col = 0; col <= gridSize; ++col) {
      bool isFree =
          (col < gridSize) ? (occupancyGrid.at<uchar>(row, col) == 0) : false;

      if (isFree && !inGap) {
        inGap = true;
        gapStart = col;
      } else if (!isFree && inGap) {
        inGap = false;
        int gapEnd = col - 1;
        int gapWidth = gapEnd - gapStart + 1;

        float gapWidthPixels = gapWidth * cellWidth;
        float rowPixelY = row * cellHeight + cellHeight / 2.0f;
        float gapWidthMeters =
            pixelWidthToMeters(gapWidthPixels, rowPixelY, frameSize, config);

        if (gapWidthMeters >= config.minGapWidth) {
          Path path;
          path.start = cv::Point(frameSize.width / 2, frameSize.height - 10);

          int gapCenterCol = (gapStart + gapEnd) / 2;
          path.end = cv::Point(
              static_cast<int>(gapCenterCol * cellWidth + cellWidth / 2),
              static_cast<int>(rowPixelY));

          path.width = gapWidthPixels;
          paths.push_back(path);
        }
      }
    }
  }

  return paths;
}

// --- Scoring ---

float PathPlanner::scorePath(const Path &path, const cv::Size &frameSize,
                             const PathConfig &config) {
  // 1) Width score — wider gaps are better
  float rowY = static_cast<float>(path.end.y);
  float widthMeters = pixelWidthToMeters(path.width, rowY, frameSize, config);
  float minWidth = config.vehicleWidth + config.safetyMargin;
  float widthScore = std::min(1.0f, widthMeters / (minWidth * 2.5f));

  // 2) Depth score — prefer gaps that are moderately far (not too close, not
  // too far)
  float horizonY = config.horizonRatio * frameSize.height;
  float normalizedDepth =
      1.0f - (rowY - horizonY) / (frameSize.height - horizonY);
  normalizedDepth = std::max(0.0f, std::min(1.0f, normalizedDepth));
  // Sweet spot around 0.3-0.5 depth
  float depthScore = 1.0f - std::abs(normalizedDepth - 0.4f) * 2.0f;
  depthScore = std::max(0.0f, depthScore);

  // 3) Alignment score — prefer paths close to center
  float centerX = frameSize.width / 2.0f;
  float deviation = std::abs(path.end.x - centerX);
  float maxDeviation = frameSize.width / 2.0f;
  float alignmentScore = 1.0f - (deviation / maxDeviation);
  alignmentScore = alignmentScore * alignmentScore; // Strongly prefer center

  // Weighted combination
  return widthScore * 0.45f + depthScore * 0.20f + alignmentScore * 0.35f;
}

void PathPlanner::classifyPath(Path &path, const cv::Size &frameSize,
                               const PathConfig &config) {
  float rowY = static_cast<float>(path.end.y);
  float widthMeters = pixelWidthToMeters(path.width, rowY, frameSize, config);
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

// --- Perspective path building ---

Path PathPlanner::buildPerspectivePath(cv::Point target,
                                       const cv::Size &frameSize,
                                       const PathConfig &config) {
  Path path;
  path.start = cv::Point(frameSize.width / 2, frameSize.height - 10);
  path.end = target;

  float horizonY = config.horizonRatio * frameSize.height;
  float vanishingX =
      frameSize.width / 2.0f; // Assume vanishing point at horizontal center

  // Create waypoints that smoothly transition from start to target
  // with perspective-correct positioning (converge toward vanishing point)
  const int numWaypoints = 8;
  path.waypoints.clear();

  for (int i = 0; i <= numWaypoints; ++i) {
    float t = static_cast<float>(i) / numWaypoints; // 0 at start, 1 at target

    // Y position: linearly interpolate
    float y = path.start.y + t * (path.end.y - path.start.y);

    // X position: blend between start x and target x, but pull toward
    // the vanishing point as we go higher (further away)
    float baseX = path.start.x + t * (path.end.x - path.start.x);

    // How much to pull toward vanishing point — increases with depth
    float depthFrac = 1.0f - (y - horizonY) / (frameSize.height - horizonY);
    depthFrac = std::max(0.0f, std::min(1.0f, depthFrac));
    float vanishPull = depthFrac * 0.3f; // Subtle pull

    float x = baseX * (1.0f - vanishPull) + vanishingX * vanishPull;

    path.waypoints.push_back(
        cv::Point(static_cast<int>(x), static_cast<int>(y)));
  }

  return path;
}

// --- Main entry point ---

std::vector<Path>
PathPlanner::findPaths(const std::vector<Detection> &detections,
                       const cv::Size &frameSize, const PathConfig &config) {
  // Create occupancy grid
  cv::Mat occupancyGrid =
      createOccupancyGrid(detections, frameSize, config.gridResolution);

  // Find potential paths through gaps
  std::vector<Path> paths = findGaps(occupancyGrid, frameSize, config);

  // If no obstacles were detected, create a default straight-ahead path
  if (detections.empty() || paths.empty()) {
    float horizonY = config.horizonRatio * frameSize.height;
    float targetY = horizonY + (frameSize.height - horizonY) * 0.35f;
    cv::Point target(frameSize.width / 2, static_cast<int>(targetY));

    Path defaultPath = buildPerspectivePath(target, frameSize, config);
    defaultPath.width = config.laneWidthAtBottom * frameSize.width;
    defaultPath.score = 1.0f;
    defaultPath.type = Path::Type::SAFE;
    defaultPath.isSafe = true;

    return {defaultPath};
  }

  // Score and classify each path
  for (auto &path : paths) {
    path.score = scorePath(path, frameSize, config);
    classifyPath(path, frameSize, config);
  }

  // Sort by score (highest first)
  std::sort(paths.begin(), paths.end(),
            [](const Path &a, const Path &b) { return a.score > b.score; });

  // Take only the single best path
  Path best = paths[0];

  // Rebuild it with perspective waypoints
  Path result = buildPerspectivePath(best.end, frameSize, config);
  result.width = best.width;
  result.score = best.score;
  result.type = best.type;
  result.isSafe = best.isSafe;

  return {result};
}

} // namespace FlightPath
