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
  float range = frameHeight - horizonY;
  if (range <= 0.0f)
    return 1.0f;

  float depth = (y - horizonY) / range; // 0 at horizon, 1 at bottom
  depth = std::max(0.0f, std::min(1.0f, depth));

  return depth * depth;
}

float PathPlanner::pixelWidthToMeters(float pixelWidth, float y,
                                      const cv::Size &frameSize,
                                      const PathConfig &config) {
  float horizonY = config.horizonRatio * frameSize.height;
  float scale =
      perspectiveScale(y, static_cast<float>(frameSize.height), horizonY);
  if (scale < 0.001f)
    scale = 0.001f;

  float pixelsPerMeterAtBottom =
      (config.laneWidthAtBottom * frameSize.width) / 3.7f;
  float pixelsPerMeterAtY = pixelsPerMeterAtBottom * scale;

  return pixelWidth / pixelsPerMeterAtY;
}

// --- ROI computation ---

cv::Rect PathPlanner::computeROI(const cv::Size &frameSize,
                                 const DetectionConfig &detectionConfig) {
  if (!detectionConfig.useROI || detectionConfig.roiWidth <= 0 ||
      detectionConfig.roiHeight <= 0) {
    // No ROI — use entire frame
    return cv::Rect(0, 0, frameSize.width, frameSize.height);
  }

  int roiX = static_cast<int>(detectionConfig.roiX * frameSize.width);
  int roiY = static_cast<int>(detectionConfig.roiY * frameSize.height);
  int roiW = static_cast<int>(detectionConfig.roiWidth * frameSize.width);
  int roiH = static_cast<int>(detectionConfig.roiHeight * frameSize.height);

  cv::Rect roi(roiX, roiY, roiW, roiH);
  roi &= cv::Rect(0, 0, frameSize.width, frameSize.height);
  return roi;
}

// --- Occupancy grid (ROI-relative) ---

cv::Mat
PathPlanner::createOccupancyGrid(const std::vector<Detection> &detections,
                                 const cv::Rect &roi, int gridSize) {
  cv::Mat grid = cv::Mat::zeros(gridSize, gridSize, CV_8UC1);

  if (detections.empty()) {
    return grid;
  }

  float cellWidth = static_cast<float>(roi.width) / gridSize;
  float cellHeight = static_cast<float>(roi.height) / gridSize;

  for (const auto &det : detections) {
    // Detection bounding boxes are in full-frame coordinates.
    // Convert to ROI-relative coordinates.
    float relX = det.boundingBox.x - roi.x;
    float relY = det.boundingBox.y - roi.y;
    float relW = det.boundingBox.width;
    float relH = det.boundingBox.height;

    // Skip detections that don't overlap the ROI
    if (relX + relW < 0 || relY + relH < 0 || relX > roi.width ||
        relY > roi.height) {
      continue;
    }

    // Clamp to ROI bounds
    float clampedX = std::max(0.0f, relX);
    float clampedY = std::max(0.0f, relY);
    float clampedR = std::min(static_cast<float>(roi.width), relX + relW);
    float clampedB = std::min(static_cast<float>(roi.height), relY + relH);

    int gridLeft = static_cast<int>(clampedX / cellWidth);
    int gridTop = static_cast<int>(clampedY / cellHeight);
    int gridRight = static_cast<int>(clampedR / cellWidth);
    int gridBottom = static_cast<int>(clampedB / cellHeight);

    gridLeft = std::max(0, std::min(gridLeft, gridSize - 1));
    gridTop = std::max(0, std::min(gridTop, gridSize - 1));
    gridRight = std::max(0, std::min(gridRight, gridSize - 1));
    gridBottom = std::max(0, std::min(gridBottom, gridSize - 1));

    // Safety buffer
    int expand = std::max(1, gridSize / 25);
    gridLeft = std::max(0, gridLeft - expand);
    gridTop = std::max(0, gridTop - expand);
    gridRight = std::min(gridSize - 1, gridRight + expand);
    gridBottom = std::min(gridSize - 1, gridBottom + expand);

    // Optimization: Use vectorized cv::rectangle instead of nested O(N^2)
    // pixel-by-pixel assignments. cv::FILLED is inclusive of the bottom-right
    // coordinate, so no +1 is needed.
    cv::rectangle(grid, cv::Point(gridLeft, gridTop),
                  cv::Point(gridRight, gridBottom), cv::Scalar(255),
                  cv::FILLED);
  }

  return grid;
}

// --- Gap finding (ROI-relative, center-out scanning) ---

// --- Trapezoid masking ---

void PathPlanner::maskOccupancyGrid(cv::Mat &grid, const cv::Rect &roi,
                                    const TrapezoidROI &trapezoid) {
  if (!trapezoid.valid)
    return;

  int gridSize = grid.rows;
  float cellWidth = static_cast<float>(roi.width) / gridSize;
  float cellHeight = static_cast<float>(roi.height) / gridSize;

  // Build a mask matching grid dimensions
  cv::Mat mask = cv::Mat::zeros(gridSize, gridSize, CV_8UC1);

  // Convert trapezoid vertices to grid coordinates (ROI-relative)
  std::vector<cv::Point> gridPoly;
  for (const auto &v : trapezoid.vertices) {
    int gx = static_cast<int>((v.x - roi.x) / cellWidth);
    int gy = static_cast<int>((v.y - roi.y) / cellHeight);
    gx = std::max(0, std::min(gridSize - 1, gx));
    gy = std::max(0, std::min(gridSize - 1, gy));
    gridPoly.push_back(cv::Point(gx, gy));
  }

  // Fill the interior of the trapezoid with white
  std::vector<std::vector<cv::Point>> polys = {gridPoly};
  cv::fillPoly(mask, polys, cv::Scalar(255));

  // Optimization: Replace O(N^2) conditional loop with natively vectorized
  // OpenCV setTo. Mark everything OUTSIDE the trapezoid as occupied (255) in
  // the grid.
  grid.setTo(255, mask == 0);
}

std::vector<Path> PathPlanner::findGaps(const cv::Mat &occupancyGrid,
                                        const cv::Rect &roi,
                                        const PathConfig &config) {
  std::vector<Path> paths;

  int gridSize = occupancyGrid.rows;
  float cellWidth = static_cast<float>(roi.width) / gridSize;
  float cellHeight = static_cast<float>(roi.height) / gridSize;
  int centerCol = gridSize / 2;

  // Scan from 25% to 80% of the ROI height (road region within the ROI)
  int startRow = static_cast<int>(gridSize * 0.25);
  int endRow = static_cast<int>(gridSize * 0.80);
  int rowStep = std::max(1, gridSize / 10);

  for (int row = startRow; row < endRow; row += rowStep) {
    // --- Check if center is blocked ---
    bool centerBlocked = false;
    int checkRadius = std::max(1, gridSize / 12);
    for (int c = centerCol - checkRadius; c <= centerCol + checkRadius; ++c) {
      int cc = std::max(0, std::min(c, gridSize - 1));
      if (occupancyGrid.at<uchar>(row, cc) != 0) {
        centerBlocked = true;
        break;
      }
    }

    if (centerBlocked) {
      // --- Scan LEFT from center for nearest clear gap ---
      int leftGapEnd = -1, leftGapStart = -1;
      for (int col = centerCol - checkRadius - 1; col >= 0; --col) {
        if (occupancyGrid.at<uchar>(row, col) == 0) {
          if (leftGapEnd < 0)
            leftGapEnd = col;
          leftGapStart = col;
        } else {
          if (leftGapEnd >= 0)
            break; // Found a gap, stop
        }
      }

      // --- Scan RIGHT from center for nearest clear gap ---
      int rightGapStart = -1, rightGapEnd = -1;
      for (int col = centerCol + checkRadius + 1; col < gridSize; ++col) {
        if (occupancyGrid.at<uchar>(row, col) == 0) {
          if (rightGapStart < 0)
            rightGapStart = col;
          rightGapEnd = col;
        } else {
          if (rightGapStart >= 0)
            break; // Found a gap, stop
        }
      }

      // Evaluate left gap
      if (leftGapEnd >= 0 && leftGapStart >= 0) {
        int gapWidth = leftGapEnd - leftGapStart + 1;
        float gapWidthPixels = gapWidth * cellWidth;
        float rowPixelY = roi.y + row * cellHeight + cellHeight / 2.0f;
        // Use full frame size for meter conversion
        cv::Size frameSize(roi.x + roi.width, roi.y + roi.height);
        float gapWidthMeters =
            pixelWidthToMeters(gapWidthPixels, rowPixelY, frameSize, config);

        if (gapWidthMeters >= config.minGapWidth) {
          Path path;
          path.start =
              cv::Point(roi.x + roi.width / 2, roi.y + roi.height - 10);
          int gapCenter = (leftGapStart + leftGapEnd) / 2;
          // Point FORWARD toward horizon, offset laterally to the gap center
          float horizonY = config.horizonRatio * (roi.y + roi.height);
          float targetY = std::max(static_cast<float>(roi.y),
                                   horizonY + (roi.height) * 0.10f);
          int targetX =
              roi.x + static_cast<int>(gapCenter * cellWidth + cellWidth / 2);
          path.end = cv::Point(targetX, static_cast<int>(targetY));
          path.width = gapWidthPixels;
          paths.push_back(path);
        }
      }

      // Evaluate right gap
      if (rightGapStart >= 0 && rightGapEnd >= 0) {
        int gapWidth = rightGapEnd - rightGapStart + 1;
        float gapWidthPixels = gapWidth * cellWidth;
        float rowPixelY = roi.y + row * cellHeight + cellHeight / 2.0f;
        cv::Size frameSize(roi.x + roi.width, roi.y + roi.height);
        float gapWidthMeters =
            pixelWidthToMeters(gapWidthPixels, rowPixelY, frameSize, config);

        if (gapWidthMeters >= config.minGapWidth) {
          Path path;
          path.start =
              cv::Point(roi.x + roi.width / 2, roi.y + roi.height - 10);
          int gapCenter = (rightGapStart + rightGapEnd) / 2;
          // Point FORWARD toward horizon, offset laterally to the gap center
          float horizonY = config.horizonRatio * (roi.y + roi.height);
          float targetY = std::max(static_cast<float>(roi.y),
                                   horizonY + (roi.height) * 0.10f);
          int targetX =
              roi.x + static_cast<int>(gapCenter * cellWidth + cellWidth / 2);
          path.end = cv::Point(targetX, static_cast<int>(targetY));
          path.width = gapWidthPixels;
          paths.push_back(path);
        }
      }
    } else {
      // --- Center is clear: scan for gap that includes center ---
      int gapStart = centerCol, gapEnd = centerCol;

      // Expand left
      while (gapStart > 0 && occupancyGrid.at<uchar>(row, gapStart - 1) == 0)
        --gapStart;
      // Expand right
      while (gapEnd < gridSize - 1 &&
             occupancyGrid.at<uchar>(row, gapEnd + 1) == 0)
        ++gapEnd;

      int gapWidth = gapEnd - gapStart + 1;
      float gapWidthPixels = gapWidth * cellWidth;
      float rowPixelY = roi.y + row * cellHeight + cellHeight / 2.0f;
      cv::Size frameSize(roi.x + roi.width, roi.y + roi.height);
      float gapWidthMeters =
          pixelWidthToMeters(gapWidthPixels, rowPixelY, frameSize, config);

      if (gapWidthMeters >= config.minGapWidth) {
        Path path;
        path.start = cv::Point(roi.x + roi.width / 2, roi.y + roi.height - 10);
        int gapCenter = (gapStart + gapEnd) / 2;
        path.end = cv::Point(
            roi.x + static_cast<int>(gapCenter * cellWidth + cellWidth / 2),
            static_cast<int>(rowPixelY));
        path.width = gapWidthPixels;
        paths.push_back(path);
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

  // 2) Depth score — prefer gaps at moderate depth
  float horizonY = config.horizonRatio * frameSize.height;
  float normalizedDepth =
      1.0f - (rowY - horizonY) / (frameSize.height - horizonY);
  normalizedDepth = std::max(0.0f, std::min(1.0f, normalizedDepth));
  float depthScore = 1.0f - std::abs(normalizedDepth - 0.4f) * 2.0f;
  depthScore = std::max(0.0f, depthScore);

  // 3) Alignment score — prefer paths close to path start x (ROI center)
  float centerX = static_cast<float>(path.start.x);
  float deviation = std::abs(path.end.x - centerX);
  float maxDeviation = frameSize.width / 2.0f;
  float alignmentScore = 1.0f - (deviation / maxDeviation);
  alignmentScore = alignmentScore * alignmentScore;

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

// --- Perspective path building (ROI-constrained) ---

Path PathPlanner::buildPerspectivePath(cv::Point target, const cv::Rect &roi,
                                       const cv::Size &frameSize,
                                       const PathConfig &config) {
  Path path;

  // Start the path at ~55% frame height (above the motorcycle dashboard)
  // instead of the very bottom of the ROI — this makes the path look like
  // it's projected onto the road surface ahead, not on the dashboard.
  int pathStartY = static_cast<int>(frameSize.height * 0.55f);
  path.start = cv::Point(roi.x + roi.width / 2, pathStartY);
  path.end = target;

  // Clamp target inside ROI
  path.end.x = std::max(roi.x, std::min(roi.x + roi.width, path.end.x));
  path.end.y = std::max(roi.y, std::min(roi.y + roi.height, path.end.y));

  // SAFETY: path must always point UPWARD on screen (toward horizon).
  // end.y must be less than start.y (higher on screen = lower pixel value).
  if (path.end.y >= path.start.y) {
    path.end.y = path.start.y - static_cast<int>(frameSize.height * 0.25f);
    path.end.y = std::max(0, path.end.y);
  }

  float horizonY = config.horizonRatio * frameSize.height;
  float vanishingX = roi.x + roi.width / 2.0f;

  // More waypoints for a longer, smoother path
  const int numWaypoints = 16;
  path.waypoints.clear();

  for (int i = 0; i <= numWaypoints; ++i) {
    float t = static_cast<float>(i) / numWaypoints;

    float y = path.start.y + t * (path.end.y - path.start.y);
    float baseX = path.start.x + t * (path.end.x - path.start.x);

    // Stronger vanishing point pull for road-surface convergence
    float depthFrac = 1.0f - (y - horizonY) / (frameSize.height - horizonY);
    depthFrac = std::max(0.0f, std::min(1.0f, depthFrac));
    float vanishPull = depthFrac * depthFrac * 0.5f;

    float x = baseX * (1.0f - vanishPull) + vanishingX * vanishPull;

    // Clamp waypoint inside frame (not just ROI — path can extend)
    x = std::max(0.0f, std::min(static_cast<float>(frameSize.width - 1), x));
    y = std::max(0.0f, std::min(static_cast<float>(frameSize.height - 1), y));

    path.waypoints.push_back(
        cv::Point(static_cast<int>(x), static_cast<int>(y)));
  }

  return path;
}

// --- Main entry point ---

std::vector<Path>
PathPlanner::findPaths(const std::vector<Detection> &detections,
                       const cv::Size &frameSize, const PathConfig &config,
                       const DetectionConfig &detectionConfig,
                       const TrapezoidROI &trapezoidROI) {
  // Compute pixel ROI: prefer trapezoid bounding rect when available
  cv::Rect roi;
  if (trapezoidROI.valid) {
    roi = trapezoidROI.boundingRect();
    roi &= cv::Rect(0, 0, frameSize.width, frameSize.height);
  } else {
    roi = computeROI(frameSize, detectionConfig);
  }

  // Create occupancy grid within the ROI
  cv::Mat occupancyGrid =
      createOccupancyGrid(detections, roi, config.gridResolution);

  // Mask grid cells outside the trapezoid if we have one
  if (trapezoidROI.valid) {
    maskOccupancyGrid(occupancyGrid, roi, trapezoidROI);
  }

  // Find potential paths through gaps (center-out scanning)
  std::vector<Path> paths = findGaps(occupancyGrid, roi, config);

  // If no obstacles detected or no viable gaps, create default straight path
  if (detections.empty() || paths.empty()) {
    float horizonY = config.horizonRatio * frameSize.height;
    float targetY = std::max(static_cast<float>(roi.y),
                             horizonY + (frameSize.height - horizonY) * 0.10f);
    cv::Point target(roi.x + roi.width / 2, static_cast<int>(targetY));

    Path defaultPath = buildPerspectivePath(target, roi, frameSize, config);
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

  // Take the single best path and rebuild with perspective waypoints.
  // CRITICAL: Override the Y target to always point FORWARD (toward horizon),
  // regardless of where the gap was found. The gap only determines the X
  // offset.
  Path best = paths[0];
  float horizonY = config.horizonRatio * frameSize.height;
  float forwardTargetY =
      std::max(static_cast<float>(roi.y),
               horizonY + (frameSize.height - horizonY) * 0.20f);
  best.end.y = static_cast<int>(forwardTargetY);
  Path result = buildPerspectivePath(best.end, roi, frameSize, config);
  result.width = best.width;
  result.score = best.score;
  result.type = best.type;
  result.isSafe = best.isSafe;

  return {result};
}

// ---------------------------------------------------------------------------
// Road-mask-aware occupancy grid
// Combines the ML road mask (road = free, off-road = occupied)
// with vehicle detection bounding boxes (vehicle = occupied)
// ---------------------------------------------------------------------------
cv::Mat PathPlanner::createOccupancyGridWithRoad(
    const std::vector<Detection> &detections, const cv::Mat &roadMask,
    const cv::Rect &roi, int gridSize) {

  cv::Mat grid = cv::Mat::zeros(gridSize, gridSize, CV_8UC1);
  float cellWidth = static_cast<float>(roi.width) / gridSize;
  float cellHeight = static_cast<float>(roi.height) / gridSize;

  // Step 1: Mark non-road cells as occupied (if road mask available)
  if (!roadMask.empty()) {
    for (int gy = 0; gy < gridSize; ++gy) {
      for (int gx = 0; gx < gridSize; ++gx) {
        // Map grid cell center to frame coordinates
        int frameX = roi.x + static_cast<int>((gx + 0.5f) * cellWidth);
        int frameY = roi.y + static_cast<int>((gy + 0.5f) * cellHeight);

        // Clamp to mask bounds
        frameX = std::max(0, std::min(roadMask.cols - 1, frameX));
        frameY = std::max(0, std::min(roadMask.rows - 1, frameY));

        // If not road → occupied
        if (roadMask.at<uchar>(frameY, frameX) == 0) {
          grid.at<uchar>(gy, gx) = 255;
        }
      }
    }
  }

  // Step 2: Mark vehicle detections as occupied (same logic as before)
  for (const auto &det : detections) {
    float relX = det.boundingBox.x - roi.x;
    float relY = det.boundingBox.y - roi.y;
    float relW = det.boundingBox.width;
    float relH = det.boundingBox.height;

    if (relX + relW < 0 || relY + relH < 0 || relX > roi.width ||
        relY > roi.height)
      continue;

    float clampedX = std::max(0.0f, relX);
    float clampedY = std::max(0.0f, relY);
    float clampedR = std::min(static_cast<float>(roi.width), relX + relW);
    float clampedB = std::min(static_cast<float>(roi.height), relY + relH);

    int gridLeft = static_cast<int>(clampedX / cellWidth);
    int gridTop = static_cast<int>(clampedY / cellHeight);
    int gridRight = static_cast<int>(clampedR / cellWidth);
    int gridBottom = static_cast<int>(clampedB / cellHeight);

    int expand = std::max(1, gridSize / 25);
    gridLeft = std::max(0, gridLeft - expand);
    gridTop = std::max(0, gridTop - expand);
    gridRight = std::min(gridSize - 1, gridRight + expand);
    gridBottom = std::min(gridSize - 1, gridBottom + expand);

    // Optimization: Use vectorized cv::rectangle instead of nested O(N^2)
    // pixel-by-pixel assignments. cv::FILLED is inclusive of the bottom-right
    // coordinate, so no +1 is needed.
    cv::rectangle(grid, cv::Point(gridLeft, gridTop),
                  cv::Point(gridRight, gridBottom), cv::Scalar(255),
                  cv::FILLED);
  }

  return grid;
}

// ---------------------------------------------------------------------------
// Temporal smoothing via EMA on waypoints
// ---------------------------------------------------------------------------
void PathPlanner::smoothPath(Path &path) {
  if (!hasPrevPath_ || prevPath_.waypoints.empty() || path.waypoints.empty()) {
    prevPath_ = path;
    hasPrevPath_ = true;
    return;
  }

  size_t numPts = std::min(path.waypoints.size(), prevPath_.waypoints.size());
  for (size_t i = 0; i < numPts; ++i) {
    path.waypoints[i].x = static_cast<int>(
        pathSmoothingAlpha_ * path.waypoints[i].x +
        (1.0f - pathSmoothingAlpha_) * prevPath_.waypoints[i].x);
    path.waypoints[i].y = static_cast<int>(
        pathSmoothingAlpha_ * path.waypoints[i].y +
        (1.0f - pathSmoothingAlpha_) * prevPath_.waypoints[i].y);
  }

  // Smooth start/end too
  path.start.x =
      static_cast<int>(pathSmoothingAlpha_ * path.start.x +
                       (1.0f - pathSmoothingAlpha_) * prevPath_.start.x);
  path.end.x = static_cast<int>(pathSmoothingAlpha_ * path.end.x +
                                (1.0f - pathSmoothingAlpha_) * prevPath_.end.x);

  prevPath_ = path;
  hasPrevPath_ = true;
}

// ---------------------------------------------------------------------------
// Road-mask-aware findPaths overload
// ---------------------------------------------------------------------------
std::vector<Path> PathPlanner::findPaths(
    const std::vector<Detection> &detections, const cv::Size &frameSize,
    const PathConfig &config, const DetectionConfig &detectionConfig,
    const cv::Mat &roadMask, const TrapezoidROI &trapezoidROI) {
  // Compute ROI
  cv::Rect roi;
  if (trapezoidROI.valid) {
    roi = trapezoidROI.boundingRect();
    roi &= cv::Rect(0, 0, frameSize.width, frameSize.height);
  } else {
    roi = computeROI(frameSize, detectionConfig);
  }

  // Build occupancy grid combining road mask + vehicle detections
  cv::Mat occupancyGrid = createOccupancyGridWithRoad(detections, roadMask, roi,
                                                      config.gridResolution);

  // Also mask outside trapezoid if available
  if (trapezoidROI.valid) {
    maskOccupancyGrid(occupancyGrid, roi, trapezoidROI);
  }

  // Find gaps through the grid
  std::vector<Path> paths = findGaps(occupancyGrid, roi, config);

  // Default straight path when road is clear
  if (detections.empty() || paths.empty()) {
    float horizonY = config.horizonRatio * frameSize.height;
    float targetY = std::max(static_cast<float>(roi.y),
                             horizonY + (frameSize.height - horizonY) * 0.10f);
    cv::Point target(roi.x + roi.width / 2, static_cast<int>(targetY));

    Path defaultPath = buildPerspectivePath(target, roi, frameSize, config);
    defaultPath.width = config.laneWidthAtBottom * frameSize.width;
    defaultPath.score = 1.0f;
    defaultPath.type = Path::Type::SAFE;
    defaultPath.isSafe = true;

    smoothPath(defaultPath);
    return {defaultPath};
  }

  // Score and classify
  for (auto &path : paths) {
    path.score = scorePath(path, frameSize, config);
    classifyPath(path, frameSize, config);
  }

  // Sort by score
  std::sort(paths.begin(), paths.end(),
            [](const Path &a, const Path &b) { return a.score > b.score; });

  // Take best path, build perspective waypoints.
  // CRITICAL: Override the Y target to always point FORWARD (toward horizon),
  // regardless of where the gap was found. The gap only determines the X
  // offset.
  Path best = paths[0];
  float horizonY2 = config.horizonRatio * frameSize.height;
  float forwardTargetY2 =
      std::max(static_cast<float>(roi.y),
               horizonY2 + (frameSize.height - horizonY2) * 0.20f);
  best.end.y = static_cast<int>(forwardTargetY2);
  Path result = buildPerspectivePath(best.end, roi, frameSize, config);
  result.width = best.width;
  result.score = best.score;
  result.type = best.type;
  result.isSafe = best.isSafe;

  // Apply temporal smoothing
  smoothPath(result);

  return {result};
}

} // namespace FlightPath
