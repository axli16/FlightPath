#include "LaneDetector.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace FlightPath {

LaneDetector::LaneDetector() {}
LaneDetector::~LaneDetector() {}

// ---------------------------------------------------------------------------
// Build a default converging perspective trapezoid (straight road view)
// Bottom: wide (~80% of frame), Top: narrow (~10% of frame) centered
// ---------------------------------------------------------------------------
TrapezoidROI LaneDetector::buildDefaultROI(int frameW, int frameH) {
  TrapezoidROI roi;
  roi.valid = true;

  // Bottom edge: wide, covering most of the road area
  int bottomMargin = static_cast<int>(frameW * 0.10f); // 10% margin each side
  // Top edge: narrow, converging toward a vanishing point at center
  int topMargin = static_cast<int>(frameW * 0.42f); // 42% margin -> 16% width

  int yBottom = frameH - 1;
  int yTop = 0;

  roi.vertices[0] = cv::Point(bottomMargin, yBottom);          // Bottom-left
  roi.vertices[1] = cv::Point(topMargin, yTop);                // Top-left
  roi.vertices[2] = cv::Point(frameW - topMargin, yTop);       // Top-right
  roi.vertices[3] = cv::Point(frameW - bottomMargin, yBottom); // Bottom-right

  return roi;
}

// ---------------------------------------------------------------------------
// Color mask: isolate yellow and white lane markings in HSV space
// ---------------------------------------------------------------------------
cv::Mat LaneDetector::createColorMask(const cv::Mat &hsvFrame,
                                      const LaneConfig &config) {
  cv::Mat yellowMask, whiteMask;
  cv::inRange(hsvFrame, config.yellowLow, config.yellowHigh, yellowMask);
  cv::inRange(hsvFrame, config.whiteLow, config.whiteHigh, whiteMask);

  cv::Mat combined;
  cv::bitwise_or(yellowMask, whiteMask, combined);
  return combined;
}

// ---------------------------------------------------------------------------
// Classify raw Hough segments into left / right by slope and position
// ---------------------------------------------------------------------------
void LaneDetector::classifyLines(const std::vector<cv::Vec4i> &lines,
                                 int frameMidX, const LaneConfig &config,
                                 std::vector<cv::Vec4i> &leftLines,
                                 std::vector<cv::Vec4i> &rightLines) {
  leftLines.clear();
  rightLines.clear();

  for (const auto &l : lines) {
    int x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];

    if (x2 == x1)
      continue; // vertical — skip

    float slope = static_cast<float>(y2 - y1) / static_cast<float>(x2 - x1);

    // Filter out nearly-horizontal lines
    if (std::abs(slope) < config.minSlopeThreshold)
      continue;

    // In image coordinates, y grows downward:
    //   Left lane line: negative slope (goes from bottom-left to upper-right
    //   ish) Right lane line: positive slope (goes from bottom-right to
    //   upper-left ish)
    float midX = (x1 + x2) / 2.0f;

    if (slope < 0 && midX < frameMidX) {
      leftLines.push_back(l);
    } else if (slope > 0 && midX > frameMidX) {
      rightLines.push_back(l);
    }
  }
}

// ---------------------------------------------------------------------------
// Check if a group of segments forms a solid (continuous) line rather than
// a dashed/dotted line. Projects all segments onto the Y axis and measures
// what fraction of the Y range is covered by actual segments.
// Solid lines have high coverage (>= minSolidCoverage), dashed lines do not.
// ---------------------------------------------------------------------------
bool LaneDetector::isSolidLine(const std::vector<cv::Vec4i> &segments,
                               float minCoverage) {
  if (segments.empty())
    return false;

  // Find the total Y span of all segments
  int yMin = INT_MAX, yMax = INT_MIN;
  for (const auto &s : segments) {
    yMin = std::min({yMin, s[1], s[3]});
    yMax = std::max({yMax, s[1], s[3]});
  }

  int span = yMax - yMin;
  if (span < 10)
    return false; // too short to judge

  // Count unique Y pixels covered by any segment
  std::vector<bool> covered(span + 1, false);
  for (const auto &s : segments) {
    int sy1 = std::min(s[1], s[3]) - yMin;
    int sy2 = std::max(s[1], s[3]) - yMin;
    for (int y = sy1; y <= sy2; ++y) {
      covered[y] = true;
    }
  }

  int coveredCount = 0;
  for (bool c : covered) {
    if (c)
      ++coveredCount;
  }

  float coverage = static_cast<float>(coveredCount) / static_cast<float>(span);
  return coverage >= minCoverage;
}

// ---------------------------------------------------------------------------
// Fit a group of segments into a single line via weighted-average
// slope/intercept. Then extrapolate to yBottom and yTop to get the two
// x-coordinates.
// ---------------------------------------------------------------------------
bool LaneDetector::fitLine(const std::vector<cv::Vec4i> &lines, int yBottom,
                           int yTop, int &xBottom, int &xTop) {
  if (lines.empty())
    return false;

  // Weighted average of slope and intercept, weighted by segment length
  double sumSlope = 0.0, sumIntercept = 0.0, sumWeight = 0.0;

  for (const auto &l : lines) {
    double x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
    if (std::abs(x2 - x1) < 1.0)
      continue;

    double slope = (y2 - y1) / (x2 - x1);
    double intercept = y1 - slope * x1;
    double length = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));

    sumSlope += slope * length;
    sumIntercept += intercept * length;
    sumWeight += length;
  }

  if (sumWeight < 1.0)
    return false;

  double avgSlope = sumSlope / sumWeight;
  double avgIntercept = sumIntercept / sumWeight;

  if (std::abs(avgSlope) < 0.01)
    return false; // near-horizontal — bad fit

  // x = (y - intercept) / slope
  xBottom = static_cast<int>((yBottom - avgIntercept) / avgSlope);
  xTop = static_cast<int>((yTop - avgIntercept) / avgSlope);

  return true;
}

// ---------------------------------------------------------------------------
// Temporal smoothing via exponential moving average on the 4 vertices
// ---------------------------------------------------------------------------
TrapezoidROI LaneDetector::smoothROI(const TrapezoidROI &current, float alpha) {
  if (!hasSmoothedROI_ || !smoothedROI_.valid) {
    // First valid detection — accept as-is
    return current;
  }

  TrapezoidROI result;
  result.valid = true;

  for (int i = 0; i < 4; ++i) {
    result.vertices[i].x =
        static_cast<int>(alpha * current.vertices[i].x +
                         (1.0f - alpha) * smoothedROI_.vertices[i].x);
    result.vertices[i].y =
        static_cast<int>(alpha * current.vertices[i].y +
                         (1.0f - alpha) * smoothedROI_.vertices[i].y);
  }

  return result;
}

// ---------------------------------------------------------------------------
// Main entry point: detect lanes → build trapezoid
// ---------------------------------------------------------------------------
TrapezoidROI LaneDetector::detectLanes(const cv::Mat &frame,
                                       const LaneConfig &config) {
  TrapezoidROI roi;
  roi.valid = false;

  if (!config.enabled || frame.empty())
    return roi;

  int frameH = frame.rows;
  int frameW = frame.cols;

  // Search region: where we look for line segments (lower portion of frame)
  int searchYTop = static_cast<int>(frameH * config.roiTopRatio);
  int searchYBottom = static_cast<int>(frameH * config.roiBottomRatio);

  // Extrapolation range: the trapezoid will span the FULL frame height
  int extrapYTop = 0;
  int extrapYBottom = frameH - 1;

  // Crop to the search region where lanes are visible
  cv::Rect searchRegion(0, searchYTop, frameW, searchYBottom - searchYTop);
  searchRegion &= cv::Rect(0, 0, frameW, frameH);
  cv::Mat regionFrame = frame(searchRegion);

  // --- Step 1a: Color-based edges (painted lane lines) ---
  cv::Mat edges;
  if (config.useColorFilter) {
    cv::Mat hsv;
    cv::cvtColor(regionFrame, hsv, cv::COLOR_BGR2HSV);
    cv::Mat colorMask = createColorMask(hsv, config);

    cv::Mat colorBlurred;
    cv::GaussianBlur(colorMask, colorBlurred,
                     cv::Size(config.gaussianKernel, config.gaussianKernel), 0);
    cv::Mat colorEdges;
    cv::Canny(colorBlurred, colorEdges, config.cannyLow, config.cannyHigh);

    // --- Step 1b: Grayscale edges (curbs, concrete boundaries) ---
    cv::Mat gray;
    cv::cvtColor(regionFrame, gray, cv::COLOR_BGR2GRAY);
    cv::Mat grayBlurred;
    cv::GaussianBlur(gray, grayBlurred,
                     cv::Size(config.gaussianKernel, config.gaussianKernel), 0);
    cv::Mat grayEdges;
    cv::Canny(grayBlurred, grayEdges, config.cannyLow, config.cannyHigh);

    // Combine both edge maps
    cv::bitwise_or(colorEdges, grayEdges, edges);
  } else {
    // Curb/barrier mode: only grayscale edges (no color filtering)
    cv::Mat gray;
    cv::cvtColor(regionFrame, gray, cv::COLOR_BGR2GRAY);
    cv::Mat grayBlurred;
    cv::GaussianBlur(gray, grayBlurred,
                     cv::Size(config.gaussianKernel, config.gaussianKernel), 0);
    cv::Canny(grayBlurred, edges, config.cannyLow, config.cannyHigh);
  }

  // --- Step 3: Hough line detection ---
  std::vector<cv::Vec4i> lines;
  cv::HoughLinesP(edges, lines, config.houghRho, config.houghTheta,
                  config.houghThreshold, config.houghMinLineLen,
                  config.houghMaxLineGap);

  if (lines.empty()) {
    if (hasSmoothedROI_ && smoothedROI_.valid)
      return smoothedROI_;
    // No edges found — return default converging perspective
    return buildDefaultROI(frameW, frameH);
  }

  // --- Step 4: Adjust coordinates back to full-frame space ---
  for (auto &l : lines) {
    l[1] += searchYTop;
    l[3] += searchYTop;
  }

  // --- Step 5: Classify into left and right ---
  int frameMidX = frameW / 2;
  std::vector<cv::Vec4i> leftLines, rightLines;
  classifyLines(lines, frameMidX, config, leftLines, rightLines);

  // --- Step 6: Filter to solid lines only ---
  // Dashed lines have low Y-coverage, solid lines have high coverage
  bool leftIsSolid = isSolidLine(leftLines, config.minSolidCoverage);
  bool rightIsSolid = isSolidLine(rightLines, config.minSolidCoverage);

  if (!leftIsSolid)
    leftLines.clear();
  if (!rightIsSolid)
    rightLines.clear();

  // --- Step 7: Fit and extrapolate each side to FULL frame height ---
  int leftXBottom, leftXTop, rightXBottom, rightXTop;
  bool hasLeft =
      fitLine(leftLines, extrapYBottom, extrapYTop, leftXBottom, leftXTop);
  bool hasRight =
      fitLine(rightLines, extrapYBottom, extrapYTop, rightXBottom, rightXTop);

  if (!hasLeft && !hasRight) {
    // No barriers detected on either side
    if (hasSmoothedROI_ && smoothedROI_.valid)
      return smoothedROI_;
    // Fall back to default converging road perspective
    return buildDefaultROI(frameW, frameH);
  }

  // If only one side detected, mirror from center to estimate the other
  if (hasLeft && !hasRight) {
    rightXBottom = 2 * frameMidX - leftXBottom;
    rightXTop = 2 * frameMidX - leftXTop;
    hasRight = true;
  } else if (!hasLeft && hasRight) {
    leftXBottom = 2 * frameMidX - rightXBottom;
    leftXTop = 2 * frameMidX - rightXTop;
    hasLeft = true;
  }

  // --- Step 8: Clamp x values to frame bounds ---
  auto clampX = [&](int x) { return std::max(0, std::min(frameW - 1, x)); };

  // Build trapezoid spanning full frame: BL, TL, TR, BR
  roi.vertices[0] =
      cv::Point(clampX(leftXBottom), extrapYBottom);          // Bottom-left
  roi.vertices[1] = cv::Point(clampX(leftXTop), extrapYTop);  // Top-left
  roi.vertices[2] = cv::Point(clampX(rightXTop), extrapYTop); // Top-right
  roi.vertices[3] =
      cv::Point(clampX(rightXBottom), extrapYBottom); // Bottom-right
  roi.valid = true;

  // --- Step 9: Temporal smoothing ---
  roi = smoothROI(roi, config.smoothingAlpha);
  smoothedROI_ = roi;
  hasSmoothedROI_ = true;

  return roi;
}

} // namespace FlightPath
