#include "Visualizer.h"
#include <cmath>
#include <iomanip>
#include <sstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace FlightPath {

Visualizer::Visualizer() {}

Visualizer::~Visualizer() {}

cv::Scalar Visualizer::getPathColor(const Path &path,
                                    const VisualConfig &config) {
  switch (path.type) {
  case Path::Type::SAFE:
    return config.hudColorPrimary; // Cyan for safe
  case Path::Type::TIGHT:
    return config.hudColorWarning; // Amber for tight
  case Path::Type::BLOCKED:
    return config.hudColorDanger; // Red for blocked
  default:
    return config.hudColorPrimary;
  }
}

// ==========================================================================
// Detection Rendering (futuristic bounding boxes)
// ==========================================================================
void Visualizer::drawDetections(cv::Mat &frame,
                                const std::vector<Detection> &detections,
                                const VisualConfig &config) {
  if (!config.showBoundingBoxes)
    return;

  for (const auto &det : detections) {
    cv::Rect box = det.boundingBox;

    // Futuristic corner brackets instead of full rectangles
    int cornerLen = std::min(box.width, box.height) / 4;
    cornerLen = std::max(10, cornerLen);
    cv::Scalar color = config.hudColorWarning; // Amber for detected vehicles

    int t = 2; // thickness

    // Top-left corner
    cv::line(frame, cv::Point(box.x, box.y),
             cv::Point(box.x + cornerLen, box.y), color, t, cv::LINE_AA);
    cv::line(frame, cv::Point(box.x, box.y),
             cv::Point(box.x, box.y + cornerLen), color, t, cv::LINE_AA);

    // Top-right corner
    cv::line(frame, cv::Point(box.x + box.width, box.y),
             cv::Point(box.x + box.width - cornerLen, box.y), color, t,
             cv::LINE_AA);
    cv::line(frame, cv::Point(box.x + box.width, box.y),
             cv::Point(box.x + box.width, box.y + cornerLen), color, t,
             cv::LINE_AA);

    // Bottom-left corner
    cv::line(frame, cv::Point(box.x, box.y + box.height),
             cv::Point(box.x + cornerLen, box.y + box.height), color, t,
             cv::LINE_AA);
    cv::line(frame, cv::Point(box.x, box.y + box.height),
             cv::Point(box.x, box.y + box.height - cornerLen), color, t,
             cv::LINE_AA);

    // Bottom-right corner
    cv::line(frame, cv::Point(box.x + box.width, box.y + box.height),
             cv::Point(box.x + box.width - cornerLen, box.y + box.height),
             color, t, cv::LINE_AA);
    cv::line(frame, cv::Point(box.x + box.width, box.y + box.height),
             cv::Point(box.x + box.width, box.y + box.height - cornerLen),
             color, t, cv::LINE_AA);

    // Thin dashed outline (draw every other segment)
    for (int i = 0; i < box.width; i += 8) {
      if ((i / 4) % 2 == 0) {
        int x1 = box.x + i;
        int x2 = std::min(box.x + i + 4, box.x + box.width);
        cv::line(frame, cv::Point(x1, box.y), cv::Point(x2, box.y),
                 cv::Scalar(color[0] * 0.4, color[1] * 0.4, color[2] * 0.4), 1,
                 cv::LINE_AA);
        cv::line(frame, cv::Point(x1, box.y + box.height),
                 cv::Point(x2, box.y + box.height),
                 cv::Scalar(color[0] * 0.4, color[1] * 0.4, color[2] * 0.4), 1,
                 cv::LINE_AA);
      }
    }

    // Label with HUD styling
    if (config.showConfidence) {
      std::stringstream label;
      label << det.className << " " << std::fixed << std::setprecision(0)
            << (det.confidence * 100) << "%";

      cv::Point labelPos(box.x, box.y - 8);
      if (labelPos.y < 15)
        labelPos.y = box.y + 18;

      // Glowing text effect: draw larger dim text behind, then crisp foreground
      cv::putText(frame, label.str(), labelPos, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                  cv::Scalar(color[0] * 0.3, color[1] * 0.3, color[2] * 0.3), 3,
                  cv::LINE_AA);
      cv::putText(frame, label.str(), labelPos, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                  color, 1, cv::LINE_AA);
    }
  }
}

// ==========================================================================
// HUD Path Rendering — the main Iron Man navigation overlay
// ==========================================================================
void Visualizer::drawHUDPath(cv::Mat &frame, const std::vector<Path> &paths,
                             const VisualConfig &config) {
  if (paths.empty())
    return;

  const Path &path = paths[0];
  if (path.type == Path::Type::BLOCKED)
    return;
  if (path.waypoints.size() < 2)
    return;

  cv::Scalar color = getPathColor(path, config);
  float frameHeight = static_cast<float>(frame.rows);

  // --- Build perspective-tapered path polygon ---
  // Path should look like it's projected onto the road surface
  std::vector<cv::Point> leftEdge, rightEdge;
  float baseHalfWidth = frame.cols * 0.07f; // Wider at the near end
  float minHalfWidth = 2.0f;

  // Reference: the path starts at ~55% frame height, not the bottom
  float pathStartY = static_cast<float>(path.waypoints.front().y);
  float pathEndY = static_cast<float>(path.waypoints.back().y);
  float pathSpan = pathStartY - pathEndY; // positive (start is lower)
  if (pathSpan < 1.0f)
    pathSpan = 1.0f;

  for (size_t i = 0; i < path.waypoints.size(); ++i) {
    const cv::Point &wp = path.waypoints[i];

    // t = 0 at path start (near), t = 1 at path end (far/horizon)
    float t = (pathStartY - static_cast<float>(wp.y)) / pathSpan;
    t = std::max(0.0f, std::min(1.0f, t));

    // Quartic taper for strong road-surface perspective convergence
    float taperedWidth = baseHalfWidth * (1.0f - t * t * t * t);
    taperedWidth = std::max(minHalfWidth, taperedWidth);

    leftEdge.push_back(cv::Point(wp.x - static_cast<int>(taperedWidth), wp.y));
    rightEdge.push_back(cv::Point(wp.x + static_cast<int>(taperedWidth), wp.y));
  }

  // Combine into polygon
  std::vector<cv::Point> polygon;
  polygon.insert(polygon.end(), leftEdge.begin(), leftEdge.end());
  polygon.insert(polygon.end(), rightEdge.rbegin(), rightEdge.rend());

  // --- 1. Semi-transparent filled path polygon ---
  {
    // Optimization: Clone only the bounding rect (ROI) instead of full frame to
    // reduce memory allocation overhead and blending bandwidth.
    cv::Rect roi = cv::boundingRect(polygon);
    roi &= cv::Rect(0, 0, frame.cols, frame.rows);

    if (roi.area() > 0) {
      cv::Mat overlay = frame(roi).clone();

      // Shift polygon points to ROI-relative coordinates
      std::vector<cv::Point> shiftedPolygon;
      for (const auto &pt : polygon) {
        shiftedPolygon.push_back(cv::Point(pt.x - roi.x, pt.y - roi.y));
      }
      std::vector<std::vector<cv::Point>> polys = {shiftedPolygon};

      // Inner fill with primary color
      cv::fillPoly(overlay, polys, color, cv::LINE_AA);

      // Blend back into original frame using the same ROI
      cv::addWeighted(overlay, config.hudPathAlpha, frame(roi),
                      1.0f - config.hudPathAlpha, 0, frame(roi));
    }
  }

  // --- 2. Glow effect on edges ---
  drawGlowEffect(frame, polygon, color, config);

  // --- 3. Scanline grid effect ---
  if (config.hudShowScanlines) {
    drawScanlines(frame, polygon, config);
  }

  // --- 4. Animated chevron arrows ---
  drawChevrons(frame, path.waypoints, config);

  // --- 5. Thin centerline with fade ---
  for (size_t i = 0; i + 1 < path.waypoints.size(); ++i) {
    float t = static_cast<float>(i) / path.waypoints.size();
    float alpha = 1.0f - t * 0.7f;
    cv::Scalar lineColor(color[0] * alpha, color[1] * alpha, color[2] * alpha);
    cv::line(frame, path.waypoints[i], path.waypoints[i + 1], lineColor, 1,
             cv::LINE_AA);
  }

  // --- 6. Arrowhead at the target end ---
  if (path.waypoints.size() >= 2) {
    cv::Point tip = path.waypoints.back();
    cv::Point prev = path.waypoints[path.waypoints.size() - 2];

    float dx = static_cast<float>(tip.x - prev.x);
    float dy = static_cast<float>(tip.y - prev.y);
    float len = std::sqrt(dx * dx + dy * dy);
    if (len > 0.0f) {
      dx /= len;
      dy /= len;

      float arrowLen = 14.0f;
      float arrowSpread = 8.0f;
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
}

// ==========================================================================
// Animated Chevrons — V-shaped arrows flowing along the path
// ==========================================================================
void Visualizer::drawChevrons(cv::Mat &frame,
                              const std::vector<cv::Point> &waypoints,
                              const VisualConfig &config) {
  if (waypoints.size() < 2)
    return;

  int numChevrons = config.hudChevronCount;
  float frameHeight = static_cast<float>(frame.rows);

  // Animation offset — chevrons scroll along the path over time
  float animOffset = std::fmod(static_cast<float>(frameCounter_) *
                                   config.hudAnimationSpeed * 0.02f,
                               1.0f);

  // Compute total path length for even spacing
  float totalLen = 0;
  std::vector<float> segLengths;
  for (size_t i = 0; i + 1 < waypoints.size(); ++i) {
    float dx = static_cast<float>(waypoints[i + 1].x - waypoints[i].x);
    float dy = static_cast<float>(waypoints[i + 1].y - waypoints[i].y);
    float segLen = std::sqrt(dx * dx + dy * dy);
    segLengths.push_back(segLen);
    totalLen += segLen;
  }
  if (totalLen < 1.0f)
    return;

  for (int c = 0; c < numChevrons; ++c) {
    // Position along path [0,1], animated
    float t = std::fmod(static_cast<float>(c) / numChevrons + animOffset, 1.0f);

    // Find the waypoint segment at this t
    float targetDist = t * totalLen;
    float accumulated = 0;
    cv::Point chevronCenter;
    float dirX = 0, dirY = -1; // default: pointing up

    for (size_t i = 0; i < segLengths.size(); ++i) {
      if (accumulated + segLengths[i] >= targetDist) {
        float segT = (targetDist - accumulated) / (segLengths[i] + 1e-9f);
        chevronCenter.x = static_cast<int>(
            waypoints[i].x + segT * (waypoints[i + 1].x - waypoints[i].x));
        chevronCenter.y = static_cast<int>(
            waypoints[i].y + segT * (waypoints[i + 1].y - waypoints[i].y));

        // Direction vector along path at this point
        dirX = static_cast<float>(waypoints[i + 1].x - waypoints[i].x);
        dirY = static_cast<float>(waypoints[i + 1].y - waypoints[i].y);
        float dl = std::sqrt(dirX * dirX + dirY * dirY);
        if (dl > 0) {
          dirX /= dl;
          dirY /= dl;
        }
        break;
      }
      accumulated += segLengths[i];
    }

    // Perspective scaling: chevrons get smaller toward the top of the frame
    float perspT =
        static_cast<float>(frame.rows - chevronCenter.y) / frameHeight;
    perspT = std::max(0.0f, std::min(1.0f, perspT));
    float scale = 1.0f - perspT * perspT * 0.8f;
    scale = std::max(0.15f, scale);

    // Chevron V-shape dimensions
    float chevWidth = frame.cols * 0.03f * scale;
    float chevHeight = frame.cols * 0.015f * scale;

    // Perpendicular direction
    float perpX = -dirY;
    float perpY = dirX;

    // V-shape: tip at center, two wings extending back and outward
    cv::Point tip(chevronCenter.x + static_cast<int>(dirX * chevHeight * 0.5f),
                  chevronCenter.y + static_cast<int>(dirY * chevHeight * 0.5f));
    cv::Point leftWing(chevronCenter.x - static_cast<int>(dirX * chevHeight +
                                                          perpX * chevWidth),
                       chevronCenter.y - static_cast<int>(dirY * chevHeight +
                                                          perpY * chevWidth));
    cv::Point rightWing(chevronCenter.x - static_cast<int>(dirX * chevHeight -
                                                           perpX * chevWidth),
                        chevronCenter.y - static_cast<int>(dirY * chevHeight -
                                                           perpY * chevWidth));

    // Alpha based on position (fade near ends) and perspective
    float alpha = scale * 0.9f;
    // Fade out at the extremes of the path
    float edgeFade = 1.0f - std::abs(t - 0.5f) * 1.6f;
    edgeFade = std::max(0.1f, std::min(1.0f, edgeFade));
    alpha *= edgeFade;

    cv::Scalar chevColor = config.hudColorPrimary;
    chevColor[0] *= alpha;
    chevColor[1] *= alpha;
    chevColor[2] *= alpha;

    // Draw chevron as two lines forming a V
    int thickness = std::max(1, static_cast<int>(2.5f * scale));
    cv::line(frame, tip, leftWing, chevColor, thickness, cv::LINE_AA);
    cv::line(frame, tip, rightWing, chevColor, thickness, cv::LINE_AA);
  }
}

// ==========================================================================
// Glow Effect — multi-layer outline with decreasing opacity
// ==========================================================================
void Visualizer::drawGlowEffect(cv::Mat &frame,
                                const std::vector<cv::Point> &polygon,
                                const cv::Scalar &color,
                                const VisualConfig &config) {
  if (polygon.size() < 3)
    return;

  std::vector<std::vector<cv::Point>> polys = {polygon};

  // Draw outer glow layers (thick, dim → thin, bright)
  int glowLayers = 4;
  for (int layer = glowLayers; layer >= 0; --layer) {
    int thickness = 1 + layer * 2;
    float alpha = config.hudGlowIntensity *
                  (1.0f - static_cast<float>(layer) / (glowLayers + 1));
    alpha = std::max(0.05f, alpha);

    cv::Scalar glowColor(color[0] * alpha, color[1] * alpha, color[2] * alpha);

    // Use secondary color for outer layers
    if (layer > 1) {
      glowColor = cv::Scalar(config.hudColorSecondary[0] * alpha * 0.5f,
                             config.hudColorSecondary[1] * alpha * 0.5f,
                             config.hudColorSecondary[2] * alpha * 0.5f);
    }

    cv::polylines(frame, polys, true, glowColor, thickness, cv::LINE_AA);
  }
}

// ==========================================================================
// Scanlines — horizontal holographic lines scrolling across the path
// ==========================================================================
void Visualizer::drawScanlines(cv::Mat &frame,
                               const std::vector<cv::Point> &polygon,
                               const VisualConfig &config) {
  if (polygon.size() < 3)
    return;

  // Get bounding rect to limit scan area
  cv::Rect bounds = cv::boundingRect(polygon);
  bounds &= cv::Rect(0, 0, frame.cols, frame.rows);

  if (bounds.area() <= 0)
    return;

  // Optimization: Allocate mask only for the polygon's bounding rectangle
  // rather than the full frame size to eliminate massive per-frame allocations.
  cv::Mat mask = cv::Mat::zeros(bounds.size(), CV_8UC1);

  // Shift polygon coordinates to be bounds-relative
  std::vector<cv::Point> shiftedPolygon;
  for (const auto &pt : polygon) {
    shiftedPolygon.push_back(cv::Point(pt.x - bounds.x, pt.y - bounds.y));
  }
  std::vector<std::vector<cv::Point>> polys = {shiftedPolygon};
  cv::fillPoly(mask, polys, cv::Scalar(255));

  // Animation: scanlines scroll upward
  int scrollOffset = (frameCounter_ * 2) % config.hudScanlineSpacing;

  cv::Scalar scanColor = config.hudColorPrimary;

  for (int y = bounds.y + scrollOffset; y < bounds.y + bounds.height;
       y += config.hudScanlineSpacing) {
    if (y < 0 || y >= frame.rows)
      continue;

    // Translate global y to mask-relative y
    int relY = y - bounds.y;
    // Explicit bounds check to prevent out-of-bounds segfaults
    if (relY < 0 || relY >= bounds.height)
      continue;

    // Find the left and right bounds of the polygon at this y
    int xMin = frame.cols, xMax = 0;

    // Use raw pointer access for hot inner loop instead of .at()
    const uchar *rowPtr = mask.ptr<uchar>(relY);

    for (int relX = 0; relX < bounds.width; ++relX) {
      if (rowPtr[relX] > 0) {
        int x = bounds.x + relX;
        if (x < frame.cols) {
          xMin = std::min(xMin, x);
          xMax = std::max(xMax, x);
        }
      }
    }

    if (xMax > xMin) {
      // Perspective-based alpha: dimmer at the top
      float perspT =
          static_cast<float>(frame.rows - y) / static_cast<float>(frame.rows);
      float alpha = 0.15f * (1.0f - perspT * 0.6f);

      cv::Scalar lineColor(scanColor[0] * alpha, scanColor[1] * alpha,
                           scanColor[2] * alpha);
      cv::line(frame, cv::Point(xMin, y), cv::Point(xMax, y), lineColor, 1,
               cv::LINE_AA);
    }
  }
}

// ==========================================================================
// Directional Arc — semicircle HUD at bottom center
// ==========================================================================
void Visualizer::drawDirectionArc(cv::Mat &frame,
                                  const std::vector<Path> &paths,
                                  const VisualConfig &config) {
  if (!config.hudShowDirectionArc)
    return;

  int centerX = frame.cols / 2;
  int centerY = frame.rows - 40;
  int radius = 60;

  // Draw arc background
  cv::ellipse(frame, cv::Point(centerX, centerY), cv::Size(radius, radius / 2),
              0, 180, 360,
              cv::Scalar(config.hudColorPrimary[0] * 0.15,
                         config.hudColorPrimary[1] * 0.15,
                         config.hudColorPrimary[2] * 0.15),
              2, cv::LINE_AA);

  // Tick marks around the arc
  for (int deg = 200; deg <= 340; deg += 20) {
    float rad = static_cast<float>(deg) * static_cast<float>(M_PI) / 180.0f;
    int innerR = radius - 5;
    int outerR = radius + 5;

    cv::Point inner(centerX + static_cast<int>(innerR * std::cos(rad)),
                    centerY + static_cast<int>((innerR / 2) * std::sin(rad)));
    cv::Point outer(centerX + static_cast<int>(outerR * std::cos(rad)),
                    centerY + static_cast<int>((outerR / 2) * std::sin(rad)));

    float tickAlpha = (deg == 260 || deg == 280) ? 0.6f : 0.25f;
    cv::Scalar tickColor(config.hudColorPrimary[0] * tickAlpha,
                         config.hudColorPrimary[1] * tickAlpha,
                         config.hudColorPrimary[2] * tickAlpha);
    cv::line(frame, inner, outer, tickColor, 1, cv::LINE_AA);
  }

  // Direction indicator triangle
  if (!paths.empty() && !paths[0].waypoints.empty()) {
    const Path &path = paths[0];

    // Compute steering angle from path deviation
    float steerAngle = 0; // 0 = straight, negative = left, positive = right
    if (path.waypoints.size() >= 2) {
      // Look at mid-path deviation from center
      int midIdx = static_cast<int>(path.waypoints.size()) / 2;
      float deviation =
          static_cast<float>(path.waypoints[midIdx].x - frame.cols / 2);
      float maxDev = static_cast<float>(frame.cols) / 4.0f;
      steerAngle = deviation / maxDev;
      steerAngle = std::max(-1.0f, std::min(1.0f, steerAngle));
    }

    // Map steer angle to arc position (180° to 360°, center at 270°)
    float indicatorDeg = 270.0f + steerAngle * 60.0f;
    float indicatorRad = indicatorDeg * static_cast<float>(M_PI) / 180.0f;

    cv::Point indicatorPos(
        centerX + static_cast<int>((radius - 12) * std::cos(indicatorRad)),
        centerY +
            static_cast<int>(((radius - 12) / 2) * std::sin(indicatorRad)));

    // Draw indicator as a small filled triangle
    int triSize = 6;
    std::vector<cv::Point> tri = {
        cv::Point(indicatorPos.x, indicatorPos.y - triSize),
        cv::Point(indicatorPos.x - triSize, indicatorPos.y + triSize),
        cv::Point(indicatorPos.x + triSize, indicatorPos.y + triSize)};
    std::vector<std::vector<cv::Point>> triPolys = {tri};
    cv::fillPoly(frame, triPolys, config.hudColorPrimary, cv::LINE_AA);

    // Draw subtle glow around indicator
    cv::circle(frame, indicatorPos, triSize + 3,
               cv::Scalar(config.hudColorPrimary[0] * 0.2,
                          config.hudColorPrimary[1] * 0.2,
                          config.hudColorPrimary[2] * 0.2),
               1, cv::LINE_AA);
  }
}

// ==========================================================================
// HUD Info Panel — modernized status display
// ==========================================================================
void Visualizer::drawHUDInfoPanel(cv::Mat &frame,
                                  const std::vector<Detection> &detections,
                                  const std::vector<Path> &paths,
                                  const VisualConfig &config) {
  int panelW = 220;
  int panelH = 90;
  int panelX = 15;
  int panelY = 15;

  if (frame.cols < panelW + 30 || frame.rows < panelH + 30)
    return;

  // Semi-transparent dark panel with border
  // Optimization: Clone only the ROI of the info panel instead of the full
  // frame
  cv::Rect roi(panelX, panelY, panelW, panelH);
  roi &= cv::Rect(0, 0, frame.cols, frame.rows);

  if (roi.area() > 0) {
    cv::Mat overlay = frame(roi).clone();

    // Draw rectangle in ROI-relative coordinates
    cv::rectangle(overlay, cv::Point(0, 0), cv::Point(roi.width, roi.height),
                  cv::Scalar(10, 10, 10), cv::FILLED);

    // Blend back into original frame
    cv::addWeighted(overlay, 0.7, frame(roi), 0.3, 0, frame(roi));
  }

  // HUD-style border (thin cyan lines)
  cv::rectangle(frame, cv::Point(panelX, panelY),
                cv::Point(panelX + panelW, panelY + panelH),
                cv::Scalar(config.hudColorPrimary[0] * 0.4,
                           config.hudColorPrimary[1] * 0.4,
                           config.hudColorPrimary[2] * 0.4),
                1, cv::LINE_AA);

  // Corner accents
  int accentLen = 12;
  cv::Scalar accentColor = config.hudColorPrimary;
  // Top-left
  cv::line(frame, cv::Point(panelX, panelY),
           cv::Point(panelX + accentLen, panelY), accentColor, 2);
  cv::line(frame, cv::Point(panelX, panelY),
           cv::Point(panelX, panelY + accentLen), accentColor, 2);
  // Top-right
  cv::line(frame, cv::Point(panelX + panelW, panelY),
           cv::Point(panelX + panelW - accentLen, panelY), accentColor, 2);
  cv::line(frame, cv::Point(panelX + panelW, panelY),
           cv::Point(panelX + panelW, panelY + accentLen), accentColor, 2);

  int textY = panelY + 25;
  int lineH = 22;

  // Status text
  cv::Scalar statusColor;
  std::string statusText;

  if (!paths.empty() && paths[0].isSafe) {
    statusText = "PATH CLEAR";
    statusColor = config.hudColorPrimary;
  } else if (!paths.empty()) {
    statusText = "TIGHT PASSAGE";
    statusColor = config.hudColorWarning;
  } else {
    statusText = "NO PATH";
    statusColor = config.hudColorDanger;
  }

  // Glow behind text
  cv::putText(frame, statusText, cv::Point(panelX + 12, textY),
              cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(statusColor[0] * 0.3, statusColor[1] * 0.3,
                         statusColor[2] * 0.3),
              3, cv::LINE_AA);
  cv::putText(frame, statusText, cv::Point(panelX + 12, textY),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, statusColor, 1, cv::LINE_AA);

  textY += lineH;

  // Detection count
  std::stringstream detText;
  detText << "OBJ: " << detections.size();
  cv::putText(frame, detText.str(), cv::Point(panelX + 12, textY),
              cv::FONT_HERSHEY_SIMPLEX, 0.4,
              cv::Scalar(config.hudColorPrimary[0] * 0.7,
                         config.hudColorPrimary[1] * 0.7,
                         config.hudColorPrimary[2] * 0.7),
              1, cv::LINE_AA);

  // Confidence
  if (!paths.empty()) {
    std::stringstream confText;
    confText << "CONF: " << std::fixed << std::setprecision(0)
             << (paths[0].score * 100) << "%";
    cv::putText(frame, confText.str(), cv::Point(panelX + 100, textY),
                cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(config.hudColorPrimary[0] * 0.7,
                           config.hudColorPrimary[1] * 0.7,
                           config.hudColorPrimary[2] * 0.7),
                1, cv::LINE_AA);
  }

  textY += lineH;

  // Animated scan bar
  int barWidth = panelW - 24;
  int barX = panelX + 12;
  float scanPos = std::fmod(static_cast<float>(frameCounter_) * 0.03f, 1.0f);
  int scanX = barX + static_cast<int>(scanPos * barWidth);
  cv::line(frame, cv::Point(barX, textY), cv::Point(barX + barWidth, textY),
           cv::Scalar(config.hudColorPrimary[0] * 0.15,
                      config.hudColorPrimary[1] * 0.15,
                      config.hudColorPrimary[2] * 0.15),
           2);
  cv::line(frame, cv::Point(scanX - 10, textY), cv::Point(scanX + 10, textY),
           cv::Scalar(config.hudColorPrimary[0] * 0.8,
                      config.hudColorPrimary[1] * 0.8,
                      config.hudColorPrimary[2] * 0.8),
           2);
}

// ==========================================================================
// ROI Drawing
// ==========================================================================
void Visualizer::drawROI(cv::Mat &frame, const DetectionConfig &detectionConfig,
                         const VisualConfig &visualConfig,
                         const TrapezoidROI &trapezoidROI) {
  if (!visualConfig.showROI)
    return;

  if (trapezoidROI.valid) {
    std::vector<cv::Point> poly = trapezoidROI.asVector();

    // Very subtle ROI overlay in HUD style
    // Optimization: Clone only the bounding rect (ROI) instead of full frame
    cv::Rect roi = cv::boundingRect(poly);
    roi &= cv::Rect(0, 0, frame.cols, frame.rows);

    if (roi.area() > 0) {
      cv::Mat overlay = frame(roi).clone();

      // Shift poly points to ROI-relative coordinates
      std::vector<cv::Point> shiftedPoly;
      for (const auto &pt : poly) {
        shiftedPoly.push_back(cv::Point(pt.x - roi.x, pt.y - roi.y));
      }
      std::vector<std::vector<cv::Point>> polys = {shiftedPoly};

      cv::fillPoly(overlay, polys, cv::Scalar(0, 40, 0), cv::LINE_AA);
      cv::addWeighted(overlay, 0.08, frame(roi), 0.92, 0, frame(roi));
    }

    // Thin outline
    std::vector<std::vector<cv::Point>> originalPolys = {poly};
    cv::polylines(frame, originalPolys, true,
                  cv::Scalar(visualConfig.hudColorPrimary[0] * 0.2,
                             visualConfig.hudColorPrimary[1] * 0.2,
                             visualConfig.hudColorPrimary[2] * 0.2),
                  1, cv::LINE_AA);
    return;
  }

  if (!detectionConfig.useROI)
    return;

  int roiX = static_cast<int>(detectionConfig.roiX * frame.cols);
  int roiY = static_cast<int>(detectionConfig.roiY * frame.rows);
  int roiWidth = static_cast<int>(detectionConfig.roiWidth * frame.cols);
  int roiHeight = static_cast<int>(detectionConfig.roiHeight * frame.rows);

  cv::Rect roi(roiX, roiY, roiWidth, roiHeight);
  roi &= cv::Rect(0, 0, frame.cols, frame.rows);

  cv::rectangle(frame, roi,
                cv::Scalar(visualConfig.hudColorPrimary[0] * 0.2,
                           visualConfig.hudColorPrimary[1] * 0.2,
                           visualConfig.hudColorPrimary[2] * 0.2),
                1, cv::LINE_AA);
}

// ==========================================================================
// Main draw entry point
// ==========================================================================
void Visualizer::draw(cv::Mat &frame, const std::vector<Detection> &detections,
                      const std::vector<Path> &paths,
                      const VisualConfig &visualConfig,
                      const DetectionConfig &detectionConfig,
                      const TrapezoidROI &trapezoidROI) {
  if (frame.empty())
    return;

  // Increment animation counter
  frameCounter_++;

  // Draw in order: ROI (subtle) → path (main HUD) → detections → HUD panel →
  // direction arc
  drawROI(frame, detectionConfig, visualConfig, trapezoidROI);
  drawHUDPath(frame, paths, visualConfig);
  drawDetections(frame, detections, visualConfig);
  drawHUDInfoPanel(frame, detections, paths, visualConfig);
  drawDirectionArc(frame, paths, visualConfig);
}

} // namespace FlightPath
