#ifndef VIDEO_PROCESSOR_H
#define VIDEO_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include "Config.h"

namespace FlightPath {

/**
 * @brief Handles video input/output operations
 * 
 * Manages reading from video files, displaying frames,
 * and optionally writing output videos.
 */
class VideoProcessor {
public:
    VideoProcessor();
    ~VideoProcessor();
    
    /**
     * @brief Open a video file for processing
     * @param videoPath Path to the video file
     * @return true if successful, false otherwise
     */
    bool open(const std::string& videoPath);
    
    /**
     * @brief Read the next frame from the video
     * @param frame Output frame
     * @return true if frame was read successfully, false if end of video
     */
    bool readFrame(cv::Mat& frame);
    
    /**
     * @brief Display a frame in a window
     * @param frame Frame to display
     * @param windowName Name of the display window
     */
    void displayFrame(const cv::Mat& frame, const std::string& windowName = "FlightPath");
    
    /**
     * @brief Initialize video writer for saving output
     * @param outputPath Path for output video file
     * @param fps Frames per second
     * @param frameSize Size of output frames
     * @return true if successful, false otherwise
     */
    bool initWriter(const std::string& outputPath, double fps, cv::Size frameSize);
    
    /**
     * @brief Write a frame to the output video
     * @param frame Frame to write
     */
    void writeFrame(const cv::Mat& frame);
    
    /**
     * @brief Get video properties
     */
    double getFPS() const { return fps_; }
    int getFrameWidth() const { return frameWidth_; }
    int getFrameHeight() const { return frameHeight_; }
    int getTotalFrames() const { return totalFrames_; }
    int getCurrentFrame() const { return currentFrame_; }
    
    /**
     * @brief Check if video is opened
     */
    bool isOpened() const { return capture_.isOpened(); }
    
    /**
     * @brief Release resources
     */
    void release();
    
private:
    cv::VideoCapture capture_;
    cv::VideoWriter writer_;
    
    double fps_;
    int frameWidth_;
    int frameHeight_;
    int totalFrames_;
    int currentFrame_;
    
    bool writerInitialized_;
};

} // namespace FlightPath

#endif // VIDEO_PROCESSOR_H
