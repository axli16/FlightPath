#include "VideoProcessor.h"
#include <iostream>

namespace FlightPath {

VideoProcessor::VideoProcessor()
    : fps_(0.0), frameWidth_(0), frameHeight_(0), 
      totalFrames_(0), currentFrame_(0), writerInitialized_(false) {
}

VideoProcessor::~VideoProcessor() {
    release();
}

bool VideoProcessor::open(const std::string& videoPath) {
    capture_.open(videoPath);
    
    if (!capture_.isOpened()) {
        std::cerr << "Error: Could not open video file: " << videoPath << std::endl;
        return false;
    }
    
    // Get video properties
    fps_ = capture_.get(cv::CAP_PROP_FPS);
    frameWidth_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
    frameHeight_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
    totalFrames_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_COUNT));
    currentFrame_ = 0;
    
    std::cout << "Video opened successfully:" << std::endl;
    std::cout << "  Resolution: " << frameWidth_ << "x" << frameHeight_ << std::endl;
    std::cout << "  FPS: " << fps_ << std::endl;
    std::cout << "  Total frames: " << totalFrames_ << std::endl;
    
    return true;
}

bool VideoProcessor::readFrame(cv::Mat& frame) {
    if (!capture_.isOpened()) {
        return false;
    }
    
    bool success = capture_.read(frame);
    if (success) {
        currentFrame_++;
    }
    
    return success;
}

void VideoProcessor::displayFrame(const cv::Mat& frame, const std::string& windowName) {
    if (frame.empty()) {
        return;
    }
    
    cv::imshow(windowName, frame);
}

bool VideoProcessor::initWriter(const std::string& outputPath, double fps, cv::Size frameSize) {
    if (outputPath.empty()) {
        return false;
    }
    
    // Use MP4V codec for .mp4 files
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    
    writer_.open(outputPath, fourcc, fps, frameSize);
    
    if (!writer_.isOpened()) {
        std::cerr << "Error: Could not open video writer: " << outputPath << std::endl;
        return false;
    }
    
    writerInitialized_ = true;
    std::cout << "Output video writer initialized: " << outputPath << std::endl;
    
    return true;
}

void VideoProcessor::writeFrame(const cv::Mat& frame) {
    if (writerInitialized_ && !frame.empty()) {
        writer_.write(frame);
    }
}

void VideoProcessor::release() {
    if (capture_.isOpened()) {
        capture_.release();
    }
    
    if (writerInitialized_) {
        writer_.release();
        writerInitialized_ = false;
    }
    
    cv::destroyAllWindows();
}

} // namespace FlightPath
