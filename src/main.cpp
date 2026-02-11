#include "Config.h"
#include "ObjectDetector.h"
#include "PathPlanner.h"
#include "VideoProcessor.h"
#include "Visualizer.h"
#include "utility/DroppingSafeQueue.h"
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <windows.h>

using namespace FlightPath;

void SetCurrentThreadName(const std::wstring &name) {
  // GetCurrentThread() gets the handle, name.c_str() gets the L"Text"
  HRESULT hr = SetThreadDescription(GetCurrentThread(), name.c_str());
  if (FAILED(hr)) {
    // Optional: Handle error if the name couldn't be set
  }
}

void printUsage(const char *programName) {
  std::cout << "FlightPath - Computer Vision Path Detection System\n\n";
  std::cout << "Usage: " << programName << " <video_path> [options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --model <path>      Path to YOLO weights file (default: "
               "models/yolov4.weights)\n";
  std::cout << "  --config <path>     Path to YOLO config file (default: "
               "models/yolov4.cfg)\n";
  std::cout << "  --names <path>      Path to class names file (default: "
               "models/coco.names)\n";
  std::cout << "  --output <path>     Save output video to file\n";
  std::cout << "  --confidence <val>  Detection confidence threshold 0-1 "
               "(default: 0.5)\n";
  std::cout << "  --gpu               Use GPU acceleration (requires "
               "CUDA-enabled OpenCV)\n";
  std::cout << "\nControls:\n";
  std::cout << "  SPACE  - Pause/Resume\n";
  std::cout << "  ESC    - Exit\n";
  std::cout << "  S      - Save current frame\n";
  std::cout << "\nExample:\n";
  std::cout << "  " << programName
            << " data/dashcam.mp4 --output output/result.mp4\n";
}

bool parseArguments(int argc, char *argv[], AppConfig &config) {
  if (argc < 2) {
    return false;
  }

  config.video.inputPath = argv[1];

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--model" && i + 1 < argc) {
      config.model.weightsPath = argv[++i];
    } else if (arg == "--config" && i + 1 < argc) {
      config.model.configPath = argv[++i];
    } else if (arg == "--names" && i + 1 < argc) {
      config.model.classNamesPath = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      config.video.outputPath = argv[++i];
      config.video.saveOutput = true;
    } else if (arg == "--confidence" && i + 1 < argc) {
      config.detection.confidenceThreshold = std::stof(argv[++i]);
    } else if (arg == "--gpu") {
      config.model.useGPU = true;
    } else if (arg == "--frameskip" && i + 1 < argc) {
      config.detection.frameSkip = std::stoi(argv[++i]);
    } else if (arg == "--input-size" && i + 1 < argc) {
      int size = std::stoi(argv[++i]);
      config.detection.inputWidth = size;
      config.detection.inputHeight = size;
    } else if (arg == "--no-crop") {
      config.video.autoCrop = false;
    } else if (arg == "--max-crop-size" && i + 1 < argc) {
      int size = std::stoi(argv[++i]);
      config.video.maxCropWidth = size;
      config.video.maxCropHeight = size;
    } else if (arg == "--help" || arg == "-h") {
      return false;
    }
  }

  return true;
}

void processFrame(AppConfig &config, DroppingSafeQueue<cv::Mat> &frameQueue, 
                    DroppingSafeQueue<FrameData> &postProcessQueue, 
                    ObjectDetector &objectDetector, PathPlanner &pathPlanner){
    int count = 0;
    cv::Mat localFrame; // Use a local frame variable for this thread

    // Cache for frame skipping optimization
    std::vector<Detection> cachedDetections;
    std::vector<Path> cachedPaths;

    while (true) {
        frameQueue.pop(localFrame);

        std::cout << "Processing frame " << count << " (DETECTING)"
                  << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        // Detect objects
        cachedDetections = objectDetector.detect(localFrame, config.detection);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "  Detection took: " << elapsed.count() << " ms"
                  << std::endl;

        // Plan paths
        start = std::chrono::high_resolution_clock::now();
        cachedPaths = pathPlanner.findPaths(cachedDetections, localFrame.size(),
                                            config.path);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "  Path planning took: " << elapsed.count() << " ms"
            << std::endl;

      // Push frame with detections (either fresh or cached)
      postProcessQueue.push(
          FrameData{localFrame.clone(), cachedDetections, cachedPaths});

      for (int i = 0; i < config.detection.frameSkip - 1; i++) {
          frameQueue.pop(localFrame);
          postProcessQueue.push(FrameData{ localFrame.clone(), cachedDetections, cachedPaths });
          std::cout << "Processing frame " << count << " (CACHED)" << std::endl;
          
      }
      count+= config.detection.frameSkip;
    }
}

int main(int argc, char *argv[]) {
  std::cout << "=== FlightPath - Computer Vision Path Detection ==="
            << std::endl;
  std::cout << "Illegal autonomous driving path visualization\n" << std::endl;

  // Parse command line arguments
  AppConfig config;
  if (!parseArguments(argc, argv, config)) {
    printUsage(argv[0]);
    return 1;
  }

  // Initialize components
  VideoProcessor videoProcessor;
  ObjectDetector objectDetector1;
  ObjectDetector objectDetector2;
  PathPlanner pathPlanner1;
  PathPlanner pathPlanner2;
  Visualizer visualizer;

  DroppingSafeQueue<cv::Mat> frameQueue;
  DroppingSafeQueue<FrameData> postProcessQueue;

  // Load video
  std::cout << "\n[1/3] Loading video..." << std::endl;
  if (!videoProcessor.open(config.video.inputPath)) {
    std::cerr << "Failed to open video file: " << config.video.inputPath
              << std::endl;
    return 1;
  }

  // Load YOLO model
  std::cout << "\n[2/3] Loading YOLO model..." << std::endl;
  if (!objectDetector1.loadModel(config.model)) {
    std::cerr << "Failed to load YOLO model" << std::endl;
    return 1;
  }

  if (!objectDetector2.loadModel(config.model)) {
      std::cerr << "Failed to load YOLO model" << std::endl;
      return 1;
  }

  // Initialize output writer if requested
  if (config.video.saveOutput) {
    cv::Size frameSize(videoProcessor.getFrameWidth(),
                       videoProcessor.getFrameHeight());
    if (!videoProcessor.initWriter(config.video.outputPath,
                                   videoProcessor.getFPS(), frameSize)) {
      std::cerr << "Warning: Could not initialize output writer" << std::endl;
      config.video.saveOutput = false;
    }
  }

  std::cout << "\n[3/3] Processing video..." << std::endl;

  // Main processing loop
  cv::Mat frame;
  int frameCount = 0;
  int savedFrameCount = 0;
  int target_fps = 30;

  auto startTime = std::chrono::high_resolution_clock::now();
  int numFrames = 0;
  bool cropMessageShown = false; // Only show crop message once

  while (videoProcessor.readFrame(frame)) {
    // Auto-crop if enabled and frame is too large
    if (config.video.autoCrop) {
      cv::Mat croppedFrame = VideoProcessor::cropToCenter(
          frame, config.video.maxCropWidth, config.video.maxCropHeight);

      // Only show message on first crop
      if (!cropMessageShown && croppedFrame.data != frame.data) {
        cropMessageShown = true;
      }

      frameQueue.push(croppedFrame.clone());
    } else if (config.video.autoScale) {
      cv::resize(frame, frame, cv::Size(config.video.targetWidth, config.video.targetHeight));
      frameQueue.push(frame.clone());
    } else {
      frameQueue.push(frame.clone()); // Clone to ensure each frame is independent
    }
    numFrames++;
  }
  std::cout << "Finished reading all frames: " << numFrames << std::endl;

  std::thread detectorThread1([&]() {
    SetCurrentThreadName(L"ProcessFrame_Thread1");
    processFrame(config, frameQueue, postProcessQueue, objectDetector1, pathPlanner1);
   
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(15));

  std::thread detectorThread2([&]() {
      SetCurrentThreadName(L"ProcessFrame_Thread2");
      processFrame(config, frameQueue, postProcessQueue, objectDetector2, pathPlanner2);

      });
  // Wait for the processing frames can get a headstart 
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  std::thread drawThread([&]() {
    SetCurrentThreadName(L"DrawFrame_Thread");

    FrameData frameData;
    std::chrono::duration<double, std::milli> frame_duration(1000.0 / target_fps);
    
    while (true) {
      auto frame_start = std::chrono::high_resolution_clock::now();
      postProcessQueue.pop(frameData);
      
      visualizer.draw(frameData.frame, frameData.detections, frameData.paths,
                      config.visual);

      if (config.video.displayWindow) {
        videoProcessor.displayFrame(frameData.frame,
                                    "FlightPath - Path Detection");

        // CRITICAL: waitKey is needed for the window to update!
        int key = cv::waitKey(1);
        if (key == 27)
          config.shouldExit = true;
      }

      if (config.video.saveOutput) {
        videoProcessor.writeFrame(frameData.frame);
      }
      std::this_thread::sleep_until(frame_start + frame_duration);
    }
  });

  // while (videoProcessor.readFrame(frame)) {
  //     if (config.shouldExit) {
  //         break;
  //     }

  //     frameCount++;

  //     // Calculate FPS
  //     auto currentTime = std::chrono::high_resolution_clock::now();
  //     std::chrono::duration<double> elapsed = currentTime - startTime;
  //     double fps = frameCount / elapsed.count();

  //     // Detect objects
  //     std::vector<Detection> detections = objectDetector.detect(frame,
  //     config.detection);

  //     // Plan paths
  //     std::vector<Path> paths = pathPlanner.findPaths(detections,
  //     frame.size(), config.path);

  //     // Visualize results
  //     visualizer.draw(frame, detections, paths, config.visual, fps);

  //     // Display frame
  //     if (config.video.displayWindow) {
  //         videoProcessor.displayFrame(frame, "FlightPath - Path Detection");
  //     }

  //     // Save to output video
  //     if (config.video.saveOutput) {
  //         videoProcessor.writeFrame(frame);
  //     }

  //     // Handle keyboard input
  //     int key = cv::waitKey(1);

  //     if (key == 27) { // ESC
  //         std::cout << "\nExiting..." << std::endl;
  //         break;
  //     } else if (key == 32) { // SPACE
  //         config.isPaused = !config.isPaused;
  //         std::cout << (config.isPaused ? "Paused" : "Resumed") << std::endl;

  //         while (config.isPaused) {
  //             key = cv::waitKey(100);
  //             if (key == 32) {
  //                 config.isPaused = false;
  //                 std::cout << "Resumed" << std::endl;
  //             } else if (key == 27) {
  //                 config.shouldExit = true;
  //                 break;
  //             }
  //         }
  //     } else if (key == 's' || key == 'S') { // Save frame
  //         std::string filename = "frame_" + std::to_string(frameCount) +
  //         ".jpg"; cv::imwrite(filename, frame); savedFrameCount++; std::cout
  //         << "Saved frame: " << filename << std::endl;
  //     }

  //     // Progress indicator
  //     if (frameCount % 30 == 0) {
  //         int totalFrames = videoProcessor.getTotalFrames();
  //         float progress = (totalFrames > 0) ? (frameCount * 100.0f /
  //         totalFrames) : 0; std::cout << "\rProcessed: " << frameCount << "
  //         frames | "
  //                  << "FPS: " << std::fixed << std::setprecision(1) << fps <<
  //                  " | "
  //                  << "Progress: " << std::setprecision(1) << progress << "%
  //                  " << std::flush;
  //     }
  // }

  detectorThread1.join();
  detectorThread2.join();
  while (true) {
    auto currentTime = std::chrono::high_resolution_clock::now();
  }

  std::cout << "Time taken" << std::endl;
  // Cleanup
  std::cout << "\n\nProcessing complete!" << std::endl;
  std::cout << "Total frames processed: " << frameCount << std::endl;

  if (config.video.saveOutput) {
    std::cout << "Output saved to: " << config.video.outputPath << std::endl;
  }

  if (savedFrameCount > 0) {
    std::cout << "Saved " << savedFrameCount << " individual frames"
              << std::endl;
  }

  videoProcessor.release();

  return 0;
}
