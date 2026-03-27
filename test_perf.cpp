#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main() {
    int gridSize = 200;
    cv::Mat mask = cv::Mat::zeros(gridSize, gridSize, CV_8UC1);
    cv::Mat grid1 = cv::Mat::zeros(gridSize, gridSize, CV_8UC1);
    cv::Mat grid2 = cv::Mat::zeros(gridSize, gridSize, CV_8UC1);

    cv::rectangle(mask, cv::Point(50, 50), cv::Point(150, 150), cv::Scalar(255), cv::FILLED);

    auto start1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10000; ++iter) {
        for (int y = 0; y < gridSize; ++y) {
            for (int x = 0; x < gridSize; ++x) {
                if (mask.at<uchar>(y, x) == 0) {
                    grid1.at<uchar>(y, x) = 255;
                }
            }
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();

    auto start2 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10000; ++iter) {
        grid2.setTo(255, mask == 0);
    }
    auto end2 = std::chrono::high_resolution_clock::now();

    std::cout << "Loop time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << "ms\n";
    std::cout << "setTo time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << "ms\n";

    // Test rect
    int gridLeft = 20, gridTop = 20, gridRight = 100, gridBottom = 100;
    cv::Mat grid3 = cv::Mat::zeros(gridSize, gridSize, CV_8UC1);
    cv::Mat grid4 = cv::Mat::zeros(gridSize, gridSize, CV_8UC1);

    auto start3 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10000; ++iter) {
        for (int y = gridTop; y <= gridBottom; ++y) {
            for (int x = gridLeft; x <= gridRight; ++x) {
                grid3.at<uchar>(y, x) = 255;
            }
        }
    }
    auto end3 = std::chrono::high_resolution_clock::now();

    auto start4 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10000; ++iter) {
        cv::rectangle(grid4, cv::Point(gridLeft, gridTop), cv::Point(gridRight, gridBottom), cv::Scalar(255), cv::FILLED);
    }
    auto end4 = std::chrono::high_resolution_clock::now();

    std::cout << "Rect loop time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3).count() << "ms\n";
    std::cout << "cv::rectangle time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end4 - start4).count() << "ms\n";

    // verify correctness
    bool same1 = true;
    for (int y = 0; y < gridSize; ++y) {
        for (int x = 0; x < gridSize; ++x) {
            if (grid1.at<uchar>(y, x) != grid2.at<uchar>(y, x)) same1 = false;
            if (grid3.at<uchar>(y, x) != grid4.at<uchar>(y, x)) same1 = false;
        }
    }
    std::cout << "Correct: " << same1 << "\n";
    return 0;
}
