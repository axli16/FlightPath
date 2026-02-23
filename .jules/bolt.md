## 2024-05-23 - Build Environment Limitations
**Learning:** The current sandbox environment lacks OpenCV headers/libraries and root access, preventing compilation and benchmark execution for this C++ project. This forces reliance on code analysis and standard practices rather than empirical verification.
**Action:** When working in constrained environments, identify critical performance anti-patterns (like allocations in hot loops) that are universally inefficient, and document the inability to verify due to environment.

## 2024-05-23 - cv::Mat Allocation in Hot Loops
**Learning:** `cv::Mat::row(i).colRange(...)` creates a new `cv::Mat` header (allocation, ref-counting) for each call. Inside a loop iterating thousands of times (e.g., YOLO anchors), this creates significant overhead compared to raw pointer arithmetic.
**Action:** Replace `cv::Mat` operations in inner loops with raw pointer access (`ptr<T>`) when just reading values.
