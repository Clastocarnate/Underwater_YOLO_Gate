# Deep-Monocular-SLAM

This project focuses on implementing various components of a Visual SLAM (Simultaneous Localization and Mapping) system.

## ToDos

1. **Feature Extraction**
   - Implement algorithms to detect and describe keypoints in images.
   - Examples: SIFT, SURF, ORB.

2. **Feature Matching**
   - Develop methods to match features between consecutive frames.
   - Examples: FLANN, BFMatcher.

3. **Visual Odometry**
   - Calculate the camera's motion by analyzing the changes in position of image features.
   - Estimate relative motion between frames.

4. **Mapping**
   - Build a consistent map of the environment using the data from visual odometry.
   - Ensure the map is updated in real-time as new data comes in.

5. **Bundle Adjustment**
   - Optimize the 3D structure and motion parameters to minimize the re-projection error.
   - Use non-linear least squares methods for optimization.

6. **Loop Closure Detection**
   - Detect when the camera revisits a previously seen location.
   - Correct the accumulated drift in the map by aligning the current view with the past view.

7. **Pose Graph Optimization**
   - Refine the estimated trajectory of the camera using loop closure constraints.
   - Optimize the entire pose graph to achieve globally consistent localization and mapping.

## Getting Started

### Prerequisites
- List the software and libraries needed to run the project.
- Example: OpenCV, Eigen, Ceres Solver, etc.

### Installation
- Step-by-step instructions to set up the project environment.
- Example:
  ```bash
  git clone https://github.com/yourusername/visual_slam_project.git
  cd visual_slam_project
  mkdir build
  cd build
  cmake ..
  make
