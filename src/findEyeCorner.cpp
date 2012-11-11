#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <queue>
#include <stdio.h>

#include "constants.h"
#include "helpers.h"

#include "findEyeCorner.h"

cv::Mat *leftCornerKernel;
cv::Mat *rightCornerKernel;

void createCornerKernels() {
  float m[4][6] = {
    {-1,-1,-1, 1, 1, 1},
    {-1,-1,-1,-1, 1, 1},
    {-1,-1,-1,-1,-1, 1},
    { 1, 1, 1, 1, 1, 1},
  };
  leftCornerKernel = new cv::Mat(4,6,CV_32F,m);
  rightCornerKernel = new cv::Mat(4,6,CV_32F);
  // flip horizontally
  cv::flip(*leftCornerKernel, *rightCornerKernel, 1);
}

void releaseCornerKernels() {
  delete leftCornerKernel;
  delete rightCornerKernel;
}

// TODO implement these
cv::Mat eyeCornerMap(cv::Mat region, bool left) {
  cv::Mat cornerMap;
  cv::filter2D(region, cornerMap, CV_32F, left ? *leftCornerKernel : *rightCornerKernel);
  return cornerMap;
}

cv::Point detectEyeCorner(cv::Mat region,bool left) {
  return cv::Point(-1,-1);
}
cv::Point2f detectSubpixelEyeCorner(cv::Mat region, bool left) {
  return cv::Point2f(-1.0,-1.0);
}