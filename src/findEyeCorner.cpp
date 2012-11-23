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

// not constant because stupid opencv type signatures
float kEyeCornerKernel[4][6] = {
  {-1,-1,-1, 1, 1, 1},
  {-1,-1,-1,-1, 1, 1},
  {-1,-1,-1,-1,-1, 1},
  { 1, 1, 1, 1, 1, 1},
};

void createCornerKernels() {
  leftCornerKernel = new cv::Mat(4,6,CV_32F,kEyeCornerKernel);
  rightCornerKernel = new cv::Mat(4,6,CV_32F);
  // flip horizontally
  cv::flip(*leftCornerKernel, *rightCornerKernel, 1);
}

void releaseCornerKernels() {
  delete leftCornerKernel;
  delete rightCornerKernel;
}

// TODO implement these
cv::Mat eyeCornerMap(const cv::Mat &region, bool left) {
  cv::Mat cornerMap;
  cv::filter2D(region, cornerMap, CV_32F, left ? *leftCornerKernel : *rightCornerKernel);
  return cornerMap;
}

cv::Point findEyeCorner(cv::Mat region,bool left) {
  cv::Mat cornerMap = eyeCornerMap(region, left);
  //imshow("Corner map",cornerMap);
  cv::Point maxP;
  cv::minMaxLoc(cornerMap, NULL,NULL,NULL,&maxP);
  // GFTT
//  std::vector<cv::Point2f> corners;
//  cv::goodFeaturesToTrack(region, corners, 500, 0.005, 20);
//  for (int i = 0; i < corners.size(); ++i) {
//    cv::circle(region, corners[i], 2, 200);
//  }
//  imshow("Corners",region);
  
  return maxP;
}
cv::Point2f findSubpixelEyeCorner(cv::Mat region, bool left) {
  return cv::Point2f(-1.0,-1.0);
}