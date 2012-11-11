#ifndef EYE_CORNER_H
#define EYE_CORNER_H

#include "opencv2/imgproc/imgproc.hpp"

#define kEyeLeft true
#define kEyeRight false

void createCornerKernels();
void releaseCornerKernels();
cv::Point detectEyeCorner(cv::Mat region,bool left);
cv::Point2f detectSubpixelEyeCorner(cv::Mat region, bool left);

#endif