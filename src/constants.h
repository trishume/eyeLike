#ifndef CONSTANTS_H
#define CONSTANTS_H

const int kSobelKernelSize = 5;
const int kWeightBlurSize = 7;
const float kWeightDivisor = 150.0;
const double kGradientThreshold = 500.0;
const int kFastEyeWidth = 55;
const float kPostProcessThreshold = 0.96;
const bool kPlotVectorField = false;
const bool kSmoothFaceImage = false;
const float kSmoothFaceFactor = 0.003;

#endif