/**
 * @file objectDetection.cpp
 * @author A. Huaman ( based in the classic facedetect.cpp in samples/c )
 * @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream
 */
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <queue>
#include <stdio.h>

/** Constants **/
const int kSobelKernelSize = 5;
const int kWeightBlurSize = 7;
const float kWeightDivisor = 150.0;
const float kGradientThreshold = 0.1;
const int kFastEyeWidth = 60;
const float kPostProcessThreshold = 0.96;

/** Function Headers */
void detectAndDisplay( cv::Mat frame );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String face_cascade_name = "../../../res/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
cv::RNG rng(12345);
cv::Mat debugImage;

int eye_percent_top = 30;
int eye_percent_side = 13;
int eye_percent_height = 20;
int eye_percent_width = 30;

/**
 * @function main
 */
int main( int argc, const char** argv ) {
  CvCapture* capture;
  cv::Mat frame;
  
  //-- 1. Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  
  cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(main_window_name, 400, 100);
  cv::namedWindow(face_window_name,CV_WINDOW_NORMAL);
  cv::moveWindow(face_window_name, 10, 100);
  cv::namedWindow("Right Eye",CV_WINDOW_NORMAL);
  cv::moveWindow("Right Eye", 10, 600);
  cv::namedWindow("Left Eye",CV_WINDOW_NORMAL);
  cv::moveWindow("Left Eye", 10, 800);
  
  //-- 2. Read the video stream
  capture = cvCaptureFromCAM( -1 );
  if( capture ) {
    while( true ) {
      frame = cvQueryFrame( capture );
      printf("Dimensions: %ix%i\n",frame.cols,frame.rows);
      // mirror it
      cv::flip(frame, frame, 1);
      frame.copyTo(debugImage);
      
      //-- 3. Apply the classifier to the frame
      if( !frame.empty() ) {
        detectAndDisplay( frame );
      }
      else {
        printf(" --(!) No captured frame -- Break!");
        break;
      }
      
      imshow(main_window_name,debugImage);
      
      int c = cv::waitKey(10);
      if( (char)c == 'c' ) { break; }
      
    }
  }
  return 0;
}

bool rectInImage(cv::Rect rect, cv::Mat image) {
  return rect.x > 0 && rect.y > 0 && rect.x+rect.width < image.cols &&
    rect.y+rect.height < image.rows;
}

float testPossibleCenterFormula(int cx, int cy, const cv::Mat &weight,
                                 const cv::Mat &gradientX, const cv::Mat &gradientY) {
  double sum = 0;
  // iterate over the gradient starts
  for (int y = 0; y < weight.rows; ++y) {
    const unsigned char *Wr = weight.ptr<unsigned char>(y);
    const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
    for (int x = 0; x < weight.cols; ++x) {
      if (x == cx && y == cy) {
        continue;
      }
      // create a vector from the possible center to the gradient origin
      double dx = x - cx;
      double dy = y - cy;
      // normalize d
      double magnitude = sqrt((dx * dx) + (dy * dy));
      dx = dx / magnitude;
      dy = dy / magnitude;
      double gx = Xr[x], gy = Yr[x];
      double dotProduct = dx*gx + dy*gy;
      // square and multiply by the weight
      sum += dotProduct * dotProduct * (Wr[x]/kWeightDivisor);
    }
  }
  
  return sum / (float)(weight.rows * weight.cols);
}

void scaleToFastSize(const cv::Mat &src,cv::Mat &dst) {
  cv::resize(src, dst, cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}

cv::Point unscalePoint(cv::Point p, cv::Rect origSize) {
  float ratio = (((float)kFastEyeWidth)/origSize.width);
  int x = round(p.x / ratio);
  int y = round(p.y / ratio);
  return cv::Point(x,y);
}

bool inMat(cv::Point p,int rows,int cols) {
  return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

void floodKillEdges(cv::Mat &mat) {
  // fill the edges so the flood fill spreads
  rectangle(mat,cv::Rect(0,0,mat.cols-1,mat.rows-1),255);
  
  std::queue<cv::Point> toDo;
  toDo.push(cv::Point(0,0));
  while (!toDo.empty()) {
    cv::Point p = toDo.front();
    toDo.pop();
    // add in every direction
    for (int dx = -1;dx <= 1; dx += 2) {
      for (int dy = -1;dy <= 1; dy += 2) {
        cv::Point np(p.x+dx,p.y+dy);
        if (inMat(np, mat.rows, mat.cols) && mat.at<float>(np) != 0.0f) {
          toDo.push(np);
        }
      }
    }
    // kill it
    mat.at<float>(p) = 0.0f;
  }
}

cv::Point findEyeCenter(cv::Mat face, cv::Rect eye, std::string debugWindow) {
  // draw the eye region
  rectangle(face,eye,1234);
  cv::Mat eyeROIUnscaled = face(eye);
  cv::Mat eyeROI;
  scaleToFastSize(eyeROIUnscaled, eyeROI);
  //-- Find the gradient
  cv::Mat gradientX;
  cv::Sobel(eyeROI, gradientX, CV_64F, 1, 0, kSobelKernelSize);
  cv::Mat gradientY;
  cv::Sobel(eyeROI, gradientY, CV_64F, 0, 1, kSobelKernelSize);
  //-- Normalize and threshold the gradient
  // find the max
  double max = 0;
//  for (int y = 0; y < eyeROI.rows; ++y) {
//    double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
//    for (int x = 0; x < eyeROI.cols; ++x) {
//      double gX = Xr[x], gY = Yr[x];
//      double magnitude = sqrt((gX * gX) + (gY * gY));
//      if (magnitude > max) {
//        max = magnitude;
//      }
//    }
//  }
  // normalize and threshold it
  double threshold = max * kGradientThreshold;
  for (int y = 0; y < eyeROI.rows; ++y) {
    double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
    for (int x = 0; x < eyeROI.cols; ++x) {
      double gX = Xr[x], gY = Yr[x];
      double magnitude = sqrt((gX * gX) + (gY * gY));
      if (magnitude > threshold) {
        Xr[x] = gX/magnitude;
        Yr[x] = gY/magnitude;
      } else {
        Xr[x] = 0;
        Yr[x] = 0;
      }
    }
  }
  //imshow(debugWindow,gradientX);
  //-- Create a blurred and inverted image for weighting
  cv::Mat weight;
  GaussianBlur( eyeROI, weight, cv::Size( kWeightBlurSize, kWeightBlurSize ), 0, 0 );
  for (int y = 0; y < weight.rows; ++y) {
    unsigned char *row = weight.ptr<unsigned char>(y);
    for (int x = 0; x < weight.cols; ++x) {
      row[x] = (255 - row[x]);
    }
  }
  //imshow(debugWindow,weight);
  //-- Run the algorithm!
  cv::Mat out(eyeROI.rows,eyeROI.cols,CV_32F);
  // for each possible center
  printf("Eye: %ix%i\n",out.cols,out.rows);
  for (int cy = 0; cy < out.rows; cy++) {
    float *Or = out.ptr<float>(cy);
    for (int cx = 0; cx < out.cols; cx++) {
      Or[cx] = testPossibleCenterFormula(cx, cy, weight, gradientX, gradientY);
    }
  }
  //imshow(debugWindow,out);
  //-- Find the maximum point
  cv::Point maxP;
  double maxVal;
  cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP);
  printf("Max: %i,%i - %f\n",maxP.x,maxP.y,maxVal);
  //-- Threshold it
  cv::threshold(out, out, maxVal * kPostProcessThreshold, 0.0f, cv::THRESH_TOZERO);
  //-- Flood fill the edges
  // fill the edges so the flood fill spreads
  //rectangle(out,cv::Rect(0,0,out.cols,out.rows),255);
  // run the flood kill from 0,0
  //cv::floodFill(out, cv::Point(0,0), 0.0,NULL,0.97,1.0,cv::FLOODFILL_FIXED_RANGE);
  //floodKillEdges(out);
  imshow(debugWindow,out);
  // redo max
  cv::minMaxLoc(out, NULL,NULL,NULL,&maxP);
  return unscalePoint(maxP,eye);
}

void findEyes(cv::Mat frame_gray, cv::Rect face) {
  cv::Mat faceROI = frame_gray(face);
  
  //-- Find eye regions and draw them
  int eye_region_width = face.width * (eye_percent_width/100.0);
  int eye_region_height = face.width * (eye_percent_height/100.0);
  int eye_region_top = face.height * (eye_percent_top/100.0);
  cv::Rect leftEyeRegion(face.width*(eye_percent_side/100.0),
                         eye_region_top,eye_region_width,eye_region_height);
  cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(eye_percent_side/100.0),
                          eye_region_top,eye_region_width,eye_region_height);
  
  //-- Find Eye Centers
  cv::Point leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
  cv::Point rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");
  // change it to face coordinates
  rightPupil.x += rightEyeRegion.x;
  rightPupil.y += rightEyeRegion.y;
  leftPupil.x += leftEyeRegion.x;
  leftPupil.y += leftEyeRegion.y;
  
  circle(faceROI, rightPupil, 3, 1234);
  circle(faceROI, leftPupil, 3, 1234);
  imshow(face_window_name, faceROI);
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( cv::Mat frame ) {
  std::vector<cv::Rect> faces;
  cv::Mat frame_gray;
  
  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  //equalizeHist( frame_gray, frame_gray );
  //cv::pow(frame_gray, 0.5, frame_gray);
  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
  
  for( int i = 0; i < faces.size(); i++ )
  {
    rectangle(debugImage, faces[i], 1234);
  }
  //-- Show what you got
  if (faces.size() > 0) {
    findEyes(frame_gray, faces[0]);
  }
}
