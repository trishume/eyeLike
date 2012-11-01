/**
 * @file objectDetection.cpp
 * @author A. Huaman ( based in the classic facedetect.cpp in samples/c )
 * @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream
 */
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>

#include "constants.h"
#include "findEyeCenter.h"

/** Constants **/


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

int eye_percent_top = 25;
int eye_percent_side = 13;
int eye_percent_height = 30;
int eye_percent_width = 35;

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
      //printf("Dimensions: %ix%i\n",frame.cols,frame.rows);
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
      if( (char)c == 'f' ) {
        imwrite("frame.png",frame);
      }
      
    }
  }
  return 0;
}

void findEyes(cv::Mat frame_gray, cv::Rect face) {
  cv::Mat faceROI = frame_gray(face);
  
  if (kSmoothFaceImage) {
    double sigma = kSmoothFaceFactor * face.width;
    GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
  }
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
