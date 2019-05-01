# vehicle_detection_and_tracking
Vehicle detection and tracking
# Goals of this project
Objective of this project is to create a Python script that can detect cars from the Video Image using a trained Classifier model by following:

Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
*  Implement a sliding-window technique and use your trained classifier to search for vehicles in images. Create a pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
*  Estimate a bounding box for vehicles detected.
*  Recreate the Video Stream with the output of pipeline

# Write up/Readme
## Readme

Frames from Video is extracted and each image is passed to Pipeline function (Process_Image) and output images are written back as Video output using Write_Videofile function
* The program consists of two parts:
  * Training SVC Linear mode
  * Identifying cars using the Linear Model
* Training of model is processed by below two functions
  * Extract_data – This function takes set of JPEG files Cars and Non Cars provided by Udacity and converts them to features
  * Train Classifier - Trains the SVC with the extracted features
* Identifying cars is processed by Find_cars function.
