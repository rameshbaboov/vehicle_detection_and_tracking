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

Below are the list of function/procedures with their implementation
Draw_boxes              - Uses cv2. Rectangle to draw boxes
Add_heat                - Adds heat map
Apply_threshold         - Uses threshold to remove false positives
draw_labeled_bboxes     - Apply labels on the image
draw_simple_chart       - Takes two images and shows them in output screen with caption
extract_data            -Extract features of training images
train_classifier        - Training of classifier
find_cars               - Utilized sliding window and identifies car
test_model_on_multi_images - Test model with multiple images for given start, stop and scale
tune_threshold_for_multi_images - A partial implementation of pipeline for tuning threshold
process_image            - Full implementation of pipeline


# Histogram of Oriented Gradients (HOG)

## Explain how (and identify where in your code) you extracted HOG features from the training images

HOG features are extracted using extract_hog_features routine. Extract_data is used to call Extract_hog_Features to extract HOG features of cars and noncar training images into arrays car_features and notcar_features.

As you can see the HOG spaces are having different features for Cars and Not Cars and this is the basic concept behind identifying Cars and Non Car images


## Explain how you settled on your final choice of HOG parameters.

I created a setup of scripts as shown below that provides extracts features and train model for various parameters using various combination of CSPACE, ORIENT and Pix_per_Cell
* # run one time only
* cspace_list = ['HSV','LUV','HLS','YUV']
* orient_list = [9,10,11,12]
* pix_per_cell_list =[8,16]

This produced various results for multiple combination. I chose the combinations that gave top accuracy and then selected an optimal combination that provided the best result


```
cpsace- RGB orient- 8 hog_channel- 0 pix_per_cell 8 spatial_size (16, 16) accuracy 0.9438
cpsace- RGB orient- 8 hog_channel- 0 pix_per_cell 8 spatial_size (32, 32) accuracy 0.9438
cpsace- RGB orient- 8 hog_channel- 0 pix_per_cell 16 spatial_size (16, 16) accuracy 0.9458
cpsace- RGB orient- 8 hog_channel- 0 pix_per_cell 16 spatial_size (32, 32) accuracy 0.9375
cpsace- RGB orient- 8 hog_channel- 0 pix_per_cell 32 spatial_size (16, 16) accuracy 0.8417
cpsace- RGB orient- 8 hog_channel- 0 pix_per_cell 32 spatial_size (32, 32) accuracy 0.8646
cpsace- RGB orient- 8 hog_channel- 1 pix_per_cell 8 spatial_size (16, 16) accuracy 0.9562
cpsace- RGB orient- 8 hog_channel- 1 pix_per_cell 8 spatial_size (32, 32) accuracy 0.9562
cpsace- RGB orient- 8 hog_channel- 1 pix_per_cell 16 spatial_size (16, 16) accuracy 0.9333
cpsace- RGB orient- 8 hog_channel- 1 pix_per_cell 16 spatial_size (32, 32) accuracy 0.9271
cpsace- RGB orient- 8 hog_channel- 1 pix_per_cell 32 spatial_size (16, 16) accuracy 0.8833
cpsace- RGB orient- 8 hog_channel- 1 pix_per_cell 32 spatial_size (32, 32) accuracy 0.8833
cpsace- RGB orient- 8 hog_channel- 2 pix_per_cell 8 spatial_size (16, 16) accuracy 0.9438
cpsace- RGB orient- 8 hog_channel- 2 pix_per_cell 8 spatial_size (32, 32) accuracy 0.9396
cpsace- RGB orient- 8 hog_channel- 2 pix_per_cell 16 spatial_size (16, 16) accuracy 0.9417
cpsace- RGB orient- 8 hog_channel- 2 pix_per_cell 16 spatial_size (32, 32) accuracy 0.9521
cpsace- RGB orient- 8 hog_channel- 2 pix_per_cell 32 spatial_size (16, 16) accuracy 0.8479
cpsace- RGB orient- 8 hog_channel- 2 pix_per_cell 32 spatial_size (32, 32) accuracy 0.8458
cpsace- RGB orient- 8 hog_channel- ALL pix_per_cell 8 spatial_size (16, 16) accuracy 0.9583
cpsace- RGB orient- 8 hog_channel- ALL pix_per_cell 8 spatial_size (32, 32) accuracy 0.9583
cpsace- RGB orient- 8 hog_channel- ALL pix_per_cell 16 spatial_size (16, 16) accuracy 0.9562
cpsace- RGB orient- 8 hog_channel- ALL pix_per_cell 16 spatial_size (32, 32) accuracy 0.9604
cpsace- RGB orient- 8 hog_channel- ALL pix_per_cell 32 spatial_size (16, 16) accuracy 0.9083
cpsace- RGB orient- 8 hog_channel- ALL pix_per_cell 32 spatial_size (32, 32) accuracy 0.8854
cpsace- RGB orient- 9 hog_channel- 0 pix_per_cell 8 spatial_size (16, 16) accuracy 0.9521
cpsace- RGB orient- 9 hog_channel- 0 pix_per_cell 8 spatial_size (32, 32) accuracy 0.9521
cpsace- RGB orient- 9 hog_channel- 0 pix_per_cell 16 spatial_size (16, 16) accuracy 0.9458
cpsace- RGB orient- 9 hog_channel- 0 pix_per_cell 16 spatial_size (32, 32) accuracy 0.9229
cpsace- RGB orient- 9 hog_channel- 0 pix_per_cell 32 spatial_size (16, 16) accuracy 0.8354
cpsace- RGB orient- 9 hog_channel- 0 pix_per_cell 32 spatial_size (32, 32) accuracy 0.85
cpsace- RGB orient- 9 hog_channel- 1 pix_per_cell 8 spatial_size (16, 16) accuracy 0.9438
cpsace- RGB orient- 9 hog_channel- 1 pix_per_cell 8 spatial_size (32, 32) accuracy 0.9583
cpsace- RGB orient- 9 hog_channel- 1 pix_per_cell 16 spatial_size (16, 16) accuracy 0.9312
cpsace- RGB orient- 9 hog_channel- 1 pix_per_cell 16 spatial_size (32, 32) accuracy 0.9292
cpsace- RGB orient- 9 hog_channel- 1 pix_per_cell 32 spatial_size (16, 16) accuracy 0.8625
cpsace- RGB orient- 9 hog_channel- 1 pix_per_cell 32 spatial_size (32, 32) accuracy 0.8604
cpsace- RGB orient- 9 hog_channel- 2 pix_per_cell 8 spatial_size (16, 16) accuracy 0.9562
cpsace- RGB orient- 9 hog_channel- 2 pix_per_cell 8 spatial_size (32, 32) accuracy 0.9562
cpsace- RGB orient- 9 hog_channel- 2 pix_per_cell 16 spatial_size (16, 16) accuracy 0.9104
cpsace- RGB orient- 9 hog_channel- 2 pix_per_cell 16 spatial_size (32, 32) accuracy 0.9396
cpsace- RGB orient- 9 hog_channel- 2 pix_per_cell 32 spatial_size (16, 16) accuracy 0.8583
cpsace- RGB orient- 9 hog_channel- 2 pix_per_cell 32 spatial_size (32, 32) accuracy 0.8396
cpsace- RGB orient- 9 hog_channel- ALL pix_per_cell 8 spatial_size (16, 16) accuracy 0.9604

```
# Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Training of classifier is done using Train_classifier with the Training data obtained from extract_cars. The svc mode is saved into a pickle file so that next time the training need not be done again

I tested the effectiveness of the model using the procedure “test_model_on_multi_images” with a predefined combination like start, stop, cspace etc


# Sliding Window Search

## Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

I re-used find_cars from Project work out and then made few changes to it. I first used function to check whether boxes are identified correctly. I tried various combination of start, stop and threshold to check if cars are identified.
Then I used tune_threshold_for_multi_images and fed saved images from test and sample video to fine tune the threshold. The routine uses a multiple combination of start stop for both X and Y axis with various combination of scale.

## Show some examples of test images to demonstrate how your pipeline is working.

Finally, I settled for the below parameters that provided better results. You can see below few output iamges. I also fine tuned threshold to avoid false positives

* cspace='HSV'
* orient= 8 #9
* pix_per_cell= 6# 8
* cell_per_block=2 #2
* hog_channel='ALL'
* spatial_size=(16,16)
* hist_bins=8
* hist_range=(0, 256)

## What did you do to optimize the performance of your classifier?

Below are few things that I did to optimize the whole performance:

1. used various combination of stop, stop and scales. I started with a smaller scale and then slowly increased. I also used X start and stop to avoid cars to be detected on unwanted places

## Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I stored the identified boxes in a class rectangle. This class helps to perform below functions

1. Saves identified boxes so that the same box can be cached. Get_bboxes and save_bboxes are used for this
2. Check_scan_status helps ot cache the image. The boxes are identified for first image and then same boxes are cached for n count of image. After this again the boxes are calculated for next image and so on.
3. This skipping is increasing performance and also flickering of boxes. Also there should be no issue as 20 frames are generated before any significant movement can be detected
4. If anytime there are no boxes detected, then boxes are cached for a predefined count
  Also there were many false positives which I avoided by tuning threshold and also ensuring that boxes with uneven sizes are ignored
  
  
I created a simple python script Image extractor that can extract a portion of video into images. I tested the pipeline using project video and then identified the places where there were false positive or no identification. I adjusted threshold and start and stop of X and Y axes to avoid these.
  
  
With this I was able to reduce the false positives less than 5 .

# Discussion
## Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

Following are the problems:
* My project is fine tuned for this video. There could be many false positives or no identifications for other videos without fine tuning.
* Also I used caching to avoid frames where there were no cars detected and used position of previous images.
* Also shadows are creating challenge in identification.
* Cars moving in the slowest lane at a far distance is also creating issues as these cars overlaps with adjacent things like hills, tress etc and result in issue and as a result these are not detected
