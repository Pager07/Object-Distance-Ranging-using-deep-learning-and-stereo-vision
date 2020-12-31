## Using Stereo Vision for Object Distance Ranging (Vehicles) ##

In this project we are dealing with the automatic detection of objects, and the estimation of their
distance from the vehicle (i.e. ranging), within stereo video imagery from an on-board forward
facing stereo camera. This can be performed by integrating the use of depth (disparity) information
recovered from an existing stereo vision algorithm with an object detection algorithms.

Knowledge of the distance of objects that have the potential to move within the scene (i.e. dynamic objects,
such as pedestrians/vehicles) assists both automatic forward motion planning and collision
avoidance within the overall autonomous control system of the vehicle.


![](./images/demo.gif)

## Project objectives ##
- [x] Perform image pre-filtering or optimization to improve either/both object detection or stereo depth estimation
- [x] Effective integration of object detection 
- [x] Effective integration of dense stereo ranging
- [x] Object range estimation strategy for challenging conditions


## How to run the project ##

INSIDE object_detection.py set the following variables.
- classes_file 
    - PATH TO: mscoco_labels.names
        - Can be found inside MASK_RCNN Folder
- text_graph 
    - PATH TO: mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
            - Can be found inside MASK_RCNN Folder
- model_weights 
    - PATH TO: frozen_inference_graph.pb
        - Can be found inside mask_rcnn_inception_v2_coco_2018_01_28 Sub-Folder
- colors_file 
    - PATH TO: colors.txt
        - Can be found inside MASK_RCNN Folder

INSIDE main.py set the following variables.
   - master_path_to_dataset
       - PATH TO: Dataset

Finally
    - python main.py

## NOTES ##
- demoVideo.mp4
    - In the video: Fewer than objects are detected because I have set the object detection confidence score threshold to 0.97. 
    - This is becuase I wanted a cleaner video for demo such that distance label on top of the polygon is not overlapped by other objects polygon. 
    - If you wish to detect more object, reset the value of following variable in object_detection.py:
        - conf_threshold


## Dataset ##

The dataset **TTBB-durham-02-10-17** which contains a set of 1449 sequential still image stereo pairs extracted from onboard stereo camera video footage. These images have been rectified
based on the camera calibration. So I did not have to perform  stereo calibration myself. I like to thanks [Toby Breckon](https://github.com/tobybreckon) for providing the dataset.



