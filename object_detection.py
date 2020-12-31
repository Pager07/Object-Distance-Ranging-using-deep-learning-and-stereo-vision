#!/usr/bin/env python
# coding: utf-8

# # Object detection and Segmentation using Mask R-CNN
#     - The code below is to define global functions later to be used in the code

# In[40]:


import cv2 as cv
import argparse
import sys
import math
import numpy as np


# In[21]:


# Load classes
classes_file = "/Users/sandeep/Desktop/MaskRCNNopencv/mscoco_labels.names"
# Load text graph and weight files for the model
text_graph = '/Users/sandeep/Desktop/MaskRCNNopencv/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
model_weights = '/Users/sandeep/Desktop/MaskRCNNopencv/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
# Load colors
colors_file = "/Users/sandeep/Desktop/MaskRCNNopencv/colors.txt"
classes = None

with open(classes_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


with open(colors_file, 'rt') as f:
    colors_str = f.read().rstrip('\n').split('\n')

colors = []
for i in range(len(colors_str)):
    rgb = colors_str[i].split(' ')
    color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
    colors.append(color)


# Load network
net = cv.dnn.readNetFromTensorflow(model_weights, text_graph)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# This is a global variable that will store the img that needs to be proccessed
frame = None
#This will store final segemented images that will be returned 
segmented_frame = None
# detected_obj details
detected_obj = {}


# In[22]:


# Confidence and mask threshold
conf_threshold = 0.98
mask_threshold = 0.4


# #The following functions: drawPred() and set_mask_frame() are some helper function of post_proccess()

# In[23]:


def drawPred(image, class_name, left, top, right, bottom, colour,z):
    # Draw a bounding box.
    cv.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    # construct label
    label = f'{class_name}:{round(z,2)}m'

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(image, (left, top - round(1.5*labelSize[1])),
        (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(image, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
    return image


# In[24]:


def set_mask_frame(frame, left, top, right, bottom, class_mask):
    class_mask = cv.resize(class_mask, (right - left + 1, bottom - top + 1))
    filtering_mask = (class_mask > mask_threshold)
    roi = frame[top:bottom+1, left:right+1][filtering_mask]
    frame[top:bottom+1, left:right+1][filtering_mask] = np.array(([255.0,255.0,255.0])).astype(np.uint8)


# #The function post_process() does the following:
#     -Processes  objects detected in the image
#         - Takes in the following input
#             - the coordinates of box for an object
#             - The binary object mask from MASK-RCNN
#         - Sets required values to  the following global variables 
#             - detected_objs
#                 - A dictionary storing details of the the detected object
#                     - coord_bounding box
#                     - class_name
#                     - class_mask
#             - segmented_frame
#                 - The binary mask of the image
#         

# In[25]:


# For each detected object in a frame, extract bounding box and mask
def postprocess(boxes, masks):
    global frame
    global segmented_frame
    num_classes = masks.shape[1]
    num_detections = boxes.shape[2]

    frame_H = frame.shape[0]
    frame_W = frame.shape[1]

    #Blank black frame, same size as frame. To be used as template of binary mask
    mask_frame = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
    
    # a capy of original image
    frame_copy = frame.copy()

    for i in range(num_detections):
        box = boxes[0, 0, i]
        mask = masks[i]
        score = box[2]
        if score > conf_threshold:
            class_id = int(box[1])

            # Extract the bounding box
            left = int(frame_W * box[3])
            top = int(frame_H * box[4])
            right = int(frame_W * box[5])
            bottom = int(frame_H * box[6])

            left = max(0, min(left, frame_W-1))
            top = max(0, min(top, frame_H-1))
            right = max(0, min(right, frame_W-1))
            bottom = max(0, min(bottom, frame_H-1))

            # Extract the mask for the object
            class_mask = mask[class_id]
            class_mask = cv.resize(class_mask, (right - left + 1, bottom - top + 1))
            #generate mask 
            set_mask_frame(mask_frame, left, top, right, bottom, class_mask)
            detected_obj[i] = {'class_name': classes[class_id],
                              'left':left, 'top':top , 'right':right , 'bottom':bottom,
                               'class_mask':class_mask}
    
    segmented_frame = mask_frame
    
    
    


# #The function process_img()
#     - Takes an image as input
#     - Returns the 
#         - A dictionary of detected objects
#         - An binary image mask

# In[41]:


def process_img(img):
    #set the global variable
    global frame,detected_obj
    frame= img
    detected_obj.clear()
    # create a 4D blob from  a frame
    # swapRB: boolean to indicate if we want to swap the first and last channel in 3 channel image.
    #       : OpenCV assumes that images are in BGR format by default but if we want to swap this order to RGB,
    blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False)

    # set input to the network
    net.setInput(blob)

    # Run the forward pass computation to get output from the output layers
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
    
    postprocess(boxes, masks)
    
    return segmented_frame, detected_obj
    
    

    

