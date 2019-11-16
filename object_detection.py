#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Given an image, this module returns
#     - Box coordinates
#     - Image with bounding boxes with lables


# In[11]:


import cv2
import argparse
import sys
import math
import numpy as np


# In[12]:


class_file = "/Users/sandeep/Desktop/Homework2019/ComputerVision Homework/python-examples-cv/coco.names"
config_file = "/Users/sandeep/Desktop/Homework2019/ComputerVision Homework/python-examples-cv/yolov3.cfg"
weights_file = "/Users/sandeep/Desktop/Homework2019/ComputerVision Homework/python-examples-cv/yolov3.weights"

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

net,output_layer_names,classes = None,None,None


# In[19]:


def set_file_dir(class_file_dir, config_file_dir, weights_file_dir):
    global class_file
    global config_file 
    global weights_file
    class_file = class_file_dir
    config_file = config_file_dir
    weights_file = weights_file_dir


# In[29]:


def drawPred(image, class_name, confidence, left, top, right, bottom, colour,z):
    # Draw a bounding box.
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    # construct label
    label = f'{class_name}:{round(confidence,2)},{round(z,2)}m'

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
        (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)


# In[30]:


def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores

    classIds = []
    confidences = []
    boxes = []
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds_nms, confidences_nms, boxes_nms)


# In[31]:


# Get the names of the output layers of the CNN network
# net : an OpenCV DNN module network object
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# In[32]:


def init_yolo():  
    # Load names of classes from file
    classesFile = class_file
    global classes 
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # load configuration and weight files for the model and load the network using them
    global net
    global output_layer_names
    net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
    output_layer_names = getOutputsNames(net)

     # defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

    # change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)


# In[33]:


def get_center_coord_of_bounding_box(top_left , bottom_right):
    col1,col2 = top_left[1],bottom_right[1]
    row1,row2 = top_left[0],bottom_right[0]
    mid_point_row = int((row1+row2)/2)
    mid_point_col = int((col1+col2)/2)
    return (mid_point_row, mid_point_col)


# In[34]:


def get_z_value(center_of_bounding_box ,coord2d_to_z_mapping):
    return coord2d_to_z_mapping[center_of_bounding_box]


# In[36]:


def process_image(frame,coord2d_to_z_mapping):
    init_yolo()
    # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
    tensor = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # set the input to the CNN network
    net.setInput(tensor)

    # runs forward inference to get output of the final output layers
    results = net.forward(output_layer_names)

    # remove the bounding boxes with low confidence
    classIDs, confidences, boxes = postprocess(frame, results, confThreshold, nmsThreshold)

    # draw resulting detections on image
    for detected_object in range(0, len(boxes)):
        box = boxes[detected_object]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        center_of_bounding_box = get_center_coord_of_bounding_box((left,top),
                                                                 (left + width, top + height)
                                                                 )
        if center_of_bounding_box in coord2d_to_z_mapping:
            z = get_z_value(center_of_bounding_box , coord2d_to_z_mapping)
            drawPred(frame, classes[classIDs[detected_object]], confidences[detected_object], left, top, left + width, top + height, (255, 178, 50) , z)


    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    return frame

    



# In[22]:


# frame = cv2.imread('/Users/sandeep/Desktop/Homework2019/ComputerVision Homework/TTBB-durham-02-10-17-sub10/left-images/1506942473.484027_L.png')
# frame, listx =process_image(frame)


# In[23]:


# listx


# In[38]:


a = {(1,2):'a'}
'a' in a


# In[39]:


round(1.11111,1)


# In[ ]:




