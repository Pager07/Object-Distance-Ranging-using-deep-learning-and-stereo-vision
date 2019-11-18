#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
import random
import csv


# In[2]:


camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres
image_centre_h = 262.0;
image_centre_w = 474.5;


# In[5]:


def project_disparity_to_3d(disparity, max_disparity,center_of_bounding_box_coord):

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;

    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then we get reasonable scaling in X and Y output if we change
    # Z to Zmax in the lines X = ....; Y = ...; below

    # Zmax = ((f * B) / 2);
    height, width = disparity.shape[:2];
    row,col = center_of_bounding_box_coord
    if disparity[row,col]>0:
        Z = (f * B) / disparity[row,col];
        return Z
    else:
        return 0




