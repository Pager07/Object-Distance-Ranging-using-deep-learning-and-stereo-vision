#!/usr/bin/env python
# coding: utf-8

# #The code below is to sets up global variable to be used by function project_disparity_to_3d()

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


# #The code converts a give disparity value to distnace in meters

# In[3]:


def project_disparity_to_3d(disparity):

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;
    if(disparity>0):
        Z = (f * B) / disparity;
        return Z
    else:
        return 0

