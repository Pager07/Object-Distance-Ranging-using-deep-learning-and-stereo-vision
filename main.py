

# #Importing Libraries

# In[ ]:


import sys
import cv2
import os 
import numpy as np
import random
from math import log10, floor

import object_detection
import stereo_to_3d
import wls_filter


# # Global variable setup

# In[ ]:


master_path_to_dataset = "/Users/sandeep/Desktop/Homework2019/ComputerVision Homework/TTBB-durham-02-10-17-sub10" # ** need to edit this **
directory_to_cycle_left = "left-images"     # edit this if needed
directory_to_cycle_right = "right-images"   # edit this if needed

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = ""; # set to timestamp to skip forward to

crop_disparity = False; # display full or cropped disparity image
pause_playback = False; # pause until key press after each image


# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

left_file_list = sorted(os.listdir(full_path_directory_left));

# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)

max_disparity = 128;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

# Used for segmenting object from background
mask_threshold = 0.3


# #The function  adaptiveHisEqulColor() 
#     - Takes in an image 
#     - Returns histogram equalized image

# In[ ]:


def adaptiveHisEqulColor(img):
    bgr = img 
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    
    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(16,16))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


# In[ ]:


def get_filtering_mask(class_mask):
    return (class_mask > mask_threshold)


# In[ ]:


def get_rgb_mask(img,filtering_mask):
    rgb_mask = np.zeros(img.shape, np.uint8)
    rgb_mask[top:bottom+1, left:right+1][filtering_mask] = np.array(([255.0,255.0,255.0])).astype(np.uint8)
    return rgb_mask


# In[ ]:


#Helper function for get_center_disparity()
def get_countour(class_mask):
    filtering_mask = get_filtering_mask(class_mask)
    filtering_mask = filtering_mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(filtering_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# # The function get_center_disparity()
#     - Takes in  
#         - coordinate of top-left point of the bounding box of an object
#             - (left,top)
#         - object class_mask
#         - disparity map
#     - Returns the disparity at the center of the object 

# In[ ]:


def get_center_disparity(left,top,class_mask,dispairty_map):
    countours = get_countour(class_mask)
    cnt = countours[0]
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    x,y = cX + left , cY + top  
    obj_center_pixel_disparity = dispairty_map[y,x]
    return obj_center_pixel_disparity


# # The function get_mean_disparity()
#     - Takes in the following:
#         - imgL (Left stereo image)
#         - coordinate of top-left and bottom-right points of the bounding box of an object
#             - (left,top)
#             - (right,bottom)
#         - object class_mask
#         - disparity map
#     - Returns the average disparity of the object

# In[ ]:


def get_mean_disparity(left,top,right,bottom,class_mask, imgL,disparity_map):
    filtering_mask = get_filtering_mask(class_mask)
#     rgb_mask = get_rgb_mask(imgL,filtering_mask)
    
#     mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2GRAY)
    mask= np.zeros(disparity_map.shape, np.uint8)
    mask[top:bottom+1, left:right+1][filtering_mask] = np.array(([255.0])).astype(np.uint8)
    obj_mean_disparity = cv2.mean(disparity_map, mask=mask)
    return obj_mean_disparity[0]


# #The function get_overlayed_disparity() 
#     - Takes in the following:
#         - Binary image mask
#         - Disparity map
#    - Returns segmented disparity map with only the object

# In[ ]:


def get_overlayed_disparity(mask, disparity_map):
    disparity_map = disparity_map.astype(np.uint8)
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    segmented_disparity = cv2.subtract(mask,disparity_map)
    segmented_disparity = cv2.subtract(mask,segmented_disparity)
    return segmented_disparity


# #The function filter_detected_object()
#     - Sometimes a particular object is detected in the left image but not right image
#     - Main objective of this functions is as follows:
#         - Keep the  detected object in left image iff the following condition are met:
#                - there exist an object of same class  within a radius of 100 pixles in the right stereo image
#               

# In[ ]:


#Helper function for filter_detected_objects()
def get_object_center_coord(left,top,class_mask):
    countours = get_countour(class_mask)
    cnt = countours[0]
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    x,y = cX + left , cY + top  
    return (x,y)


# In[ ]:


#Helper function for filter_detected_objects()
def check_point_lie(radius,center_x,center_y,x,y):
    if ((x-center_x)^2 + (y - center_y)^2) <= (radius^2):
        return True 
    else:
        return False
    


# In[ ]:


def filter_detected_object(detected_objs_L,detected_objs_R):
    final_detected_objs = {}
    counter = 0 
    for detection_number_L in range(len(detected_objs_L)):
        
        detected_object = detected_objs_L[detection_number_L]
        #Get object detection results
        class_name  = detected_object['class_name']
        class_mask = detected_object['class_mask']
        left , top = detected_object['left'],detected_object['top']
        (center_x,center_y) = get_object_center_coord(left,top,class_mask)
        
        for detection_number_R in range(len(detected_objs_R)):
            detected_object_R = detected_objs_R[detection_number_R]
            #Get object detection results
            class_name_R  = detected_object['class_name']
            class_mask_R = detected_object['class_mask']
            left_R , top_R = detected_object['left'],detected_object['top']
            (x,y) = get_object_center_coord(left_R,top_R,class_mask_R)
            
            is_inside_circle = check_point_lie(200,center_x,center_y,x,y)
            same_class = (class_name == class_name_R)
            if is_inside_circle and same_class:
                final_detected_objs[counter] = detected_object
                counter = counter + 1
    return final_detected_objs
                
                
            


# # Looping over image files

# - Load the image file
# - For left and right images 
#     - Pass the image to Mask RCNN and do object detection
#         - Should return the follwingg 
#             - 
#             - [1:{class_name:'dog'  , left-top: (x,y) ,right-bottom: (x1,y1), mask:}]
#     - Remove the background and only keep the objects in the image 
#             - Get back and white mask,(DONE) 
#             - get object only(DONE)
# - Find the disparity map of the using the segmented Left and Right image file
# - Overlay the mask with countour with disaprity map to find 
#     - Center of countour region 
#     - Mean of countour region
#     - Median of countour region
# - Edit project disparity to 3d such that
#        - for each detected object it only calculates 3d point of given coords (center)
#        - x,y,z
#              - You do not need x and y 
#              - You only need compute and return z 
# - Pass the image to Mask RCNN module agian to draw the rectangle and z value on it
#         - draw_pred needs: image,class_name,(left-top-coord),(right-bottom-coord), colour
#         - Thus, you need to store 

# In[ ]:


for filename_left in left_file_list:

    # from the left image filename get the correspondoning right image
    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    # # for sanity print out these filenames
    # print(full_path_filename_left);
    # print(full_path_filename_right);
    # print();

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
#         print("-- files loaded successfully");
        # print();
        imgL_copy = imgL[:]
        
         #uncomment below to use adaptive histogram equalization on input images
        imgL = adaptiveHisEqulColor(imgL)
        imgR = adaptiveHisEqulColor(imgR)
        
        #uncomment below to use bilateral-filtering on input images 
        imgL = cv2.bilateralFilter(imgL,5,75,75)
        imgR = cv2.bilateralFilter(imgR,5,25,25)
        
        mask_imgL, detected_objects_list_L = object_detection.process_img(imgL)
        mask_imgR, detected_objects_list_R = object_detection.process_img(imgR)
        
        
        #uncommnet to filter the detected_objects_L using detected_objects_list_R
#         detected_objects_list_L = filter_detected_object(detected_objects_list_L,detected_objects_list_R)  

       
        
        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images
        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);
        
        # Applying Weighted least squares filter to generate more continuous disparity map
        disparity = wls_filter.wsl_disparity_filter(grayL,grayR);
        
        # uncomment below to use median filter 
        disparity = cv2.medianBlur(disparity,9)
        
        
        dispNoiseFilter = 5; # increase for more agressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);
        _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
        
        disparity_scaled = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        disparity_scaled = np.uint8(disparity_scaled)
      
        closest_object_distance = 0
        closest_object_class = 'No Object Detected'
        for detection_number in range(len(detected_objects_list_L)):
            detected_object = detected_objects_list_L[detection_number]
            #Get object detection results
            class_name  = detected_object['class_name']
            class_mask = detected_object['class_mask']
            left , top = detected_object['left'],detected_object['top']
            right, bottom = detected_object['right'], detected_object['bottom']
            
# #           #get disparity value at the center of the object
#             center_disparity = get_center_disparity(left ,
#                                                     top,
#                                                     class_mask, 
#                                                     disparity_scaled)
            
            #get the average disparity value of object. 
            #I Passed in the class_mask such that I can segment the region of object in disparity map
            # And take the mean of that region
            mean_disparity = get_mean_disparity(left,
                                                top,
                                                right,
                                                bottom,
                                                class_mask,
                                                imgL,
                                                disparity_scaled)
            
            
            #Given a disparity value calculate the z value
            distance = stereo_to_3d.project_disparity_to_3d(mean_disparity)
            


            if distance != 0:
                imgL_copy = object_detection.drawPred(imgL_copy,
                                                class_name,
                                                left,top,
                                                right,bottom,
                                                (255, 178, 50),
                                                distance
                                                )
                if detection_number == 0:
                    closest_object_distance = distance
                    closest_object_class = class_name
                elif detection_number > 0:
                    if distance < closest_object_distance:
                        closest_object_distance = distance
                        closest_object_class = class_name
                        
        print(filename_left)
        if distance != 0:
            print(filename_left+': nearest detected scene object (' + str( round(distance, -int(floor(log10(abs(distance))))) ) + 'm)') 
        else:
            print(filename_left+': nearest detected scene object (' + str(0.0) + 'm)')     

         
        cv2.imshow('detection',imgL_copy)
        
        
        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break; # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty.png", disparity_scaled);
            cv2.imwrite("left.png", imgL);
            cv2.imwrite("right.png", imgR);
        elif (key == ord('c')):     # crop
            crop_disparity = not(crop_disparity);
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback);
    else:
        print("-- files skipped (perhaps one is missing or not PNG)");
        print();


# # Gamma Correction (NOT USED)

# In[ ]:


def adjust_gamma(image):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    table = []
    for i in np.arange(0, 256):
        if i <= 150:
            gamma = 1.5/1
            inv_gamma = 1/gamma
            table.append(((i/ 255.0) ** inv_gamma)*255.0)
        elif i >= 250:
            #map brighter pixles to dark value
            gamma = 1/5
            inv_gamma = 1/gamma
            table.append(((i/255.0) ** inv_gamma)*255.0)
        else:
            inv_gamma = 1/1
            table.append(((i/255.0) ** inv_gamma)*255.0)

    table = np.array(table).astype('uint8')    

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

