import cv2
import os
import numpy as np
import random

import object_detection
import stereo_to_3d


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


# In[ ]:


counter = 0
for filename_left in left_file_list:

    # from the left image filename get the correspondoning right image
    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);


    # for sanity print out these filenames
    print(full_path_filename_left);
    print(full_path_filename_right);
    print();

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        print("-- files loaded successfully");
        print();


        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images
        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);

        # compute disparity image from undistorted and rectified stereo images
        # that we have loaded
        disparity = stereoProcessor.compute(grayL,grayR);
        dispNoiseFilter = 5; # increase for more agressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);
        _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
        disparity_scaled = (disparity / 16.).astype(np.uint8);

        # project to a 3D colour point cloud (with or without colour)
        points = stereo_to_3d.project_disparity_to_3d(disparity_scaled, max_disparity,imgL);
        points = np.array(points)
        map_2d_coord_to_Z = {}
        for row in points:
            coords_3d = row[0:3]
            coords_2d  = stereo_to_3d.project_3D_points_to_2D_image_points([coords_3d])
            coords_2d_tuple = (int(coords_2d[0][0]),int(coords_2d[0][1]))
#             print(coords_2d_tuple)
            map_2d_coord_to_Z[coords_2d_tuple] = coords_3d[2]   
        yolo_image_output = object_detection.process_image(imgL, map_2d_coord_to_Z)
        cv2.imshow('Distance detection',yolo_image_output)

        key = cv2.waitKey(40 * (not (pause_playback))) & 0xFF;  # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):  # exit
            break;  # exit
        elif (key == ord('s')):  # save
            cv2.imwrite("yolo-image-ouput.png", yolo_image_output);
        elif (key == ord('c')):  # crop
            crop_disparity = not (crop_disparity);
        elif (key == ord(' ')):  # pause (on next frame)
            pause_playback = not (pause_playback);
    else:
        print("-- files skipped (perhaps one is missing or not PNG)");
        print();


# In[ ]:




