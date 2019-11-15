#####################################################################

import cv2
import os
import numpy as np
import random
import object_detection

# code from stereo_disparity.py
#####################################################################
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
#####################################################################

# Code from stereo_to_3d.py
#####################################################################

# fixed camera parameters for this stereo setup (from calibration)

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0;
image_centre_w = 474.5;

## project_disparity_to_3d : project a given disparity image
## (uncropped, unscaled) to a set of 3D points with optional colour

if __name__ == "__main__":
    for filename_left in left_file_list:

        # skip forward to start a file we specify by timestamp (if this is set)
        if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
            continue;
        elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
            skip_forward_file_pattern = "";


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
            cv2.imshow('left image',imgL)
            imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
            cv2.imshow('right image',imgR)
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
            points = project_disparity_to_3d(disparity_scaled, max_disparity);
            points = np.array(points)
            map_2d_coord_to_Z = {}
            for row in points:
                3d_coords = row[0:2 + 1]
                2d_coord  = tuple(project_3D_points_to_2D_image_points(3d_coords))
                map_2d_coord_to_Z[2d_coord] = 3d_coords[2]
            yolo_image_output = object_detection.process_image(imgL , map_2d_coord_to_Z)
            
            
        else:
            print("-- files skipped (perhaps one is missing or not PNG)");
            print();











