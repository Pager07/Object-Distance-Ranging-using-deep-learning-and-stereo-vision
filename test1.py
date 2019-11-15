import cv2
import object_detection
frame = cv2.imread('/Users/sandeep/Desktop/Homework2019/ComputerVision Homework/TTBB-durham-02-10-17-sub10/left-images/1506942473.484027_L.png')
a,b  = object_detection.process_image(frame)
print(b)

