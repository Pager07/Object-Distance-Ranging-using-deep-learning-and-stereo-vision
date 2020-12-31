
THE ZIP FILE CONTAINS:
	- demoVideo.mp4
	- object_detection.py
	- stereo_to_3d.py
	- wls_filter.py
	- main.py
	- Mask_RCNN Folder
		- mask_rcnn_inception_v2_coco_2018_01_28 Sub-Folder

STEPS TO RUN THE WORK:

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

NOTES
- demoVideo.mp4
		- In the video: Fewer than objects are detected because I have set the object detection confidence score threshold to 0.97. 
		- This is becuase I wanted a cleaner video for demo such that distance label on top of the polygon is not overlapped by other objects polygon. 
		- If you wish to detect more object, reset the value of following variable in object_detection.py:
			- conf_threshold