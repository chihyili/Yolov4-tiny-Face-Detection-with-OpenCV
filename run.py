import numpy as np
import sys
import os
import cv2
import glob
import ipdb
from pathlib import Path
from model import yolo


config_file = "./config/config.cfg"

# face detection
model_path = "weight/fd_yolov4-tiny.weights"
label = "face"

# save output video 
SAVE_RESULT_VID = True


if len(sys.argv) != 2:
    print("Please enter: python run_human_detection.py frame_folder")
    exit()
else:
    frame_folder = sys.argv[1]


yolo_detect = yolo.Yolo(model_path,config_file,label)

folder = os.path.join(frame_folder,"*jpg")
image_list  = glob.glob(folder)

if SAVE_RESULT_VID:
    image = cv2.imread(image_list[0])
    image_out = yolo_detect.run(image)
    w = image_out.shape[1]
    h = image_out.shape[0]
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_out = cv2.VideoWriter('%s_detection.mp4'%label, fourcc, fps, (w, h), True)
    

for image_path in image_list:

    image = cv2.imread(image_path)
    image_out = yolo_detect.run(image)


    cv2.imshow("%s detectoin"%label,image_out)
    if SAVE_RESULT_VID:
        print(image_out.shape)
        vid_out.write(image_out)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        vid_out.release()
        break


