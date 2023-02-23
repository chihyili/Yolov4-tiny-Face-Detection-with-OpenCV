import numpy as np
import cv2
import argparse

from model import yolov4-tiny

label = "face"

# save output video 
SAVE_RESULT_VID = True
def parser():
    parser = argparse.ArgumentParser(description="YOLO Face Detection")
    parser.add_argument("--input", type=str,
                        help="video source")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="./weights/FD_v1.weights",
                        help="yolo weights path")
    parser.add_argument("--config_file", default="./config/config.cfg",
                        help="path to config file")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display")
    return parser.parse_args()

def set_saved_video(fps, out_filename):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_out = cv2.VideoWriter(out_filename, fourcc, fps, (416, 416), True)
    return vid_out

if __name__ == '__main__':
    args = parser()

    yolo_detect = yolov4-tiny.Yolo(args.weights, args.config_file, label)
    capture = cv2.VideoCapture(args.input)
    fps = capture.get(cv2.CAP_PROP_FPS)

    if args.out_filename is not None:
        vid_out = set_saved_video(fps, args.out_filename)

    if not capture.isOpened():
        print("Cannot open video")
        exit()
    else:
        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                image_out = yolo_detect.run(frame)
                if not args.dont_show:
                    cv2.imshow("%s detectoin"%label, image_out)
                    cv2.waitKey(1)
                if args.out_filename is not None:
                    vid_out.write(image_out)
            else:
                cv2.destroyAllWindows()
                vid_out.release()
                capture.release()


