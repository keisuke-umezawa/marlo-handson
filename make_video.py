import cv2
import glob

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video = cv2.VideoWriter('video.mp4', fourcc, 20.0, (640, 640))

paths = sorted(glob.glob('./results/*.png'))
for p in paths:
    img = cv2.imread(p)
    img = cv2.resize(img, (640, 640))
    video.write(img)

video.release()
