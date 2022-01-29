import cv2 as cv
import numpy as np
from datetime import datetime

# video capture
capture = cv.VideoCapture("theaf")


# background segmentation
fgbg = cv.createBackgroundSubtractorMOG2(50, 200, True)

# keeps track of frame
frame_count = 0
while (1):
    # return value and current frame
    ret, frame = capture.read()
    # check if current frame exist
    if not ret:
        break
    frame_count += 1
    # resizing the frame
    resized_frame = cv.resize(frame, (0, 0), fx=1.1, fy=1.1)
    # get the foreground object
    fg_object = fgbg.apply(resized_frame)
    # count all non-zero pixels within the object
    count = np.count_nonzero(fg_object)
    print("Frame:%d, Pixel Count: %d" % (frame_count, count))
    # how many pixels you want to detect to be considered "movement"
    if (frame_count > 1 and count > 1000):
        cv.putText(resized_frame, "someone is stealing ur things!!", (10, 50),
                   cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    cv.imshow("Frame", resized_frame)
    cv.imshow("Object", fg_object)
    key = cv.waitKey(30) & 0xff
    if key == 27:
        break
capture.release()
cv.destroyAllWindows()
