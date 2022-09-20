import os
from math import tan, radians
import cv2
from functions import *

videodir = os.getcwd() + "\\line_track.mp4"
video = cv2.VideoCapture(videodir)

max_gap = 40

while True:
    ret, frame = video.read()
    if not ret:
        print("Video generation error")
        cv2.destroyAllWindows()
        video.release()
        break

    # preprocess frame to find lines
    frame_processed = preprocess(frame, (5, 5), (100, 200))
    print(frame_processed.shape)
    frame_processed = ROI(frame_processed, (frame_processed.shape[1]*0.2, frame_processed.shape[1]*0.8),
                          (frame_processed.shape[0]*0.2, frame_processed.shape[0]*0.8))

    size_x, size_y = frame_processed.shape[0], frame_processed.shape[1]

    lines_right, lines_left, lines_side_left, lines_side_right = get_lines(frame_processed, (size_x, size_y), max_gap)

    lines = []
    lines.extend(lines_right)
    lines.extend(lines_left)
    lines.extend(lines_side_left)
    lines.extend(lines_side_right)

    frame_line = draw_lines(frame, BGRColorPalette.RED.value, *lines)

    frame_last = cv2.addWeighted(src1=frame, alpha=0.8, src2=frame_line, beta=1, gamma=0.)

    cv2.imshow("frame", frame_last)


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):

        cv2.destroyAllWindows()
        video.release()
        break

    elif key == ord("p"):

        while True:

            key_cancel = cv2.waitKey(1) & 0xFF

            if key_cancel == ord("p"):
                print("p pressed")
                break
