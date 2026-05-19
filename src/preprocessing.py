import cv2
import numpy as np

def load_frames(folder):
    import os
    frames = []
    files = sorted(os.listdir(folder))

    for f in files:
        path = os.path.join(folder, f)
        frame = cv2.imread(path)
        frames.append(frame)

    return frames


def bgr_to_ycbcr(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)


def preprocess_frames(frames):
    return [bgr_to_ycbcr(f) for f in frames]

