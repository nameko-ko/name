import cv2
import numpy as np

def assemble_frames(frame_list, rows, cols):
    frame_height, frame_width, _ = frame_list[0].shape
    canvas = np.zeros((frame_height * rows, frame_width * cols, 3), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(frame_list):
                canvas[i * frame_height: (i + 1) * frame_height, j * frame_width: (j + 1) * frame_width] = frame_list[idx]

    return canvas
