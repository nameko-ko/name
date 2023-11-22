import cv2
import numpy as np
def assemble_frames_with_order_numbers(frame_list, rows, cols):
    frame_height, frame_width, _ = frame_list[0].shape
    canvas = np.zeros((frame_height * rows, frame_width * cols, 3), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(frame_list):
                frame_with_number = frame_list[idx].copy()  # Create a copy of the frame
                order_number = idx  # Display order number starting from 0
                # Display order number on the frame
                cv2.putText(frame_with_number, str(order_number), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                canvas[i * frame_height: (i + 1) * frame_height, j * frame_width: (j + 1) * frame_width] = frame_with_number

    return canvas