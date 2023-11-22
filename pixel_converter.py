import cv2

def get_video_dpi(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            dpi = frame.shape[1]  # 幅をDPI情報として使用
            cap.release()
            return dpi
        else:
            cap.release()
            return None
    except Exception as e:
        return str(e)

def pixel_to_mm(pixel_value, dpi):
    mm_per_inch = 25.4
    mm_value = (pixel_value / dpi) * mm_per_inch
    return mm_value