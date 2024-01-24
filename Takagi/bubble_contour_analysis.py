#!/usr/bin/env python
# coding: utf-8

# In[11]:


import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import pixel_converter
from assemble_frames_with_order_numbers import assemble_frames_with_order_numbers
from table_image_generator import create_table_image, find_min_max_avg_table

def create_directories(base_path, video_base_name, folder_names):
    """動画ごとのディレクトリとサブフォルダを作成する"""
    video_directory = os.path.join(base_path, video_base_name)
    os.makedirs(video_directory, exist_ok=True)
    for folder_name in folder_names:
        os.makedirs(os.path.join(video_directory, folder_name), exist_ok=True)
    return video_directory

def get_video_info(filename):
    """動画の基本情報を取得する"""
    cap = cv2.VideoCapture(filename)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, total_frame, fps


def preprocess_frame(frame):
    """フレームの前処理を行う"""
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.bilateralFilter(gray_image, 7, 15, 15)
    ret, binary = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return binary

def detect_and_filter_contours(binary, conditions):
    """輪郭の検出とフィルタリングを行う"""
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if conditions["min_contour_area"] <= area <= conditions["max_contour_area"]:
            filtered_contours.append(contour)
    return filtered_contours

# その他の関数（find_reference_point, calculate_angle_from_reference, draw_angle_points）は既に定義されているため省略
def find_reference_point(contour, centroid):
    cx, cy = centroid
    reference_point = None
    min_y = float('inf')  # 最小のy座標値を見つけるために無限大で初期化
    for point in contour:
        x, y = point[0]
        if x == cx and y < cy:  # 重心の真上に位置する点を探す
            if y < min_y:  # より上に位置する点を見つけた場合
                min_y = y
                reference_point = (x, y)
    return reference_point


def calculate_angle_from_reference(contour, centroid, reference_point):
    angles = []
    cx, cy = centroid
    rx, ry = reference_point
    reference_angle = math.degrees(math.atan2(ry - cy, rx - cx))

    for point in contour:
        x, y = point[0]
        angle = math.degrees(math.atan2(y - cy, x - cx)) - reference_angle
        if angle < 0:
            angle += 360
        angles.append(angle)
    return angles

def draw_angle_points(image, contours, angles):
    target_angles = [0, 90, 180, 270]  # 対象とする角度
    colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]  # 各角度に対応する色

    for contour, angle_set in zip(contours, angles):
        for target_angle, color in zip(target_angles, colors):
            # 対象の角度に最も近い点を見つける
                        closest_point_index = min(range(len(contour)), key=lambda i: abs(angle_set[i] - target_angle))
                        closest_point = contour[closest_point_index]
                        cv2.circle(image, tuple(closest_point[0]), 5, color, -1)

    return image

def calculate_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cy  # y座標のみを返す
    else:
        return None

def draw_contours_and_centroid(frame, contours, save_directory):
    for idx, contour in enumerate(contours):
        # 重心の計算
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        # 輪郭と重心を描画
        contour_image = frame.copy()
        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)
        cv2.circle(contour_image, (cx, cy), 5, (0, 0, 255), -1)

        # 描画した画像を保存
        save_path = os.path.join(save_directory, f"contour_centroid_{frame_count}_{idx}.bmp")
        cv2.imwrite(save_path, contour_image)

def draw_and_save_angle_images(frame, contours, angles, save_directory):
    for idx, (contour, angle_set) in enumerate(zip(contours, angles)):
        angle_image = draw_angle_points(frame.copy(), [contour], [angle_set])
        
        # 描画した画像を保存
        save_path = os.path.join(save_directory, f"angle_image_{frame_count}_{idx}.bmp")
        cv2.imwrite(save_path, angle_image)
        
def calculate_vertical_velocity(prev_cy, current_cy, dt):
    if prev_cy is None or current_cy is None:
        return None
    dy = current_cy - prev_cy
    velocity = dy / dt  # 単位はピクセル/秒
    return velocity

def pixel_to_mm(pixel_value, dpi):
    mm_per_inch = 25.4
    mm_value = (pixel_value / dpi) * mm_per_inch
    return mm_value

def calculate_sphere_volume_from_radius(radius):
    # 球の体積を計算
    volume = (4/3) * math.pi * radius**3
    return volume

def process_spherical_bubble(contour):
    # 重心の計算
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    else:
        # ゼロ除算を避けるための処理
        cx, cy = 0, 0

    # 最大半径の計算
    max_radius = max(math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in contour[:, 0])

    # 球の体積の計算
    volume = calculate_sphere_volume_from_radius(max_radius)
    return volume

# メイン処理
# filename = "C:\\Users\\flow\\Desktop\\bubbledate\\spherical bubbleGray.mp4"
filename = "C:\\Users\\flow\\Desktop\\今さんデータ\\indefinite(1).mp4"
base_path = "C:\\Users\\flow\\Desktop\\bubble_rasing_result\\"
video_base_name = os.path.splitext(os.path.basename(filename))[0]
folder_names = ["original", "gray", "binary", "contours", "angles"]

# 動画情報の取得
video_directory = create_directories(base_path, video_base_name, folder_names)
width, height, total_frame, fps = get_video_info(filename)
dpi = pixel_converter.get_video_dpi(filename)
# 動画情報の出力
print(f"Video: {filename}")
print(f"Resolution: {width}x{height}")
print(f"Total Frames: {total_frame}")
print(f"FPS: {fps}")
print(f"DPI: {dpi}")
print(f"video_directory:", video_directory)


# 各種画像保存用ディレクトリの作成
directories = {
    "original": os.path.join(video_directory, "original"),
    "gray": os.path.join(video_directory, "gray"),
    "binary": os.path.join(video_directory, "binary"),
    "contours": os.path.join(video_directory, "contours"),
    "angles": os.path.join(video_directory, "angles")
}
for dir_path in directories.values():
    os.makedirs(dir_path, exist_ok=True)
    print(dir_path)

# ビデオ処理
cap = cv2.VideoCapture(filename)
frame_count = 0
test_frame_limit = 10  # テストに使用するフレーム数

while frame_count < test_frame_limit:
    ret, frame = cap.read()
    if not ret:
        break
    print(f"Processing frame {frame_count}/{total_frame}")

    # フレームの前処理
    binary = preprocess_frame(frame)

    # 輪郭検出とフィルタリング
    conditions = {"min_contour_area": 5000, "max_contour_area": 20000}
    filtered_contours = detect_and_filter_contours(binary, conditions)

    # 角度計算と描画
    for contour in filtered_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            reference_point = find_reference_point(contour, (cx, cy))
            angles = calculate_angle_from_reference(contour, (cx, cy), reference_point)
            
            draw_and_save_angle_images(frame, [contour], [angles], directories["angles"])
            draw_contours_and_centroid(frame, [contour], directories["contours"])
        
        
    # オリジナル、グレースケール、二値化画像の保存
    cv2.imwrite(os.path.join(directories["original"], f"frame_{frame_count}.bmp"), frame)
    cv2.imwrite(os.path.join(directories["gray"], f"frame_{frame_count}.bmp"), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cv2.imwrite(os.path.join(directories["binary"], f"frame_{frame_count}.bmp"), binary)
    
    frame_count += 1

cap.release()



