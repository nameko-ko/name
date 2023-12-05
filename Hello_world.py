import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize

def SearchCF(params,d,theta_cal,r_obs):
    a,b,c = params
    r_cal = []
    for i,_ in enumerate(theta_cal): #関数部
        theta_prime = (d/3 * theta_cal[i]-c) % (np.pi/1.5)
        r_cal.append(a * (b * (1 / np.cos(theta_prime - np.pi/3) - 2) + 2))
    MSE = 0 #初期化
    for r_obs_i, r_cal_i in zip (r_obs,r_cal):
        MSE += (r_obs_i - r_cal_i) ** 2 / r_cal_i ** 2
    MSE /= len(r_obs)
    return MSE #計算した相対誤差を返す

def drawContour(params,d):
    a,b,c = params
    r = [] #初期化
    #d = 3
    theta = np.linspace(0, 2*np.pi,1000)
    for i,_ in enumerate(theta):
        theta_prime = (d/3 * theta[i] - c) % (np.pi/1.5)
        r.append(a * (b * (1 / np.cos(theta_prime - np.pi/3) - 2) + 2))
    #極座標変換
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    #グラフプロット
    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot(2,2,1)
    plt.rcParams["font.family"] = "Arial"
    ax.plot(x,y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    textstr = '\n'.join((
        r'$a=%.2f$' % (a, ),
        r'$b=%.2f$' % (b, ),
        r'$c=%.2f$' % (c, ),
        r'$d=%.2f$' % (d, ),
        r'$r=a(b(sec(θ - π/3)-2)+2)$'
    ))
    props = dict(boxstyle='round',facecolor='white',alpha = 0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    ax.set_xlim([-80,80])
    ax.set_ylim([-50,50])
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

#main関数
cap = cv2.VideoCapture('video.mp4')
number_list = []
num = 0
frame_count = 0  # フレーム番号のカウント

for i in range(4) #while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        #読み込みエラー時の分岐
        if frame is None:
            print(f"Error: Failed to load image from here")
            sys.exit(1)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        blurred_image = cv2.bilateralFilter(gray_image,7,15,15) #variable
        denoised_image = cv2.medianBlur(blurred_image,5) #variable
        ret,binary = cv2.threshold(denoised_image,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #変数retは使用した閾値、変数binaryは二値画像を格納
        contours,_ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #返り値の二つ目の階層情報は捨てる
        #以下、面積フィルタリング
        conditions = {"min_contour_area" : 2000, "max_contour_area" : 5000}
        filtered_contours = [] #初期化
        moment_data = [] #初期化
        for _, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)
            if (
                ("min_contour_area" not in conditions or contour_area >= conditions["min_contour_area"]) and
                ("max_contour_area" not in conditions or contour_area <= conditions["max_contour_area"])
            ):
                filtered_contours.append(contour)
                M = cv2.moments(contour)
                filtered_cx = int(M['m10'] / M['m00']) if M["m00"] != 0 else 0
                filtered_cy = int(M['m01'] / M['m00']) if M["m00"] != 0 else 0
                moment_data.append((filtered_cx, filtered_cy))
                
        #main関数の続き
        result_image = color_image.copy() #result_imageの上にプロットしていく
        d = None
        if d == None: 
            for contour, moment in zip(filtered_contours, moment_data):
            M = cv2.moments(contour)
            rel_con = []
            if M["m00"] != 0:
                filtered_cx, filtered_cy = moment
                # 重心に点を描画
                cv2.circle(result_image, (filtered_cx, filtered_cy), 5, (0, 0, 255), -1)
                rel_con = contour - moment
            # 抽出したcontourの相対座標をx, yリストに分離
            con_x = []
            con_y = []
            for i, con in enumerate(rel_con):
                con_x.append(con[0][0])
                con_y.append(con[0][1]) 
            # r, θの計算
            r_obs = []
            theta_obs = []
            for i, _ in enumerate(con_x):
                r_obs.append(np.sqrt(con_x[i]**2 + con_y[i]**2))
                theta_obs.append(np.arctan2(con_y[i], con_x[i]))
            #パラメータ探索
            #dは0~5で離散的に探索する
            for i in range(5):
                initial_guess = [21, 0.8, 1.5]
                result = minimize(SearchCF, initial_guess, args=(d,theta_obs, r_obs), method='Nelder-Mead', options={'disp': True})
                # 結果の表示
                print("Optimal parameters:", result.x)
                print("Minimum value:", result.fun)
                #drawContour(result.x,d)
                if result.fun < MSE_standard:
                    MSE_standard = result.fun
                    for j,item in enumerate(result.x):
                        params_result[j] =  result.x[j]
                    d = j
            drawContour(params_result,d)
            continue

        for contour, moment in zip(filtered_contours, moment_data):
            M = cv2.moments(contour)
            rel_con = []
            if M["m00"] != 0:
                filtered_cx, filtered_cy = moment
                # 重心に点を描画
                cv2.circle(result_image, (filtered_cx, filtered_cy), 5, (0, 0, 255), -1)
                rel_con = contour - moment
            # 抽出したcontourの相対座標をx, yリストに分離
            con_x = []
            con_y = []
            for i, con in enumerate(rel_con):
                con_x.append(con[0][0])
                con_y.append(con[0][1]) 
            # r, θの計算
            r_obs = []
            theta_obs = []
            for i, _ in enumerate(con_x):
                r_obs.append(np.sqrt(con_x[i]**2 + con_y[i]**2))
                theta_obs.append(np.arctan2(con_y[i], con_x[i]))
            #パラメータ探索
            initial_guess = [21, 0.8, 1.5]
            result = minimize(SearchCF, initial_guess, args=(d,theta_obs, r_obs), method='Nelder-Mead', options={'disp': True})
            # 結果の表示
            print("Optimal parameters:", result.x)
            print("Minimum value:", result.fun)
            drawContour(result.x,d)


# 結果の表示
#plt.imshow(result_image)
#plt.title('Processed Image')
#plt.savefig("/Users/nametakouhei/name/画像ファイル/Processed Image.png")
#plt.show()