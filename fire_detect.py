import numpy as np
import cv2

# 初始化背景减法器和结构元素
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

# 定义火焰颜色范围
lower_fire = np.array([18, 50, 50], dtype='uint8')
upper_fire = np.array([35, 255, 255], dtype='uint8')

'''lower_smoke = [100, 100, 100]
upper_smoke = [124, 180, 180]'''

def detect_fire(frame):
    # 转换为 HSV 颜色空间并模糊处理
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (21, 21), 0)
    
    '''lower_smoke_array = np.array(lower_smoke, dtype='uint8')
    upper_smoke_array = np.array(upper_smoke, dtype='uint8')
    smoke_mask = cv2.inRange(hsv, lower_smoke_array, upper_smoke_array)'''
    
    # 火焰颜色掩码
    fire_mask = cv2.inRange(blur, lower_fire, upper_fire)
    
    # 背景减法
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    fmask = fgbg.apply(gray_blur)
    
    # 形态学操作
    fmask = cv2.medianBlur(fmask, 5)
    fmask = cv2.erode(fmask, kernel, iterations=2)
    fmask = cv2.dilate(fmask, kernel, iterations=2)
    
    # 结合火焰掩码和背景掩码
    combined_mask = cv2.bitwise_and(fmask, fire_mask)
    
    # 轮廓检测
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # 面积阈值
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return frame
