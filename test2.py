import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
def calculate_apple_maturity(image,x1,y1,x2,y2):

    image = image[y1:y2, x1:x2]


    radius = min(image.shape[0]/2, image.shape[1]/2)
    mianji = math.pi * radius ** 2
    # 将图像转换为HSV颜色空间
    try:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    except:
        return "低成熟度"
    # 定义红色的范围
    lower_red_range1 = np.array([0, 40, 40])
    upper_red_range1 = np.array([10, 255, 255])
    lower_red_range2 = np.array([170, 40, 40])
    upper_red_range2 = np.array([180, 255, 255])

    # 创建红色的掩膜
    red_mask1 = cv2.inRange(hsv_image, lower_red_range1, upper_red_range1)
    red_mask2 = cv2.inRange(hsv_image, lower_red_range2, upper_red_range2)

    # 合并红色的掩膜
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    red_area_size = cv2.countNonZero(red_mask)

    if red_area_size < mianji / 3:
        return "低成熟度"
    else:
        # 获取红色区域内每个像素的颜色深度
        red_color_depth = hsv_image[red_mask > 0, 2]

        # 计算所有红色位置的颜色深度综合
        total_color_depth = np.sum(red_color_depth)

        # 将平均颜色深度映射到 [0, 1] 范围作为成熟度
        maturity = total_color_depth / (255 * red_area_size)

        if maturity < 0.5:
            return "中成熟度"
        else:
            return "高成熟度"

