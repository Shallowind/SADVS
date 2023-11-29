import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
def calculate_apple_maturity(image_path):
    image = cv2.imread(image_path)

    # 将图像转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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

    cv2.imshow("red_mask", red_mask)



if __name__ == '__main__':
    image_path = "F:/sm/0.jpg"
    calculate_apple_maturity(image_path)
    cv2.waitKey(0)