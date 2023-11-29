import os
import detect_yolov5
from utils.myutil import Globals

# 根据面积求质量
def zhi(area):
    k = 1.3
    density = 235
    return k * area * density


if __name__ == '__main__':
    # 加载文件路径
    path = "F:/sm/Attachment/Attachment/Attachment 3"
    # 遍历所有图片
    for root, dirs, files in os.walk(path):
        files = sorted(files, key=lambda x: int(x.split('.')[0].split(' ')[-1][1:-1]))
        for file in files:
            if file.endswith(".jpg"):
                # 使用YOLOv5模型进行预测
                detect_yolov5.run(source=os.path.join(root, file), weights="F:/motion-monitor-x/weights/yolov5s.pt",
                                  show_label=None, project="F:/sm/Attachment/Attachment/Attachment 4",
                                  save_img=True, show_window=None, classes=29,
                                  max_det=200, conf_thres=0.15,
                                  iou_thres=0.4, line_thickness=2)

    # 保存每张图片中苹果的数量
    with open("F:/sm/111.txt", "w") as file:
        for i in range(200):
            for j in range(len(Globals.apple_xy[i])):
                file.write(str(Globals.apple_cs[i][j]) + "\n")

    # 保存每张图片中苹果的数量位置、质量和成熟度
    with open("F:/sm/222.txt", "w") as file:
        for i in range(200):
            file.write(str(len(Globals.apple_xy[i])) + "\n")
            for j in range(len(Globals.apple_xy[i])):
                file.write(str(zhi(Globals.apple_mianji[i][j])) + "\n" +
                           str(Globals.apple_xy[i][j][0]) + "\n" +
                           str(zhi(Globals.apple_mianji[i][j])) + "\n" +
                           str(Globals.apple_cs[i]) + "\n")






