import os
import detect_yolov5
from json import loads
from utils.myutil import Globals
def zhi(m):
    s = m
    while s < 150 or s > 400:
        if s < 150:
            s *= 2
        else:
            s /= 2
    return s

if __name__ == '__main__':
    with open("F:/motion-monitor-x/labels/yolov5/字典.txt", 'r', encoding='utf-8') as f:
        content = f.read()
    Globals.yolov5_dict = loads(content)

    path = "F:/sm/Attachment/Attachment/Attachment 3"
    for root, dirs, files in os.walk(path):
        files = sorted(files, key=lambda x: int(x.split('.')[0].split(' ')[-1][1:-1]))
        for file in files:
            if file.endswith(".jpg"):
                detect_yolov5.run(source=os.path.join(root, file), weights="F:/motion-monitor-x/weights/yolov5s.pt",
                                  show_label=None, project="F:/sm/Attachment/Attachment/Attachment 4",
                                  save_img=True, show_window=None, classes=[29, 47, 51],
                                  max_det=200, conf_thres=0.05,
                                  iou_thres=0.4, line_thickness=2)

    with open("F:/sm/111.txt", "w") as file:
        for apple_num in Globals.apple_num:
            file.write(str(apple_num) + "\n")

    # with open("F:/sm/111.txt", "w") as file:
    #     for i in range(200):
    #         for j in range(len(Globals.apple_xy[i])):
    #             file.write(str(Globals.apple_cs[i][j]) + "\n")

    # with open("F:/sm/222.txt", "w") as file:
    #     for i in range(200):
    #         for j in range(len(Globals.apple_xy[i])):
    #             file.write(str(zhi(Globals.apple_mianji[i][j])) + "\n")

    with open("F:/sm/222.txt", "w") as file:
        for i in range(20705):
            file.write(str(i+1) + " " + str(Globals.apple_num[i]) + "\n")
            for j in range(len(Globals.apple_xy[i])):
                file.write(str(Globals.apple_xy[i][j]) + " "
                            + str(Globals.apple_mianji[i][j]) + " "
                            + str(zhi(Globals.apple_mianji[i][j])) + " "
                            + str(Globals.apple_cs[i][j]) + "\n")




