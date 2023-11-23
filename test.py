import sys
import os
import detect_yolov5
from json import loads
from utils.myutil import Globals
if __name__ == '__main__':
    with open("F:/motion-monitor-x/labels/yolov5/字典.txt", 'r', encoding='utf-8') as f:
        content = f.read()
    Globals.yolov5_dict = loads(content)

    path = "F:/sm/Attachment/Attachment/Attachment 1"
    for root, dirs, files in os.walk(path):
        files = sorted(files, key=lambda x: int(x.split('.')[0]))
        for file in files:
            if file.endswith(".jpg"):
                detect_yolov5.run(source=os.path.join(root, file), weights="F:/motion-monitor-x/weights/yolov5s.pt",
                                  show_label=None, project="F:/sm/0",
                                  save_img=True, show_window=None, classes=[29, 47, 51],
                                  max_det=200, conf_thres=0.05,
                                  iou_thres=0.4, line_thickness=2)

    print(Globals.apple_num)
    with open("F:/sm/111.txt", "w") as file:
        for apple_num in Globals.apple_num:
            file.write(str(apple_num) + "\n")
    print(Globals.apple_xy)
    with open("F:/sm/222.txt", "w") as file:
        for apple_xy in Globals.apple_xy:
            file.write(str(apple_xy) + "\n")