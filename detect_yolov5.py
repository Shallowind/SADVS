import argparse
import os
import platform
import sys
import math
from pathlib import Path
import torch
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.myutil import Globals
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights='yolov5s.pt',  # 模型路径或triton URL
        source='data/images',  # 文件/目录/URL/glob/screen/0（摄像头）
        data='',  # dataset.yaml路径
        imgsz=(640, 640),  # 推断尺寸（高度，宽度）
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # NMS IOU阈值
        max_det=1000,  # 每张图片的最大检测数
        device='',  # cuda设备，例如0或0,1,2,3或cpu
        view_img=False,  # 显示结果
        save_txt=False,  # 保存结果到*.txt
        save_conf=False,  # 保存标签中的置信度
        save_crop=False,  # 保存裁剪的预测框
        save_img=True,  # 保存图像
        classes=None,  # 按类别过滤：--class 0，或--class 0 2 3
        agnostic_nms=False,  # 类别无关的NMS
        augment=False,  # 增强的推断
        visualize=False,  # 可视化特征
        update=False,  # 更新所有模型
        project='result',  # 结果保存到project/name
        name='',  # 结果保存到project/name
        exist_ok=True,  # 已存在的project/name可以，不要增加计数
        line_thickness=3,  # 边框厚度（像素）
        hide_labels=False,  # 隐藏标签
        hide_conf=False,  # 隐藏置信度
        half=False,  # 使用FP16半精度推断
        dnn=False,  # 使用OpenCV DNN进行ONNX推断
        vid_stride=1,  # 视频帧率步长
        show_label=None,  # 显示标签
        use_camera=False,  # 使用摄像头
        show_window=None  # 显示窗口
):
    # 将source转换为字符串类型
    source = str(source)
    # 增加路径
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 如果路径已存在，则忽略错误
    # 创建目录
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # 如果save_txt为True，则创建web_dir/labels目录，否则创建web_dir目录
    # 判断source是否为数字
    webcam = source.isnumeric()
    # 选择设备
    device = select_device(device)
    # 加载模型
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # 获取模型的步长、名称和点数
    stride, names, pt = model.stride, model.names, model.pt
    # 检查图像大小
    imgsz = check_img_size(imgsz, s=stride)  # 将imgsz与stride进行比较并返回比较结果

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)  # 将numpy数组转换成torch张量，并放到指定设备上
            im = im.half() if model.fp16 else im.float()  # 将张量类型转换为半精度（fp16）或单精度（fp32）
            im /= 255  # 将图像像素值从0-255归一化到0.0-1.0的范围
            if len(im.shape) == 3:
                im = im[None]  # 增加批处理维度

        # 推理
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)  # 进行模型推理

        # 非极大值抑制
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms,
                                       max_det=max_det)  # 对预测结果进行非极大值抑制处理

        # 第二阶段分类器（可选）
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # 处理预测结果

        for i, det in enumerate(pred):  # 对于每张图片
            seen += 1  # 增加计数器
            if webcam:  # 如果是webcam模式
                p, im0, frame = path[i], im0s[i].copy(), dataset.count  # 获取路径、原始图像和帧数
                s += f'{i}: '  # 拼接字符串
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)  # 获取路径、原始图像和帧数
            p = Path(p)  # 将路径转换为Path对象
            save_path = str(save_dir / p.name)  # 获取保存路径
            s += '%gx%g ' % im.shape[2:]  # 拼接字符串
            annotator = Annotator(im0, line_width=line_thickness, example=str(names) + '汉字')  # 创建标注器对象

            if len(det):
                # 如果检测结果不为空，则对框进行重新缩放，将图像大小从img_size调整为im0大小
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印结果
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # 整数类别
                    name = Globals.yolov5_dict[names[c]]
                    label = None if hide_labels else (name if hide_conf else f'{name} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            im1 = im0.astype("uint8")
            show = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * 3, QImage.Format_RGB888)
            if show_window is not None:
                # label_size = show_label.size()
                # label_size.setWidth(label_size.width() - 10)
                # label_size.setHeight(label_size.height() - 10)
                # scaled_image = showImage.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                # pixmap = QPixmap.fromImage(scaled_image)
                # show_label.setPixmap(pixmap)
                # show_label.setAlignment(Qt.AlignCenter)

                scale_factor = min(show_window.player_2.width() / showImage.width(),
                                   show_window.player_2.height() / showImage.height())

                # 计算新的宽度和高度
                new_width = int(showImage.width() * scale_factor)
                new_height = int(showImage.height() * scale_factor)

                # 设置新的最大大小
                show_window.camera_2.setMaximumSize(new_width, new_height)

                show_window.camera_2.setPixmap(QPixmap(showImage))
                show_window.camera_2.setScaledContents(True)

            # 如果需要保存图像
            if save_img:
                # 如果数据集模式为'image'
                if dataset.mode == 'image':
                    # 使用OpenCV保存图像到指定路径
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    # 如果视频路径不等于保存路径时（即需要保存到新的路径）
                    if vid_path[i] != save_path:  # new video
                        # 更新视频路径
                        vid_path[i] = save_path
                        # 如果视频写入器不为None（即已经存在）
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            # 释放之前的视频写入器
                            vid_writer[i].release()  # release previous video writer
                        # 判断是否有视频输入流
                        if vid_cap:  # video
                            # 获取视频的帧率
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            # 获取视频的宽度
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            # 获取视频的高度
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            # 如果是流媒体，则固定帧率为30，宽度为im0的宽度，高度为im0的高度
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        # 修改保存路径的后缀为'.mp4'
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        # 创建新的视频写入器
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    # 将图像写入视频
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # 关闭摄像头->退出
        if not Globals.camera_running and use_camera:
            dataset.cap.release()  # 释放摄像头
            break
