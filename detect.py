import argparse
import os
import platform
import sys
import time
import threading
import random
import json

import numpy as np
from pathlib import Path
import torch
from PyQt5.QtWidgets import QMessageBox

from deep_sort.deep_sort import DeepSort
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image, )
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from mainwindow import MainWindow
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.myutil import Globals
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


def func_slowfast(vid_cap, idx, stack, yolo_pred, img_size, device, video_model):
    # 获取视频的帧率
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    # 打印正在处理的秒数
    print(f"processing {idx // fps}th second clips")
    # 将图像栈中的图像转换为张量，并拼接为一个视频剪辑
    stack_tensor = [to_tensor(img) for img in stack]
    clip = torch.cat(stack_tensor).permute(-1, 0, 1, 2)
    # 如果有YOLO预测框
    if yolo_pred[0].shape[0]:
        # 对视频剪辑和YOLO预测框进行AVA推理转换
        inputs, inp_boxes, _ = ava_inference_transform(clip, yolo_pred[0][:, 0:4], crop_size=img_size)
        # 在YOLO预测框中添加一个全零行
        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
        # 根据设备放置输入数据
        if isinstance(inputs, list):
            inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
        else:
            inputs = inputs.unsqueeze(0).to(device)
        # 不进行梯度计算
        with torch.no_grad():
            # 使用视频模型进行预测
            slowfaster_preds = video_model(inputs, inp_boxes.to(device))
            # 将预测结果转为CPU张量
            slowfaster_preds = slowfaster_preds.cpu()
        # 返回预测结果
        return slowfaster_preds



def my_uniform_temporal_subsample(
        x: torch.Tensor, num_samples: int, temporal_dim: int = -3
) -> torch.Tensor:
    """
    从视频的时序维度均匀抽样得到num_samples个索引。
    当num_samples大于视频时域维度的大小时，将根据最近邻插值方法进行抽样。

    参数:
        x (torch.Tensor): 大于一时维度的视频张量，张量类型包括int、long、float、complex等。
        num_samples (int): 要选择的等距样本数量
        temporal_dim (int): 执行时域抽样的维度。

    返回:
        一个具有降采样时域维度的类似x的张量。
    """

    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    # 如果num_samples > t，则通过最近邻插值进行抽样
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    indices = indices.cuda()
    return torch.index_select(x, temporal_dim, indices)

# 对跟踪器进行更新，并返回更新后的结果。
def deepsort_update(Tracker, pred, xywh, np_img):
    outputs = Tracker.update(xywh, pred[:, 4:5], pred[:, 5].tolist(), cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    return outputs


def to_tensor(img):
    # 将图像从BGR格式转换为RGB格式，并使用torch库创建张量
    img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 在张量上增加一个维度，以便于处理
    return img.unsqueeze(0)



def ava_inference_transform(
        clip,
        boxes,
        num_frames=32,  # 如果使用slowfast_r50_detection，将此更改为32、4 for slow。
        crop_size=640,
        data_mean=[0.45, 0.45, 0.45],
        data_std=[0.225, 0.225, 0.225],
        slow_fast_alpha=4,  # 如果使用slowfast_r50_detection，将此更改为4，None for slow。
):
    """
     对输入的视频帧和目标框进行推断前的转换处理

     参数：
         - clip：输入的视频帧
         - boxes：输入的目标框
         - num_frames：均衡抽样时的帧数，默认为32
         - crop_size：视频帧的裁剪尺寸，默认为640
         - data_mean：数据的均值，默认为[0.45, 0.45, 0.45]
         - data_std：数据的标准差，默认为[0.225, 0.225, 0.225]
         - slow_fast_alpha：慢速和快速路径的采样比例，默认为4

     返回：
         - 转换后的视频帧
         - 转换后的目标框（以numpy数组形式）
         - ROI框的副本
     """
    # boxes = np.array(boxes)
    # boxes = np.floor(boxes)
    roi_boxes = boxes.copy()
    clip = clip.cuda()
    clip = my_uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width)
    clip, boxes = short_side_scale_with_boxes(clip, size=crop_size, boxes=boxes, )
    clip = normalize(clip,
                     np.array(data_mean, dtype=np.float32),
                     np.array(data_std, dtype=np.float32), )
    boxes = clip_boxes_to_image(boxes, clip.shape[2], clip.shape[3])
    if slow_fast_alpha is not None:
        fast_pathway = clip
        slow_pathway = torch.index_select(clip, 1,
                                          torch.linspace(0, clip.shape[1] - 1,
                                                         clip.shape[1] // slow_fast_alpha).long().cuda())
        clip = [slow_pathway, fast_pathway]

    return clip, torch.from_numpy(boxes), roi_boxes


@smart_inference_mode()
def run(
        weights='yolov5s.pt',  # 模型路径或triton URL
        source='data/images',  # 文件/目录/URL/glob/screen/0（摄像头）
        data='',  # dataset.yaml路径
        imgsz=(640, 640),  # 推断尺寸（高度，宽度）
        conf_thres=0.4,  # 置信度阈值
        iou_thres=0.4,  # NMS IOU阈值
        max_det=100,  # 每张图片的最大检测数
        device='',  # cuda设备，例如0或0,1,2,3或cpu
        view_img=False,  # 显示结果
        save_txt=False,  # 保存到*.txt结果
        save_conf=False,  # 保存标签中的置信度
        save_crop=False,  # 保存裁剪的预测框
        save_img=True,  # 保存图像
        classes=None,  # 按类别过滤：--class 0，或--class 0 2 3
        agnostic_nms=False,  # 无类别NMS
        augment=False,  # 增强推理
        visualize=False,  # 可视化特征
        update=False,  # 更新所有模型
        project='result',  # 结果保存到project/name
        name='',  # 结果保存到project/name
        exist_ok=True,  # 已存在的project/name可以，不需要自增
        line_thickness=2,  # 边框厚度（像素）
        hide_labels=False,  # 隐藏标签
        hide_conf=False,  # 隐藏置信度
        half=False,  # 使用FP16半精度推理
        dnn=False,  # 使用OpenCV DNN进行ONNX推理
        vid_stride=1,  # 视频帧率步长
        show_label=None,  # 显示标签
        use_camera=False,  # 使用摄像头
        show_window=None,  # 显示窗口
        select_labels=None  # 选择标签
):

    source = str(source)  # 将source转换为字符串类型
    # 目录操作
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 增加路径序号
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录
    webcam = source.isnumeric()  # 判断source是否为数字
    # 加载模型
    device = select_device(device)  # 选择设备
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # 加载模型
    stride, names, pt = model.stride, model.names, model.pt  # 获取模型的步长、名称和pt值
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像大小
    # 数据加载器
    bs = 1  # 批次大小
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # 加载实时视频流数据
        bs = len(dataset)  # 更新批次大小为实时视频流的帧数
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # 加载图像数据
    vid_path, vid_writer = [None] * bs, [None] * bs  # 初始化视频路径和视频写入器的列表


    # 运行推理
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # 模型热身
    video_model = slowfast_r50_detection(True).eval().to(device)  # 创建视频模型
    id_to_ava_labels = {}
    id_to_labels = {}
    ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")  # 读取标签映射
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())  # 初始化变量 seen、windows、dt
    stack = []  # 创建空列表 stack
    color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]  # 创建颜色映射表 color_map
    idx = 0  # 初始化变量 idx
    deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")  # 创建 DeepSort 追踪器
    dict_text = {}  # 创建空字典 dict_text
    dict_text_persec = {}  # 创建空字典 dict_text_persec
    a = time.time()  # 记录当前时间到变量 a

    for path, im, im0s, vid_cap, s in dataset:
        # 对于数据集中的每个路径、图像、原始图像、视频捕获和帧编号进行循环
        if not Globals.detection_run:
            # 如果全局变量detection_run为False，则跳出循环
            break
        if use_camera:
            # 如果使用相机
            vid_cap = dataset.cap
        time_frame_st = time.time()
        # 记录当前时间
        stack.append(im0s[0] if use_camera else im0s)
        # 如果使用相机，则将im0s[0]添加到stack中，否则将im0s添加到stack中
        with dt[0]:
            # 使用dt[0]上下文管理器
            im = torch.from_numpy(im).to(model.device)
            # 将im转换为torch张量，并移动到模型的设备上
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            # 如果模型使用混合精度，则将im转换为半精度浮点数，否则转换为单精度浮点数
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 将im的像素值从0-255范围内的无符号整数转换为0.0-1.0范围内的浮点数
            if len(im.shape) == 3:
                # 如果im的形状长度为3
                im = im[None]  # expand for batch dim
                # 在batch维度上扩展im


        # 推理
        with dt[1]:
            # 如果visualize为True，则调用increment_path函数生成保存目录，并将结果赋值给visualize变量；否则将False赋值给visualize变量
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 调用model预测图像im的结果，并将结果赋值给pred变量
            pred = model(im, augment=augment, visualize=visualize)

        # 非极大值抑制
        with dt[2]:
            # 对pred进行非极大值抑制处理，并将结果赋值给pred变量
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 第二阶段分类器（可选）
        # pred = utils.general.apply_classifier(pred, video_model, im, im0s)
        # 将pred转换为xywh格式，并将结果赋值给xywh变量
        xywh = [xyxy2xywh(pred[0].cpu())]
        # 记录开始进行第二阶段分类的时间
        time_yolo_end = time.time()
        # 处理预测结果
        deepsort_outputs = []
        # 遍历预测结果pred
        for j in range(len(pred)):
            # 调用deepsort_update函数更新deepsort_tracker，并将结果赋值给temp变量
            temp = deepsort_update(deepsort_tracker, pred[j].cpu(), xywh[j][:, 0:4],
                                   im0s[j] if use_camera else [im0s][j])
            # 如果temp为空
            if len(temp) == 0:
                # 将空数组赋值给temp
                temp = np.ones((0, 8))
            # 将temp转换为float32类型并添加到deepsort_outputs列表中
            deepsort_outputs.append(temp.astype(np.float32))

        # 将deepsort_outputs赋值给yolo_pred变量
        yolo_pred = deepsort_outputs

        # 遍历yolo_pred中的每一行
        for i, det in enumerate(yolo_pred):
            # 将im的形状应用于yolo_pred的每一行的前四列，进行缩放，并将结果赋值给yolo_pred中的同一行
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s[0].shape if use_camera else im0s.shape).round()
        # 记录结束进行deepsort的时间
        time_deepsort_end = time.time()


        # 判断是否为每秒一帧的图像
        if idx % int(vid_cap.get(cv2.CAP_PROP_FPS)) == 0 and idx != 0:
            # 获取当前视频的帧率
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            # 打印当前处理到第几秒的片段
            print(f"正在处理第{idx // fps}秒的片段")
            # 将图像栈中的图像转换为tensor，并拼接为视频片段
            stack_tensor = [to_tensor(img) for img in stack]
            clip = torch.cat(stack_tensor).permute(-1, 0, 1, 2)
            # 将当前片段对应的字幕字典存入全局字典
            dict_text[(idx // fps)] = dict_text_persec
            Globals.dict_text[(idx // fps)] = dict_text_persec
            # 打印当前片段对应的字幕
            print(dict_text_persec)


            if yolo_pred[0].shape[0]:  # 如果yolo_pred的第一个维度的长度大于0
                inputs, inp_boxes, _ = ava_inference_transform(clip, yolo_pred[0][:, 0:4],
                                                               crop_size=dataset.img_size[0])  # 对clip进行ava_inference_transform，返回inputs, inp_boxes以及_
                inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)  # 在维度1上将torch.zeros(inp_boxes.shape[0], 1)和inp_boxes进行拼接，赋值给inp_boxes
                if isinstance(inputs, list):  # 如果inputs是list类型
                    inputs = [inp.unsqueeze(0).to(device) for inp in inputs]  # 对inputs中的每个元素inp进行unsqueeze(0)，然后转为device类型，将结果赋值给inputs
                else:
                    inputs = inputs.unsqueeze(0).to(device)  # 对inputs进行unsqueeze(0)，然后转为device类型，将结果赋值给inputs
                with torch.no_grad():  # 不进行梯度计算
                    slowfaster_preds = video_model(inputs, inp_boxes.to(device))  # 对inputs和inp_boxes进行video_model，不进行梯度计算，将结果赋值给slowfaster_preds
                    slowfaster_preds = slowfaster_preds.cpu()  # 将slowfaster_preds转为cpu类型，将结果赋值给slowfaster_preds
                for tid, avalabel in zip(yolo_pred[0][:, 5].tolist(), np.argmax(slowfaster_preds, axis=1).tolist()):  # 对yolo_pred的第一个维度的第5列进行tolist，以及np.argmax(slowfaster_preds, axis=1).tolist()进行迭代，将迭代出的值分别赋值给tid和avalabel
                    id_to_ava_labels[tid] = ava_labelnames[avalabel + 1]  # 将ava_labelnames[avalabel + 1]赋值给id_to_ava_labels[tid]
                    id_to_labels[tid] = avalabel + 1  # 将avalabel + 1赋值给id_to_labels[tid]


            if show_window is not None:
                show_window.ui.action_list.clear()
                # 清空动作标签列表

                # 遍历每帧的输出
                for action in dict_text_persec:
                    # 过滤动作标签
                    try:
                        # 如果动作标签在选择的标签列表中
                        if id_to_labels[action] in select_labels:
                            # 在动作标签列表中添加条目
                            show_window.ui.action_list.addItem(
                                f"时间：{idx // fps} 动作：{action}-{dict_text_persec[action]}")
                    except KeyError:
                        # 继续循环，跳过不存在的动作标签
                        continue

                    # try: if id_to_labels[action] in select_labels: show_window.ui.action_list.addItem(f"时间：{idx //
                    # fps} 动作：{action}-{dict_text_persec[action]}")


        MainWindow.drawLineChart(show_window)
        MainWindow.drawPieChart(show_window)
        del dict_text_persec
        dict_text_persec = {}

        idx += 1
        if len(stack) >= 30:
            del stack[0]

        for i, det in enumerate(yolo_pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=line_thickness, example=str(names) + '汉字', font_size=20)
            if len(det):
                # 遍历所有识别框
                for j, (*box, cls, trackid, vx, vy) in enumerate(yolo_pred[0]):
                    if int(cls) != 0:
                        ava_label = ''
                    elif trackid in id_to_ava_labels.keys():
                        ava_label = id_to_ava_labels[trackid]
                        # 如果cls不为0，将ava_label置为空字符串
                        # 如果trackid在id_to_ava_labels字典的键中，将ava_label置为对应的值
                    elif trackid not in id_to_labels.keys():
                        ava_label = 'Unknow'
                    else:
                        ava_label = 'Unknow'
                        # 如果trackid在id_to_labels字典的键中，将ava_label置为对应的值
                        # 否则，将ava_label置为'Unknow'

                    text = '{} {} {}'.format(int(trackid), Globals.yolov5_dict[names[int(cls)]],
                                             Globals.yolo_slowfast_dict[ava_label])
                    # 将trackid、cls对应的name和ava_label对应的name格式化为字符串，赋值给text

                    dict_text_persec[int(trackid)] = '{} {}'.format(Globals.yolov5_dict[names[int(cls)]],
                                                                    Globals.yolo_slowfast_dict[ava_label])
                    # 将cls对应的name和ava_label对应的name格式化为字符串，赋值给dict_text_persec中的对应trackid键

                    color = color_map[int(cls)]
                    # 将cls对应的color赋值给color

                    annotator.box_label(box, text, color=color)
                    # 根据box、text和color绘制标注框和文本


            # 将注释结果转换为图像，并将图像缩小后显示在标签上
            im0 = annotator.result()
            im1 = im0.astype("uint8")
            show = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)

            label_size = show_label.size()
            label_size.setWidth(label_size.width() - 10)
            label_size.setHeight(label_size.height() - 10)
            scaled_image = showImage.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            pixmap = QPixmap.fromImage(scaled_image)
            show_label.setPixmap(pixmap)
            show_label.setAlignment(Qt.AlignCenter)

            # 保存结果（包含检测结果的图像）
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)  # 保存图像到指定路径
                else:  # 'video' 或 'stream'
                    if vid_path[i] != save_path:  # 如果是新的视频
                        vid_path[i] = save_path  # 更新视频路径
                        if isinstance(vid_writer[i], cv2.VideoWriter):  # 如果存在之前的视频写入器
                            vid_writer[i].release()  # 释放之前的视频写入器
                        if vid_cap:  # 如果是视频
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
                        else:  # 如果是流媒体
                            fps, w, h = 30, im0.shape[1], im0.shape[0]  # 设置默认的帧率、宽度和高度
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # 强制将结果视频的后缀设置为'.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))  # 创建视频写入器
                    vid_writer[i].write(im0)  # 将图像写入视频

        # 打印时间（只进行推断）
        LOGGER.info(
            f"{s}{'' if len(det) else '(没有检测结果), '}{time.time() - time_frame_st:.3f}s {time_yolo_end - time_frame_st:.3f}s {time_deepsort_end - time_yolo_end:.3f}s {time.time() - time_deepsort_end:.3f}s")

        # 如果摄像头不再运行并且使用摄像头
        if not Globals.camera_running and use_camera:
            dataset.cap.release()  # 释放摄像头
            break

        # 打开'pred_results.json'文件以供写入
    with open('pred_results.json', 'w') as json_file:
        # 将dict_text字典写入json_file文件中
        json.dump(dict_text, json_file)

