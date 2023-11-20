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
from deep_sort.deep_sort import DeepSort
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image, )
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from utils.myutil import Globals
from mainwindow import MainWindow
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.myutil import Globals
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


def func_slowfast(vid_cap, idx, stack, yolo_pred, img_size, device, video_model):
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    print(f"processing {idx // fps}th second clips")
    stack_tensor = [to_tensor(img) for img in stack]
    clip = torch.cat(stack_tensor).permute(-1, 0, 1, 2)
    # stack = []
    if yolo_pred[0].shape[0]:
        inputs, inp_boxes, _ = ava_inference_transform(clip, yolo_pred[0][:, 0:4], crop_size=img_size)
        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
        if isinstance(inputs, list):
            inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
        else:
            inputs = inputs.unsqueeze(0).to(device)
        with torch.no_grad():
            slowfaster_preds = video_model(inputs, inp_boxes.to(device))
            slowfaster_preds = slowfaster_preds.cpu()
        return slowfaster_preds


def my_uniform_temporal_subsample(
        x: torch.Tensor, num_samples: int, temporal_dim: int = -3
) -> torch.Tensor:
    """
    Uniformly subsamples num_samples indices from the temporal dimension of the video.
    When num_samples is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.

    Args:
        x (torch.Tensor): A video tensor with dimension larger than one with torch
            tensor type includes int, long, float, complex, etc.
        num_samples (int): The number of equispaced samples to be selected
        temporal_dim (int): dimension of temporal to perform temporal subsample.

    Returns:
        An x-like Tensor with subsampled temporal dimension.
    """
    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    indices = indices.cuda()
    return torch.index_select(x, temporal_dim, indices)


def deepsort_update(Tracker, pred, xywh, np_img):
    outputs = Tracker.update(xywh, pred[:, 4:5], pred[:, 5].tolist(), cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    return outputs


def to_tensor(img):
    img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img.unsqueeze(0)


def ava_inference_transform(
        clip,
        boxes,
        num_frames=32,  # if using slowfast_r50_detection, change this to 32, 4 for slow
        crop_size=640,
        data_mean=[0.45, 0.45, 0.45],
        data_std=[0.225, 0.225, 0.225],
        slow_fast_alpha=4,  # if using slowfast_r50_detection, change this to 4, None for slow
):
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
        weights='yolov5s.pt',  # model path or triton URL
        source='data/images',  # file/dir/URL/glob/screen/0(webcam)
        data='',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.4,  # confidence threshold
        iou_thres=0.4,  # NMS IOU threshold
        max_det=100,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_img=True,
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='result',  # save results to project/name
        name='',  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        show_label=None,
        use_camera=False,
        show_window=None,
        select_labels=None
):
    source = str(source)
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    webcam = source.isnumeric()
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
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
    video_model = slowfast_r50_detection(True).eval().to(device)
    id_to_ava_labels = {}
    id_to_labels = {}
    ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    stack = []
    color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]
    idx = 0
    deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    dict_text = {}
    dict_text_persec = {}
    a = time.time()
    for path, im, im0s, vid_cap, s in dataset:
        if not Globals.detection_run:
            break
        if use_camera:
            vid_cap = dataset.cap
        time_frame_st = time.time()
        stack.append(im0s[0] if use_camera else im0s)
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, video_model, im, im0s)
        xywh = [xyxy2xywh(pred[0].cpu())]
        time_yolo_end = time.time()
        # Process predictions
        deepsort_outputs = []
        for j in range(len(pred)):
            temp = deepsort_update(deepsort_tracker, pred[j].cpu(), xywh[j][:, 0:4],
                                   im0s[j] if use_camera else [im0s][j])
            if len(temp) == 0:
                temp = np.ones((0, 8))
            deepsort_outputs.append(temp.astype(np.float32))

        yolo_pred = deepsort_outputs

        for i, det in enumerate(yolo_pred):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s[0].shape if use_camera else im0s.shape).round()
        time_deepsort_end = time.time()


        if idx % int(vid_cap.get(cv2.CAP_PROP_FPS)) == 0 and idx != 0:
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            print(f"正在处理第{idx // fps}秒的片段")
            stack_tensor = [to_tensor(img) for img in stack]
            clip = torch.cat(stack_tensor).permute(-1, 0, 1, 2)
            dict_text[(idx // fps)] = dict_text_persec
            Globals.dict_text[(idx // fps)] = dict_text_persec
            print(dict_text_persec)

            if yolo_pred[0].shape[0]:
                inputs, inp_boxes, _ = ava_inference_transform(clip, yolo_pred[0][:, 0:4],
                                                               crop_size=dataset.img_size[0])
                inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
                if isinstance(inputs, list):
                    inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                else:
                    inputs = inputs.unsqueeze(0).to(device)
                with torch.no_grad():
                    slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                    slowfaster_preds = slowfaster_preds.cpu()
                for tid, avalabel in zip(yolo_pred[0][:, 5].tolist(), np.argmax(slowfaster_preds, axis=1).tolist()):
                    id_to_ava_labels[tid] = ava_labelnames[avalabel + 1]
                    id_to_labels[tid] = avalabel + 1

            if show_window is not None:
                show_window.ui.action_list.clear()
                # 启动输出动作标签
                for action in dict_text_persec:
                    # 过滤动作标签
                    try:
                        if id_to_labels[action] in select_labels:
                            show_window.ui.action_list.addItem(
                                f"时间：{idx // fps} 动作：{action}-{dict_text_persec[action]}")
                    except KeyError:
                        continue

                    # try:
                    #     if id_to_labels[action] in select_labels:
                    #         show_window.ui.action_list.addItem(f"时间：{idx // fps} 动作：{action}-{dict_text_persec[action]}")

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
            annotator = Annotator(im0, line_width=line_thickness, example=str(names)+'汉字')
            if len(det):

                for j, (*box, cls, trackid, vx, vy) in enumerate(yolo_pred[0]):
                    # c = int(cls)  # integer class
                    if int(cls) != 0:
                        ava_label = ''
                    elif trackid in id_to_ava_labels.keys():
                        ava_label = id_to_ava_labels[trackid]
                        #.split(' ')[0]
                        # 过滤动作标签
                        if id_to_labels[trackid] not in select_labels:
                            continue
                    else:
                        ava_label = 'Unknow'

                    text = '{} {} {}'.format(int(trackid), Globals.yolov5_dict[names[int(cls)]],
                                             Globals.yolo_slowfast_dict[ava_label])

                    dict_text_persec[int(trackid)] = '{} {}'.format(Globals.yolov5_dict[names[int(cls)]],
                                                                    Globals.yolo_slowfast_dict[ava_label])

                    color = color_map[int(cls)]
                    # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(box, text, color=color)

            # Stream results
            im0 = annotator.result()
            im1 = im0.astype("uint8")
            show = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)

            label_size = show_label.size()
            label_size.setWidth(label_size.width() - 10)
            label_size.setHeight(label_size.height() - 10)
            scaled_image = showImage.scaled(label_size, Qt.KeepAspectRatio)
            pixmap = QPixmap.fromImage(scaled_image)
            show_label.setPixmap(pixmap)
            show_label.setAlignment(Qt.AlignCenter)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(
            f"{s}{'' if len(det) else '(no detections), '}{time.time() - time_frame_st:.3f}s {time_yolo_end - time_frame_st:.3f}s {time_deepsort_end - time_yolo_end:.3f}s {time.time() - time_deepsort_end:.3f}s")

        # 关闭摄像头->退出
        if not Globals.camera_running and use_camera:
            dataset.cap.release()  # 释放摄像头
            break
    print("总用时: {:.3f} s".format(time.time() - a))
    with open('pred_results.json', 'w') as json_file:
        json.dump(dict_text, json_file)
