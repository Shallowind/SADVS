def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj


import torch.jit
script_method1 = torch.jit.script_method
script1 = torch.jit.script
torch.jit.script_method = script_method
torch.jit.script = script


import argparse
import cv2
import os
import time
import numpy as np
import torch
import imageio
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage

from .dataset.transforms import BaseTransform
from .utils.misc import load_weight
from .utils.box_ops import rescale_bboxes
from .utils.vis_tools import vis_detection
from .config import build_dataset_config, build_model_config
from .models import build_model


def build_args(size: int = 224, show: bool = False, cuda: bool = True, save: str = './result',
               video: str = 'D:\VScode\motion-monitor-x\easyowov2\example.mp4', version: str = 'yowo_v2_large', gif: bool = False,
               dataset: str = 'ava_v2.2',
               vis_thresh: float = 0.3, pose: bool = False, conf_thresh: float = 0.1,
               nms_thresh: float = 0.5, topk: int = 40, len_clip: int = 16, memory: bool = False
               ) -> argparse.Namespace:
    args_list = argparse.Namespace(
        img_size=size,  # 图像的大小，用于图像处理
        show=show,  # 是否显示处理后的视频
        cuda=cuda,  # 是否使用CUDA进行计算
        save_folder=save,  # 保存处理后的视频的文件夹
        vis_thresh=vis_thresh,  # 可视化阈值，用于确定哪些检测结果应该被显示
        video=video,  # 要处理的视频文件的路径
        gif=gif,  # 是否生成GIF
        dataset=dataset,  # 使用的数据集
        pose=pose,  # 是否显示姿态
        version=version,  # 模型的版本
        weight='D:\VScode\motion-monitor-x\easyowov2\path\yowo_v2_large_ava.pth',  # 模型权重的路径
        conf_thresh=conf_thresh,  # 置信度阈值，用于确定哪些检测结果应该被保留
        nms_thresh=nms_thresh,  # 非极大值抑制的阈值，用于去除重叠的检测结果
        topk=topk,  # 保留的最大检测结果数量
        len_clip=len_clip,  # 视频剪辑的长度
        batch_size=4,  # batch size
        memory=memory  # 是否使用内存
    )
    return args_list


def multi_hot_vis(args, frame, out_bboxes, orig_w, orig_h, class_names, act_pose=False):
    # visualize detection results
    for bbox in out_bboxes:
        x1, y1, x2, y2 = bbox[:4]
        if act_pose:
            # only show 14 poses of AVA.
            cls_conf = bbox[5:5 + 14]
        else:
            # show all actions of AVA.
            cls_conf = bbox[5:]

        # rescale bbox
        x1, x2 = int(x1 * orig_w), int(x2 * orig_w)
        y1, y2 = int(y1 * orig_h), int(y2 * orig_h)

        # score = obj * cls
        det_conf = float(bbox[4])
        cls_scores = np.sqrt(det_conf * cls_conf)

        indices = np.where(cls_scores > args.vis_thresh)
        scores = cls_scores[indices]
        indices = list(indices[0])
        scores = list(scores)

        if len(scores) > 0:
            # draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw text
            blk = np.zeros(frame.shape, np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            coord = []
            text = []
            text_size = []

            for _, cls_ind in enumerate(indices):
                text.append("[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind]))
                text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.5, thickness=1)[0])
                coord.append((x1 + 3, y1 + 14 + 20 * _))
                cv2.rectangle(blk, (coord[-1][0] - 1, coord[-1][1] - 12),
                              (coord[-1][0] + text_size[-1][0] + 1, coord[-1][1] + text_size[-1][1] - 4), (0, 255, 0),
                              cv2.FILLED)
            frame = cv2.addWeighted(frame, 1.0, blk, 0.5, 1)
            for t in range(len(text)):
                cv2.putText(frame, text[t], coord[t], font, 0.5, (0, 0, 0), 1)

    return frame


@torch.no_grad()
def detect(args, model, device, transform, class_names, class_colors,
           save_size: (int, int) = (960, 720), save_name: str = 'detection', fps: int = 20, show_window=None):
    # path to save
    save_path = os.path.join(args.save_folder)
    os.makedirs(save_path, exist_ok=True)

    # path to video
    path_to_video = os.path.join(args.video)

    # video
    video = cv2.VideoCapture(path_to_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    save_name = os.path.join(save_path, save_name + '.avi')
    out = cv2.VideoWriter(save_name, fourcc, fps, save_size)

    # run
    video_clip = []
    image_list = []
    while (True):
        ret, frame = video.read()

        if ret:
            # to RGB
            frame_rgb = frame[..., (2, 1, 0)]
            print(frame_rgb.shape)

            # to PIL image
            frame_pil = Image.fromarray(frame_rgb.astype(np.uint8))

            # prepare
            if len(video_clip) <= 0:
                for _ in range(args.len_clip):
                    video_clip.append(frame_pil)

            video_clip.append(frame_pil)
            del video_clip[0]

            # orig size
            orig_h, orig_w = frame.shape[:2]

            # transform
            x, _ = transform(video_clip)
            # List [T, 3, H, W] -> [3, T, H, W]
            x = torch.stack(x, dim=1)
            x = x.unsqueeze(0).to(device)  # [B, 3, T, H, W], B=1

            t0 = time.time()
            # inference
            outputs = model(x)
            print("inference time ", time.time() - t0, "s")

            # vis detection results
            if args.dataset in ['ava_v2.2']:
                batch_bboxes = outputs
                # batch size = 1
                bboxes = batch_bboxes[0]
                # multi hot
                frame = multi_hot_vis(
                    args=args,
                    frame=frame,
                    out_bboxes=bboxes,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    class_names=class_names,
                    act_pose=args.pose
                )
            elif args.dataset in ['ucf24']:
                batch_scores, batch_labels, batch_bboxes = outputs
                # batch size = 1
                scores = batch_scores[0]
                labels = batch_labels[0]
                bboxes = batch_bboxes[0]
                # rescale
                bboxes = rescale_bboxes(bboxes, [orig_w, orig_h])
                # one hot
                frame = vis_detection(
                    frame=frame,
                    scores=scores,
                    labels=labels,
                    bboxes=bboxes,
                    vis_thresh=args.vis_thresh,
                    class_names=class_names,
                    class_colors=class_colors
                )
            # save

            im0 = frame
            im1 = im0.astype("uint8")
            show = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * 3, QImage.Format_RGB888)
            scale_factor = min(show_window.player_2.width() / showImage.width(),
                               show_window.player_2.height() / showImage.height())
            # 计算新的宽度和高度
            new_width = int(showImage.width() * scale_factor)
            new_height = int(showImage.height() * scale_factor)
            # 设置新的最大大小
            show_window.camera_2.setMaximumSize(new_width, new_height)
            show_window.camera_2.setPixmap(QPixmap(showImage))
            show_window.camera_2.setScaledContents(True)

            frame_resized = cv2.resize(frame, save_size)
            out.write(frame_resized)

            if args.gif:
                gif_resized = cv2.resize(frame, (200, 150))
                gif_resized_rgb = gif_resized[..., (2, 1, 0)]
                image_list.append(gif_resized_rgb)

            if args.show:
                # show
                cv2.imshow('key-frame detection', frame)
                cv2.waitKey(1)

        else:
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()

    # generate GIF
    if args.gif:
        save_name = os.path.join(save_path, 'detect.gif')
        print('generating GIF ...')
        imageio.mimsave(save_name, image_list, fps=fps)
        print('GIF done: {}'.format(save_name))


def run_easydemo(
        show_window=None,
        size: int = 224, show: bool = False, cuda: bool = True, save: str = './result',
        video: str = 'result\example.mp4', version: str = 'yowo_v2_large', gif: bool = False,
        dataset: str = 'ava_v2.2',
        vis_thresh: float = 0.3, pose: bool = False, conf_thresh: float = 0.1,
        nms_thresh: float = 0.5, topk: int = 40, len_clip: int = 16, memory: bool = False
) -> None:
    np.random.seed(100)
    args = argparse.Namespace(
        img_size=size,  # 图像的大小，用于图像处理
        show=show,  # 是否显示处理后的视频
        cuda=cuda,  # 是否使用CUDA进行计算
        save_folder=save,  # 保存处理后的视频的文件夹
        vis_thresh=vis_thresh,  # 可视化阈值，用于确定哪些检测结果应该被显示
        video=video,  # 要处理的视频文件的路径
        gif=gif,  # 是否生成GIF
        dataset=dataset,  # 使用的数据集
        pose=pose,  # 是否显示姿态
        version=version,  # 模型的版本
        weight='easyowov2\path\yowo_v2_large_ava.pth',  # 模型权重的路径
        conf_thresh=conf_thresh,  # 置信度阈值，用于确定哪些检测结果应该被保留
        nms_thresh=nms_thresh,  # 非极大值抑制的阈值，用于去除重叠的检测结果
        topk=topk,  # 保留的最大检测结果数量
        len_clip=len_clip,  # 视频剪辑的长度
        memory=memory  # 是否使用内存
    )

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    class_names = d_cfg['label_map']
    num_classes = d_cfg['valid_num_classes']

    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # transform
    basetransform = BaseTransform(img_size=args.img_size)

    # build model
    model, _ = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=num_classes,
        trainable=False
    )

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)

    # to eval
    model = model.to(device).eval()

    # run
    detect(args=args,
           model=model,
           device=device,
           transform=basetransform,
           class_names=class_names,
           class_colors=class_colors,
           show_window=show_window)
