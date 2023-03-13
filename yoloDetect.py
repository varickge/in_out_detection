import cv2
import torch
import torchvision
import numpy as np
from utils.torch_utils import select_device
from utils.general import non_max_suppression
from models.common import DetectMultiBackend
import matplotlib.pyplot as plt
from utils.general import scale_boxes
from custom_utils import *
import logging
import logging.config
import os
import argparse
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from utils.augmentations import letterbox


class yoloDetect:
    def __init__(
        self,
        device=select_device("cuda:0"),
        conf_thres=0.4,
        iou_thres=0.5,
        max_det=1000,
        agnostic=False,
        half=False,
        line_thickness=3,
    ):
        self.half = half
        self.device = device
        self.max_det = max_det
        self.agnostic = agnostic
        self.iou_thres = iou_thres
        self.line_thickness = line_thickness

    def __call__(
        self,
        model,
        frame,
        img_size,
        crop_size=(None, None, None, None),
        scale_with=None,
        conf_thres=0.25,
        classes = 0
    ):
        img = frame.copy()
        if not None in crop_size:
            img = img[crop_size[0] : crop_size[1], crop_size[2] : crop_size[3]]

        #         img = torch.from_numpy(img).to(self.device).permute(2, 0, 1)
        if not None in img_size:
            img = letterbox(img, img_size, stride=32, auto=True)[0]
            img = img.transpose((2, 0, 1))  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)  # contiguous

        img = torch.from_numpy(img).to(self.device)
        if self.half:
            img = img.to(torch.float16)
        else:
            img = img.to(torch.float32)

        if len(img.shape) == 3:
            img = img[None]
        preds = model(img / 255)
        pred = non_max_suppression(
            preds,
            conf_thres,
            self.iou_thres,
            classes=classes,
            agnostic=self.agnostic,
            max_det=self.max_det,
        )[0]
        bboxes = np.zeros((0, 5))

        if len(pred):
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], scale_with).round()

            # Write results
            for *xyxy, conf, cls in reversed(pred):
                bboxes = np.concatenate(
                    [
                        bboxes,
                        np.array(
                            [
                                cls.cpu().numpy(),
                                xyxy[0].cpu().numpy(),
                                xyxy[1].cpu().numpy(),
                                xyxy[2].cpu().numpy(),
                                xyxy[3].cpu().numpy(),
                            ]
                        )[None],
                    ],
                    axis=0,
                )

        return bboxes
