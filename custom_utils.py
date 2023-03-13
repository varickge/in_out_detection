import torch
import torchvision
import numpy as np


def round_box(x1, x2, y1, y2, imgsz=(720, 1280), round_to=32, round_up_from=10):
    w = x2 - x1
    h = y2 - y1

    to_add_w = round_to - w % 32
    to_add_h = round_to - h % 32

    if to_add_w <= round_to - round_up_from:
        x2 = x2 + (to_add_w // 2)
        x1 = x1 - to_add_w + (to_add_w // 2)
    elif to_add_w > round_to - round_up_from and to_add_w < 32:
        x2 = x2 - ((round_to - to_add_w) // 2)
        x1 = x1 + ((round_to - to_add_w) - ((round_to - to_add_w) // 2))

    if to_add_h <= round_to - round_up_from:
        y2 = y2 + to_add_h
    elif to_add_h > round_to - round_up_from and to_add_h < 32:
        y2 = y2 - (round_to - to_add_h)

    if 0 > x1:
        x2 = x2 + abs(x1)
        x1 = 0

    if x2 > imgsz[1]:
        x1 = x1 - (x2 - imgsz[1])
        x2 = imgsz[1]

    if 0 > y1:
        y2 = y2 + abs(x1)
        y1 = 0

    if y2 > imgsz[0]:
        y1 = y1 - (y2 - imgsz[0])
        y2 = imgsz[0]

    return int(x1), int(x2), int(y1), int(y2)


def box_iom(boxes1: torch.Tensor, boxes2: torch.Tensor):
    area1 = torchvision.ops.box_area(boxes1)
    area2 = torchvision.ops.box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    min_area = torch.min(area1[:, None, ...], area2)

    return inter / min_area
