# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'


import random

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def plot_one_box(xyxy, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def match_template(image, template, threshold=0.8):
    """
    返回矩形的左上角,右下角坐标,可能有多个矩形
    [((x1,y1),(x2,y2))]
    """
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)
    matches = []
    for pt in zip(*loc[::-1]):
        # match_score = result[pt[1], pt[0]]
        bottom_right = (pt[0] + template.shape[1], pt[1] + template.shape[0])
        matches.append((pt, bottom_right))
    return matches


def match_template_one(image, template, threshold=0.8):
    """
    模板匹配,返回置信度最高的一个
    """
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        pt = max_loc
        bottom_right = (pt[0] + template.shape[1], pt[1] + template.shape[0])
        return [(pt, bottom_right)]
    else:
        return []


def compare_images(img1, img2):
    # print('img1的尺寸',img1.shape)
    # print('img2的尺寸',img2.shape)
    # 确保图片尺寸相同(18x18)
    # img1 = cv2.resize(img1, (18, 18))
    # img2 = cv2.resize(img2, (18, 18))

    # 计算结构相似性指数(SSIM)
    # 转换为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    similarity_score = ssim(gray1, gray2)

    return similarity_score
