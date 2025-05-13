# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'

import os

import cv2
import config as config_
from utils.utilities import match_template_by_roi

# 定义目标检测参数
object_detect_param = {
    "death": {
        "description": "幽灵状态",
        "template": cv2.imread(os.path.normpath(f'{config_.project_base_path}/assets/img/death.png')),
        "roi_xywh": (478, 422, 110, 26),
        "threshold": 0.8
    }
}


def object_detection_cv(image):
    """目标检测"""
    results = {}

    # 死亡状态 todo 异步
    death_detect = object_detect_param["death"]
    results["death"] = match_template_by_roi(image, death_detect['roi_xywh'], death_detect['template'], death_detect['threshold'])

    return results
