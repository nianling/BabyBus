# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'


import os
import os
import re

import cv2
import numpy as np

import config as config_
from logger_config import logger
from utils.utilities import match_template_one


# from pytesseract import pytesseract


def try_again_is_disabled(img_again):
    """
    判断"再次挑战"是灰色还是金色,决定是否可以继续再次挑战
    # todo 计算两种情况的平均颜色值
    """

    hsv_roi = cv2.cvtColor(img_again, cv2.COLOR_BGR2HSV)
    # 定义灰色和金色的HSV范围
    gray_lower = np.array([0, 0, 70])
    gray_upper = np.array([180, 50, 220])
    gold_lower = np.array([20, 100, 100])
    gold_upper = np.array([30, 255, 255])
    # 创建掩膜
    gray_mask = cv2.inRange(hsv_roi, gray_lower, gray_upper)
    gold_mask = cv2.inRange(hsv_roi, gold_lower, gold_upper)
    # 计算掩膜中的非零像素数
    gray_count = cv2.countNonZero(gray_mask)
    gold_count = cv2.countNonZero(gold_mask)
    if gray_count > gold_count:
        again_is_gray = True
        logger.debug("再次挑战按钮禁用了")
    else:
        again_is_gray = False
    return again_is_gray


# todo 截好图传参，判断图片
def detect_try_again_button(full_screen_img):
    """
    识别"再次挑战"按钮的情况,按钮是否识别到,文本是否识别到,是否可以点击
    """
    # full_screen = window_utils.capture_window_BGRX(handle)
    gray_screenshot = cv2.cvtColor(full_screen_img, cv2.COLOR_BGR2GRAY)
    template_again = cv2.imread(os.path.normpath(f'{config_.project_base_path}/assets/img/try_again_gold_1067_600.jpg'),
                                cv2.IMREAD_COLOR)
    template_again_gray = cv2.cvtColor(template_again, cv2.COLOR_BGR2GRAY)
    matches = match_template_one(gray_screenshot, template_again_gray, threshold=0.9)
    logger.debug(f'模板匹配"再次挑战": {len(matches)}')
    btn_exist = len(matches) > 0  # 按钮是否存在
    text_exist = False  # 文本是否识别到
    clickable = False  # 是否可以点击
    if len(matches) > 0:
        # 获取匹配区域的尺寸
        for (x1, y1), (x2, y2) in matches:
            # 截取匹配区域
            roi = full_screen_img[y1:y2, x1:x2]
            # 对截取的区域进行文字识别
            # text = pytesseract.image_to_string(roi, lang='chi_sim')
            # logger.debug(f"识别的文本: {text}")
            # if text.strip() == '再次挑战' or '挑战' in text:
            text_exist = True
            again_is_gray = try_again_is_disabled(roi)
            logger.debug(f"再次挑战禁用: {again_is_gray}")
            if not again_is_gray:
                clickable = True
    return btn_exist, text_exist, clickable


def detect_1and1_next_map_button(full_screen_img):
    """
    识别"再次挑战"按钮的情况,按钮是否识别到,文本是否识别到,是否可以点击
    """
    # full_screen = window_utils.capture_window_BGRX(handle)
    gray_screenshot = cv2.cvtColor(full_screen_img, cv2.COLOR_BGR2GRAY)
    template_again = cv2.imread(
        os.path.normpath(f'{config_.project_base_path}/assets/img/to_next_daily_map_1607_600.jpg'),
        cv2.IMREAD_COLOR)
    template_again_gray = cv2.cvtColor(template_again, cv2.COLOR_BGR2GRAY)
    matches = match_template_one(gray_screenshot, template_again_gray, threshold=0.9)
    logger.debug(f'模板匹配 "前往下一地下城": {len(matches)}')
    btn_exist = len(matches) > 0  # 按钮是否存在
    text_exist = False  # 文本是否识别到
    clickable = False  # 是否可以点击
    if len(matches) > 0:
        # 获取匹配区域的尺寸
        for (x1, y1), (x2, y2) in matches:
            # 截取匹配区域
            roi = full_screen_img[y1:y2, x1:x2]
            # 对截取的区域进行文字识别
            # text = pytesseract.image_to_string(roi, lang='chi_sim')
            # logger.debug(f"识别的文本: {text}")
            # if text.strip() == '再次挑战' or '挑战' in text:
            text_exist = True
            again_is_gray = try_again_is_disabled(roi)
            logger.debug(f"再次挑战禁用: {again_is_gray}")
            if not again_is_gray:
                clickable = True
    return btn_exist, text_exist, clickable


def extract_fatigue_number(text):
    """
    从识别文本中提取正确的疲劳值
    :param text:
    :return:
    """
    if "/" not in text:
        return None
    # 去除空格
    cleaned_text = re.sub(r'\s+', '', text)
    cleaned_text = re.sub(r'[^\d/]', '', cleaned_text)

    # 提取第一个数字（斜杠前的部分）
    match = re.match(r'^(\d+)/', cleaned_text)
    return int(match.group(1)) if match else None
