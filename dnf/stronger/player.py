# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'


import os
import time

import cv2
import easyocr
import numpy as np
from pynput.keyboard import Key
from pynput.mouse import Button

import config as config_
from dnf.stronger.method import (
    extract_fatigue_number
)
from logger_config import logger
from utils import keyboard_utils as kbu
from utils import mouse_utils as mu
from utils import window_utils as window_utils
from utils.utilities import match_template, compare_images

reader = None


# todo x,y如果传了，没传自己获取
def sale_equipment_to_sailiya(x, y):
    """
    一键出售装备,给赛丽亚
    """
    logger.debug('一键出售装备,给赛丽亚')
    # 点击赛丽亚
    mu.do_smooth_move_to(x + 556, y + 189)
    time.sleep(0.2)
    mu.do_click(Button.left)
    time.sleep(0.1)
    # 点击弹出的商店
    cur_x, cur_y = mu.get_current_position()
    mu.do_smooth_move_to(cur_x + 51, cur_y + 51)
    time.sleep(0.1)
    mu.do_click(Button.left)
    time.sleep(0.1)
    # 点击装备栏,准备卖出
    mu.do_smooth_move_to(x + 632, y + 275)
    time.sleep(0.2)
    mu.do_click(Button.left)
    time.sleep(0.1)
    # 按下a,一键出售
    kbu.do_press('a')
    # 按下a,一键选择出售
    kbu.do_press('a')
    # 点击卖出
    kbu.do_press(Key.space)
    time.sleep(0.2)

    # 计算弹出的确认按钮位置
    # cur_x, cur_y = mu.get_current_position()
    # mu.do_smooth_move_to(cur_x - 92, cur_y)
    # time.sleep(0.5)
    # mu.do_click(Button.left)
    kbu.do_press(Key.space)
    time.sleep(0.1)
    # 两下esc完全关闭商店
    kbu.do_press(Key.esc)
    time.sleep(0.2)
    kbu.do_press(Key.esc)
    time.sleep(0.2)


def transfer_materials_to_account_vault(x, y):
    """
    转移材料到账号金库
    """
    logger.debug('转移材料到账号金库')
    # 点击仓库
    mu.do_smooth_move_to(x + 366, y + 353)
    time.sleep(0.1)
    mu.do_click(Button.left)
    time.sleep(0.1)
    # 两下tab切换到账号金库
    kbu.do_press(Key.tab)
    time.sleep(0.1)
    kbu.do_press(Key.tab)
    time.sleep(0.1)
    # 按下a,一键转移物品
    kbu.do_press('a')
    time.sleep(0.1)
    # 空格,确定
    kbu.do_press(Key.space)
    time.sleep(1.5)
    kbu.do_press(Key.space)
    time.sleep(0.1)

    # esc 取消仓库
    kbu.do_press(Key.esc)
    time.sleep(0.2)


def transfer_gold_to_account_vault(x, y):
    """
    转移金币到账号金库
    """
    logger.debug('转移金币到账号金库')
    # 点击仓库
    mu.do_smooth_move_to(x + 366, y + 353)
    time.sleep(0.1)
    mu.do_click(Button.left)
    time.sleep(0.1)
    # 两下tab切换到账号金库
    kbu.do_press(Key.tab)
    time.sleep(0.1)
    kbu.do_press(Key.tab)
    time.sleep(0.1)

    # 整理金币到账号金库
    mu.do_smooth_move_to(x + 413, y + 465)
    time.sleep(0.1)
    mu.do_click(Button.left)
    time.sleep(0.1)
    mu.do_click(Button.left)
    time.sleep(0.1)

    # esc 取消仓库
    kbu.do_press(Key.esc)
    time.sleep(0.2)


def finish_daily_challenge(x, y, daily1and1=False):
    """
    点击每日任务
    """
    # 点击畅玩任务(1446,1108)，从下往上依次点击一遍（需要移动鼠标）(1006,838) 1007,711  1004,599 1006,494
    logger.debug('点击畅玩任务,完成每日任务')
    mu.do_smooth_move_to(x + 767, y + 542)  # 不等比
    time.sleep(0.1)
    mu.do_click(Button.left)
    time.sleep(0.1)
    if daily1and1:
        mu.do_smooth_move_to(x + 494, y + 444)
        time.sleep(0.1)
        mu.do_click(Button.left)
        time.sleep(0.1)
    mu.do_smooth_move_to(x + 494, y + 361)
    time.sleep(0.1)
    mu.do_click(Button.left)
    time.sleep(0.1)
    mu.do_smooth_move_to(x + 494, y + 295)
    time.sleep(0.1)
    mu.do_click(Button.left)
    time.sleep(0.1)
    mu.do_smooth_move_to(x + 494, y + 230)
    time.sleep(0.1)
    mu.do_click(Button.left)
    time.sleep(0.1)
    mu.do_smooth_move_to(x + 494, y + 165)
    time.sleep(0.1)
    mu.do_click(Button.left)
    time.sleep(0.1)
    mu.do_click(Button.left)
    time.sleep(0.1)
    mu.do_click(Button.left)
    time.sleep(0.1)
    # esc关闭畅玩任务
    kbu.do_press(Key.esc)
    time.sleep(0.2)


def teleport_to_sailiya(x, y):
    """
    刷了图的情况下(图标位置不一样,刷图第三个,不刷图第四个),鼠标移动到瞬移赛丽亚旅馆
    都一样了 都是第四个
    """
    logger.debug('瞬移赛丽亚旅馆')
    # 鼠标移动到瞬移赛丽亚旅馆
    mu.do_smooth_move_to(x + 818, y + 543)
    time.sleep(0.2)
    mu.do_click(Button.left)
    time.sleep(0.2)
    kbu.do_press(Key.space)
    time.sleep(0.2)
    kbu.do_press_with_time(Key.down, 140, 200)


def clik_to_quit_game(handle, x, y):
    """
    结束游戏
    """
    logger.debug('结束游戏')
    time.sleep(0.5)

    kbu.do_press(Key.esc)
    time.sleep(0.5)

    mu.do_smooth_move_to(x + 679, y + 497)
    time.sleep(0.3)
    mu.do_click(Button.left)
    time.sleep(0.3)

    full_screen = window_utils.capture_window_BGRX(handle)
    gray_screenshot = cv2.cvtColor(full_screen, cv2.COLOR_BGRA2GRAY)
    template_again = cv2.imread(os.path.normpath(f'{config_.project_base_path}/assets/img/quit_game.png'), cv2.IMREAD_COLOR)
    template_again_gray = cv2.cvtColor(template_again, cv2.COLOR_BGR2GRAY)
    matches = match_template(gray_screenshot, template_again_gray, threshold=0.9)
    top_left, bottom_right = matches[0]
    x1, y1 = top_left
    x2, y2 = bottom_right
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    mu.do_move_to(x + center_x, y + center_y)
    time.sleep(0.3)
    mu.do_click(Button.left)
    time.sleep(0.3)


# todo todo 截好图传参(先移动鼠标,)，判断图片
def do_ocr_fatigue(handle, x, y, model):
    """
    easyocr，识别疲劳值
    :param model:
    :param handle:
    :param x:
    :param y:
    :return:
    """
    # mu.do_smooth_move_to(x + 1038, y + 713)
    # mu.do_smooth_move_to(x + 865, y + 594)
    mu.do_smooth_move_to(x + 875, y + 594)
    time.sleep(0.2)
    # img_fatigue = window_utils.capture_window_BGRX(handle, (985, 689, 92, 17))
    # img_fatigue = window_utils.capture_window_BGRX(handle, (976, 687, 111, 18)) # 整体部分 976, 687 1087 705
    # img_fatigue = window_utils.capture_window_BGRX(handle, (1032, 687, 55, 18))  # 数值部分
    # img_fatigue = window_utils.capture_window_BGRX(handle, (1035, 687, 52, 18))  # 数值部分
    img_fatigue = window_utils.capture_window_BGRX(handle, (880, 574, 43, 13))  # 数值部分

    # 放大图像
    resize = cv2.resize(img_fatigue, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # 转换为灰度图
    gray = cv2.cvtColor(resize, cv2.COLOR_BGRA2GRAY)

    # 二值化
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 去噪
    denoised = cv2.medianBlur(thresh, 3)

    # 边缘增强
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # # 膨胀和腐蚀
    # kernel = np.ones((2, 2), np.uint8)
    # dilated = cv2.dilate(sharpened, kernel, iterations=1)
    # eroded = cv2.erode(dilated, kernel, iterations=1)

    # # 对比度调整
    # adjusted = cv2.convertScaleAbs(eroded, alpha=1.5, beta=0)

    # cv2.imshow('img_fatigue', img_fatigue)
    # cv2.imwrite('./img_fatigue.png', img_fatigue)
    #
    # cv2.imshow('resize', resize)
    # cv2.imwrite('./resize.png', resize)
    #
    # cv2.imshow('gray', gray)
    # cv2.imwrite('./gray.png', gray)
    #
    # cv2.imshow('thresh', thresh)
    # cv2.imwrite('./thresh.png', thresh)
    #
    # cv2.imshow('denoised', denoised)
    # cv2.imwrite('./denoised.png', denoised)
    #
    # cv2.imshow('sharpened', sharpened)
    # cv2.imwrite('./sharpened.png', sharpened)

    # cv2.imshow('eroded', eroded)
    # cv2.imshow('adjusted', adjusted)

    # cv2.waitKey(0)
    global reader

    if reader is None:
        if model is not None:
            reader = model
        else:
            reader = easyocr.Reader(['en'])

    result = reader.readtext(sharpened, allowlist="0123456789/", detail=0)
    logger.info('识别文本:{}', result)

    mu.do_smooth_move_to(x + 1027, y + 561)

    # 解析结果并提取匹配模式的文本
    for detection in result:
        # logger.debug('---------->', detection)
        # text = detection
        # if pattern_pl.search(text):
        #     matched_text = pattern_pl.search(text).group()
        #     logger.debug(f"识别疲劳值->:{matched_text}")
        #     parts = matched_text.split("/")
        #     return int(parts[0])
        fatigue = extract_fatigue_number(detection)
        if fatigue is not None:
            logger.debug(f"识别疲劳值1->:{fatigue}")
            return fatigue
    if len(result) > 0:
        fatigue = result[0].strip()
        logger.debug(f"识别疲劳值2->:{fatigue}")
        return int(fatigue)
    logger.error("识别疲劳值为空!")
    return None


def do_ocr_fatigue_retry(handle, x, y, model, retry=1, default_fatigue=10):
    """
    允许重试的疲劳值获取
    :param model:
    :param y:
    :param x:
    :param handle:
    :param retry:
    :param default_fatigue:
    :return:
    """
    for attempt in range(retry):
        result = do_ocr_fatigue(handle, x, y, model)  # todo
        if result is not None:
            return result
        else:
            logger.warning(f"第{attempt + 1}次识别疲劳值失败,重试中...")
    logger.warning(f"识别疲劳值失败...")
    return default_fatigue


# todo todo 截好图传参，判断图片
# todo 只截取部分匹配，或者直接对比
def detect_return_town_button_when_choose_map(full_screen):
    """
    识别选择地图页面的'返回城镇'按钮
    """
    # full_screen = window_utils.capture_window_BGRX(handle)
    gray_screenshot = cv2.cvtColor(full_screen, cv2.COLOR_BGRA2GRAY)
    template_again = cv2.imread(
        os.path.normpath(f'{config_.project_base_path}/assets/img/return_button_on_choose_map_1067_600.jpg'),
        cv2.IMREAD_COLOR)
    template_again_gray = cv2.cvtColor(template_again, cv2.COLOR_BGR2GRAY)
    matches = match_template(gray_screenshot, template_again_gray, threshold=0.95)
    return len(matches) > 0


def from_sailiya_to_abyss(x, y):
    """
    传送到深渊/妖气门口
    :return:
    """
    # 打开地图
    kbu.do_press('n')
    time.sleep(0.2)
    # 切换至收藏
    kbu.do_press(Key.f2)
    time.sleep(0.2)
    # 默认第一个 深渊
    mu.do_smooth_move_to(x + 145, y + 120)
    time.sleep(0.1)
    mu.do_click(Button.left)
    time.sleep(2.2)


def crusader_to_battle(x, y):
    """
    奶爸,切换输出护石
    """
    logger.warning("奶爸,准备切换锤子护石...")
    time.sleep(2)
    # 奶爸切换锤子护石
    # 打开物品栏
    kbu.do_press('i')
    time.sleep(0.2)
    # 点击护石
    # mu.do_smooth_move_to(x + 926, y + 98)
    mu.do_smooth_move_to(x + 785, y + 70)
    time.sleep(0.2)
    mu.do_click(Button.left)
    time.sleep(0.2)
    # 点开护石方案列表
    # mu.do_smooth_move_to(x + 976, y + 123)
    mu.do_smooth_move_to(x + 828, y + 93)
    time.sleep(0.2)
    mu.do_click(Button.left)
    time.sleep(0.2)
    # 默认第二项是锤子
    # mu.do_smooth_move_to(x + 1071, y + 187)
    mu.do_smooth_move_to(x + 870, y + 149)
    time.sleep(0.2)
    mu.do_click(Button.left)
    time.sleep(0.2)
    # 点击应用
    # mu.do_smooth_move_to(x + 1112, y + 218)
    mu.do_smooth_move_to(x + 947, y + 174)
    time.sleep(0.2)
    mu.do_click(Button.left)
    time.sleep(0.5)
    # esc 关闭装备栏
    kbu.do_press(Key.esc)
    time.sleep(0.2)
    logger.warning("奶爸,切换锤子护石完毕...")


def goto_white_map(x, y):
    """
    去白图，（跌宕群岛）
    :param x:
    :param y:
    :return:
    """

    # 有可能本来就选中了，需要重置并选级别
    # 先重置可能得级别
    kbu.do_press(Key.left)
    time.sleep(0.15)
    kbu.do_press(Key.left)
    time.sleep(0.15)
    kbu.do_press(Key.left)
    time.sleep(0.15)
    kbu.do_press(Key.left)
    time.sleep(0.15)
    kbu.do_press(Key.right)
    time.sleep(0.15)
    kbu.do_press(Key.right)
    time.sleep(0.1)

    # mu.do_smooth_move_to(x + 357, y + 106)  # 妖怪追踪
    # mu.do_smooth_move_to(x + 551, y + 176)  # 妖气歼灭
    mu.do_smooth_move_to(x + 620, y + 305)  # 跌宕群岛
    # mu.do_smooth_move_to(x + 835, y + 309)  # 萧索的回廊
    mu.do_click(Button.left)
    time.sleep(0.5)

    # 先重置可能得级别
    kbu.do_press(Key.left)
    time.sleep(0.2)
    kbu.do_press(Key.left)
    time.sleep(0.2)
    kbu.do_press(Key.left)
    time.sleep(0.2)
    kbu.do_press(Key.left)
    time.sleep(0.2)
    kbu.do_press(Key.right)
    time.sleep(0.2)
    kbu.do_press(Key.right)
    time.sleep(0.2)

    # 确认进入
    mu.do_click(Button.left)
    time.sleep(0.2)

    time.sleep(2)


def goto_white_map_level(x, y, press_cnt=2):
    """
    去白图，（跌宕群岛）
    :param press_cnt:
    :param x:
    :param y:
    :return:
    """

    # 有可能本来就选中了，需要重置并选级别
    # 先重置可能得级别
    kbu.do_press(Key.left)
    time.sleep(0.15)
    kbu.do_press(Key.left)
    time.sleep(0.15)
    kbu.do_press(Key.left)
    time.sleep(0.15)
    kbu.do_press(Key.left)
    time.sleep(0.15)
    for i in range(press_cnt):
        kbu.do_press(Key.right)
        time.sleep(0.15)

    # mu.do_smooth_move_to(x + 357, y + 106)  # 妖怪追踪
    # mu.do_smooth_move_to(x + 551, y + 176)  # 妖气歼灭
    mu.do_smooth_move_to(x + 620, y + 305)  # 跌宕群岛
    # mu.do_smooth_move_to(x + 835, y + 309)  # 萧索的回廊
    mu.do_click(Button.left)
    time.sleep(0.5)

    # 先重置可能得级别
    kbu.do_press(Key.left)
    time.sleep(0.2)
    kbu.do_press(Key.left)
    time.sleep(0.2)
    kbu.do_press(Key.left)
    time.sleep(0.2)
    kbu.do_press(Key.left)
    time.sleep(0.2)
    for i in range(press_cnt):
        kbu.do_press(Key.right)
        time.sleep(0.15)

    # 确认进入
    mu.do_click(Button.left)
    time.sleep(0.2)

    time.sleep(2)


def goto_zhuizong(x, y):
    """
    去白图，（跌宕群岛）
    :param x:
    :param y:
    :return:
    """
    mu.do_smooth_move_to(x + 357, y + 106)  # 妖怪追踪
    # mu.do_smooth_move_to(x + 551, y + 176)  # 妖气歼灭
    # mu.do_smooth_move_to(x + 620, y + 305)  # 跌宕群岛
    # mu.do_smooth_move_to(x + 835, y + 309)  # 萧索的回廊
    mu.do_click(Button.left)
    time.sleep(0.5)

    # 确认进入
    mu.do_click(Button.left)
    time.sleep(0.2)

    time.sleep(2)


def goto_jianmie(x, y):
    """
    去白图，（跌宕群岛）
    :param x:
    :param y:
    :return:
    """
    # mu.do_smooth_move_to(x + 357, y + 106)  # 妖怪追踪
    mu.do_smooth_move_to(x + 551, y + 176)  # 妖气歼灭
    # mu.do_smooth_move_to(x + 620, y + 305)  # 跌宕群岛
    # mu.do_smooth_move_to(x + 835, y + 309)  # 萧索的回廊
    mu.do_click(Button.left)
    time.sleep(0.5)

    # 确认进入
    mu.do_click(Button.left)
    time.sleep(0.2)

    time.sleep(2)


def goto_abyss(x, y):
    """
    去白图，（跌宕群岛）
    :param x:
    :param y:
    :return:
    """
    mu.do_smooth_move_to(x + 720, y + 470)
    mu.do_click(Button.left)
    time.sleep(0.5)

    # 确认进入
    mu.do_click(Button.left)
    time.sleep(0.2)

    time.sleep(2)


def goto_daily_1and1(x, y):
    """
    去白图，（跌宕群岛）
    :param x:
    :param y:
    :return:
    """
    logger.debug('点击畅玩任务')
    mu.do_smooth_move_to(x + 767, y + 542)  # 不等比
    time.sleep(0.2)
    mu.do_click(Button.left)
    time.sleep(0.2)

    mu.do_smooth_move_to(x + 494, y + 444)
    time.sleep(0.2)
    mu.do_click(Button.left)
    time.sleep(0.2)

    # 按下之后到了选择地图页面
    kbu.do_press(Key.space)
    time.sleep(1)

    # 空格进入地图
    kbu.do_press(Key.space)
    time.sleep(0.2)

    # 按下可能存在的角色绑定提示
    kbu.do_press(Key.space)
    time.sleep(0.2)

    time.sleep(2)


def detect_aolakou(full_screen):
    """
    识别"再次挑战"按钮的情况,按钮是否识别到,文本是否识别到,是否可以点击
    """
    gray_screenshot = cv2.cvtColor(full_screen, cv2.COLOR_BGRA2GRAY)
    template_again = cv2.imread(os.path.normpath(f'{config_.project_base_path}/assets/img/activity/act_aolakou1.jpg'),
                                cv2.IMREAD_COLOR)
    template_again_gray = cv2.cvtColor(template_again, cv2.COLOR_BGR2GRAY)
    matches = match_template(gray_screenshot, template_again_gray, threshold=0.8)
    if len(matches) > 0:
        print("有普通奥拉扣！！")
        return True

    template_again = cv2.imread(os.path.normpath(f'{config_.project_base_path}/assets/img/activity/act_aolakou2.jpg'),
                                cv2.IMREAD_COLOR)
    template_again_gray = cv2.cvtColor(template_again, cv2.COLOR_BGR2GRAY)
    matches = match_template(gray_screenshot, template_again_gray, threshold=0.8)
    if len(matches) > 0:
        print("有特殊奥拉扣！！")
        return True

    return False


def detect_daily_1and1_clickable(full_screen):
    """
    检查每日1+1是否可点击
    :param full_screen:
    :return:
    """
    x1, y1 = 475, 431
    x2, y2 = 514 + 1, 452 + 1
    daily_crop = full_screen[y1:y2, x1:x2]
    daily_img = cv2.imread(os.path.normpath(f'{config_.project_base_path}/assets/img/daily_1and1_goto.png'))
    scores = compare_images(daily_crop, daily_img)
    # print(scores)
    return scores > 0.6


def hide_right_bottom_icon(full_screen, x, y):
    x1, y1 = 736, 534
    x2, y2 = 748 + 1, 549 + 1
    icon_crop = full_screen[y1:y2, x1:x2]
    icon_img = cv2.imread(os.path.normpath(f'{config_.project_base_path}/assets/img/activity_icon.png'))
    scores = compare_images(icon_crop, icon_img)
    if scores < 0.5:
        pass
    else:
        mu.do_move_and_click(x + 744, y + 577)
        time.sleep(0.1)


def show_right_bottom_icon(full_screen, x, y):
    x1, y1 = 736, 534
    x2, y2 = 748 + 1, 549 + 1
    icon_crop = full_screen[y1:y2, x1:x2]
    icon_img = cv2.imread(os.path.normpath(f'{config_.project_base_path}/assets/img/activity_icon.png'))
    scores = compare_images(icon_crop, icon_img)
    if scores > 0.6:
        pass
    else:
        mu.do_move_and_click(x + 744, y + 577)
        time.sleep(0.1)


def buy_from_mystery_shop(full_screen, x, y):
    """
    神秘商店购买
    """
    logger.debug('出现神秘商店！')
    gray_screenshot = cv2.cvtColor(full_screen, cv2.COLOR_BGRA2GRAY)
    template_again = cv2.imread(os.path.normpath(f'{config_.project_base_path}/assets/img/ticket.png'),
                                cv2.IMREAD_COLOR)
    template_again_gray = cv2.cvtColor(template_again, cv2.COLOR_BGR2GRAY)
    matches = match_template(gray_screenshot, template_again_gray, threshold=0.85)
    logger.debug(f"发现门票{len(matches)}个。{matches}")

    for top_left, bottom_right in matches:
        x1, y1 = top_left
        x2, y2 = bottom_right
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        mu.do_move_to(x + center_x, y + center_y)
        time.sleep(0.2)
        mu.do_click(Button.left)
        time.sleep(0.2)
        mu.do_click(Button.left)
        time.sleep(0.2)
        logger.debug("购买门票一次")


def buy_tank_from_mystery_shop(full_screen, x, y):
    """
    神秘商店购买
    """
    logger.debug('出现神秘商店！')
    gray_screenshot = cv2.cvtColor(full_screen, cv2.COLOR_BGRA2GRAY)
    template_again = cv2.imread(os.path.normpath(f'{config_.project_base_path}/assets/img/tank_legend.png'), cv2.IMREAD_COLOR)
    template_again_gray = cv2.cvtColor(template_again, cv2.COLOR_BGR2GRAY)
    matches = match_template(gray_screenshot, template_again_gray, threshold=0.85)
    logger.debug(f"发现罐子{len(matches)}个。{matches}")

    if len(matches) > 0:
        cv2.imwrite(f'tank_{time.time()}.jpg', full_screen)

    for top_left, bottom_right in matches:
        x1, y1 = top_left
        x2, y2 = bottom_right
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        mu.do_move_to(x + center_x, y + center_y)
        time.sleep(0.2)
        mu.do_click(Button.left)
        time.sleep(0.2)
        mu.do_click(Button.left)
        time.sleep(0.2)
        logger.debug("购买罐子一次")
