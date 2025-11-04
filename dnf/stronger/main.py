# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'

import itertools
import math
import os
import pathlib
import random
import re
import threading
import time
from datetime import datetime
import queue
import traceback
import concurrent.futures
import copy

import cv2
import keyboard as kboard
import numpy as np
import torch
import winsound
from pynput import keyboard
from pynput.keyboard import Key
from pynput.mouse import Button
from ultralytics import YOLO

import config as config_
import dnf.dnf_config as dnf
import map_util as map_util
import skill_util as skill_util
from dnf.stronger.det_result import DetResult
from dnf.stronger.method import (
    detect_try_again_button,
    detect_1and1_next_map_button,
    find_densest_monster_cluster,
    get_closest_obj,
    exist_near,
    get_objs_in_range,
    find_door_by_position,
    get_opposite_direction
)
from dnf.stronger.player import (
    transfer_materials_to_account_vault,
    finish_daily_challenge_by_all,
    teleport_to_sailiya,
    clik_to_quit_game,
    do_ocr_fatigue_retry,
    detect_return_town_button_when_choose_map,
    from_sailiya_to_abyss,
    crusader_to_battle,
    goto_daily_1and1,
    goto_white_map,
    goto_zhuizong,
    goto_jianmie,
    detect_daily_1and1_clickable,
    hide_right_bottom_icon,
    show_right_bottom_icon,
    goto_white_map_level,
    buy_from_mystery_shop,
    process_mystery_shop,
    activity_live,
    do_recognize_fatigue,
    receive_mail, match_and_click,
    close_new_day_dialog,
    detect_aolakou, calc_role_height, detect_try_again_conflict,
)
from dnf.stronger.role_config import SubClass, BaseClass
from logger_config import logger
from dnf.stronger.role_list import get_role_config_list
from utils import keyboard_utils as kbu
from utils import mouse_utils as mu
from utils import window_utils as window_utils
from utils.custom_thread_pool_excutor import SingleTaskThreadPool
from utils.fixed_length_queue import FixedLengthQueue
from utils.keyboard_move_controller import MovementController
from utils.utilities import plot_one_box
from utils.window_utils import WindowCapture
from dnf.stronger.path_finder import PathFinder
from utils.utilities import match_template_by_roi, match_template
from utils.mail_sender import EmailSender
from dnf.mail_config import config as mail_config
from dnf.stronger.object_detect import object_detection_cv
from utils.utilities import hex_to_bgr
from dnf.stronger.skill_util import get_skill_initial_images
from dnf.stronger.role_config import class_icon_map

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

#  >>>>>>>>>>>>>>>> 运行时相关的参数 >>>>>>>>>>>>>>>>

show = False  # 查看检测结果

# 脚本执行完之后,结束游戏
quit_game_after_finish = False
# 睡觉去了,让脚本执行完之后,自己关机
shutdown_pc_after_finish = False

# 买罐子
buy_tank_type = 0  # buy_type: 0不买，1买传说，2买史诗，3买史诗+传说
# 买铃铛
buy_bell_ticket = 0  # buy_type: 0，不买，1买粉罐子，2买传说罐子，3买粉+传说罐子
# 买闪闪明
buy_shanshanming = 2  # buy_type: 0，不买，1买粉罐子，2买传说罐子，3买粉+传说罐子

# 执行脚本的第一个角色_编号
first_role_no = 1
last_role_no = 20
# 游戏模式 1:白图（跌宕群岛），2:每日1+1，3:妖气追踪，4:妖怪歼灭，
# 5:先1+1再白图，6:先1+1在妖气追踪
game_mode = 1

# 使用此处统一配置预留的疲劳值
enable_uniform_pl = False
uniform_default_fatigue_reserved = 17

weights = os.path.join(config_.project_base_path, 'weights/stronger.pt')  # 模型存放的位置
# <<<<<<<<<<<<<<<< 运行时相关的参数 <<<<<<<<<<<<<<<<

#  >>>>>>>>>>>>>>>> 脚本所需要的变量 >>>>>>>>>>>>>>>>
# 每秒最大处理帧数
max_fps = 10

# 游戏窗口位置
x, y = 0, 0
handle = -1

# 全局变量 暂停
pause_event = threading.Event()
pause_event.set()  # 初始设置为未暂停状态

# 当前按下的按键集合
current_keys_control = set()

# 全局变量，停止组合键是否按下,用于控制脚本运行
stop_be_pressed = False
# 唤醒继续运行
continue_pressed = False

# reader = easyocr.Reader(['en'])
# 疲劳值识别
pattern_pl = re.compile(r'\d+/\d+')

color_red = (0, 0, 255)  # 红色
color_green = (0, 255, 0)  # 绿色
color_blue = (255, 0, 0)  # 蓝色
color_yellow = (0, 255, 255)  # 黄色
color_purple = (255, 0, 255)  # 紫色

# ---------------------------------------------------------
model = YOLO(weights)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# if device.type != 'cpu':
#     model.half()  # to FP16
names = [
    'boss',
    'card',
    'continue',
    'door',
    'gold',
    'hero',
    'loot',
    'menu',
    'monster',
    'elite-monster',
    'shop',
    'shop-mystery',
    'sss',
    'door-boss'
]

name_colors = [
    {
        "name": "boss",
        "id": 1,
        "color": "#523294",
        "type": "rectangle",
        "attributes": []
    },
    {
        "name": "card",
        "id": 2,
        "color": "#5b98c6",
        "type": "rectangle",
        "attributes": []
    },
    {
        "name": "continue",
        "id": 3,
        "color": "#4c7a1d",
        "type": "rectangle",
        "attributes": []
    },
    {
        "name": "door",
        "id": 4,
        "color": "#4398ef",
        "type": "rectangle",
        "attributes": []
    },
    {
        "name": "gold",
        "id": 5,
        "color": "#f2cb53",
        "type": "rectangle",
        "attributes": []
    },
    {
        "name": "hero",
        "id": 6,
        "color": "#fefe30",
        "type": "rectangle",
        "attributes": []
    },
    {
        "name": "loot",
        "id": 7,
        "color": "#a8e898",
        "type": "rectangle",
        "attributes": []
    },
    {
        "name": "menu",
        "id": 8,
        "color": "#268674",
        "type": "rectangle",
        "attributes": []
    },
    {
        "name": "monster",
        "id": 9,
        "color": "#fcb5fc",
        "type": "rectangle",
        "attributes": []
    },
    {
        "name": "elite-monster",
        "id": 10,
        "color": "#33ddff",
        "type": "rectangle",
        "attributes": []
    },
    {
        "name": "shop",
        "id": 11,
        "color": "#c8b3cb",
        "type": "rectangle",
        "attributes": []
    },
    {
        "name": "shop-mystery",
        "id": 12,
        "color": "#909950",
        "type": "rectangle",
        "attributes": []
    },
    {
        "name": "sss",
        "id": 13,
        "color": "#b5b5b0",
        "type": "rectangle",
        "attributes": []
    },
    {
        "name": "door-boss",
        "id": 14,
        "color": "#ea6a4b",
        "type": "rectangle",
        "attributes": []
    }
]
name_colors = [hex_to_bgr(d['color']) for d in name_colors]

colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
# ----------------------------------------------------------
boss_h = 120  # boss高度处理
monster_h = 57  # 普通怪高度处理
em_h = 100  # 精英怪高度处理
door_h = 32  # 门高度处理
loot_h = 0  # 掉落物高度处理

attack_x = 300  # 打怪命中范围，x轴距离
attack_y = 90  # 打怪命中范围，y轴距离

door_hit_x = 25  # 过门命中范围，y轴距离
door_hit_y = 15  # 过门命中范围，x轴距离

pick_up_x = 25  # 捡材料命中范围，x轴距离
pick_up_y = 15  # 捡材料命中范围，y轴距离

# <<<<<<<<<<<<<<<< 脚本所需要的变量 <<<<<<<<<<<<<<<<
mover = MovementController()
executor = SingleTaskThreadPool()
img_executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
tool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
mail_sender = EmailSender(mail_config)  # 初始化邮件发送器

# 创建一个队列，用于主线程和展示线程之间的通信
result_queue = queue.Queue()


# 展示线程的函数
def display_results():
    while True:
        try:
            # 从队列中获取检测结果
            frame_with_detections = result_queue.get()
            if frame_with_detections is None:  # 如果接收到None，退出线程
                break

            # 使用OpenCV显示检测结果
            cv2.imshow("Game Capture", frame_with_detections)

            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            logger.error(f"展示显示报错: {e}")

    # 清理资源
    cv2.destroyAllWindows()


# # 启动展示线程
# display_thread = threading.Thread(target=display_results, daemon=True)
# display_thread.start()

#  >>>>>>>>>>>>>>>> 方法定义 >>>>>>>>>>>>>>>>

def on_press(key):
    global stop_be_pressed, continue_pressed, x, y
    if key in dnf.key_stop_script or key in dnf.key_pause_script:
        current_keys_control.add(key)
        if all(k in current_keys_control for k in dnf.key_stop_script):
            formatted_keys = ', '.join(item.name for item in dnf.key_stop_script)
            logger.warning(f"监听到组合键 [{formatted_keys}]，停止脚本...")
            threading.Thread(target=lambda: winsound.PlaySound(config_.sound2, winsound.SND_FILENAME)).start()
            stop_be_pressed = True
            return False  # 停止监听

        if all(k in current_keys_control for k in dnf.key_pause_script):
            formatted_keys = ', '.join(item.name for item in dnf.key_pause_script)
            logger.warning(f"监听到组合键 [{formatted_keys}]，暂停or继续?")
            threading.Thread(target=lambda: winsound.PlaySound(config_.sound3, winsound.SND_FILENAME)).start()
            if pause_event.is_set():
                logger.warning(f"按下 [{formatted_keys}]键，暂停运行...")
                pause_event.clear()  # 暂停
                mover._release_all_keys()
                time.sleep(0.2)
                mover._release_all_keys()
            else:
                logger.warning(f"按下 [{formatted_keys}] 键，唤醒运行...")
                x, y, _, _ = window_utils.get_window_rect(handle)
                mu.do_move_to(x + 250, y + 150)
                time.sleep(0.1)
                mu.do_click(Button.left)
                continue_pressed = True
                pause_event.set()  # 继续
            time.sleep(0.5)  # 防止重复触发


def on_release(key):
    try:
        if key in current_keys_control:
            current_keys_control.remove(key)
    except KeyError as e:
        logger.error(f"Error occurred: {e}")
        pass


# 创建一个线程来监听键盘输入
def start_keyboard_listener():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def analyse_det_result(results, hero_height, img):
    global show
    if results is not None and len(results):
        boss_xywh_list = []
        monster_xywh_list = []
        elite_monster_xywh_list = []

        loot_xywh_list = []
        gold_xywh_list = []
        door_xywh_list = []
        door_boss_xywh_list = []

        hero_conf = -1
        hero_xywh = None

        card_num = 0
        continue_exist = False
        shop_exist = False
        shop_mystery_exist = False
        menu_exist = False
        sss_exist = False

        result = results[0]
        for box in result.boxes:
            cls = int(box.cls)
            xywh = box.xywh[0].tolist()
            xyxy = box.xyxy[0].tolist()
            conf = box.conf[0]

            # 高度处理
            if names[cls] == "hero":
                xywh[1] += hero_height

                if conf > hero_conf:  # 找一个置信度最大的hero,记录索引
                    hero_conf = conf
                    hero_xywh = xywh

            if names[cls] == "boss":
                # xywh[1] += b_h
                xywh[1] = xyxy[3] - 20

                boss_xywh_list.append(xywh)

            if names[cls] == "monster":
                xywh[1] += monster_h

                monster_xywh_list.append(xywh)

            if names[cls] == "elite-monster":
                # xywh[1] += em_h
                xywh[1] = xyxy[3] - 20

                elite_monster_xywh_list.append(xywh)

            if names[cls] == "door":
                xywh[1] += door_h

                door_xywh_list.append(xywh)

            if names[cls] == "door-boss":
                xywh[1] += door_h

                door_boss_xywh_list.append(xywh)

            if names[cls] == "loot":
                xywh[1] += loot_h
                # 处理半拉子框的情况
                if xywh[2] > 111 and xywh[3] < 110:
                    if (xyxy[1] + 60) > xywh[1]:
                        xywh[1] = xyxy[1] + 60
                loot_xywh_list.append(xywh)

            if names[cls] == 'gold':
                xywh[1] += loot_h
                if xywh[2] > 111 and xywh[3] < 110:
                    if (xyxy[1] + 60) > xywh[1]:
                        xywh[1] = xyxy[1] + 60

                gold_xywh_list.append(xywh)

            if names[cls] == "continue":
                continue_exist = True

            if names[cls] == "card":
                card_num = card_num + 1

            if names[cls] == "shop":
                shop_exist = True

            if names[cls] == "shop-mystery":
                shop_mystery_exist = True

            if names[cls] == "menu":
                menu_exist = True

            if names[cls] == "sss":
                sss_exist = True

            # 在原图上画框
            if show and img is not None:
                label = '%s %.2f' % (names[int(cls)], conf)
                # plot_one_box(box.xyxy[0], img, label=label, color=colors[int(cls)], line_thickness=2)
                # plot_one_box(box.xyxy[0], img, label=label, color=hex_to_bgr(name_colors[int(cls)]['color']), line_thickness=2)
                plot_one_box(box.xyxy[0], img, label=label, color=name_colors[int(cls)], line_thickness=2)

        res = DetResult()
        res.monster_xywh_list = monster_xywh_list
        res.elite_monster_xywh_list = elite_monster_xywh_list
        res.boss_xywh_list = boss_xywh_list
        res.loot_xywh_list = loot_xywh_list
        res.gold_xywh_list = gold_xywh_list
        res.door_xywh_list = door_xywh_list
        res.door_boss_xywh_list = door_boss_xywh_list

        res.hero_xywh = hero_xywh
        # res.hero_conf = hero_conf

        res.card_num = card_num
        res.continue_exist = continue_exist
        res.shop_exist = shop_exist
        res.shop_mystery_exist = shop_mystery_exist
        res.menu_exist = menu_exist
        res.sss_exist = sss_exist

        # 给角色绘制定位圆点,方便查看
        if show:
            if res.hero_xywh:
                # 处理后的中心
                cv2.circle(img, (int(hero_xywh[0]), int(hero_xywh[1])), 1, color_green, 2)
                # 推理后的中心
                cv2.circle(img, (int(hero_xywh[0]), int(hero_xywh[1] - hero_height)), 1, color_red, 2)

            for a in (res.loot_xywh_list + res.gold_xywh_list):
                # 掉落物
                cv2.circle(img, (int(a[0]), int(a[1])), 1, color_green, 2)
                cv2.circle(img, (int(a[0]), int(a[1] - loot_h)), 1, color_red, 2)

            for a in (res.door_xywh_list + res.door_boss_xywh_list):
                # 门口
                cv2.circle(img, (int(a[0]), int(a[1])), 1, color_green, 2)
                cv2.circle(img, (int(a[0]), int(a[1] - door_h)), 1, color_red, 2)

            for a in (res.monster_xywh_list):
                # 怪
                cv2.circle(img, (int(a[0]), int(a[1])), 1, color_green, 2)
                cv2.circle(img, (int(a[0]), int(a[1] - monster_h)), 1, color_red, 2)

        return res


def judge_is_target_door(current_room, door_box, hero_box, next_room_direction, allow_directions, path_stack, d, img0):
    """
    判断是否是目标门
    :param current_room:
    :param door_box:
    :param next_room_direction:
    :param allow_directions:
    :param path_stack:
    :param d:
    :param img0:
    :return:
    """
    if door_box is None:
        logger.debug("判断是否是目标门，空的，否")
        return False
    if len(allow_directions) == len(d.door_xywh_list + d.door_boss_xywh_list):
        # 一屏全部出现了，肯定是目标门
        logger.debug("判断是否是目标门，全部出现，是")
        return True
    else:

        previous = None
        if current_room in [item[0] for item in path_stack]:
            for ii in range(len(path_stack) - 1, 0, -1):
                if path_stack[ii][0] == current_room:
                    previous = path_stack[ii - 1][1]
                    break

        # last_room = get_last_room_info(current_room, path_history)
        # previous = None if not last_room else last_room.direction

        if len(allow_directions) == 2 and previous == 'RIGHT' and door_box[0] > img0.shape[1] * 3 // 4 and door_box[0] - hero_box[0] > 170:
            logger.debug("判断是否是目标门，2门入口门在左，可能处于右，是")
            return True
        if len(allow_directions) == 2 and previous == 'LEFT' and door_box[0] < img0.shape[1] // 4 and hero_box[0] - door_box[0] > 170:
            logger.debug("判断是否是目标门，2门入口门在右，可能处于左，是")
            return True

        if next_room_direction == 'RIGHT' and door_box[0] > img0.shape[1] * 3 // 4:
            logger.debug("判断是否是目标门，目标右，处于右，是")
            return True
        elif next_room_direction == 'LEFT' and door_box[0] < img0.shape[1] // 5:
            logger.debug("判断是否是目标门，目标左，处于左，是")
            return True
        else:
            # if previous == "RIGHT" and door_box[0] < img0.shape[1] // 6:
            if previous == "RIGHT" and door_box[0] < img0.shape[1] * 7 // 50:
                logger.debug("判断是否是目标门，太靠左了贴着入口，否")
                return False
            elif previous == "LEFT" and door_box[0] > img0.shape[1] * 5 // 6:
                logger.debug("判断是否是目标门，太靠右了贴着入口，否")
                return False
            else:
                if next_room_direction == 'DOWN' and door_box[1] > img0.shape[0] * 775 // 1000 and (img0.shape[1] // 7 < door_box[0] < img0.shape[1] * 6 // 7):
                    logger.debug("判断是否是目标门，目标下，可能是")
                    return True
                if next_room_direction == 'UP' and door_box[1] < img0.shape[0] * 0.72 and (img0.shape[1] // 7 < door_box[0] < img0.shape[1] * 6 // 7):
                    logger.debug("判断是否是目标门，目标上，可能是")
                    return True
    logger.warning("判断是否是目标门，无法判断，否")
    return False


def minimap_analyse(capturer):
    # 分析小地图
    cols, rows = 0, 0
    cur_row, cur_col = 0, 0
    map_crop = None
    boss_room = (-1, -1)
    current_room = (-1, -1)
    map_error_cnt = 0
    analyse_map_error = True
    while analyse_map_error:
        try:
            img0 = capturer.capture()

            # 分析小地图的行列
            cols = map_util.get_colum_count(img0)
            rows = map_util.get_row_count(img0)
            # logger.warning("分析小地图的行列{},{}", rows, cols)

            # 裁剪小地图区域
            map_crop = map_util.get_small_map_region_img(img0, rows, cols)

            # 获取boss房间位置，0基
            boss_room = map_util.get_boss_from_crop(map_crop, rows, cols)
            # logger.info('boss房间是 {}', boss_room)
            current_room = map_util.current_room_index_cropped(map_crop, rows, cols)  # 实际上没有用，只是打印看一下位置
            # logger.info('当前房间是 {}', current_room)
            cur_row, cur_col = current_room
        except Exception as e:
            logger.error(e)
            traceback.print_exc()

        analyse_map_error = boss_room is None or current_room is None or boss_room == (-1, -1) or current_room == (-1, -1)
        if analyse_map_error:
            map_error_cnt = map_error_cnt + 1
            # cv2.imwrite(f'errorDetectMap_init_{map_error_cnt}.jpg', img0)
            # logger.error(f"分析小地图的行列init，第 {map_error_cnt} 次出错,行列是 {rows} , {cols}")
            # logger.error("暂停2秒继续重试！！")
            time.sleep(0.4)
        else:
            map_error_cnt = 0

        if analyse_map_error and map_error_cnt > 20:
            logger.error("分析小地图的行列init多次出错了 废了！！！")
            return None
    return boss_room, (rows, cols), current_room


# <<<<<<<<<<<<<<<< 方法定义 <<<<<<<<<<<<<<<<


def main_script():
    global x, y, handle, show, game_mode
    # ################### 主流程开始 ###############################
    logger.info("_____________________准备_____________________")
    time.sleep(1)

    # 获取游戏窗口的位置，和大小
    handle = window_utils.get_window_handle(dnf.window_title)
    x, y, width, height = window_utils.get_window_rect(handle)
    logger.info("获取游戏窗口位置和大小...{},{},{},{}", x, y, width, height)
    capturer = WindowCapture(handle)

    # 获取角色配置列表
    role_list = get_role_config_list()
    logger.info("读取角色配置列表...")
    logger.info(f"共有{len(role_list)}个角色...")

    pause_event.wait()
    # 检查每日弹窗
    close_new_day_dialog(handle, x, y)

    pause_event.wait()  # 暂停
    # 遍历角色, 循环刷图
    for i in range(len(role_list)):
        pause_event.wait()  # 暂停

        role = copy.deepcopy(role_list[i])
        # 判断,从指定的角色开始,其余的跳过
        if first_role_no != -1 and (i + 1) < first_role_no:
            logger.warning(f'[跳过]-【{i + 1}】[{role.name}]...')
            continue
        logger.warning(f'第【{i + 1}】个角色，【{role.name}】 开始了')
        oen_role_start_time = datetime.now()

        if i + 1 > 20 and game_mode == 2:
            logger.warning(f'前20个每日1+1已经结束了')
            mode_name = (
                "白图" if game_mode == 1 else
                "每日1+1" if game_mode == 2 else
                "妖气追踪" if game_mode == 3 else
                "妖怪歼灭" if game_mode == 4 else
                "未知模式"
            )
            email_subject = f"{mode_name} 任务执行结束 {pathlib.Path(__file__).stem.replace('main', '').strip() if 'main' in pathlib.Path(__file__).stem else ''}"
            email_content = email_subject
            mail_receiver = mail_config.get("receiver")
            if mail_receiver:
                tool_executor.submit(lambda: (
                    mail_sender.send_email(email_subject, email_content, mail_receiver),
                    logger.info("任务执行结束")
                ))
            break

        # 读取角色配置
        hero_height = role.height  # 高度
        # 读取疲劳值配置
        if enable_uniform_pl:
            role.fatigue_reserved = uniform_default_fatigue_reserved
        skill_images = {}

        # 等待加载角色完成
        time.sleep(4)

        # 检查每日弹窗
        if datetime.now().hour == 0:
            close_new_day_dialog(handle, x, y)

        # # 确保展示右下角的图标
        # show_right_bottom_icon(capturer.capture(), x, y)

        logger.info(f'设置的拥有疲劳值: {role.fatigue_all}')

        # ocr_fatigue = do_ocr_fatigue_retry(handle, x, y, reader, 5)
        ocr_fatigue = do_recognize_fatigue(capturer.capture())
        logger.info(f'识别的拥有疲劳值: {ocr_fatigue}')
        if ocr_fatigue is not None:
            if role.fatigue_all != ocr_fatigue:
                logger.warning(f'更新疲劳值--->(计算): {role.fatigue_all},(识别): {ocr_fatigue}')
            role.fatigue_all = ocr_fatigue

        # 角色当前疲劳值
        current_fatigue = role.fatigue_all
        fatigue_cost = 16  # 一把消耗的疲劳值
        if game_mode == 3 or game_mode == 4:
            fatigue_cost = 8

        logger.info(f'{role.name},拥有疲劳值:{role.fatigue_all},预留疲劳值:{role.fatigue_reserved}')

        # 如果需要刷图,这选择副本,进入副本
        need_fight = current_fatigue - fatigue_cost >= role.fatigue_reserved if role.fatigue_reserved > 0 else current_fatigue > 0

        # 判断1+1是否能点
        if need_fight and game_mode == 2:
            mu.do_move_and_click(x + 767, y + 542)
            time.sleep(0.4)
            daily_1and1_clickable = detect_daily_1and1_clickable(capturer.capture())
            time.sleep(0.4)
            kbu.do_press(Key.esc)
            time.sleep(0.2)
            if not daily_1and1_clickable:
                logger.warning("1+1点不了,跳过...")
            need_fight = daily_1and1_clickable

        if need_fight:
            pause_event.wait()  # 暂停
            # todo 奶爸刷图,切换输出加点
            # if '奶爸' in role.name:
            #     logger.info("是奶爸,准备切换加点...")
            #     crusader_to_battle(x, y)

            pause_event.wait()  # 暂停
            # 默认是站在赛丽亚房间

            # 识别当前职业
            kbu.do_press('k')
            time.sleep(2)
            skill_panel_img = capturer.capture()
            skill_panel_img = skill_panel_img[360:450, 700:920]
            skill_panel_img = cv2.cvtColor(skill_panel_img, cv2.COLOR_BGRA2GRAY)

            # 从role_list中找到对应的角色配置
            find_role_config = False
            for class_code, icon in class_icon_map.items():
                matches = match_template(skill_panel_img, icon, threshold=0.85)
                if len(matches) > 0:
                    logger.info(f"当前职业编号是是: {class_code}")
                    for job in SubClass:
                        code = job.code
                        if code == class_code:
                            logger.info("识别当前职业是 " + job.name)
                            for cc in role_list:
                                if cc.sub_class == job:
                                    logger.info(f"从角色配置池中找到对应的角色配置,{cc.no}-{cc.sub_class}-{cc.name}")
                                    role_bak = role
                                    role = cc
                                    role.height = role_bak.height
                                    role.fatigue_reserved = role_bak.fatigue_reserved
                                    role.fatigue_all = role_bak.fatigue_all
                                    find_role_config = True
                                    break
                            if not find_role_config and role.sub_class_auto:
                                logger.debug("未找到对应职业，缺省配置角色，并且允许自动配置角色高度和技能")
                                role.height = BaseClass.get_base_class(job).height
                                role.custom_priority_skills = skill_util.default_all_skills
                            break
                    break
                else:
                    logger.debug("未识别当前职业!!")
            logger.debug(f"最终生效职业是：序号：{role.no}-名称：{role.name}-高度：{role.height}")
            logger.debug(f"{role}")
            time.sleep(0.5)
            kbu.do_press(Key.esc)
            time.sleep(0.5)

            calc_height = calc_role_height(capturer.capture(), x, y)
            if calc_height:
                logger.info(f"计算出的角色高度: {calc_height}，原高度：{role.height}")
                role.height = calc_height
                hero_height = role.height

            # 获取技能栏截图
            skill_images = get_skill_initial_images(capturer.capture())

            if game_mode != 2:
                # N 点第一个
                logger.info("传送到风暴门口,选地图...")
                # 传送到风暴门口
                from_sailiya_to_abyss(x, y)
                kbu.do_press_with_time(Key.right, 500, 50)
                kbu.do_press_with_time(Key.left, 1000, 50)
                kbu.do_press_with_time(Key.down, 1000, 50)
                kbu.do_press_with_time(Key.up, 1500, 50)
                time.sleep(0.5)
                time.sleep(1.5)  # 先等自己移动到深渊图

            if game_mode == 2:
                goto_daily_1and1(x, y)
            elif game_mode == 1:
                goto_white_map_level(x, y, role.white_map_level)
            elif game_mode == 3:
                goto_zhuizong(x, y)
            elif game_mode == 4:
                goto_jianmie(x, y)

            pause_event.wait()  # 暂停

            # 检查是否成功进入地图
            img0 = capturer.capture()
            enter_map_success = not detect_return_town_button_when_choose_map(img0)
            # 进不去
            if not enter_map_success:
                logger.error(f'第【{i + 1}】【{role.name}】，进不去地图,结束当前角色')
                time.sleep(0.2)
                # esc 关闭地图选择界面
                kbu.do_press(Key.esc)
                time.sleep(0.2)
                need_fight = False

        # 刷图流程开始>>>>>>>>>>
        logger.warning(f'第【{i + 1}】个角色【{role.name}】已经进入地图,刷图打怪循环开始...')

        # # 隐藏掉右下角的图标
        # if need_fight:
        #     hide_right_bottom_icon(capturer.capture(), x, y)

        # ##############################
        # 记录一下刷图次数
        fight_count = 0
        # 角色刷完结束
        finished = False
        exception_mail_notify_timer = None

        # todo 循环进图开始>>>>>>>>>>>>>>>>>>>>>>>>
        # 一直循环
        pause_event.wait()  # 暂停
        while not finished and need_fight:  # 循环进图，再次挑战
            if exception_mail_notify_timer:
                exception_mail_notify_timer.cancel()
            exception_mail_notify_timer = threading.Timer(300, mail_sender.send_email, ("刷图异常提醒", "刷图异常提醒，长时间未动，及时介入处理。", mail_config.get("receiver")))
            exception_mail_notify_timer.start()
            logger.debug("启动刷图异常提醒定时器")

            # 先要等待地图加载 todo 改动态识别
            # time.sleep(4.5)
            pause_event.wait()  # 暂停
            try:
                t1 = time.time()
                time.sleep(2)  # 防止太快
                load_map_task = tool_executor.submit(minimap_analyse, capturer)
                load_map_success = load_map_task.result(timeout=5)
                if load_map_success:
                    logger.info(f"地图加载完成！{(time.time() - t1):.2f}s")
            except Exception as e:
                logger.error("地图加载任务异常")
                logger.error(e)
                traceback.print_exc()

            # 不管了,全部释放掉
            mover._release_all_keys()

            pause_event.wait()  # 暂停

            fight_count += 1
            logger.info(f'【{role.name}】 刷图,第 {fight_count} 次，开始...')
            one_game_start = time.time()
            mu.do_move_to(x + width / 4, y + height / 4)  # 重置鼠标位置

            # # 记录疲劳值
            # current_fatigue_ocr = do_ocr_fatigue_retry(handle, x, y, reader, 5)  # 识别疲劳值
            current_fatigue_ocr = do_recognize_fatigue(capturer.capture())  # 识别疲劳值
            logger.info(f'当前还有疲劳值(识别): {current_fatigue_ocr}')

            global continue_pressed
            if continue_pressed:
                # exception_count = 0  # 主动唤醒过,重置异常次数
                continue_pressed = False

            pause_event.wait()  # 暂停

            # 上Buff
            logger.info(f'准备上Buff..')
            if role.buff_effective:
                for buff in role.buffs:
                    kbu.do_buff(buff)
            else:
                logger.info(f'不需要上Buff..')

            # 分析小地图
            cols, rows = 0, 0
            cur_row, cur_col = 0, 0
            map_crop = None
            boss_room = (-1, -1)
            current_room = (-1, -1)

            map_error_cnt = 0
            analyse_map_error = True
            while analyse_map_error:
                try:
                    img0 = capturer.capture()

                    # 分析小地图的行列
                    cols = map_util.get_colum_count(img0)
                    rows = map_util.get_row_count(img0)
                    logger.warning("分析小地图的行列{},{}", rows, cols)

                    # 裁剪小地图区域
                    map_crop = map_util.get_small_map_region_img(img0, rows, cols)

                    # 获取boss房间位置，0基
                    boss_room = map_util.get_boss_from_crop(map_crop, rows, cols)
                    logger.info('boss房间是 {}', boss_room)
                    current_room = map_util.current_room_index_cropped(map_crop, rows, cols)  # 实际上没有用，只是打印看一下位置
                    logger.info('当前房间是 {}', current_room)
                    cur_row, cur_col = current_room
                except Exception as e:
                    logger.error(e)
                    traceback.print_exc()

                analyse_map_error = boss_room is None or current_room is None or boss_room == (-1, -1) or current_room == (-1, -1)
                if analyse_map_error:
                    map_error_cnt = map_error_cnt + 1
                    # cv2.imwrite(f'errorDetectMap_init_{map_error_cnt}.jpg', img0)
                    logger.error(f"分析小地图的行列init，第 {map_error_cnt} 次出错,行列是 {rows} , {cols}")
                    logger.error("暂停2秒继续重试！！")
                    time.sleep(1)
                else:
                    map_error_cnt = 0

                if analyse_map_error and map_error_cnt > 20:
                    logger.error("分析小地图的行列init多次出错了 废了！！！")
                    # cv2.imwrite(f'errorDetectMap_init_{map_error_cnt}.jpg', capturer.capture())
                    break

            allow_directions = map_util.get_allow_directions(map_crop, cur_row, cur_col)

            # 初始化
            finder = PathFinder(rows, cols, boss_room)

            logger.info(f'准备打怪..')

            # todo 循环打怪过图，过房间 循环开始////////////////////////////////
            fq = FixedLengthQueue(max_length=30)
            room_idx_list = FixedLengthQueue(max_length=100)
            stuck_room_idx = None
            hero_pos_is_stable = False

            collect_loot_pressed = False  # 按过移动物品了
            collect_loot_pressed_time = 0
            boss_appeared = False  # 遭遇boss了
            sss_appeared = False  # 已经结算了
            door_absence_time = 0  # 什么也没识别到的时间(没识别到门)
            boss_door_appeared = False
            path_stack = []  # ((x,y),direction) 房间，去下一个房间的方向
            card_esc_time = 0
            card_appear_time = 0
            hero_stuck_pos = {}  # 卡住的位置 ((r,c),[(x,y),(x,y)])
            die_time = 0
            in_boss_room = False
            delay_break = 0

            frame_time = time.time()
            while True:  # 循环打怪过图，过房间
                # 限制处理速率
                if max_fps:
                    if time.time() - frame_time < 1.0 / max_fps:
                        time.sleep(0.02)
                        continue
                    frame_time = time.time()

                pause_event.wait()  # 暂停

                # 截图
                img0 = capturer.capture()

                # 识别
                cv_det_task = None
                if boss_appeared or in_boss_room or boss_door_appeared or game_mode == 2:
                    cv_det_task = img_executor.submit(object_detection_cv, img0)
                img4show = img0.copy()
                # 执行推理
                results = model.predict(
                    source=img0,
                    device=device,
                    imgsz=640,
                    conf=0.7,
                    iou=0.2,
                    verbose=False
                )

                if results is None or len(results) == 0 or len(results[0].boxes) == 0:
                    # logger.info('模型没有识别到物体')
                    if not sss_appeared:
                        mover.move(target_direction=random.choice(kbu.single_direct))
                    continue

                # # todo
                # if show:
                #     annotated_frame = results[0].plot()
                #     # 将结果放入队列，供展示线程使用
                #     result_queue.put(annotated_frame)

                # print('results[0].boxes', results[0].boxes)
                # 分析推理结果,组装类别数据
                det = analyse_det_result(results, hero_height, img4show)
                # logger.debug(f'det_res是什么 {det}')
                # logger.debug(f'doors:{det.door_xywh_list}')

                hero_xywh = det.hero_xywh
                monster_xywh_list = det.monster_xywh_list
                elite_monster_xywh_list = det.elite_monster_xywh_list
                boss_xywh_list = det.boss_xywh_list
                loot_xywh_list = det.loot_xywh_list
                gold_xywh_list = det.gold_xywh_list
                door_xywh_list = det.door_xywh_list
                door_boss_xywh_list = det.door_boss_xywh_list

                card_num = det.card_num
                continue_exist = det.continue_exist
                shop_exist = det.shop_exist
                shop_mystery_exist = det.shop_mystery_exist
                menu_exist = det.menu_exist
                sss_exist = det.sss_exist

                if stuck_room_idx is not None:
                    logger.debug("材料卡住了,loot_xywh_list置空")
                    loot_xywh_list = []
                    gold_xywh_list = []

                if sss_exist or continue_exist or shop_exist:
                    # logger.debug(f"出现翻牌{sss_exist}，再次挑战了{continue_exist}")
                    if not sss_appeared:
                        sss_appeared = True
                        logger.warning(f'【{role.name}】 刷图,第 {fight_count} 次，打怪结束，耗时...{(time.time() - one_game_start):.1f}秒')
                if door_boss_xywh_list:
                    if not boss_door_appeared:
                        logger.info(f"出现boss门了")
                        boss_door_appeared = True
                if boss_xywh_list:
                    if not boss_appeared:
                        logger.info(f"出现boss了")
                        boss_appeared = True
                        in_boss_room = True

                if cv_det_task:
                    cv_det = cv_det_task.result()
                    if cv_det and cv_det["death"]:

                        logger.warning(f"角色死了")

                        if time.time() - die_time > 11:
                            die_time = time.time()
                            logger.warning(f"死亡提醒!!")
                            # 声音提醒 不要
                            # 邮件提醒
                            mode_name = (
                                "白图" if game_mode == 1 else
                                "每日1+1" if game_mode == 2 else
                                "妖气追踪" if game_mode == 3 else
                                "妖怪歼灭" if game_mode == 4 else
                                "未知模式"
                            )
                            email_subject = f"{mode_name} {role.name}阵亡通知书"
                            email_content = f"鏖战{mode_name}，角色【{role.name}】不幸阵亡，及时查看处理。"
                            mail_receiver = mail_config.get("receiver")
                            if mail_receiver:
                                tool_executor.submit(lambda: (
                                    mail_sender.send_email(email_subject, email_content, mail_receiver),
                                    logger.info("角色死亡 已经发送邮件提醒了")
                                ))
                            else:
                                logger.warning("角色死亡 邮件提醒没有配置,跳过")

                        logger.warning(f"检测到死了，准备复活")
                        time.sleep(8)  # 拖慢点复活
                        kbu.do_press('x')
                        time.sleep(0.1)
                        kbu.do_press('x')
                        time.sleep(0.1)

                if hero_xywh:
                    fq.enqueue((hero_xywh[0], hero_xywh[1]))
                    hero_pos_is_stable = fq.coords_is_stable(threshold=10, window_size=10)
                    if hero_pos_is_stable and not sss_appeared and stuck_room_idx is None:
                        random_direct = random.choice(random.choice([kbu.single_direct, kbu.double_direct]))
                        logger.warning('可能卡住不能移动了{},随机跑个方向看看-->{}', hero_xywh, random_direct)  # todo 方向处理
                        kbd_current_direction = mover.get_current_direction()
                        # 先看是不是在门上
                        if len(door_xywh_list + door_boss_xywh_list) > 0 and exist_near(hero_xywh, door_xywh_list + door_boss_xywh_list, 100):
                            # allow_directions_, next_room_direction_ = None, None
                            # try:
                            #     # 裁剪小地图区域
                            #     map_crop = map_util.get_small_map_region_img(img0, rows, cols)
                            #     # 当前房间位置
                            #     current_room = map_util.current_room_index_cropped(map_crop, rows, cols)
                            #     logger.info('小卡，当前房间是 {}', current_room)
                            #     allow_directions_ = map_util.get_allow_directions(map_crop, cur_row, cur_col)
                            #     next_room_direction_ = finder.get_next_direction((cur_row, cur_col), allow_directions_)
                            # except Exception as e:
                            #     logger.error(e)
                            #     traceback.print_exc()
                            stand_on_door = exist_near(hero_xywh, door_xywh_list + door_boss_xywh_list, 100)
                            if hero_xywh[0] > img0.shape[1] * 4 // 5 and stand_on_door[0] > img0.shape[1] * 4 // 5:
                                logger.debug("人在右边2")
                                random_direct = "LEFT"
                            elif hero_xywh[0] < img0.shape[1] * 1 // 5 and stand_on_door[0] < img0.shape[1] * 1 // 5:
                                logger.debug("人在左边2")
                                random_direct = "RIGHT"
                                if 300 < hero_xywh[1] < 390:
                                    logger.debug("人在左边2且上边")
                                    random_direct = random.choice(['RIGHT', 'RIGHT_DOWN'])
                            elif hero_xywh[0] > img0.shape[1] * 4 // 5:
                                logger.debug("人在右边")
                                random_direct = random.choice(list(filter(lambda x1: x1 != "RIGHT" and x1 != kbd_current_direction, kbu.single_direct)))
                            elif hero_xywh[0] < img0.shape[1] * 1 // 5:
                                logger.debug("人在左边")
                                random_direct = random.choice(list(filter(lambda x1: x1 != "LEFT" and x1 != kbd_current_direction, kbu.single_direct)))
                            elif hero_xywh[1] > img0.shape[0] * 3 // 5:
                                logger.debug("人在下面")
                                random_direct = random.choice(list(filter(lambda x1: x1 != "DOWN" and x1 != kbd_current_direction, kbu.single_direct)))
                            else:
                                logger.debug("人在上面")
                                random_direct = random.choice(list(filter(lambda x1: x1 != "UP" and x1 != kbd_current_direction, kbu.single_direct)))
                        else:
                            logger.debug(f"x轴上位置：{hero_xywh[0]:.2f},{(hero_xywh[0] / img0.shape[1]):.2f}---,current:{kbd_current_direction}")
                            logger.debug(f"y轴上位置：{hero_xywh[1]:.2f},{(hero_xywh[1] / img0.shape[0]):.2f}---,current:{kbd_current_direction}")

                            if hero_xywh[1] < 400 and hero_xywh[0] > 850 and (kbd_current_direction is None or "UP" in kbd_current_direction or "RIGHT" in kbd_current_direction):
                                logger.warning("人在右边3")
                                mover.move(target_direction="LEFT")
                                time.sleep(1.2)
                                logger.warning("强制向左1秒")
                            elif hero_xywh[0] > img0.shape[1] * 3 // 4 and (kbd_current_direction is None or "RIGHT" in kbd_current_direction):
                                logger.warning("人在右边2")
                                mover.move(target_direction="LEFT")
                                time.sleep(1.2)
                                logger.warning("强制向左1秒")
                                random_direct = random.choice(list(filter(lambda x1: x1 != "RIGHT" and x1 != kbd_current_direction, kbu.single_direct)))
                            elif hero_xywh[1] < 400 and hero_xywh[0] < 200 and (kbd_current_direction is None or "UP" in kbd_current_direction or "LEFT" in kbd_current_direction):
                                logger.warning("人在左边3")
                                mover.move(target_direction="RIGHT")
                                time.sleep(1.2)
                                logger.warning("强制向右1秒")
                            elif hero_xywh[0] < img0.shape[1] * 1 // 5 and (kbd_current_direction is None or "LEFT" in kbd_current_direction):
                                logger.warning("人在左边2")
                                mover.move(target_direction="RIGHT")
                                time.sleep(1.2)
                                logger.warning("强制向右1秒")
                                random_direct = random.choice(list(filter(lambda x1: x1 != "LEFT" and x1 != kbd_current_direction, kbu.single_direct)))
                            else:
                                try:
                                    # 裁剪小地图区域
                                    map_crop = map_util.get_small_map_region_img(img0, rows, cols)
                                    # 当前房间位置
                                    current_room = map_util.current_room_index_cropped(map_crop, rows, cols)
                                    logger.warning('小卡，当前房间是 {}', current_room)
                                    # if not hero_stuck_pos[current_room]:
                                    #     hero_stuck_pos[current_room] = []
                                    #     hero_stuck_pos[current_room].append(hero_xywh)
                                    if current_room != (-1, -1):
                                        previous = None
                                        if current_room in [item[0] for item in path_stack]:
                                            for ii in range(len(path_stack) - 1, 0, -1):
                                                if path_stack[ii][0] == current_room:
                                                    previous = path_stack[ii - 1][1]
                                                    logger.info('小卡，当前房间finder过，之前是向【{}】走，走过来的', previous)
                                                    if hero_xywh[0] < 100 and door_xywh_list and len(door_xywh_list) == 1 and door_xywh_list[0][0] < 100:
                                                        logger.debug("左侧处理")
                                                        random_direct = random.choice(list(filter(lambda x1: x1 != get_opposite_direction(previous) and x1 != kbd_current_direction and x1 not in ["DOWN", "LEFT"], kbu.single_direct)))
                                                    elif 882 < hero_xywh[0] < 888 and 300 < hero_xywh[1] < 305 and kbd_current_direction == "DOWN" and previous == "RIGHT":
                                                        logger.debug("人可能被卡在右上了")
                                                        random_direct = "LEFT_DOWN"
                                                    elif hero_xywh[0] > 888 and previous == "RIGHT":
                                                        logger.debug("已经向右走到底了，回头")
                                                        random_direct = random.choice(["LEFT", "LEFT_DOWN", "LEFT_UP"])
                                                        mover.move(target_direction=random_direct)
                                                        time.sleep(round(random.uniform(0.6, 0.8), 1))
                                                    elif hero_xywh[0] < 179 and previous == "LEFT":
                                                        logger.debug("已经向左到底了，回头")
                                                        random_direct = random.choice(["RIGHT", "RIGHT_DOWN", "RIGHT_UP"])
                                                        mover.move(target_direction=random_direct)
                                                        time.sleep(round(random.uniform(0.6, 0.8), 1))
                                                    else:
                                                        logger.debug("else 了", previous)
                                                        random_direct = random.choice(list(filter(lambda x1: x1 != get_opposite_direction(previous) and x1 != kbd_current_direction, kbu.single_direct)))
                                                    break
                                        else:
                                            logger.info('小卡，当前房间未finder过，之前是向【{}】走，走过来的', path_stack[-1][1])
                                            previous = path_stack[-1][1]
                                            random_direct = random.choice(
                                                list(filter(lambda x1: x1 != get_opposite_direction(path_stack[-1][1]) and x1 != kbd_current_direction, kbu.single_direct)))
                                            if hero_xywh[0] < 70:
                                                random_direct = random.choice(["RIGHT", random_direct])
                                            elif hero_xywh[0] > 950:
                                                random_direct = random.choice(["LEFT", random_direct])

                                        if hero_xywh[1] < 400 and kbd_current_direction == "UP" and previous == "RIGHT" and hero_xywh[0] < 630:
                                            logger.debug("走到底，上小卡处理1。。")
                                            random_direct = random.choice(["RIGHT", "RIGHT_DOWN"])
                                        elif hero_xywh[1] < 400 and kbd_current_direction == "UP" and previous == "LEFT" and hero_xywh[0] > 420:
                                            logger.debug("走到底，上小卡处理2。。")
                                            random_direct = random.choice(["LEFT", "LEFT_DOWN"])

                                except Exception as e:
                                    logger.error(e)
                                    traceback.print_exc()

                        fq.clear()  # 重置历史记录
                        room_idx_list.clear()
                        stuck_room_idx = None
                        logger.warning('可能卡住不能移动了,随机跑个方向看看-->{}', random_direct)  # todo 方向处理
                        mover.move(target_direction=random_direct)
                        time.sleep(round(random.uniform(0.2, 0.6), 1))
                        continue
                else:  # todo 没有识别到角色
                    if not sss_appeared:
                        random_direct = random.choice(kbu.single_direct)
                        logger.warning('未检测到角色,随机跑个方向看看{}', random_direct)
                        # mover._release_all_keys()
                        mover.move(target_direction=random_direct)
                    else:
                        logger.info('未检测到角色,已经结算了')
                        if not collect_loot_pressed and (sss_exist or continue_exist or shop_exist or shop_mystery_exist):
                            # kbu.do_press_with_time(Key.left, 3000, 100)
                            mover.move(target_direction="LEFT")
                            time.sleep(0.1)
                    # continue

                # ############################### 判断-准备打怪 ######################################
                wait_for_attack = hero_xywh and (monster_xywh_list or boss_xywh_list or elite_monster_xywh_list) and not sss_appeared
                monster_box = None
                monster_in_range = False
                role_attack_center = None
                best_attack_point = None
                if wait_for_attack:

                    if stuck_room_idx:
                        stuck_room_idx = None
                        room_idx_list.clear()

                    role_attack_center = (hero_xywh[0], hero_xywh[1])
                    if mover.get_current_direction() is None or "RIGHT" in mover.get_current_direction():
                        role_attack_center = (hero_xywh[0] + role.attack_center_x, hero_xywh[1])
                    else:
                        role_attack_center = (hero_xywh[0] - role.attack_center_x, hero_xywh[1])

                    # 距离最近的怪 todo 改成最近的堆
                    # monster_box, _ = get_closest_obj(itertools.chain(monster_xywh_list, boss_xywh_list), role_attack_center)
                    monster_box = find_densest_monster_cluster(monster_xywh_list + boss_xywh_list + elite_monster_xywh_list, role_attack_center)

                    if show:
                        # 怪(堆中心) 蓝色
                        cv2.circle(img4show, (int(monster_box[0]), int(monster_box[1])), 5, color_blue, 4)
                    # 怪处于攻击范围内
                    # monster_in_range = abs(hero_xywh[0] - monster_box[0]) < attx and abs(hero_xywh[1] - monster_box[1]) < atty

                    if role.attack_center_x:
                        if mover.get_current_direction() is None or "RIGHT" in mover.get_current_direction():
                            monster_in_range = (monster_box[0] > role_attack_center[0]
                                                and abs(role_attack_center[0] - monster_box[0]) < attack_x
                                                and abs(role_attack_center[1] - monster_box[1]) < attack_y
                                                ) or (
                                                       monster_box[0] < role_attack_center[0]
                                                       and abs(role_attack_center[0] - monster_box[0]) < (role.attack_center_x * 0.65)
                                                       and abs(role_attack_center[1] - monster_box[1]) < attack_y
                                               )
                        else:
                            monster_in_range = (monster_box[0] < role_attack_center[0]
                                                and abs(role_attack_center[0] - monster_box[0]) < attack_x
                                                and abs(role_attack_center[1] - monster_box[1]) < attack_y
                                                ) or (
                                                   (monster_box[0] > role_attack_center[0]
                                                    and abs(role_attack_center[0] - monster_box[0]) < (role.attack_center_x * 0.65)
                                                    and abs(role_attack_center[1] - monster_box[1]) < attack_y
                                                    )
                                               )
                    else:
                        if mover.get_current_direction() is None or "RIGHT" in mover.get_current_direction():
                            monster_in_range = (monster_box[0] > role_attack_center[0]
                                                and abs(role_attack_center[0] - monster_box[0]) < attack_x
                                                and abs(role_attack_center[1] - monster_box[1]) < attack_y
                                                )
                        else:
                            monster_in_range = (monster_box[0] < role_attack_center[0]
                                                and abs(role_attack_center[0] - monster_box[0]) < attack_x
                                                and abs(role_attack_center[1] - monster_box[1]) < attack_y
                                                )

                    # if fought_boss:
                    #     monster_in_range = abs(hero_xywh[0] - monster_box[0]) < 300 and abs(hero_xywh[1] - monster_box[1]) < 200
                    if show and monster_in_range:
                        # 怪处于攻击范围内,给角色一个标记
                        cv2.circle(img4show, (int(hero_xywh[0]), int(hero_xywh[1])), 10, color_yellow, 2)

                # # todo 待考虑
                # if not wait_for_attack and not sss_appeared:
                #     cur_row, cur_col = map_util.current_room_index_cropped(map_crop, rows, cols)
                #     current_room = (cur_row, cur_col)
                #     map_crop = map_util.get_small_map_region_img(img0, rows, cols)

                # ############################ 判断-准备进入下一个房间 ####################################
                # todo 门开了 = map_util.门开了()
                wait_for_next_room = (hero_xywh
                                      and ((door_xywh_list or door_boss_xywh_list) and not monster_xywh_list and not elite_monster_xywh_list and not boss_xywh_list and not loot_xywh_list and not gold_xywh_list)
                                      and not sss_appeared)
                next_room_direction = None
                door_box = None
                door_in_range = False
                if wait_for_next_room:

                    door_absence_time = 0
                    # 根据小地图分析 下一个房间所在的方向(上校左右)
                    try:
                        map_door_error_cnt = 0
                        analyse_map_door_error = True
                        while analyse_map_door_error:
                            allow_directions = []
                            in_boss_room = False
                            try:
                                img00 = capturer.capture()

                                # 裁剪小地图区域
                                map_crop = map_util.get_small_map_region_img(img00, rows, cols)
                                # 当前房间位置
                                current_room = map_util.current_room_index_cropped(map_crop, rows, cols)
                                logger.info('当前房间是 {}', current_room)
                                cur_row, cur_col = current_room

                                if current_room == boss_room or (boss_door_appeared and current_room == (-1, -1)):
                                    in_boss_room = True

                                allow_directions = map_util.get_allow_directions(map_crop, cur_row, cur_col)
                                logger.debug(f"allow_directions:{allow_directions}")
                            except Exception as e:
                                logger.error(e)
                                traceback.print_exc()

                            analyse_map_door_error = not allow_directions
                            if analyse_map_door_error:
                                if in_boss_room:
                                    logger.info("在boss房间分析出错，无视")
                                    break
                                map_door_error_cnt = map_door_error_cnt + 1
                                # cv2.imwrite(f'errorDetectMap_door_{map_door_error_cnt}.jpg', map_crop)
                                logger.error(f"分析小地图的行列door，第 {map_door_error_cnt} 次出错,行列是 {rows} , {cols}")
                                logger.error("暂停2秒继续重试！！")
                                time.sleep(2)
                            else:
                                map_door_error_cnt = 0
                                next_room_direction = finder.get_next_direction((cur_row, cur_col), allow_directions)
                                logger.debug(f"next_room_direction:{next_room_direction}")

                                if path_stack and path_stack[-1][0] == current_room:
                                    pass
                                else:
                                    if next_room_direction:
                                        logger.debug(f"加入path, 当前房间是 {current_room}, 模板方向是 {next_room_direction}")
                                        path_stack.append((current_room, next_room_direction))

                            if analyse_map_door_error and map_door_error_cnt > 5:
                                logger.error("分析小地图的行列door多次出错了 废了！！！")
                                break

                        if next_room_direction is None or current_room is None:
                            # 没正确的分析出小地图信息,跳过
                            logger.warning('没正确的分析出小地图信息,跳过')
                            continue
                    except Exception as e:
                        logger.warning(f'小地图分析异常报错,跳过.{e}')
                        traceback.print_exc()
                        # boss_room = map_util.get_boss_room(window_utils.capture_window_BGRX(handle))
                        # logger.info('boss房间是 {}', boss_room)
                        continue

                    if stuck_room_idx is not None and stuck_room_idx == current_room:  # 已经被卡住了，且还位于被卡房间（材料置空--无意义--能进这个逻辑，材料list肯定已经是空的）
                        logger.debug("已经被材料时有时无卡住了,忽略材料")
                        loot_xywh_list = []
                        gold_xywh_list = []
                        wait_for_next_room = hero_xywh and (
                                (door_xywh_list or door_boss_xywh_list) and not monster_xywh_list and not elite_monster_xywh_list and not boss_xywh_list and not loot_xywh_list and not gold_xywh_list)
                    elif stuck_room_idx is not None and stuck_room_idx != current_room:  # 已经被卡住了，且不在被卡房间，（出去了，置空）
                        stuck_room_idx = None
                        room_idx_list.enqueue(current_room)  # 记录识别的房间位置
                    else:  # 还没有被卡住
                        room_idx_list.enqueue(current_room)  # 记录识别的房间位置
                        room_is_same = room_idx_list.room_is_same(min_size=80)
                        if room_is_same and not hero_pos_is_stable:  # 之前没卡住，刚刚计算得到卡住
                            logger.warning(f"可能可能可能可能被材料时有时无 卡住了 当前房间{current_room}")
                            stuck_room_idx = current_room
                            room_idx_list.clear()
                        else:  # 之前没卡住，现在也没卡住
                            stuck_room_idx = None

                    # 找这个方向上最远的门
                    door_box = find_door_by_position(door_xywh_list + door_boss_xywh_list, next_room_direction)

                    door_in_range = abs(door_box[1] - hero_xywh[1]) < door_hit_y * 2 and abs(door_box[0] - hero_xywh[0]) < door_hit_x  # todo 门的范围问题
                    if show and door_box:
                        # 给目标门口画一个点
                        cv2.circle(img4show, (int(door_box[0]), int(door_box[1])), 1, color_blue, 3)

                # ####################### 判断-准备拾取材料 #############################################
                # wait_for_pickup = hero_xywh and (not monster_xywh_list and hero_xywh and (loot_xywh_list or gold_xywh_list) and not continue_exist)
                wait_for_pickup = hero_xywh and (loot_xywh_list or gold_xywh_list) and (
                        not monster_xywh_list and not elite_monster_xywh_list and not boss_xywh_list)
                material_box = None
                loot_in_range = False
                material_min_distance = float("inf")
                material_is_gold = False
                if wait_for_pickup:
                    # 距离最近的掉落物
                    material_box, material_min_distance = get_closest_obj(
                        itertools.chain(loot_xywh_list, gold_xywh_list), det.hero_xywh)
                    if material_box in gold_xywh_list:
                        material_is_gold = True
                    if show and material_box:
                        # 给目标掉落物画一个点
                        cv2.circle(img4show, (int(material_box[0]), int(material_box[1])), 2, color_blue, 3)
                    # 材料处于拾取范围
                    loot_in_range = abs(material_box[1] - hero_xywh[1]) < pick_up_y and abs(material_box[0] - hero_xywh[0]) < pick_up_x
                    if show and loot_in_range:
                        # 材料处于拾取范围,给角色一个标记
                        cv2.circle(img4show, (int(hero_xywh[0]), int(hero_xywh[1])), 10, color_yellow, 2)

                # 截图展示前的处理完毕,进行显示
                if show:
                    # img4show = cv2.resize(img4show, (756, 425))
                    # cv2.namedWindow('Game Capture', cv2.WINDOW_NORMAL)
                    # cv2.resizeWindow('Game Capture', 756, 425)

                    # result_queue.put(img4show)

                    cv2.imshow('Game Capture', img4show)
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        show = False
                        cv2.destroyAllWindows()

                    # pass
                # ######################### 判断完毕,进行逻辑处理 ########################################################

                # 逻辑处理-找门进入下个房间>>>>>>>>>>>>>>>>>>>>>>>>>>
                if wait_for_next_room:

                    pause_event.wait()  # 暂停

                    is_target_door = judge_is_target_door(current_room, door_box, hero_xywh, next_room_direction, allow_directions, path_stack, det, img0)
                    logger.info(f"判断目标门：{is_target_door}")

                    # todo 门还要处理，做追踪？
                    # if len(allow_directions) > len(door_xywh_list + door_boss_xywh_list):
                    if not is_target_door:
                        # 尚未出现目标门,需要继续移动寻找 todo 当前画面一个门也没有的时候进不来这个逻辑
                        if next_room_direction == 'RIGHT' and (not door_box or door_box[0] < img0.shape[1] * 4 // 5):  # 右侧四分之一还没有门出现,继续往右
                            logger.debug("目标房间在右边---->右侧四分之一还没有门出现,继续往右")
                            # todo 防止走向目标门的过程中,误入其他门(主要是左右跑的时候,误入了上方或下方的门)
                            mover.move(target_direction="RIGHT")
                            continue
                        if next_room_direction == 'LEFT' and (not door_box or door_box[0] > img0.shape[1] // 5):  # 左侧四分之一还没有门出现,继续往左
                            logger.debug("目标房间在左边---->左侧四分之一还没有门出现,继续往左")
                            mover.move(target_direction="LEFT")
                            continue
                        if next_room_direction == 'DOWN' and (not door_box or door_box[1] <= img0.shape[0] * 775 // 1000 or (door_box and (door_box[0] < img0.shape[1] // 7 or door_box[0] > img0.shape[1] * 6 // 7))):
                            logger.debug("目标房间在下边---->下侧四分之一还没有门出现,继续往下")
                            mover.move(target_direction="DOWN")
                            continue
                        if next_room_direction == 'UP' and (not door_box or door_box[1] > img0.shape[0] * 0.72 or (door_box and (door_box[0] < img0.shape[1] // 7 or door_box[0] > img0.shape[1] * 6 // 7))):
                            logger.debug("目标房间在上边---->上侧二分之一还没有门出现,继续往上")
                            mover.move(target_direction="UP")
                            continue

                    # 门在命中范围内,等待过图即可
                    if door_in_range:
                        # 不管了,全部释放掉
                        mover._release_all_keys()
                        logger.info("门在命中范围内,等待过图")
                        time.sleep(0.1)
                        if stuck_room_idx is not None:
                            # todo 除歼灭不存在 跳过材料时无卡住的逻辑
                            logger.debug("等三秒直接跳过材料")
                            time.sleep(1)
                            # 可能没过去，随便走两步，(todo 根据角色位置，决定往哪里走)
                            if next_room_direction == 'RIGHT':
                                logger.debug("先向左走两步")
                                mover.move(target_direction="LEFT")
                                time.sleep(0.5)
                            if next_room_direction == 'LEFT':
                                logger.debug("先向右走两步")
                                mover.move(target_direction="RIGHT")
                                time.sleep(0.5)
                            if next_room_direction == 'UP':
                                logger.debug("先向下走两步")
                                mover.move(target_direction="DOWN")
                                time.sleep(0.5)
                            if next_room_direction == 'DOWN':
                                logger.debug("先向上走两步")
                                mover.move(target_direction="UP")
                                time.sleep(0.5)
                            # stuck_room_idx = None
                            # room_idx_list.clear()
                        continue

                    # 已经确定目标门,移动到目标位置
                    # 目标在角色的右上方
                    if door_box[1] - hero_xywh[1] < 0 and door_box[0] - hero_xywh[0] > 0:
                        # y方向上处于范围内,只需要x方向移动
                        if abs(door_box[1] - hero_xywh[1]) < door_hit_y:
                            # print("y方向上处于范围内,只需要x方向移动")
                            mover.move(target_direction="RIGHT")
                        # x轴上的距离比较远,斜方向移动
                        elif abs(hero_xywh[1] - door_box[1]) < abs(door_box[0] - hero_xywh[0]):
                            # print("x轴上的距离比较远,斜方向移动")
                            mover.move(target_direction="RIGHT_UP")
                        # y轴上的距离也比较远,只进行y轴上的移动
                        elif abs(hero_xywh[1] - door_box[1]) >= abs(door_box[0] - hero_xywh[0]):
                            mover.move(target_direction="UP")
                    # 目标在角色的左上方
                    elif door_box[1] - hero_xywh[1] < 0 and door_box[0] - hero_xywh[0] < 0:
                        # y方向上处于范围内,只需要x方向移动
                        if abs(door_box[1] - hero_xywh[1]) < door_hit_y:
                            mover.move(target_direction="LEFT")
                        # x轴上的距离比较远,斜方向移动
                        elif abs(hero_xywh[1] - door_box[1]) < abs(hero_xywh[0] - door_box[0]):
                            mover.move(target_direction="LEFT_UP")
                        # y轴上的距离也比较远,只进行y轴上的移动
                        elif abs(hero_xywh[1] - door_box[1]) >= abs(hero_xywh[0] - door_box[0]):
                            mover.move(target_direction="UP")
                    # 目标在角色的左下方
                    elif door_box[1] - hero_xywh[1] > 0 and door_box[0] - hero_xywh[0] < 0:
                        # y方向上处于范围内,只需要x方向移动
                        if abs(door_box[1] - hero_xywh[1]) < door_hit_y:
                            mover.move(target_direction="LEFT")
                        # x轴上的距离比较远,斜方向移动
                        elif abs(door_box[1] - hero_xywh[1]) < abs(hero_xywh[0] - door_box[0]):
                            mover.move(target_direction="LEFT_DOWN")
                        # y轴上的距离也比较远,只进行y轴上的移动
                        elif abs(door_box[1] - hero_xywh[1]) >= abs(hero_xywh[0] - door_box[0]):
                            mover.move(target_direction="DOWN")
                    # 目标在角色的右下方
                    elif door_box[1] - hero_xywh[1] > 0 and door_box[0] - hero_xywh[0] > 0:
                        # y方向上处于范围内,只需要x方向移动
                        if abs(door_box[1] - hero_xywh[1]) < door_hit_y:
                            mover.move(target_direction="RIGHT")
                        # x轴上的距离比较远,斜方向移动
                        elif abs(door_box[1] - hero_xywh[1]) < abs(door_box[0] - hero_xywh[0]):
                            mover.move(target_direction="RIGHT_DOWN")
                        # y轴上的距离也比较远,只进行y轴上的移动
                        elif abs(door_box[1] - hero_xywh[1]) >= abs(door_box[0] - hero_xywh[0]):
                            mover.move(target_direction="DOWN")

                    continue
                # 逻辑处理-找门进入下个房间<<<<<<<<<<<<<<<<<<<<<<<<<

                # 逻辑处理-有怪要打怪>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                if wait_for_attack:
                    # 处于攻击范围
                    if monster_in_range and mover.get_current_direction() is not None:

                        # 不管了,全部释放掉
                        mover._release_all_keys()

                        # 调整方向,面对怪
                        if hero_xywh[0] - monster_box[0] > 100:
                            logger.debug('面对怪,朝左，再放技能')
                            kbu.do_press(Key.left)
                        elif monster_box[0] > hero_xywh[0] > 100:
                            logger.debug('面对怪,朝右，再放技能')
                            kbu.do_press(Key.right)
                        time.sleep(0.05)

                        skill_name = None
                        if role.powerful_skills and (boss_xywh_list):
                            # skill_name = skill_util.suggest_skill_powerful(role, img0)
                            skill_name = skill_util.get_available_skill_from_list_by_match(skills=role.powerful_skills, img0=img0, skill_images=skill_images)
                        if skill_name is None:
                            # 推荐技能
                            # skill_name = skill_util.suggest_skill(role, img0)
                            skill_name = skill_util.suggest_skill_by_img_match(role, img0, skill_images)
                        skill_util.cast_skill(skill_name)
                        # 小等一下 比如等怪死
                        if skill_name == 'x':
                            ...
                        else:
                            time.sleep(0.1)
                        continue

                    pause_event.wait()  # 暂停
                    # 目标在角色右上方
                    if monster_box[1] - role_attack_center[1] < 0 and monster_box[0] - role_attack_center[0] > 0:
                        # y方向已经处于攻击范围,只需要x方向移动
                        if abs(monster_box[1] - role_attack_center[1]) < attack_y:
                            mover.move(target_direction="RIGHT")
                        # x轴上的距离比较远,斜方向移动
                        elif abs(role_attack_center[1] - monster_box[1]) < abs(monster_box[0] - role_attack_center[0]):
                            mover.move(target_direction="RIGHT_UP")
                        # y轴上的距离也比较远,只进行y轴上的移动
                        elif abs(role_attack_center[1] - monster_box[1]) >= abs(monster_box[0] - role_attack_center[0]):
                            mover.move(target_direction="UP")

                    # 目标在角色左上方
                    elif monster_box[1] - role_attack_center[1] < 0 and monster_box[0] - role_attack_center[0] < 0:
                        # y方向已经处于攻击范围,只需要x方向移动
                        if abs(monster_box[1] - role_attack_center[1]) < attack_y:
                            mover.move(target_direction="LEFT")
                        # x轴上的距离比较远,斜方向移动
                        elif role_attack_center[1] - monster_box[1] < role_attack_center[0] - monster_box[0]:
                            mover.move(target_direction="LEFT_UP")
                        # y轴上的距离也比较远,只进行y轴上的移动
                        elif role_attack_center[1] - monster_box[1] >= role_attack_center[0] - monster_box[0]:
                            mover.move(target_direction="UP")

                    # 目标在角色左下方
                    elif monster_box[1] - role_attack_center[1] > 0 and monster_box[0] - role_attack_center[0] < 0:
                        # y方向已经处于攻击范围,只需要x方向移动
                        if abs(monster_box[1] - role_attack_center[1]) < attack_y:
                            mover.move(target_direction="LEFT")
                        # x轴上的距离比较远,斜方向移动
                        elif monster_box[1] - role_attack_center[1] < role_attack_center[0] - monster_box[0]:
                            mover.move(target_direction="LEFT_DOWN")
                        # y轴上的距离也比较远,只进行y轴上的移动
                        elif monster_box[1] - role_attack_center[1] >= role_attack_center[0] - monster_box[0]:
                            mover.move(target_direction="DOWN")

                    # 目标在角色右下方
                    elif monster_box[1] - role_attack_center[1] > 0 and monster_box[0] - role_attack_center[0] > 0:
                        # y方向已经处于攻击范围,只需要x方向移动
                        if abs(monster_box[1] - role_attack_center[1]) < attack_y:
                            mover.move(target_direction="RIGHT")
                        # x轴上的距离比较远,斜方向移动
                        elif monster_box[1] - role_attack_center[1] < monster_box[0] - role_attack_center[0]:
                            mover.move(target_direction="RIGHT_DOWN")
                        # y轴上的距离也比较远,只进行y轴上的移动
                        elif monster_box[1] - role_attack_center[1] >= monster_box[0] - role_attack_center[0]:
                            mover.move(target_direction="DOWN")

                    continue
                # 逻辑处理-有怪要打怪<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                # 逻辑处理-出现菜单>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                if menu_exist:
                    kbu.do_press(Key.esc)
                    logger.info("关闭菜单")
                    time.sleep(0.1)
                    continue
                # 逻辑处理-出现菜单<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                # 逻辑处理-捡材料>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                if wait_for_pickup:
                    if gold_xywh_list:
                        # logger.error(f"有金币金币金币!  {gold_xywh_list}")
                        pass
                    if sss_appeared and not collect_loot_pressed:
                        logger.info("预先移动物品到脚下")
                        # 不管了,全部释放掉
                        mover._release_all_keys()

                        collect_loot_pressed = True
                        collect_loot_pressed_time = time.time()

                        executor.submit(lambda: (
                            logger.info("预先移动物品到脚下"),
                            time.sleep(2.1),
                            kbu.do_press(dnf.Key_collect_loot),
                            time.sleep(0.1),
                            kbu.do_press_with_time('x', 4000 if game_mode == 4 else 2000, 50),
                            logger.info("预先长按x 按完x了"),
                        ))

                        continue
                    elif sss_appeared and collect_loot_pressed and time.time() - collect_loot_pressed_time < 7:
                        tt = time.time()
                        if 0.1 < tt - int(tt) < 0.2:  # 0.6 < tt - int(tt) < 0.75
                            logger.info(f"已经预先按下移动物品了，10s内忽略拾取...{int(7 - (time.time() - collect_loot_pressed_time))}")
                        continue

                    # 掉落物在范围内,直接拾取
                    if loot_in_range:
                        # 不管了,全部释放掉
                        mover._release_all_keys()
                        time.sleep(0.1)
                        kbu.do_press("x")
                        logger.debug("捡东西按完x了")
                        continue

                    # # 如果被材料卡在当前房间了,忽略材料
                    # if stuck_room_idx:
                    #     logger.error("捡东西---》被材料卡在当前房间了,忽略材料")
                    #     continue

                    # 掉落物不在范围内,需要移动
                    byWalk = False
                    if material_min_distance < 150:
                        byWalk = True
                    # slow_pickup = not material_is_gold or material_min_distance < 100
                    slow_pickup = material_min_distance < 100

                    # todo 靠近门口的的,小碎步去捡
                    # door_is_near = exist_near(material_box, door_xywh_list, threshold=200)
                    door_is_near = False
                    near_door_list = get_objs_in_range(material_box, door_xywh_list + door_boss_xywh_list, threshold=200)

                    if near_door_list:
                        logger.warning("存在距离材料很近的门！")
                        for door in near_door_list:
                            # 如果材料位于门和角色之间
                            if (
                                    (door[0] <= material_box[0] <= hero_xywh[0] or door[0] >= material_box[0] >= hero_xywh[0])
                                    and (
                                    (door[1] <= material_box[1] <= hero_xywh[1] and abs(material_box[0] - hero_xywh[0]) < 170)
                                    or (door[1] >= material_box[1] >= hero_xywh[1] and abs(material_box[0] - hero_xywh[0]) < 170)
                                    or (abs(door[1] - material_box[1]) < 100 and abs(door[1] - hero_xywh[1]) < 100 and abs(material_box[1] - hero_xywh[1]) < 100)
                            )
                            ):
                                # logger.error(f"门:{door}, 材料：{material_box}， 角色：{hero_xywh}")
                                door_is_near = True
                            elif (
                                    (door[1] <= material_box[1] <= hero_xywh[1] or door[1] >= material_box[1] >= hero_xywh[1])
                                    and (
                                            (door[0] <= material_box[0] <= hero_xywh[0] and abs(material_box[0] - hero_xywh[0]) < 170)
                                            or (door[0] >= material_box[0] >= hero_xywh[0] and abs(material_box[0] - hero_xywh[0]) < 170)
                                            or (abs(door[0] - material_box[0]) < 100 and abs(door[0] - hero_xywh[0]) < 100 and abs(material_box[0] - hero_xywh[0]) < 100)
                                    )
                            ):
                                # logger.error(f"门:{door}, 材料：{material_box}， 角色：{hero_xywh}")
                                door_is_near = True

                        if door_is_near:
                            logger.info("材料离门口太近了!!")
                            if gold_xywh_list:
                                logger.info(f"是金币离门口太近了!!!  {gold_xywh_list}")
                            byWalk = True
                            if not slow_pickup:
                                slow_pickup = True
                        else:
                            logger.info("但是材料角色门口 不影响")

                    pause_event.wait()  # 暂停
                    move_mode = 'walking' if byWalk else 'running'
                    # todo 抽取方法, 根据距离判断做直线还是斜线, 根据距离判断走还是跑
                    # 目标在角色的上右方
                    if material_box[1] - hero_xywh[1] < 0 and material_box[0] - hero_xywh[0] > 0:
                        # y方向已经处于攻击范围, 只需要x方向移动
                        if abs(material_box[1] - hero_xywh[1]) < pick_up_y:
                            mover.move_stop_immediately(target_direction="RIGHT", move_mode=move_mode, stop=slow_pickup)
                        # x轴上的距离比较远,斜方向移动
                        elif hero_xywh[1] - material_box[1] < material_box[0] - hero_xywh[0]:
                            mover.move_stop_immediately(target_direction="RIGHT_UP", move_mode=move_mode, stop=slow_pickup)
                        # y轴上的距离也比较远,只进行y轴上的移动
                        elif hero_xywh[1] - material_box[1] >= material_box[0] - hero_xywh[0]:
                            mover.move_stop_immediately(target_direction="UP", move_mode=move_mode, stop=slow_pickup)
                            # break
                    # 目标在角色的左上方
                    elif material_box[1] - hero_xywh[1] < 0 and material_box[0] - hero_xywh[0] < 0:
                        # y方向已经处于攻击范围, 只需要x方向移动
                        if abs(material_box[1] - hero_xywh[1]) < pick_up_y:
                            mover.move_stop_immediately(target_direction="LEFT", move_mode=move_mode, stop=slow_pickup)
                        # x轴上的距离比较远,斜方向移动
                        elif hero_xywh[1] - material_box[1] < hero_xywh[0] - material_box[0]:
                            mover.move_stop_immediately(target_direction="LEFT_UP", move_mode=move_mode, stop=slow_pickup)
                        # y轴上的距离也比较远,只进行y轴上的移动
                        elif hero_xywh[1] - material_box[1] >= hero_xywh[0] - material_box[0]:
                            mover.move_stop_immediately(target_direction="UP", move_mode=move_mode, stop=slow_pickup)
                            # break
                    # 目标在角色的左下方
                    elif material_box[1] - hero_xywh[1] > 0 and material_box[0] - hero_xywh[0] < 0:
                        # y方向已经处于攻击范围, 只需要x方向移动
                        if abs(material_box[1] - hero_xywh[1]) < pick_up_y:
                            mover.move_stop_immediately(target_direction="LEFT", move_mode=move_mode, stop=slow_pickup)
                        # x轴上的距离比较远,斜方向移动
                        elif material_box[1] - hero_xywh[1] < hero_xywh[0] - material_box[0]:
                            mover.move_stop_immediately(target_direction="LEFT_DOWN", move_mode=move_mode, stop=slow_pickup)
                        # y轴上的距离也比较远,只进行y轴上的移动
                        elif material_box[1] - hero_xywh[1] >= hero_xywh[0] - material_box[0]:
                            mover.move_stop_immediately(target_direction="DOWN", move_mode=move_mode, stop=slow_pickup)
                    # 目标在角色的右下方
                    elif material_box[1] - hero_xywh[1] > 0 and material_box[0] - hero_xywh[0] > 0:
                        # y方向已经处于攻击范围, 只需要x方向移动
                        if abs(material_box[1] - hero_xywh[1]) < pick_up_y:
                            mover.move_stop_immediately(target_direction="RIGHT", move_mode=move_mode, stop=slow_pickup)
                        # x轴上的距离比较远,斜方向移动
                        elif material_box[1] - hero_xywh[1] < material_box[0] - hero_xywh[0]:
                            mover.move_stop_immediately(target_direction="RIGHT_DOWN", move_mode=move_mode, stop=slow_pickup)
                        # y轴上的距离也比较远,只进行y轴上的移动
                        elif material_box[1] - hero_xywh[1] >= material_box[0] - hero_xywh[0]:
                            mover.move_stop_immediately(target_direction="DOWN", move_mode=move_mode, stop=slow_pickup)

                    continue
                # 逻辑处理-捡材料<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                # 逻辑处理-出现再次挑战>>>>>>>>>>>>>>>>>>>>>>>>>>>
                if continue_exist:
                    # 不管了,全部释放掉
                    mover._release_all_keys()

                    aolakou = False
                    if game_mode == 2 or game_mode == 1:
                        aolakou = detect_aolakou(results[0].orig_img)
                    # todo 前多少角色买奥拉扣
                    if aolakou and role.no <= 0:
                        mu.do_move_to(x + 337, y + 209)
                        time.sleep(0.2)
                        mu.do_click(Button.left)
                        time.sleep(0.2)
                        mu.do_click(Button.left)
                        time.sleep(0.2)

                    # 如果商店开着,需要esc关闭
                    if shop_mystery_exist or shop_exist or aolakou:
                        if shop_mystery_exist:
                            # cv2.imwrite(f'./shop_imgs/mystery_Shop_{datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")}.jpg', img0)
                            time.sleep(0.5)
                            process_mystery_shop(capturer.capture(), x, y, buy_tank_type, buy_bell_ticket, buy_shanshanming)  # 重新截图，防止前面截的帧有干扰不清晰
                            logger.info("神秘商店处理完毕")
                        kbu.do_press(Key.esc)
                        logger.info("商店开着,需要esc关闭")
                        time.sleep(0.1)
                        continue

                    try_again_conflict = detect_try_again_conflict(capturer.capture())
                    if try_again_conflict:
                        logger.warning("再次挑战，有冲突，准备ESC！！！")
                        kbu.do_press(Key.esc)
                        time.sleep(0.3)
                        logger.warning("再次挑战，有冲突，已经ESC！！！")
                        continue

                    # 不存在掉落物了,就再次挑战
                    if not loot_xywh_list and not gold_xywh_list:
                        logger.warning("出现再次挑战,并且没有掉落物了,终止")
                        # time.sleep(3)  # 等待加载地图
                        if delay_break < 3:
                            # 延迟break，终止掉当前刷一次图的循环，多花0.3秒再次进行检测，处理商店和掉落物
                            delay_break = delay_break + 1
                            time.sleep(0.1)
                            continue

                        break  # 终止掉当前刷一次图的循环

                    # 聚集物品,按x
                    if (loot_xywh_list or gold_xywh_list) and not collect_loot_pressed:
                        if not collect_loot_pressed:
                            logger.info("中间移动物品到脚下")
                            kbu.do_press(dnf.Key_collect_loot)
                            collect_loot_pressed = True
                            collect_loot_pressed_time = time.time()
                            time.sleep(0.1)
                            kbu.do_press_with_time('x', 4000 if game_mode == 4 else 2000, 50)
                            logger.info("中间长按x 按完x了")
                        continue
                    continue
                # 逻辑处理-出现再次挑战<<<<<<<<<<<<<<<<<<<<<<<<<<<

                # 逻辑处理-出现翻牌>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                if card_num >= 3:
                    if not card_appear_time:
                        card_appear_time = time.time()

                    # 如果商店开着,需要esc关闭
                    if shop_mystery_exist:
                        # cv2.imwrite(f'./shop_imgs/mystery_Shop_{datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")}.jpg', img0)
                        time.sleep(0.5)
                        process_mystery_shop(capturer.capture(), x, y, buy_tank_type, buy_bell_ticket, buy_shanshanming)  # 重新截图，防止前面截的帧有干扰不清晰

                        logger.info("翻牌时有神秘商店，处理完毕")

                    if time.time() - card_appear_time > 0.5:
                        if not card_esc_time:
                            card_esc_time = time.time()
                            # 按下esc跳过翻牌
                            kbu.do_press(Key.esc)
                            logger.debug(f"关闭翻牌,shop_mystery_exist:{shop_mystery_exist},shop_exist:{shop_exist}")
                        elif time.time() - card_esc_time >= 1.5:
                            # 按下esc跳过翻牌
                            kbu.do_press(Key.esc)
                            logger.debug(f"再次关闭翻牌,shop_mystery_exist:{shop_mystery_exist},shop_exist:{shop_exist}")
                        else:
                            logger.debug("翻牌已经esc过，先等等1.5s再关闭")
                    else:
                        logger.debug("翻牌刚刚出现，先等等再关闭")

                    # 不管了,全部释放掉
                    mover._release_all_keys()
                    time.sleep(0.1)  # todo 翻牌睡两秒可行?

                    continue
                # 逻辑处理-出现翻牌<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                # 逻辑处理-什么都没有>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                if (not gold_xywh_list and not loot_xywh_list and not monster_xywh_list and not elite_monster_xywh_list and not boss_xywh_list
                    and not door_xywh_list and not door_boss_xywh_list and card_num < 3 and not continue_exist) and not sss_appeared:  # todo boss
                    pause_event.wait()  # 暂停
                    # 情况1:漏怪了,并且视野内看不到怪了,随机久了肯定能看到怪 todo 还是得做？匹配
                    # 情况2:翻牌附近
                    # 情况3:打完当前房间了,当前视野内没有门
                    if not door_absence_time:
                        door_absence_time = time.time()
                    if hero_xywh is not None:
                        logger.warning("除了角色什么也没识别到")
                        direct = random.choice(random.choice([kbu.single_direct, kbu.double_direct]))
                        try:
                            map_crop = map_util.get_small_map_region_img(img0, rows, cols)
                            cur_row, cur_col = map_util.current_room_index_cropped(map_crop, rows, cols)
                            allow_directions = map_util.get_allow_directions(map_crop, cur_row, cur_col)
                            logger.debug(f"未识别到尝试allow_directions:{allow_directions}")
                            if not allow_directions:
                                # cv2.imwrite("no_allow_directions_full1.jpg", img0)
                                # cv2.imwrite("no_allow_directions_crop1.jpg", map_crop)
                                logger.debug(f'小地图没找到对应的图{(rows, cols)},{(cur_row, cur_col)}！！！！')
                                time.sleep(1)
                                img00 = capturer.capture()
                                map_crop = map_util.get_small_map_region_img(img00, rows, cols)
                                cur_row, cur_col = map_util.current_room_index_cropped(map_crop, rows, cols)
                                allow_directions = map_util.get_allow_directions(map_crop, cur_row, cur_col)
                                current_room = (cur_row, cur_col)

                            next_room_direction = finder.get_next_direction((cur_row, cur_col), allow_directions)
                            logger.debug("计算方向2", next_room_direction)
                            logger.info(f"除了角色什么也没识别到,当前房间: {cur_row},{cur_col},允许方向: {allow_directions}, 下个方向: {next_room_direction}")

                            # previous = None
                            # if current_room in [item[0] for item in path_stack]:
                            #     for ii in range(len(path_stack) - 1, 0, -1):
                            #         if path_stack[ii][0] == current_room:
                            #             previous = path_stack[ii - 1][1]
                            #             break
                            #
                            # if hero_xywh[1] < 360 and mover.get_current_direction() == "UP":
                            #     if previous == "RIGHT" and hero_xywh[0] < img0.shape[1] * 3 // 5:
                            #         direct = "RIGHT"
                            #     elif previous == "LEFT" and hero_xywh[0] > img0.shape[1] * 2 // 5:
                            #         direct = "LEFT"

                            if next_room_direction:
                                direct = next_room_direction
                        except Exception as e:
                            logger.warning(f"捕获到异常: {e}")
                            traceback.print_exc()
                            logger.warning('小地图分析异常报错,跳过2')
                            direct = random.choice(random.choice([kbu.single_direct, kbu.double_direct]))

                        if door_absence_time and time.time() - door_absence_time > 180:
                            logger.warning('什么都没检测到(没有门)已经3分钟了,随机方向')
                            direct = random.choice(random.choice([kbu.single_direct, kbu.double_direct]))

                        logger.info(f"尝试方向--->{direct}")
                        # mover._release_all_keys()
                        mover.move(target_direction=direct)

                        pass
                    else:
                        random_direct = random.choice(random.choice([kbu.single_direct, kbu.double_direct]))
                        logger.warning('角色也没识别到,什么都没识别到,随机跑个方向看看-->{}', random_direct)
                        # mover._release_all_keys()
                        mover.move(target_direction=random_direct)
                    continue
                # 逻辑处理-什么都没有<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # todo 循环打怪过图 循环结束////////////////////////////////
            logger.warning("循环打怪过图 循环结束////////////////////////////////")

            pause_event.wait()  # 暂停
            need_wait_collect_finish = False
            if not collect_loot_pressed:
                executor.submit(lambda: (
                    logger.info("最后移动物品到脚下"),
                    mover._release_all_keys(),
                    time.sleep(0.1),
                    kbu.do_press(dnf.Key_collect_loot),
                    time.sleep(0.1),
                    kbu.do_press_with_time('x', 4000 if game_mode == 4 else 2000, 0),
                    logger.info("最后长按x 按完x了")
                ))
                need_wait_collect_finish = True

            pause_event.wait()  # 暂停
            # 疲劳值判断
            # current_fatigue = do_ocr_fatigue_retry(handle, x, y, reader, 5)
            current_fatigue = do_recognize_fatigue(capturer.capture())
            if role.fatigue_reserved > 0 and (current_fatigue - fatigue_cost) < role.fatigue_reserved:
                # 再打一把就疲劳值就不够预留的了
                logger.info(f'再打一把就疲劳值就不够预留的{role.fatigue_reserved}了')
                logger.info(f'刷完{fight_count}次了，结束...')
                if need_wait_collect_finish:
                    time.sleep(1.6)
                # 返回城镇
                kbu.do_press(dnf.key_return_to_town)
                time.sleep(5)
                finished = True
                # break

            if current_fatigue <= 0:
                # 再打一把就疲劳值就不够预留的了
                logger.info(f'没有疲劳值了')
                logger.info(f'刷完{fight_count}次了，结束...')
                if need_wait_collect_finish:
                    time.sleep(1.6)
                # 返回城镇
                kbu.do_press(dnf.key_return_to_town)
                time.sleep(5)
                finished = True
                # break

            pause_event.wait()  # 暂停
            # 识别"再次挑战"按钮是否存在,是否可以点击
            # btn_exist, text_exist, btn_clickable = detect_try_again_button(capturer.capture())
            btn_exist, text_exist, btn_clickable = detect_try_again_button(capturer.capture()) if game_mode != 2 else detect_1and1_next_map_button(capturer.capture())
            # 没的刷了,不能再次挑战了
            if (game_mode != 2 and text_exist and not btn_clickable) or (game_mode == 2 and not btn_exist):
                pause_event.wait()  # 暂停
                logger.info(f'刷了{fight_count}次了,再次挑战禁用状态,不能再次挑战了...')
                if need_wait_collect_finish:
                    time.sleep(1.6)
                # 返回城镇
                kbu.do_press(dnf.key_return_to_town)
                time.sleep(5)
                finished = True
            else:
                # 按下再次挑战
                if game_mode == 2:
                    kbu.do_press(Key.space)
                    time.sleep(2)
                    logger.warning('等两秒 再按空格继续下一个每日的图')
                    time.sleep(2)
                    kbu.do_press(Key.space)
                    time.sleep(2)
                    kbu.do_press(Key.space)
                    time.sleep(2)
                else:
                    kbu.do_press(dnf.key_try_again)
                    logger.warning("按下再次挑战了")

        # todo 循环进图结束<<<<<<<<<<<<<<<<<<<<<<<

        # # 瞬移到赛丽亚房间
        # teleport_to_sailiya()
        # time.sleep(0.5)

        time_diff = datetime.now() - oen_role_start_time
        logger.warning(f'第【{i + 1}】个角色【{role.name}】刷图打怪循环结束...总计耗时: {(time_diff.total_seconds() / 60):.1f} 分钟')
        if exception_mail_notify_timer:
            exception_mail_notify_timer.cancel()
        # 刷图流程结束<<<<<<<<<<
        # # 展示掉右下角的图标
        # show_right_bottom_icon(capturer.capture(), x, y)

        pause_event.wait()  # 暂停
        # 如果刷图了,则完成每日任务,整理背包
        if fight_count > 0:
            logger.info('刷了图之后,进行整理....')
            # 检查每日弹窗
            if datetime.now().hour == 0:
                close_new_day_dialog(handle, x, y)

            pause_event.wait()  # 暂停

            # 瞬移到赛丽亚房间
            teleport_to_sailiya(x, y)

            pause_event.wait()  # 暂停
            # 完成每日任务
            if game_mode == 2 or ((game_mode == 3 or game_mode == 1) and i < 20):
                finish_daily_challenge_by_all(x, y, game_mode == 2)

            # pause_event.wait()  # 暂停
            # # 一键出售装备,给赛丽亚
            # sale_equipment_to_sailiya()

            # 收邮件
            if datetime.now().weekday() in dnf.receive_mail_days:
                logger.info('日期匹配，今日触发收邮件')
                receive_mail(capturer.capture(), x, y)

            pause_event.wait()  # 暂停
            # 转移材料到账号金库
            transfer_materials_to_account_vault(x, y)

        pause_event.wait()  # 暂停
        # 准备重新选择角色
        # if i < len(role_list) - 1:
        if i < last_role_no - 1:
            logger.warning("准备重新选择角色")
            # esc打开菜单
            time.sleep(0.5)
            # kbu.do_press(Key.esc)
            mu.do_smooth_move_to(x + 832, y + 576)  # 通过点击菜单按钮打开菜单
            time.sleep(0.2)
            mu.do_click(Button.left)
            time.sleep(0.5)

            pause_event.wait()  # 暂停
            # 鼠标移动到选择角色，点击 偏移量（1038,914）
            img_menu = capturer.capture()
            template_choose_role = cv2.imread(os.path.normpath(f'{config_.project_base_path}/assets/img/choose_role.png'), cv2.IMREAD_GRAYSCALE)
            match_and_click(img_menu, x, y, template_choose_role, (506, 504))
            # 等待加载角色选择页面
            time.sleep(5)

            # 默认停留在刚才的角色上，直接按一次右键，空格
            kbu.do_press(Key.right)
            time.sleep(0.2)
            kbu.do_press(Key.space)
            time.sleep(0.2)
        else:
            logger.warning("已经刷完最后一个角色了，结束脚本")
            mode_name = (
                "白图" if game_mode == 1 else
                "每日1+1" if game_mode == 2 else
                "妖气追踪" if game_mode == 3 else
                "妖怪歼灭" if game_mode == 4 else
                "未知模式"
            )
            email_subject = f"{mode_name} 任务执行结束 {pathlib.Path(__file__).stem.replace('main', '').strip() if 'main' in pathlib.Path(__file__).stem else ''}"
            email_content = email_subject
            mail_receiver = mail_config.get("receiver")
            if mail_receiver:
                tool_executor.submit(lambda: (
                    mail_sender.send_email(email_subject, email_content, mail_receiver),
                    logger.info("任务执行结束")
                ))
            break


# 等待按键,启动
logger.info(".....python主线程 启动..........")
logger.warning(f".....请按下 {dnf.key_start_script} 组合键开始脚本...")
kboard.wait(dnf.key_start_script)  # 等待按下组合键
winsound.PlaySound(config_.sound1, winsound.SND_FILENAME)
# winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
logger.warning(f".....{dnf.key_start_script} ok....触发开始了........")

# 创建并启动脚本线程
script_task_thread = threading.Thread(target=main_script)
script_task_thread.daemon = True
script_task_thread.start()
start_time = datetime.now()
logger.info('')
logger.info(f'脚本开始: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')

# 创建并启动监听中断按键线程
listener_thread = threading.Thread(target=start_keyboard_listener)
listener_thread.daemon = True
listener_thread.start()

# 等待脚本线程结束或检测到中断信号
while script_task_thread.is_alive():
    if stop_be_pressed:
        mover._release_all_keys()
        logger.warning(f"监听到组合键被按下,[stop_be_pressed=={stop_be_pressed}],不再阻塞,继续执行主线程代码直至退出")
        break
    time.sleep(1)

end_time = datetime.now()
logger.info(f'脚本开始: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
logger.info(f'脚本结束: {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
time_delta = end_time - start_time
logger.info(f'总计耗时: {(time_delta.total_seconds() / 60):.1f} 分钟')

# 脚本正常执行完,不是被组合键中断的,并且配置了退出游戏
if not stop_be_pressed and quit_game_after_finish:
    logger.info("正在退出游戏...")
    clik_to_quit_game(handle, x, y)
    time.sleep(5)
    window_utils.kill_process_by_hwnd(handle)  # 如果没退出，就强杀掉进程
    time.sleep(5)

logger.info("python主线程已停止.....")

if not stop_be_pressed and quit_game_after_finish and shutdown_pc_after_finish:
    logger.info("一分钟之后关机...")
    # os.system("shutdown /r /t 60")  # 60后秒重启
    os.system("shutdown /s /t 60")  # 60后秒关机
