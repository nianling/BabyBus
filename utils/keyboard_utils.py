# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'


import random
import time

from pynput.keyboard import Controller, Key, KeyCode
from dnf.dnf_config import direct_dic

keyboard = Controller()


direct_set = {"UP", "DOWN", "LEFT", "RIGHT", "RIGHT_UP", "RIGHT_DOWN", "LEFT_UP", "LEFT_DOWN"}
single_direct = ["LEFT", "RIGHT", "UP", "DOWN"]
double_direct = ["RIGHT_UP", "RIGHT_DOWN", "LEFT_UP", "LEFT_DOWN"]

def do_release(key):
    keyboard.release(key)


def do_press(key):
    """
    按键，按下和抬起之间的间隔，按完之后也等待一下
    :param key:
    :return:
    """
    # 生成一个随机数
    random_float = random.uniform(40, 60)
    do_press_with_time(key, random_float, random_float)


def do_press_with_time(key, duration: float, after_release: float):
    """
    按键，指定按下和抬起之间的间隔（毫秒）,按完之后等待指定时间
    :param after_release: 抬起之后等待多久（毫秒）
    :param duration: 长按时间，按下之后多久抬起（毫秒）
    :param key: 按键
    :return:
    """
    keyboard.press(key)
    time.sleep(duration / 1000)
    keyboard.release(key)
    if after_release:
        time.sleep(after_release / 1000)


def do_skill(key):
    """
    按技能键，按完之后等一秒左右让技能动作结束
    :param key:
    :return:
    """
    if key == '' or key is None:
        return
    random_float = random.uniform(980, 1100)
    do_skill_with_time(key, random_float)


def do_skill_with_time(key, wait_time: float):
    """
    按技能键，按完之后等待一定时间
    :param key: 按键
    :param wait_time: 按完之后等多久（毫秒）
    :return:
    """
    if key == '' or key is None:
        return
    do_press(key)
    time.sleep(wait_time / 1000)


def do_command_wait_time(key_arr, wait_time: float):
    """
    指令组合键，之后等待指定时间
    :param key_arr: 指令组合键，数组,如[前,前,空格]
    :param wait_time: 等待时间
    :return:
    """
    for key in key_arr:
        if key == ' ' or key == '':
            time.sleep(0.1)
            continue
        do_press(key)
    time.sleep(wait_time)


def do_concurrent_command_wait_time(key_arr, wait_time: float):
    """
    指令组合键，之后等待指定时间
    :param key_arr: 指令组合键，数组,如[前,前,空格]
    :param wait_time: 等待时间
    :return:
    """
    for key in key_arr:
        keyboard.press(key)
        time.sleep(0.05)

    for key in key_arr:
        keyboard.release(key)
    time.sleep(wait_time)


def do_buff(key_arr):
    """
    上Buff，之后等待随机时间
    :param key_arr: Buff技能组合，数组,如[前,前,空格]
    :return:
    """
    wait_time = random.uniform(1, 1.5)
    do_command_wait_time(key_arr, wait_time)


def do_run(key, span):
    """
    跑起来，跑多久（秒）
    :param key: 按方向键
    :param span: 持续时间,跑多久(秒)
    """
    keyboard.press(key)
    time.sleep(random.uniform(50, 100) / 1000)
    keyboard.release(key)
    time.sleep(random.uniform(50, 100) / 1000)

    keyboard.press(key)
    time.sleep(span)  # 控制跑多久
    keyboard.release(key)
    time.sleep(random.uniform(50, 100) / 1000)

def release_all_direct():
    for d in single_direct:
        keyboard.release(direct_dic[d])
        time.sleep(0.02)


def move(direct, walk=False, pressed_direct_cache=None, press_delay=0.1, release_delay=0.1, pickup=False):
    # todo 跑/走状态的切换
    result = direct
    release_delay = 0.05
    press_delay = 0.05
    # 本次按下的是单方向
    if direct in single_direct:
        # 之前有按下的方向键还未松开
        if pressed_direct_cache is not None:
            # 当前要去的方向,与之前按下的方向不一致,先松开
            if pressed_direct_cache != direct:
                # # 之前按下的是双方向,先松开
                # if pressed_direct_cache not in single_direct:
                #     keyboard.release(direct_dic[pressed_direct_cache.strip().split("_")[0]])
                #     keyboard.release(direct_dic[pressed_direct_cache.strip().split("_")[1]])
                # # 之前按的是单方向,先松开
                # else:
                #     keyboard.release(direct_dic[pressed_direct_cache])

                # 不管了,上下左右全都松开一遍
                release_all_direct()

                # 按下方向键
                keyboard.press(direct_dic[direct])

                # 如果是跑,则需要再按一次方向键
                if not walk:
                    time.sleep(press_delay)
                    keyboard.release(direct_dic[direct])
                    time.sleep(release_delay)
                    keyboard.press(direct_dic[direct])
                else:
                    # 是走,不需要多按一次
                    if pickup:
                        time.sleep(0.05)
                        keyboard.release(direct_dic[direct])
                        result = None
                    pass

            # 之前就是这个方向,不需要操作
            else:
                pass

        # 之前没有按下的键
        else:
            # 按下相应的方向键
            keyboard.press(direct_dic[direct])
            # 如果是跑,则需要再按一次方向键
            if not walk:
                time.sleep(press_delay)
                keyboard.release(direct_dic[direct])
                time.sleep(release_delay)
                keyboard.press(direct_dic[direct])
            else:
                # 是走,不需要多按一次
                if pickup:
                    time.sleep(0.05)
                    keyboard.release(direct_dic[direct])
                    result = None
                pass
    else:
        left_or_right = direct.strip().split("_")[0]
        up_or_down = direct.strip().split("_")[1]
        if pressed_direct_cache is not None:

            if pressed_direct_cache != direct:  # 当前要去的方向,与之前按下的方向不一致,先松开

                # if pressed_direct_cache not in single_direct:  # 不在四个方向之内,之前按的是斜方向,都松开
                #     keyboard.release(direct_dic[pressed_direct_cache.strip().split("_")[0]])
                #     keyboard.release(direct_dic[pressed_direct_cache.strip().split("_")[1]])
                # else:  # 之前按的是单方向,先松开
                #     keyboard.release(direct_dic[pressed_direct_cache])

                # 不管了,上下左右全都松开一遍
                release_all_direct()

                # 先按两次左或者右,让角色跑起来
                if not walk:
                    keyboard.press(direct_dic[left_or_right])
                    time.sleep(press_delay)
                    keyboard.release(direct_dic[left_or_right])
                    time.sleep(release_delay)
                    keyboard.press(direct_dic[left_or_right])
                    time.sleep(press_delay)
                else:
                    keyboard.press(direct_dic[left_or_right])

                keyboard.press(direct_dic[up_or_down])

                if walk and pickup:
                    time.sleep(0.05)
                    keyboard.release(direct_dic[left_or_right])
                    keyboard.release(direct_dic[up_or_down])
                    result = None
                pass

            else:
                # 本来就是这个方向
                pass
        else:
            # 跑,需要先跑起来
            if not walk:
                keyboard.press(direct_dic[left_or_right])
                time.sleep(press_delay)

                keyboard.release(direct_dic[left_or_right])
                time.sleep(release_delay)

                keyboard.press(direct_dic[left_or_right])
                time.sleep(press_delay)
            else:
                keyboard.press(direct_dic[left_or_right])

            keyboard.press(direct_dic[up_or_down])

            if walk and pickup:
                time.sleep(0.05)
                keyboard.release(direct_dic[left_or_right])
                keyboard.release(direct_dic[up_or_down])
                result = None
            pass

    return result
