# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'


import math
import random
import time

from pynput.mouse import Controller
from pynput.mouse import Button

mouse = Controller()


def do_smooth_move_to(x, y):
    """
    尽量平滑的移动，仿真的去移动鼠标
    :param x:
    :param y:
    :return:
    """
    current_x, current_y = mouse.position
    steps = 10  # 将移动分解为多个步骤

    # 根据移动距离决定拆分步骤
    distance = math.sqrt((current_x - x) ** 2 + (current_y - y) ** 2)
    steps = min(math.ceil(distance / 100), steps)
    steps = max(steps, 1)

    step_x = (x - current_x) / steps
    step_y = (y - current_y) / steps

    for step in range(steps):
        next_x = current_x + step_x + random.uniform(-1, 1)  # 加入一些随机，不要保持直线
        next_y = current_y + step_y + random.uniform(-1, 1)
        if step == (steps - 1):
            next_x = x
            next_y = y
        mouse.position = (int(next_x), int(next_y))
        time.sleep(random.uniform(0.01, 0.02))  # 控制移动速度，随机调整每次移动的时间间隔
    time.sleep(0.1)


def do_move_to(x, y):
    mouse.position = (x, y)
    time.sleep(0.1)


def do_move_and_click(x, y):
    """
    移过去并鼠标左键点击
    :param x:
    :param y:
    :return:
    """
    do_move_to(x, y)
    do_click(Button.left)
    time.sleep(0.1)


def do_click(key):
    """
    按键，按下和抬起之间的间隔，按完之后也等待一下
    :param key:
    :return:
    """
    # 生成一个随机数
    random_float = random.uniform(40, 60)
    do_click_with_time(key, random_float, random_float)


def do_click_with_time(key, duration: float, after_release: float):
    """
    按键，指定按下和抬起之间的间隔（毫秒）,按完之后等待指定时间
    :param after_release: 抬起之后等待多久（毫秒）
    :param duration: 长按时间，按下之后多久抬起（毫秒）
    :param key: 按键
    :return:
    """
    mouse.press(key)
    time.sleep(duration / 1000)
    mouse.release(key)
    time.sleep(after_release / 1000)


def get_current_position():
    """
    获取当前鼠标坐标(x,y)
    """
    return mouse.position
