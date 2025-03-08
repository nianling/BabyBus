# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'


from pynput.keyboard import Key, Controller
import time
from enum import Enum


class Direction(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    RIGHT_UP = "RIGHT_UP"
    RIGHT_DOWN = "RIGHT_DOWN"
    LEFT_UP = "LEFT_UP"
    LEFT_DOWN = "LEFT_DOWN"


class MoveMode(Enum):
    WALKING = "walking"
    RUNNING = "running"


class MovementController:
    def __init__(self):
        self.keyboard = Controller()
        self.current_direction = None
        self.current_mode = None
        self.pressed_keys = set()

    def _press_key(self, key):
        # todo 不需要判断
        self.keyboard.press(key)
        if key not in self.pressed_keys:
            self.pressed_keys.add(key)
        time.sleep(0.02)

    def _release_key(self, key):
        # todo 不需要判断
        self.keyboard.release(key)
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
        time.sleep(0.02)

    def _release_all_keys(self):
        # todo all keys
        for key in list(self.pressed_keys):
            self._release_key(key)
        for key in ({Key.up, Key.down, Key.left, Key.right} - self.pressed_keys):
            self._release_key(key)
        self.pressed_keys.clear()
        self.current_direction = None
        self.current_mode = None


    def _get_direction_keys(self, direction):
        direction_map = {
            Direction.UP: [Key.up],
            Direction.DOWN: [Key.down],
            Direction.LEFT: [Key.left],
            Direction.RIGHT: [Key.right],
            Direction.RIGHT_UP: [Key.right, Key.up],
            Direction.RIGHT_DOWN: [Key.right, Key.down],
            Direction.LEFT_UP: [Key.left, Key.up],
            Direction.LEFT_DOWN: [Key.left, Key.down]
        }
        return direction_map[Direction(direction)]

    def _get_main_direction(self, direction):
        """获取方向的主方向（用于跑步）"""
        if "RIGHT" in direction:
            return "RIGHT"
        elif "LEFT" in direction:
            return "LEFT"
        return direction

    def _handle_walking_direction_change(self, target_direction):
        """处理走路状态下的方向改变"""
        if not self.current_direction:
            # 如果当前没有方向，直接按下所需的所有键
            for key in self._get_direction_keys(target_direction):
                self._press_key(key)
            return

        # 获取目标方向和当前方向的按键集合
        target_keys = set(self._get_direction_keys(target_direction))
        current_keys = set(self._get_direction_keys(self.current_direction))

        # 需要新按下的键和需要释放的键
        keys_to_press = target_keys - current_keys
        keys_to_release = current_keys - target_keys

        # 释放不需要的键
        for key in keys_to_release:
            self._release_key(key)
        # 按下新增的键
        for key in keys_to_press:
            self._press_key(key)
            
    def _setup_running(self, direction):
        """设置跑步状态"""
        # 确保所有键都释放
        self._release_all_keys()

        # 获取主方向键（左或右）
        main_key = Key.right if "RIGHT" in direction else Key.left if "LEFT" in direction else None

        if main_key:
            # 双击实现跑步
            self._press_key(main_key)
            # time.sleep(0.05)
            self._release_key(main_key)
            # time.sleep(0.05)
            self._press_key(main_key)

        # 添加垂直方向
        if "UP" in direction:
            self._press_key(Key.up)
        elif "DOWN" in direction:
            self._press_key(Key.down)


    def move_stop_immediately(self, target_direction, move_mode='running', stop=False):
        self.move(target_direction, move_mode)
        if stop:
            time.sleep(0.04)
            self._release_all_keys()

    def move(self, target_direction, move_mode='running'):
        """
        移动角色到指定方向和移动模式

        Args:
            target_direction (str): 目标方向
            move_mode (str): 移动模式 "walking" 或 "running"
        """
        # 如果方向和模式都没变，无需操作
        if target_direction == self.current_direction and move_mode == self.current_mode:
            return

        target_direction = Direction(target_direction)
        move_mode = MoveMode(move_mode)

        # 如果是要进行跑步状态
        if move_mode == MoveMode.RUNNING:
            current_main = self._get_main_direction(self.current_direction) if self.current_direction else None
            target_main = self._get_main_direction(target_direction.value)

            # 如果主方向发生改变或者之前不是跑步状态，需要重新设置跑步
            if (current_main != target_main or self.current_mode != MoveMode.RUNNING.value):
                self._setup_running(target_direction.value)
            else: # 主方向没有变化，并且之前就是跑步状态
                # 主方向相同，只需要处理垂直方向的变化
                if "UP" in (self.current_direction or "") and "UP" not in target_direction.value:
                    self._release_key(Key.up)
                elif "DOWN" in (self.current_direction or "") and "DOWN" not in target_direction.value:
                    self._release_key(Key.down)

                if "UP" in target_direction.value and ("UP" not in (self.current_direction or "")):
                    self._press_key(Key.up)
                elif "DOWN" in target_direction.value and ("DOWN" not in (self.current_direction or "")):
                    self._press_key(Key.down)
        else: # 要进行走路状态
            # 从跑步切换到走路状态
            if self.current_mode == MoveMode.RUNNING.value:
                # 从跑步切换到走路，需要重置所有按键
                self._release_all_keys()
            self._handle_walking_direction_change(target_direction.value)

        # 更新当前状态
        self.current_direction = target_direction.value
        self.current_mode = move_mode.value

    def get_current_direction(self):
        return self.current_direction
