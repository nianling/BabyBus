# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'

from pynput import keyboard
from pynput.keyboard import Key, KeyCode
import weights

window_title = "地下城与勇士：创新世纪"

# 定义小键盘数字1的KeyCode
numpad_1 = KeyCode.from_vk(97)
# 定义小键盘数字2的KeyCode
numpad_2 = KeyCode.from_vk(98)
# 定义上下左右方向键
direct_dic = {"UP": Key.up, "DOWN": Key.down, "LEFT": Key.left, "RIGHT": Key.right}

# 游戏按键定义 再次挑战
# key_try_again = numpad_1
key_try_again = Key.f10
# 游戏按键定义 返回城镇
key_return_to_town = Key.f12
# 游戏按键定义 移动物品
# Key_collect_loot = numpad_2
Key_collect_loot = '0'

# 定义脚本暂停组合键
key_pause_script = {keyboard.Key.delete}
# 定义终止脚本组合键 {keyboard.Key.ctrl_l, keyboard.Key.alt_l, keyboard.Key.f10}
key_stop_script = {keyboard.Key.end}
# 定义任务开始组合键 'ctrl+alt+f11'
key_start_script = '='

# 截图日志
enable_picture_log = True

# 技能栏快捷键配置，第一排第二排依次填写
skill_hotkeys = [
    # 第一排
    "q",
    "w",
    "e",
    "r",
    "t",
    "v",
    Key.ctrl_l,

    # 第二排
    "a",
    "s",
    "d",
    "f",
    "g",
    Key.tab,
    "h",
]

# 配置每周几领取邮件，0周一，1周二，2周三，3周四，4周五，5周六，6周日
receive_mail_days = [3, 6]
