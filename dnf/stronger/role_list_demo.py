# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'

from pynput.keyboard import Key

from dnf.stronger.role_config import RoleConfig as R, SubClass, BaseClass
from dnf.stronger.role_config import Skill as S


def get_role_config_list() -> list[R]:
    # 总疲劳值
    default_fatigue_all = 188
    # 保留的疲劳值
    # default_fatigue_reserved = 30
    default_fatigue_reserved = 0

    role_configs = []

    role_configs.append(R(name='刺客', no=1,
                          buffs=[
                              [Key.right, Key.right, Key.space]
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='q', animation_time=0.9),
                              S(hot_key='s', animation_time=0.5),
                              S(hot_key='e', animation_time=0.5),
                              S(hot_key='v', animation_time=0.6),
                          ],
                          height=155,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          ))

    role_configs.append(R(name='剑魔', no=2,
                          buffs=[
                              [Key.right, Key.right, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='f', animation_time=0.3),
                              S(hot_key='q', animation_time=0.5),
                              S(hot_key=Key.tab, animation_time=0.5),
                              S(hot_key='w', animation_time=0.4),
                              S(hot_key='e', animation_time=0.4),
                              S(hot_key='v', animation_time=0.4),
                              S(hot_key='a', animation_time=0.5),
                          ],
                          height=149,
                          # attack_center_x=250,
                          attack_center_x=100,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          ))

    role_configs.append(R(name='剑魂', no=3,
                          buffs=[
                              [Key.right, Key.right, Key.space]
                          ],
                          candidate_hotkeys=['e'],
                          custom_priority_skills=[
                              S(hot_key='q', animation_time=0.4),
                              S(hot_key='r', animation_time=0.3),
                              S(hot_key='d', animation_time=0.3),
                              S(hot_key='a', animation_time=0.4),
                              S(hot_key='g', animation_time=0.5),
                              S(hot_key=Key.tab, animation_time=0.7),
                          ],
                          height=160,
                          attack_center_x=40,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          sub_class=BaseClass.男鬼剑.剑魂,
                          ))

    role_configs.append(R(name='奶爸', no=4,
                          buffs=[
                              # [Key.left, Key.right, Key.space],
                              ['s']
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='a', animation_time=0.3),
                              S(hot_key='w', animation_time=0.3),
                              S(name='神圣之光', hot_key='v', animation_time=0.3),
                              S(hot_key='g', animation_time=0.4),
                              S(hot_key='r', animation_time=0.4),
                              S(hot_key='f', animation_time=0.4),
                              S(hot_key='t', animation_time=0.4),
                          ],
                          height=164,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          sub_class=BaseClass.男圣职.奶爸,
                          ))

    role_configs.append(R(name='瞎子', no=5,
                          buffs=[
                              [Key.down, Key.down, Key.space]
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='q', animation_time=0.3),
                              S(hot_key='t', animation_time=0.6),
                              S(hot_key=Key.tab, animation_time=0.5),
                              S(hot_key='h', animation_time=0.3),
                              S(hot_key='f', animation_time=0.4),
                              S(hot_key='e', animation_time=0.4),
                              S(hot_key='r', animation_time=0.4),
                          ],
                          height=160,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          sub_class=BaseClass.男鬼剑.阿修罗,
                          ))

    role_configs.append(R(name='旅人', no=6,
                          buffs=[
                              [Key.right, Key.right, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(name='高原雾花', hot_key='d', animation_time=0.8),
                              S(name='迷雾箭雨', hot_key='w', animation_time=0.3),
                              S(name='浓雾暴雨', hot_key=Key.tab, animation_time=0.6),
                              S(name='细雨', hot_key='a', animation_time=0.4),
                              S(name='旋风', hot_key='f', animation_time=0.4),
                              S(hot_key='r', animation_time=0.5),
                              S(hot_key='t', animation_time=0.4),
                          ],
                          height=138,
                          attack_center_x=100,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          sub_class=BaseClass.弓箭手.旅人,
                          ))

    role_configs.append(R(name='光兵', no=7,
                          buffs=[
                              [Key.up, Key.up, Key.space],
                              [Key.right, Key.right, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='d', animation_time=0.4),
                              S(hot_key='g', animation_time=0.6),
                              S(hot_key='e', animation_time=0.4),
                              S(hot_key='v', animation_time=0.4),
                              S(hot_key='r', animation_time=0.9),
                              S(hot_key='a', animation_time=0.5),
                              S(hot_key='t', animation_time=0.9),
                          ],
                          height=141,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          sub_class=BaseClass.女格斗.女气功,
                          ))

    role_configs.append(R(name='红眼', no=8,
                          buffs=[
                              [Key.right, Key.right, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='e', animation_time=0.4),
                              S(hot_key='w', animation_time=0.6),
                              S(hot_key='d', animation_time=0.3),
                              S(hot_key='q', animation_time=0.5),
                              S(hot_key='r', animation_time=0.4),
                              S(hot_key='t', animation_time=0.5),
                              S(hot_key=Key.tab, animation_time=0.5),
                          ],
                          height=160,
                          attack_center_x=100,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          sub_class=BaseClass.男鬼剑.狂战士,
                          ))

    role_configs.append(R(name='魔道', no=9,
                          buffs=[
                              [Key.up, Key.up, Key.space],
                              [Key.down, Key.up, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='d', animation_time=0.3),
                              S(hot_key='q', animation_time=0.3),
                              S(hot_key='f', animation_time=0.4),
                              S(hot_key='h', animation_time=0.4),
                              S(hot_key='t', animation_time=0.9),
                              S(hot_key='e', animation_time=1.2),
                              S(hot_key='a', animation_time=0.4),
                              S(hot_key='s', animation_time=0.4),
                          ],
                          height=138,
                          attack_center_x=80,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='s', animation_time=0.4),
                              S(hot_key='t', animation_time=0.4),
                              S(hot_key='e', animation_time=0.4),
                          ],
                          sub_class=BaseClass.女法师.魔道,
                          ))

    role_configs.append(R(name='奶弓', no=10,
                          buffs=[
                              [Key.right, Key.right, Key.space]
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(name='加速', command=['w', 'a'], cd=2, animation_time=0.6),
                              S(name='净化', command=['f', 's'], cd=2, animation_time=0.6),
                              S(hot_key='s', animation_time=0.6),
                              S(hot_key='r', animation_time=0.4),
                              S(hot_key='d', animation_time=0.4),
                              S(hot_key='t', animation_time=0.4),
                              S(hot_key='a', animation_time=0.4),
                          ],
                          height=138,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='r', animation_time=0.4),
                              S(hot_key='s', animation_time=0.6),
                              S(hot_key='h', animation_time=0.6),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='二觉', hot_key='v', animation_time=2),
                          ],
                          sub_class=BaseClass.弓箭手.奶弓,
                          ))

    role_configs.append(R(name='奶妈', no=11,
                          buffs=[
                              [Key.right, Key.right, Key.space],
                              [Key.up, Key.right, Key.space]
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='q', animation_time=0.8),
                              S(name='勇气颂歌', command=['r', 'x', 'x', 'x', 'c'], hot_key='r', hotkey_cd_command_cast=True),
                              S(name='圣洁之翼', command=['e', 'x', 'x', 'x', 'x', 'x', 'x', 'e'], hot_key='e', hotkey_cd_command_cast=True),
                              S(hot_key='v', animation_time=0.3),
                              S(hot_key='d', animation_time=0.5),
                              S(hot_key='f', animation_time=0.5),
                          ],
                          height=151,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          sub_class=BaseClass.女圣职.奶妈,
                          ))

    role_configs.append(R(name='剑影', no=12,
                          buffs=[
                              [Key.up, Key.up, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='r', animation_time=0.3),
                              S(hot_key='a', animation_time=0.5),
                              S(hot_key='v', animation_time=0.3),
                              S(hot_key='h', animation_time=0.4),
                              S(hot_key='q', animation_time=0.3),
                              S(hot_key='s', animation_time=0.3),
                              S(hot_key='t', animation_time=0.6),
                          ],
                          height=160,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          sub_class=BaseClass.男鬼剑.剑影,
                          ))

    role_configs.append(R(name='奶萝', no=13,
                          buffs=[
                              [Key.right, Key.right, Key.space],
                              [Key.up, Key.up, Key.space]
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='q', animation_time=0.4),
                              S(hot_key='d', animation_time=0.3),
                              S(hot_key='f', animation_time=0.3),
                              S(hot_key='w', animation_time=0.5),
                              S(hot_key='e', animation_time=1),
                              S(hot_key='a', animation_time=0.4),
                              S(hot_key='r', animation_time=0.3),
                          ],
                          height=127,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='g', animation_time=1),
                              S(hot_key='e', animation_time=1),
                              S(hot_key='a', animation_time=0.4),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='二觉', hot_key='t', animation_time=2),
                          ],
                          sub_class=BaseClass.女法师.小魔女,
                          ))

    role_configs.append(R(name='刃影', no=14,
                          buffs=[
                              [Key.right, Key.right, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          height=149,
                          custom_priority_skills=[
                              S(hot_key='v', animation_time=0.3),
                              S(hot_key='q', animation_time=0.3),
                              S(hot_key='e', animation_time=0.3),
                              S(hot_key='w', animation_time=0.3),
                              S(hot_key='a', animation_time=0.3),
                              S(hot_key=Key.tab, animation_time=0.9),
                              S(hot_key='r', animation_time=0.6),
                              S(hot_key='s', animation_time=0.3),
                          ],
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          sub_class=BaseClass.女鬼剑.刃影,
                          ))

    role_configs.append(R(name='奶枪', no=15,
                          buffs=[
                              [Key.up, Key.up, Key.space],
                              [Key.right, Key.right, Key.space]
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='d', animation_time=0.9),
                              S(hot_key='s', animation_time=0.8),
                              S(hot_key='t', animation_time=0.3),
                              S(hot_key='r', animation_time=0.4),
                              S(hot_key='a', animation_time=0.4),
                              S(hot_key='f', animation_time=0.4),
                              S(hot_key='v', animation_time=0.4),
                              S(hot_key='e', animation_time=0.5),
                          ],
                          height=158,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='t', animation_time=0.3),
                              S(hot_key='r', animation_time=0.4),
                              S(hot_key='e', animation_time=0.4),
                              S(hot_key='v', animation_time=0.4),
                              S(hot_key='d', animation_time=0.4),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='二觉', hot_key='g', animation_time=2),
                          ],
                          sub_class=BaseClass.女枪手.奶枪,
                          ))

    role_configs.append(R(name='奇美拉', no=16,
                          buffs=[
                              [Key.down, Key.down, Key.space],
                              [Key.up, Key.up, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='e', animation_time=0.4),
                              S(hot_key='v', animation_time=0.5),
                              S(hot_key='a', animation_time=0.5),
                              S(hot_key='f', animation_time=0.6),
                              S(hot_key=Key.tab, animation_time=0.4),
                              S(hot_key='w', animation_time=0.4),
                              S(hot_key='g', animation_time=0.4),
                              S(hot_key='d', animation_time=0.5),
                              S(hot_key='q', animation_time=0.9),
                          ],
                          height=138,
                          attack_center_x=50,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='g', animation_time=0.4),
                              S(hot_key='e', animation_time=0.4),
                              S(hot_key='w', animation_time=0.4),
                              S(hot_key='q', animation_time=0.9),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='二觉', hot_key='t', animation_time=2),
                          ],
                          sub_class=BaseClass.弓箭手.奇美拉,
                          ))

    role_configs.append(R(name='剑帝', no=17,
                          buffs=[
                              [Key.right, Key.right, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='q', animation_time=0.4),
                              S(hot_key='s', animation_time=0.7),
                              S(hot_key='d', animation_time=0.5),
                              S(hot_key='f', animation_time=0.5),
                              S(hot_key='w', animation_time=0.3),
                              S(hot_key='e', animation_time=0.8),
                              S(hot_key='t', animation_time=0.6),
                              S(hot_key='r', animation_time=0.8),
                              S(hot_key=Key.tab, animation_time=0.8),
                          ],
                          height=149,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='r', animation_time=0.8),
                              S(hot_key='t', animation_time=0.6),
                              S(hot_key='g', animation_time=0.7),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                          ],
                          sub_class=BaseClass.女鬼剑.剑帝,
                          ))

    role_configs.append(R(name='蓝拳', no=18,
                          buffs=[
                              [Key.up, Key.down, Key.space],
                              [Key.left, Key.right, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          height=164,
                          custom_priority_skills=[
                              S(hot_key=Key.tab, animation_time=0.9),
                              S(hot_key='h', animation_time=0.5),
                              S(hot_key='e', animation_time=0.4),
                              S(hot_key='w', animation_time=0.8),
                              S(hot_key='r', animation_time=0.5),
                              S(hot_key='v', animation_time=0.8),
                              S(hot_key='t', animation_time=0.7),
                              S(name='破碎', command=['z', 'x', 'x', 'x', 'x', 'x'], cd=6, animation_time=0.2),
                          ],
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='w', animation_time=0.8),
                              S(hot_key='v', animation_time=0.8),
                              S(hot_key='e', animation_time=0.4),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='二觉', hot_key='g', animation_time=2),
                          ],
                          sub_class=BaseClass.男圣职.蓝拳,
                          ))

    role_configs.append(R(name='弹药', no=19,
                          buffs=[
                              [Key.right, Key.right, Key.space, Key.right],
                              [Key.left, Key.right, Key.space]
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='f', animation_time=0.4),
                              S(hot_key='s', animation_time=0.4),
                              S(hot_key='e', animation_time=0.3),
                              S(hot_key='r', animation_time=0.4),
                              S(hot_key='w', animation_time=0.5),
                              S(hot_key='t', animation_time=0.4),
                              S(hot_key='g', animation_time=0.9),
                              S(hot_key='a', animation_time=0.4),
                          ],
                          height=158,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='w', animation_time=0.5),
                              S(hot_key='g', animation_time=0.9),
                              S(hot_key='r', animation_time=0.4),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='二觉', hot_key='v', animation_time=2),
                          ],
                          sub_class=BaseClass.女枪手.女弹药,
                          ))

    role_configs.append(R(name='漫游', no=20,
                          buffs=[
                              [Key.left, Key.left, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='g', animation_time=1.2),
                              S(hot_key='e', animation_time=0.6),
                              S(hot_key='r', animation_time=0.3),
                              S(hot_key='d', animation_time=1),
                              S(hot_key='w', animation_time=0.9),
                              S(hot_key='t', animation_time=0.6),
                              S(hot_key='s', animation_time=0.3),
                              S(hot_key='f', animation_time=0.3),
                              S(hot_key='q', animation_time=0.4),
                          ],
                          height=170,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='t', animation_time=0.4),
                              S(hot_key='r', animation_time=0.3),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='二觉', hot_key='h', animation_time=2),
                          ],
                          sub_class=BaseClass.男枪手.男漫游,
                          ))

    role_configs.append(R(name='龙骑', no=21,
                          buffs=[
                              [Key.up, Key.up, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='a', animation_time=0.6),
                              S(hot_key='s', animation_time=0.8),
                              S(hot_key='q', animation_time=0.6),
                              S(hot_key='e', animation_time=1),
                              S(hot_key='t', animation_time=0.6),
                              S(hot_key='w', animation_time=0.6),
                              S(hot_key='f', animation_time=0.7),
                              S(hot_key='r', animation_time=0.7),
                              S(hot_key='g', animation_time=0.7),
                          ],
                          height=138,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='e', animation_time=1),
                              S(hot_key='r', animation_time=0.7),
                              S(hot_key='f', animation_time=0.7),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='二觉', hot_key='h', animation_time=2),
                          ],
                          white_map_level=0,
                          sub_class=BaseClass.守护者.龙神,
                          ))

    role_configs.append(R(name='冰结', no=22,
                          buffs=[
                              # [Key.up, Key.down, Key.space],
                              [Key.left, Key.right, Key.space]
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='v', animation_time=0.3),
                              S(hot_key='g', animation_time=0.5),
                              S(hot_key='t', animation_time=0.4),
                              S(hot_key='e', animation_time=0.4),
                              S(hot_key='r', animation_time=0.3),
                              S(hot_key='h', animation_time=0.6),
                              S(hot_key='d', animation_time=0.6),
                              S(hot_key='a', animation_time=0.8),
                              S(hot_key=Key.tab, animation_time=0.8),
                          ],
                          height=141,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='g', animation_time=0.4),
                              S(hot_key='r', animation_time=0.3),
                              S(hot_key='h', animation_time=0.6),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='二觉', hot_key='f', animation_time=2),
                          ],
                          white_map_level=0,
                          sub_class=BaseClass.男法师.冰结师,
                          ))

    role_configs.append(R(name='爆破', no=23,
                          buffs=[
                              # [Key.up, Key.down, Key.space],
                              [Key.right, Key.right, Key.space, '', '', Key.left, '', '', '', '']
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='v', animation_time=0.4),
                              S(hot_key='s', animation_time=0.4),
                              S(hot_key='q', animation_time=0.6),
                              S(hot_key='w', animation_time=0.4),
                              S(hot_key='r', animation_time=0.4),
                              S(hot_key='f', animation_time=0.4),
                              S(hot_key='t', animation_time=0.4),
                              S(hot_key='a', animation_time=0.4),
                              S(hot_key='h', animation_time=0.9),
                          ],
                          height=141,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='h', animation_time=0.6),
                              S(hot_key='v', animation_time=0.4),
                              S(hot_key='q', animation_time=0.4),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='二觉', hot_key='g', animation_time=2),
                          ],
                          white_map_level=0,
                          sub_class=BaseClass.男法师.魔皇,
                          ))

    role_configs.append(R(name='精灵', no=24,
                          buffs=[
                              [Key.up, Key.up, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(name='1', command=['f', ' ', ' ', ' ', Key.space], hot_key='f', hotkey_cd_command_cast=True, animation_time=0.4),
                              S(name='1', command=['a', ' ', ' ', ' ', Key.space], hot_key='a', hotkey_cd_command_cast=True, animation_time=0.4),
                              S(name='1', command=['w', ' ', ' ', ' ', Key.space], hot_key='w', hotkey_cd_command_cast=True, animation_time=0.6),
                              S(name='1', command=['v', ' ', ' ', ' ', Key.space], hot_key='v', hotkey_cd_command_cast=True, animation_time=0.4),
                              S(name='1', command=['r', ' ', ' ', ' ', Key.space], hot_key='r', hotkey_cd_command_cast=True, animation_time=0.4),
                              S(name='1', command=['s', ' ', ' ', ' ', Key.space], hot_key='s', hotkey_cd_command_cast=True, animation_time=0.4),
                              S(name='1', command=['d', ' ', ' ', ' ', Key.space], hot_key='d', hotkey_cd_command_cast=True, animation_time=0.4),
                              S(name='1', command=['e', ' ', ' ', ' ', Key.space], hot_key='e', hotkey_cd_command_cast=True, animation_time=0.4),
                          ],
                          height=137,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(name='1', command=['s', ' ', ' ', ' ', Key.space], hot_key='s', hotkey_cd_command_cast=True, animation_time=0.4),
                              S(name='1', command=['w', ' ', ' ', ' ', Key.space], hot_key='w', hotkey_cd_command_cast=True, animation_time=0.4),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                          ],
                          white_map_level=0,
                          sub_class=BaseClass.守护者.精灵骑士,
                          ))

    role_configs.append(R(name='风法', no=25,
                          buffs=[
                              [Key.left, Key.left, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='s', animation_time=0.4),
                              S(hot_key='d', animation_time=0.4),
                              S(hot_key='w', animation_time=0.5),
                              S(hot_key='f', animation_time=0.5),
                              S(hot_key='r', animation_time=0.6),
                              S(hot_key='t', animation_time=0.4),
                              S(hot_key='v', animation_time=0.4),
                          ],
                          height=141,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='w', animation_time=0.5),
                              S(hot_key='r', animation_time=0.7),
                              S(hot_key='q', animation_time=0.6),
                              S(hot_key='e', animation_time=0.7),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='2觉', hot_key='h', animation_time=2),
                          ],
                          white_map_level=0,
                          sub_class=BaseClass.男法师.风法,
                          ))

    role_configs.append(R(name='忍者', no=26,
                          buffs=[
                              [Key.right, Key.right, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='f', animation_time=0.9),
                              S(hot_key='d', animation_time=1.2),
                              S(hot_key='g', animation_time=0.5),
                              S(hot_key='w', animation_time=0.6),
                              S(hot_key='r', animation_time=0.6),
                              S(hot_key='v', animation_time=0.8),
                              S(hot_key='t', animation_time=0.8),
                              S(hot_key='s', animation_time=0.9),
                              S(hot_key='q', animation_time=1.2),
                          ],
                          height=155,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='v', animation_time=0.6),
                              S(hot_key='d', animation_time=0.6),
                              S(hot_key='s', animation_time=0.6),
                              S(hot_key='t', animation_time=0.6),
                              S(hot_key='f', animation_time=0.6),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                          ],
                          white_map_level=0,
                          sub_class=BaseClass.暗夜.忍者,
                          ))

    role_configs.append(R(name='死灵', no=27,
                          buffs=[
                              # [Key.right, Key.right, Key.space],
                              [Key.down, Key.down, Key.space]
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='d', animation_time=0.4),
                              S(hot_key='a', animation_time=0.6),
                              S(hot_key='e', animation_time=0.4),
                              S(hot_key='q', animation_time=0.4),
                              S(hot_key=Key.tab, animation_time=0.6),
                              S(hot_key='t', animation_time=0.4),
                              S(hot_key='v', animation_time=0.4),
                              S(hot_key='w', animation_time=0.4),
                              S(hot_key='s', animation_time=0.4),

                          ],
                          height=155,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='e', animation_time=0.4),
                              S(hot_key='t', animation_time=0.4),
                              S(hot_key='v', animation_time=0.4),
                              S(hot_key='d', animation_time=0.4),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='2觉', command=['g'], animation_time=2),
                          ],
                          white_map_level=0,
                          sub_class=BaseClass.暗夜.死灵,
                          ))

    dark_skills = [
        S(name='e', hot_key='e', cd=12.1, animation_time=0.4),
        S(name='w', hot_key='w', cd=12.9, animation_time=0.4),
        S(name='tab', hot_key=Key.tab, cd=16.2, animation_time=1.2),
        S(name='r', hot_key='r', cd=28.5, animation_time=1.2),
        S(name='t', hot_key='t', cd=32, animation_time=0.6),
        S(name='v', hot_key='v', cd=42.5, animation_time=0.9),
        S(name='q', hot_key='q', cd=7.1, animation_time=0.4),
        S(name='a', hot_key='a', cd=36.2, animation_time=1),
        S(name='s', hot_key='s', cd=19.2, animation_time=0.6),
        S(name='d', hot_key='d', cd=14.3, animation_time=0.6),
        S(name='f', hot_key='f', cd=21.4, animation_time=0.4),
        S(name='g', hot_key='g', cd=42.8, animation_time=0.4),
    ]
    role_configs.append(R(name='黑武', no=28,
                          buffs=[
                              [Key.right, Key.right, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=dark_skills,
                          # custom_priority_skills=[
                          #     S(name='e', hot_key='e', cd=10.7, animation_time=0.4),
                          #     S(name='tab', hot_key=Key.tab, cd=16.2, animation_time=0.4),
                          #     S(name='w', hot_key='w', cd=11.4, animation_time=0.4),
                          #     S(name='r', hot_key='r', cd=28.5, animation_time=1),
                          #     S(name='t', hot_key='t', cd=32, animation_time=0.6),
                          #     S(name='v', hot_key='v', cd=36.2, animation_time=1),
                          #     S(name='a', hot_key='a', cd=28.9, animation_time=0.6),
                          #     S(name='s', hot_key='s', cd=19.2, animation_time=0.6),
                          #     S(name='d', hot_key='d', cd=14.3, animation_time=0.6),
                          #     S(name='q', hot_key='q', cd=7.1, animation_time=0.4),
                          #     S(name='f', hot_key='f', cd=21.4, animation_time=0.4),
                          #     S(name='g', hot_key='g', cd=42.8, animation_time=0.4),
                          # ],
                          height=160,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              list(filter(lambda x1: x1.hot_key == 'v', dark_skills))[0],
                              # S(name='v', hot_key='v', cd=36.2, animation_time=1),
                              list(filter(lambda x1: x1.hot_key == 'a', dark_skills))[0],
                              # S(name='a', hot_key='a', cd=28.9, animation_time=0.6),
                              list(filter(lambda x1: x1.hot_key == 'g', dark_skills))[0],
                              # S(name='g', hot_key='g', cd=42.8, animation_time=0.4),
                              S(name='3觉', command=[Key.left, Key.left, Key.right, Key.right, 'z'], cd=290, animation_time=3),
                          ],
                          white_map_level=0,
                          sub_class=BaseClass.外传.黑暗武士,
                          ))

    role_configs.append(R(name='机械', no=29,
                          buffs=[
                              # [Key.right, Key.right, Key.space],
                              [Key.up, Key.down, Key.space],
                              [Key.left, Key.right, Key.space]
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='h', animation_time=0.3),
                              S(hot_key='r', animation_time=0.3),
                              S(hot_key=Key.tab, animation_time=0.5),
                              S(name='1', command=['w', ' ', ' ', ' ', ' ', 'q'], hot_key='w', hotkey_cd_command_cast=True, animation_time=0.3),
                              S(hot_key='e', animation_time=0.3),
                              S(hot_key='t', animation_time=0.3),
                              S(name='1', command=['s', ' ', ' ', ' ', ' ', 'q'], hot_key='s', hotkey_cd_command_cast=True, animation_time=0.3),
                              S(name='1', command=['f', ' ', ' ', ' ', ' ', 'q'], hot_key='f', hotkey_cd_command_cast=True, animation_time=0.3),
                              S(name='1', command=['v', ' ', ' ', ' ', ' ', 'q'], hot_key='v', hotkey_cd_command_cast=True, animation_time=0.3),
                          ],
                          height=169,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='e', animation_time=0.3),
                              S(hot_key='h', animation_time=0.3),
                              S(hot_key=Key.tab, animation_time=0.5),
                              S(hot_key='r', animation_time=0.3),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='2觉', command=['g'], animation_time=2),
                          ],
                          white_map_level=0,
                          sub_class=BaseClass.男枪手.男机械,
                          ))

    role_configs.append(R(name='暗帝', no=30,
                          buffs=[
                              [Key.right, Key.right, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='w', animation_time=0.8),
                              S(hot_key='e', animation_time=0.6),
                              S(hot_key='v', animation_time=0.6),
                              S(hot_key='t', animation_time=0.6),
                              S(hot_key='d', animation_time=0.6),
                              S(hot_key='g', animation_time=0.5),
                              S(hot_key='a', animation_time=0.8),
                              S(hot_key='r', animation_time=0.4),
                          ],
                          height=149,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='e', animation_time=0.3),
                              S(hot_key='r', animation_time=0.3),
                              S(hot_key='a', animation_time=0.3),
                              S(hot_key='v', animation_time=0.9),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='2觉', command=['h'], animation_time=2),
                          ],
                          white_map_level=0,
                          sub_class=BaseClass.女鬼剑.暗帝,
                          ))

    role_configs.append(R(name='毒王', no=31,
                          buffs=[
                              # [Key.right, Key.right, Key.space],
                              # [Key.down, Key.down, Key.space],
                              ['h']
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='e', animation_time=0.3),
                              S(hot_key='f', animation_time=0.7),
                              S(hot_key='g', animation_time=0.4),
                              S(hot_key='a', animation_time=0.3),
                              S(hot_key='t', animation_time=0.4),
                              S(hot_key='v', animation_time=0.4),
                              S(hot_key='s', animation_time=0.5),
                              S(hot_key='w', animation_time=0.4),
                              S(hot_key='q', animation_time=0.4),
                          ],
                          height=141,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='t', animation_time=0.4),
                              S(hot_key='s', animation_time=0.5),
                              S(hot_key='v', animation_time=0.4),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                          ],
                          white_map_level=0,
                          sub_class=BaseClass.女格斗.女毒王,
                          ))

    role_configs.append(R(name='影舞', no=32,
                          buffs=[
                              # [Key.right, Key.right, Key.space],
                              [Key.up, Key.down, Key.space]
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='q', animation_time=0.5),
                              S(hot_key='d', animation_time=1),
                              S(hot_key=Key.tab, animation_time=0.5),
                              S(hot_key='v', animation_time=0.5),
                              S(hot_key='s', animation_time=0.5),
                              S(hot_key='a', animation_time=0.5),
                              S(hot_key='t', animation_time=0.5),
                              S(hot_key='e', animation_time=0.9),
                              S(hot_key='w', animation_time=0.6),
                          ],
                          height=155,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='q', animation_time=0.5),
                              S(hot_key=Key.tab, animation_time=0.5),
                              S(hot_key='v', animation_time=0.5),
                              S(hot_key='s', animation_time=0.9),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                          ],
                          white_map_level=0,
                          sub_class=BaseClass.暗夜.影舞者,
                          ))

    role_configs.append(R(name='四姨', no=33,
                          buffs=[
                              [Key.up, Key.up, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='d', animation_time=0.3),
                              S(hot_key='h', animation_time=0.3),
                              S(hot_key='e', animation_time=0.6),
                              S(hot_key='q', animation_time=0.7),
                              S(hot_key='v', animation_time=0.3),
                              S(name='1', command=['a', ' ', ' ', 'a', Key.space], hot_key='a', hotkey_cd_command_cast=True, animation_time=0.3),
                              S(hot_key='r', animation_time=0.5),
                              S(hot_key='t', animation_time=0.5),
                              S(hot_key='s', animation_time=0.5),
                          ],
                          height=151,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='w', animation_time=0.6),
                              S(hot_key='q', animation_time=0.7),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3)
                          ],
                          white_map_level=0,
                          sub_class=BaseClass.女圣职.诱魔者,
                          ))

    role_configs.append(R(name='专家', no=34,
                          buffs=[
                              [Key.down, Key.down, Key.space]
                          ],
                          buff_effective=False,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='w', animation_time=0.8),
                              S(hot_key='q', animation_time=0.4),
                              S(hot_key=Key.tab, animation_time=0.4),
                              S(hot_key='f', animation_time=0.6),
                              S(hot_key='a', animation_time=0.4),
                              S(hot_key='s', animation_time=0.4),
                              S(hot_key='r', animation_time=0.4),
                          ],
                          height=164,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key=Key.tab, animation_time=0.4),
                              S(hot_key='f', animation_time=0.4),
                              S(command=['d', ' ', ' ', ' ', 'd'], hot_key='d', hotkey_cd_command_cast=True, animation_time=0.4),
                              S(hot_key='r', animation_time=0.4),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3)
                          ],
                          sub_class=BaseClass.枪剑士.能源专家,
                          ))

    role_configs.append(R(name='奶妈2', no=35,
                          buffs=[
                              [Key.right, Key.right, Key.space],
                              [Key.up, Key.right, Key.space]
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='q', animation_time=0.8),
                              S(name='勇气颂歌', command=['r', 'x', 'x', 'x', 'c'], hot_key='r', hotkey_cd_command_cast=True),
                              S(name='圣洁之翼', command=['e', 'x', 'x', 'x', 'x', 'x', 'x', 'e'], hot_key='e', hotkey_cd_command_cast=True),
                              S(hot_key='v', animation_time=0.3),
                              S(hot_key='d', animation_time=0.5),
                              S(hot_key='f', animation_time=0.5),
                          ],
                          height=151,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          sub_class=BaseClass.女圣职.奶妈,
                          ))

    role_configs.append(R(name='鬼泣', no=36,
                          buffs=[
                              [Key.down, Key.down, Key.space]
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='d', animation_time=0.4),
                              S(hot_key='s', animation_time=0.4),
                              S(hot_key='e', animation_time=0.4),
                              S(hot_key='w', animation_time=0.4),
                              S(hot_key='r', animation_time=0.4),
                              S(hot_key='t', animation_time=0.4),
                              S(hot_key='g', animation_time=0.4),
                              S(hot_key=Key.tab, animation_time=0.7),
                          ],
                          height=160,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='g', animation_time=0.4),
                              S(hot_key='r', animation_time=0.4),
                              S(hot_key='t', animation_time=0.4),
                              S(hot_key='s', animation_time=0.4),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3)
                          ],
                          sub_class=BaseClass.男鬼剑.鬼泣,
                          ))

    role_configs.append(R(name='斗萝', no=37,
                          buffs=[
                              [Key.right, Key.right, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='w', animation_time=0.3),
                              S(hot_key='s', animation_time=0.6),
                              S(hot_key='a', animation_time=0.4),
                              S(hot_key='f', animation_time=0.4),
                              S(hot_key='h', animation_time=0.5),
                              S(hot_key='v', animation_time=0.4),
                              S(hot_key=Key.tab, animation_time=0.5),
                          ],
                          height=127,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='f', animation_time=0.4),
                              S(name='d', command=['d', ' ', ' ', ' ', ' ', ' ', 'd'], hot_key='d', hotkey_cd_command_cast=True, animation_time=0.3),
                              S(hot_key='g', animation_time=0.6),
                              S(hot_key=Key.tab, animation_time=0.5),
                              S(name='一觉', hot_key='e', animation_time=2),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                          ],
                          sub_class=BaseClass.女法师.战法,
                          ))

    role_configs.append(R(name='赵云', no=38,
                          buffs=[
                              [Key.up, Key.down, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          height=157,
                          custom_priority_skills=[
                              S(hot_key='q', animation_time=0.4),
                              S(hot_key='v', animation_time=1),
                              S(hot_key='r', animation_time=0.5),
                              S(hot_key='e', animation_time=0.5),
                              S(hot_key='s', animation_time=0.4),
                              S(hot_key='d', animation_time=0.4),
                              S(hot_key='w', animation_time=0.8),
                          ],
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='q', animation_time=0.4),
                              S(hot_key='v', animation_time=0.8),
                              S(hot_key='e', animation_time=0.5),
                              S(hot_key=Key.tab, animation_time=0.5),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='二觉', hot_key='g', animation_time=2),
                          ],
                          sub_class=BaseClass.魔枪士.赵云,
                          ))

    role_configs.append(R(name='帕拉丁', no=39,
                          buffs=[
                              [Key.left, Key.left, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='v', animation_time=0.5),
                              S(hot_key='a', animation_time=0.8),
                              S(hot_key='s', animation_time=0.5),
                              S(hot_key='e', animation_time=0.5),
                              S(hot_key='d', animation_time=1),
                              S(hot_key='q', animation_time=0.5),
                              S(hot_key='w', animation_time=0.7),
                              S(hot_key=Key.tab, animation_time=0.9),
                          ],
                          height=138,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='s', animation_time=0.5),
                              S(hot_key='q', animation_time=0.5),
                              Key.tab,
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='二觉', hot_key='g', animation_time=2),
                          ],
                          sub_class=BaseClass.守护者.帕拉丁,
                          ))

    role_configs.append(R(name='魔灵', no=40,
                          buffs=[
                              # [Key.left, Key.left, Key.space]
                              ['a'],
                              ['s']
                          ],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='d', animation_time=0.8),
                              S(hot_key='g', animation_time=0.8),
                              S(hot_key='q', animation_time=0.7),
                              S(hot_key='f', animation_time=0.8),
                              S(hot_key='e', animation_time=0.8),
                              S(hot_key=Key.tab, animation_time=0.9),
                              S(hot_key='w', animation_time=0.7),
                              S(hot_key='r', animation_time=0.7),
                              S(hot_key='t', animation_time=0.7),
                              S(hot_key='a', animation_time=0.3),
                              S(hot_key='s', animation_time=0.3),
                          ],
                          height=138,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='e', animation_time=0.8),
                              S(hot_key='t', animation_time=0.7),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(name='二觉', hot_key='h', animation_time=2),
                          ],
                          white_map_level=0,
                          sub_class=BaseClass.守护者.混沌魔灵,
                          ))

    role_configs.append(R(name='柔道', no=41,
                          buffs=[],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='v', animation_time=0.6),
                              S(hot_key='e', animation_time=0.9),
                              S(hot_key=Key.tab, animation_time=1),
                              S(hot_key='f', animation_time=1),
                              S(hot_key='w', animation_time=1),
                              S(hot_key='s', animation_time=0.7),
                              S(hot_key='d', animation_time=0.7),
                              S(hot_key='a', animation_time=0.6),
                          ],
                          height=141,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                              S(hot_key='q', animation_time=0.8),
                              S(name='二觉', hot_key='h', animation_time=2),
                          ],
                          sub_class=BaseClass.女格斗.女柔道,
                          ))

    role_configs.append(R(name='次元', no=42,
                          buffs=[
                              [Key.left, Key.right, Key.space]
                          ],
                          buff_effective=False,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(hot_key='w', animation_time=0.3),
                              S(hot_key='q', animation_time=0.5),
                              S(hot_key='e', animation_time=0.4),
                              S(hot_key='r', animation_time=0.4),
                              S(hot_key='t', animation_time=0.3),
                              S(hot_key='v', animation_time=0.6),
                              S(hot_key='a', animation_time=0.6),
                              S(hot_key='s', animation_time=0.8),
                              S(hot_key='d', animation_time=0.8),
                          ],
                          height=141,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved,
                          powerful_skills=[
                              S(hot_key='q', animation_time=0.4),
                              S(hot_key='v', animation_time=0.3),
                              S(hot_key='a', animation_time=0.6),
                              S(name='三觉', hot_key=Key.ctrl_l, animation_time=3),
                          ],
                          white_map_level=0,
                          sub_class=BaseClass.男法师.次元,
                          ))

    role_configs.append(R(name='缺省配置角色', no=43,
                          sub_class_auto=True
                          ))

    role_configs.append(R(name='缺省配置角色', no=44,
                          sub_class_auto=True
                          ))


    return role_configs


if __name__ == "__main__":
    role_list = get_role_config_list()
    print(role_list[1])
    cike = role_list[0]

    print(cike)
