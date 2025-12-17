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
                          buffs=[[Key.right, Key.right, 'z'], [Key.right, Key.right, Key.space]],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              'q',
                              's',
                              'e',
                              'v'
                          ],
                          height=155,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='剑魔', no=2,
                          buffs=[[Key.right, Key.right, Key.space]],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              'f',
                              'q',
                              Key.tab,
                              'w',
                              'e',
                              'v',
                              'a'
                          ],
                          height=149,
                          # attack_center_x=250,
                          attack_center_x=100,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='剑魂', no=3,
                          buffs=[[Key.right, Key.right, Key.space]],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(name='拔刀', hot_key='r', cd=8.9),
                              # S(name='大拔', command=[' ', ' ', Key.right, Key.space], cd=29.8),
                              'q',
                              # S(name='95', command=[Key.up, Key.space], cd=35.7, concurrent=True),
                              'd',
                              # S(name='心剑', command=[Key.down, Key.space], cd=20.1, concurrent=True),
                              'a',
                              'g',
                              'w',
                          ],
                          height=160,
                          attack_center_x=60,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='奶爸', no=4,
                          buffs=[[Key.left, Key.right, Key.space], ['s']],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              'a',
                              'w',
                              S(name='神圣之光', command=['v'], cd=8.3),
                              'g',
                              'r',
                              'f',
                              't'
                          ],
                          height=164,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='瞎子', no=5,
                          buffs=[[Key.right, Key.right, Key.space],
                                 [Key.down, Key.down, Key.space]],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              'q',
                              't',
                              Key.tab,
                              'h',
                              'f',
                              'e',
                              'r'
                          ],
                          # height=212,
                          height=160,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='旅人', no=6,
                          buffs=[[Key.right, Key.right, Key.space]],
                          buff_effective=True,
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(name='高原雾花', command=['d', 'd', 'd', 'd', 'v'], cd=8.9),
                              S(name='迷雾箭雨', command=['w'], cd=3),
                              S(name='浓雾暴雨', command=[Key.tab], cd=13.4),
                              S(name='细雨', command=['a'], cd=5.2),
                              S(name='旋风', command=['f'], cd=33.5),
                              'r',
                              't',
                          ],
                          height=138,
                          attack_center_x=200,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='光兵', no=7,
                          buffs=[[Key.up, Key.up, Key.space], [Key.right, Key.right, Key.space]],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              'd',
                              'g',
                              'e',
                              'v',
                              'r',
                              'a',
                              't'
                          ],
                          height=141,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='红眼', no=8,
                          buffs=[[Key.right, Key.right, Key.space]],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              'e',
                              'w',
                              'd',
                              'q',
                              'r',
                              't',
                              Key.tab
                          ],
                          height=160,
                          attack_center_x=100,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='魔道', no=9,
                          buffs=[[Key.up, Key.up, Key.space], [Key.down, Key.up, Key.space]],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              'd',
                              S(name='舒露露', command=['q', 'x', 'x', 'x', 'x', 'x', 'q'], cd=10.5),
                              'f',
                              'h',
                              'w',
                              's',
                              'a',
                              'e'

                          ],
                          height=138,
                          attack_center_x=80,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='奶弓', no=10,
                          buffs=[[Key.right, Key.right, Key.space]],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(name='加速', command=['w', 'a'], cd=2),
                              S(name='净化', command=['f', 's'], cd=2),
                              's',
                              'r',
                              'd',
                              't',
                              'a'
                          ],
                          height=138,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='奶妈', no=11,
                          buffs=[[Key.right, Key.right, Key.space], [Key.up, Key.right, Key.space]],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              'q',
                              S(name='勇气颂歌', command=['r', 'x', 'x', 'x', 'c'], cd=15.6),
                              S(name='圣洁之翼', command=['e', 'x', 'x', 'x', 'x', 'x', 'x', 'e'], cd=28.2),
                              'v',
                              'd',
                              'f'
                          ],
                          height=151,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='剑影', no=12,
                          buffs=[[Key.up, Key.up, Key.space]],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              'r',
                              'd',
                              'a',
                              'v',
                              'h',
                              'q',
                              's',
                              't'
                          ],
                          height=160,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='奶萝', no=13,
                          buffs=[[Key.right, Key.right, Key.space], [Key.up, Key.up, Key.space]],
                          candidate_hotkeys=['q', 'w', 'e', 'r', 'a', 's', 'd', 'f', 'g'],
                          custom_priority_skills=[
                              'd',
                              'f',
                              'w',
                              'e',
                              'a',
                              'q',
                              'r'
                          ],
                          height=127,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='刃影', no=14,
                          buffs=[[Key.right, Key.right, Key.space]],
                          candidate_hotkeys=['x'],
                          height=149,
                          custom_priority_skills=[
                              'v',
                              'q',
                              'e',
                              'w',
                              'a',
                              Key.tab,
                              'r',
                              's'
                          ],
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='蓝拳', no=15,
                          buffs=[[Key.up, Key.down, Key.space], [Key.left, Key.right, Key.space]],
                          candidate_hotkeys=['x'],
                          height=164,
                          custom_priority_skills=[
                              Key.tab,
                              'v',
                              'h',
                              'w',
                              S(name='破碎', command=['z', 'x', 'x', 'x', 'x', 'x'], cd=6),
                              'r',
                              's'
                          ],
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='弹药', no=16,
                          buffs=[[Key.right, Key.right, Key.space, Key.right],
                                 [Key.left, Key.right, Key.space]],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              's',
                              'e',
                              'g',
                              't',
                              'r',
                              'a',
                              'f',
                              'q'
                          ],
                          height=158,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='斗萝', no=17,
                          buffs=[[Key.right, Key.right, Key.space]],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              'w',
                              's',
                              'a',
                              'f',
                              'h',
                              'v',
                              'h'
                          ],
                          height=127,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='赵云', no=18,
                          buffs=[[Key.up, Key.down, Key.space]],
                          candidate_hotkeys=['x'],
                          height=157,
                          custom_priority_skills=[
                              'a',
                              'q',
                              'v',
                              'r',
                              'e',
                              's',
                              'd',
                              'w'
                          ],
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='帕拉丁', no=19,
                          buffs=[
                              [Key.left, Key.left, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              'a',
                              's',
                              'e',
                              'd',
                              'v',
                              'w'
                          ],
                          height=138,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='漫游', no=20,
                          buffs=[
                              [Key.left, Key.left, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              'g',
                              't',
                              'r',
                              's',
                              'd',
                              'f',
                              'w',
                              'q'
                          ],
                          height=170,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='冰结', no=21,
                          buffs=[[Key.up, Key.down, Key.space],
                                 [Key.left, Key.right, Key.space]],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              'v',
                              't',
                              'r',
                              Key.tab,
                              'h',
                              's',
                              'e',
                              'a',
                              'w'
                          ],
                          height=141,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='龙骑', no=22,
                          buffs=[
                              [Key.up, Key.up, Key.space]
                          ],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              'a',
                              's',
                              'd',
                              'f',
                              'q',
                              'e',
                              'r',
                              't'
                          ],
                          height=138,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    role_configs.append(R(name='机械', no=23,
                          buffs=[[Key.right, Key.right, Key.space],
                                 [Key.up, Key.down, Key.space],
                                 [Key.left, Key.right, Key.space]],
                          candidate_hotkeys=['x'],
                          custom_priority_skills=[
                              S(name='1', command=['w', ' ', ' ', ' ', ' ', 'q'], cd=21.7),
                              'e',
                              S(name='1', command=['t', ' ', ' ', ' ', ' ', 'q'], cd=32.5),
                              S(name='1', command=['f', ' ', ' ', ' ', ' ', 'q'], cd=21.7),
                              S(name='1', command=['s', ' ', ' ', ' ', ' ', 'q'], cd=8.7),
                              S(name='1', hot_key='e'),
                              S(name='1', hot_key='r'),
                              S(name='1', command=['v', ' ', ' ', ' ', ' ', 'q'], cd=4.7)
                          ],
                          height=169,
                          fatigue_all=default_fatigue_all,
                          fatigue_reserved=default_fatigue_reserved
                          ))

    #
    # role_configs.append(R(name='爆破', no=24,
    #                       buffs=[[Key.up, Key.down, Key.space],
    #                              [Key.right, Key.right, Key.space, Key.left]
    #                              ],
    #                       candidate_hotkeys=['x'],
    #                       custom_priority_skills=[
    #                           'v',
    #                           'w',
    #                           'e',
    #                           'a',
    #                           's'
    #                       ],
    #                       height=141,
    #                       fatigue_all=default_fatigue_all,
    #                       fatigue_reserved=default_fatigue_reserved
    #                       ))
    #
    #
    # role_configs.append(R(name='风法', no=25,
    #                       buffs=[
    #                           [Key.left, Key.left, Key.space]
    #                       ],
    #                       candidate_hotkeys=['x'],
    #                       custom_priority_skills=[
    #                           's',
    #                           'd',
    #                           'w',
    #                           'f',
    #                           'r',
    #                           't'
    #                       ],
    #                       height=141,
    #                       fatigue_all=default_fatigue_all,
    #                       fatigue_reserved=default_fatigue_reserved
    #                       ))

    # role_configs.append(R(name='精灵', no=26,
    #                       buffs=[
    #                           [Key.up, Key.up, Key.space]
    #                       ],
    #                       candidate_hotkeys=['x'],
    #                       custom_priority_skills=[
    #                           S(name='1', command=['r', ' ', ' ', ' ', Key.space], cd=3.5),
    #                           S(name='1', command=['w', ' ', ' ', ' ', Key.space], cd=19.5),
    #                           S(name='1', command=['f', ' ', ' ', ' ', Key.space], cd=11.4),
    #                           S(name='1', command=['v', ' ', ' ', ' ', Key.space], cd=25.3),
    #                           S(name='1', command=['a', ' ', ' ', ' ', Key.space], cd=7.6),
    #                           S(name='1', command=['s', ' ', ' ', ' ', Key.space], cd=30.4),
    #                           S(name='1', command=['d', ' ', ' ', ' ', Key.space], cd=22.8),
    #                           S(name='1', command=['e', ' ', ' ', ' ', Key.space], cd=5)
    #                       ],
    #                       height=137,
    #                       fatigue_all=default_fatigue_all,
    #                       fatigue_reserved=default_fatigue_reserved
    #                       ))

    return role_configs


if __name__ == "__main__":
    role_list = get_role_config_list()
    print(role_list[1])
    cike = role_list[0]

    print(cike)


