# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import cv2

import config as config_


@dataclass(frozen=True)
class SubClassInfo:
    code: str = field(metadata={"description": "编号"})
    cname: str = field(metadata={"description": "职业名称"})
    # awakening_skill: Optional[np.ndarray] = field(metadata={"description": "觉醒图标"})


class SubClass(Enum):
    @property
    def cname(self):
        return self.value.cname

    @property
    def code(self):
        return self.value.code

    # @property
    # def awakening_skill(self):
    #     return self.value.awakening_skill

    剑魂 = SubClassInfo('1-1', "剑魂1")
    狂战士 = SubClassInfo('1-2', "狂战士")
    鬼泣 = SubClassInfo('1-3', "鬼泣")
    阿修罗 = SubClassInfo('1-4', "阿修罗")
    剑影 = SubClassInfo('1-5', "剑影")

    女气功 = SubClassInfo('1-1', "UNK")
    女柔道 = SubClassInfo('1-2', "UNK")
    女散打 = SubClassInfo('1-3', "UNK")
    女毒王 = SubClassInfo('1-4', "UNK")

    剑宗 = SubClassInfo('3-1', "UNK")
    剑魔 = SubClassInfo('3-2', "UNK")
    剑帝 = SubClassInfo('3-3', "UNK")
    暗帝 = SubClassInfo('3-4', "UNK")
    刃影 = SubClassInfo('3-5', "UNK")

    男气功 = SubClassInfo('4-1', "UNK")
    男柔道 = SubClassInfo('4-2', "UNK")
    男散打 = SubClassInfo('4-3', "UNK")
    男街霸 = SubClassInfo('4-4', "UNK")

    男漫游 = SubClassInfo('5-1', "UNK")
    男大枪 = SubClassInfo('5-2', "UNK")
    男弹药 = SubClassInfo('5-3', "UNK")
    男机械 = SubClassInfo('5-4', "UNK")
    合金战士 = SubClassInfo('5-5', "UNK")

    女漫游 = SubClassInfo('6-1', "UNK")
    女大枪 = SubClassInfo('6-2', "UNK")
    女弹药 = SubClassInfo('6-3', "UNK")
    女机械 = SubClassInfo('6-4', "UNK")
    奶枪 = SubClassInfo('6-5', "UNK")

    冰结师 = SubClassInfo('7-1', "UNK")
    魔皇 = SubClassInfo('7-2', "UNK")
    风法 = SubClassInfo('7-3', "UNK")
    血法 = SubClassInfo('7-4', "UNK")
    次元 = SubClassInfo('7-5', "UNK")

    元素 = SubClassInfo('8-1', "UNK")
    战法 = SubClassInfo('8-2', "UNK")
    小魔女 = SubClassInfo('8-3', "UNK")
    魔道 = SubClassInfo('8-4', "UNK")
    召唤 = SubClassInfo('8-5', "UNK")

    奶爸 = SubClassInfo('9-1', "UNK")
    蓝拳 = SubClassInfo('9-2', "UNK")
    驱魔 = SubClassInfo('9-3', "UNK")
    复仇 = SubClassInfo('9-4', "UNK")

    奶妈 = SubClassInfo('10-1', "UNK")
    异端审判者 = SubClassInfo('10-2', "UNK")
    巫女 = SubClassInfo('10-3', "UNK")
    诱魔者 = SubClassInfo('10-4', "UNK")

    忍者 = SubClassInfo('11-1', "UNK")
    刺客 = SubClassInfo('11-2', "UNK")
    影舞者 = SubClassInfo('11-3', "UNK")
    死灵 = SubClassInfo('11-4', "UNK")

    帕拉丁 = SubClassInfo('12-1', "UNK")
    精灵骑士 = SubClassInfo('12-2', "UNK")
    龙神 = SubClassInfo('12-3', "UNK")
    混沌魔灵 = SubClassInfo('12-4', "UNK")

    赵云 = SubClassInfo('13-1', "UNK")
    关羽 = SubClassInfo('13-2', "UNK")
    光枪 = SubClassInfo('13-3', "UNK")
    暗枪 = SubClassInfo('13-4', "UNK")

    战线佣兵 = SubClassInfo('14-1', "UNK")
    特工 = SubClassInfo('14-2', "UNK")
    暗刃 = SubClassInfo('14-3', "UNK")
    能源专家 = SubClassInfo('14-4', "UNK")

    黑暗武士 = SubClassInfo('15-1', "UNK")

    旅人 = SubClassInfo('16-1', "UNK")
    奶弓 = SubClassInfo('16-2', "UNK")
    妖护使 = SubClassInfo('16-3', "UNK")
    猎人 = SubClassInfo('16-4', "UNK")
    奇美拉 = SubClassInfo('16-5', "UNK")


@dataclass
class Skill:
    name: str = field(default="", metadata={"description": "技能名称"})
    hot_key: Optional[object] = field(default=None, metadata={"description": "技能快捷键"})
    command: Optional[List[object]] = field(default_factory=list, metadata={"description": "指令"})
    concurrent: Optional[bool] = field(default=False, metadata={"description": "是否需要同时按住"})
    cd: Optional[float] = field(default=0, metadata={"description": "技能CD,秒"})
    recent_use_time: Optional[float] = field(default=0, metadata={"description": "最近使用的时间,秒级时间戳"})
    animation_time: Optional[float] = field(default=0.7, metadata={"description": "技能动作演出时间"})
    hotkey_cd_command_cast: Optional[bool] = field(default=False, metadata={"description": "使用快捷键判断CD，且使用指令释放技能"})

    def __post_init__(self):
        if not self.name and self.hot_key is not None:
            self.name = str(self.hot_key)


@dataclass
class RoleConfig:
    """
    RoleConfig 类用于配置角色的相关属性。
    """

    name: str = field(metadata={"description": "角色名称"})
    no: int = field(metadata={"description": "序号"})
    buffs: List[List[object]] = field(metadata={"description": "buff列表"})
    candidate_hotkeys: List[object] = field(metadata={"description": "允许释放的快捷栏列表上的技能"})
    custom_priority_skills: Optional[List[object]] = field(default_factory=list, metadata={"description": "自定义的技能列表"})
    height: int = field(default=180, metadata={"description": "身高，默认值为180"})
    fatigue_all: int = field(default=188, metadata={"description": "总疲劳值，默认值为188"})
    fatigue_reserved: int = field(default=30, metadata={"description": "保留的疲劳值"})
    attack_center_x: Optional[int] = field(default=0, metadata={"description": "角色攻击中心点距离"})
    attack_range_x: Optional[int] = field(default=0, metadata={"description": "攻击范围x"})
    attack_range_y: Optional[int] = field(default=0, metadata={"description": "攻击范围y"})
    buff_effective: Optional[bool] = field(default=False, metadata={"description": "是否上buff"})
    powerful_skills: Optional[List[object]] = field(default_factory=list, metadata={"description": "强力技能列表"})
    white_map_level: int = field(default=2, metadata={"description": "白图等级，默认勇士，（0普通，1冒险，2勇士，依次类推）"})
    sub_class: Optional[SubClass] = field(default=None, metadata={"description": "职业"})
    sub_class_auto: Optional[bool] = field(default=False, metadata={"description": "是否自动选择职业"})


class BaseClass:
    """BaseClass.男鬼剑.剑魂"""

    class 男鬼剑:
        height = 160
        剑魂 = SubClass.剑魂
        狂战士 = SubClass.狂战士
        鬼泣 = SubClass.鬼泣
        阿修罗 = SubClass.阿修罗
        剑影 = SubClass.剑影

    class 女格斗:
        height = 141
        女气功 = SubClass.女气功
        女柔道 = SubClass.女柔道
        女散打 = SubClass.女散打
        女毒王 = SubClass.女毒王

    class 女鬼剑:
        height = 149
        剑宗 = SubClass.剑宗
        剑魔 = SubClass.剑魔
        剑帝 = SubClass.剑帝
        暗帝 = SubClass.暗帝
        刃影 = SubClass.刃影

    class 男格斗:
        height = 138
        男气功 = SubClass.男气功
        男柔道 = SubClass.男柔道
        男散打 = SubClass.男散打
        男街霸 = SubClass.男街霸

    class 男枪手:
        height = 170
        男漫游 = SubClass.男漫游
        男大枪 = SubClass.男大枪
        男弹药 = SubClass.男弹药
        男机械 = SubClass.男机械
        合金战士 = SubClass.合金战士

    class 女枪手:
        height = 158
        女漫游 = SubClass.女漫游
        女大枪 = SubClass.女大枪
        女弹药 = SubClass.女弹药
        女机械 = SubClass.女机械
        奶枪 = SubClass.奶枪

    class 男法师:
        height = 141
        冰结师 = SubClass.冰结师
        魔皇 = SubClass.魔皇
        风法 = SubClass.风法
        血法 = SubClass.血法
        次元 = SubClass.次元

    class 女法师:
        height = 127
        元素 = SubClass.元素
        战法 = SubClass.战法
        小魔女 = SubClass.小魔女
        魔道 = SubClass.魔道
        召唤 = SubClass.召唤

    class 男圣职:
        height = 164
        奶爸 = SubClass.奶爸
        蓝拳 = SubClass.蓝拳
        驱魔 = SubClass.驱魔
        复仇 = SubClass.复仇

    class 女圣职:
        height = 151
        奶妈 = SubClass.奶妈
        异端审判者 = SubClass.异端审判者
        巫女 = SubClass.巫女
        诱魔者 = SubClass.诱魔者

    class 暗夜:
        height = 155
        忍者 = SubClass.忍者
        刺客 = SubClass.刺客
        影舞者 = SubClass.影舞者
        死灵 = SubClass.死灵

    class 守护者:
        height = 138
        帕拉丁 = SubClass.帕拉丁
        精灵骑士 = SubClass.精灵骑士
        龙神 = SubClass.龙神
        混沌魔灵 = SubClass.混沌魔灵

    class 魔枪士:
        height = 157
        赵云 = SubClass.赵云
        关羽 = SubClass.关羽
        光枪 = SubClass.光枪
        暗枪 = SubClass.暗枪

    class 枪剑士:
        height = 164
        战线佣兵 = SubClass.战线佣兵
        特工 = SubClass.特工
        暗刃 = SubClass.暗刃
        能源专家 = SubClass.能源专家

    class 外传:
        height = 160
        黑暗武士 = SubClass.黑暗武士

    class 弓箭手:
        height = 138
        旅人 = SubClass.旅人
        奶弓 = SubClass.奶弓
        妖护使 = SubClass.妖护使
        猎人 = SubClass.猎人
        奇美拉 = SubClass.奇美拉

    # @classmethod
    # def get_subclass(cls, base_name: str, sub_name: str):
    #     """通过字符串获取进阶职业"""
    #     base = getattr(cls, base_name, None)
    #     if base:
    #         return getattr(base, sub_name, None)
    #     return None

    @classmethod
    def get_base_class(cls, subclass_enum: SubClass):
        """
        根据 SubClass 枚举对象，返回它所属的 BaseClass 子类（如 BaseClass.男鬼剑）
        找不到，返回 None
        """
        for attr_name in dir(BaseClass):
            attr = getattr(BaseClass, attr_name)
            # 筛选 BaseClass 的内部类（排除内置属性和方法）
            if isinstance(attr, type) and attr.__module__ == BaseClass.__module__:
                # 检查该内部类是否包含这个枚举成员
                for field_name in dir(attr):
                    if getattr(attr, field_name, None) is subclass_enum:
                        return attr  # 返回这个“基类”，如 BaseClass.男鬼剑
        return None

class_icon_map = {}
folder_path = f'{config_.project_base_path}/assets/img/game/class'
print("初始化职业觉醒技能图标。。。")
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg')):
        file_name = os.path.splitext(filename)[0]
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            class_icon_map[file_name] = img

# 直接运行这个脚本
if __name__ == '__main__':
    print(SubClass.剑魂.code)
    print(BaseClass.男鬼剑.剑魂.name)
    print(BaseClass.男鬼剑.剑魂.code)
    print(BaseClass.女格斗.女散打.code)

    print(BaseClass.get_base_class(SubClass.剑魂).height)

    class_code = '16-1'
    for job in SubClass:
        code = job.code
        if code == class_code:
            print(job.name)
            break
