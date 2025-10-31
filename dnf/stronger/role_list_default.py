# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'

from pynput.keyboard import Key

from dnf.stronger.role_config import RoleConfig as R, SubClass, BaseClass
from dnf.stronger.role_config import Skill as S


def get_role_config_list() -> list[R]:
    role_configs = []

    for i in range(1, 44):
        role_configs.append(R(name='缺省配置角色', no=i, sub_class_auto=True))

    return role_configs


if __name__ == "__main__":
    role_list = get_role_config_list()
    print(role_list[1])
    cike = role_list[0]

    print(cike)
