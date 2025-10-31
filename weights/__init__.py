# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'

import os
import shutil
import urllib.error
import urllib.request

import config as config_
from utils.utilities import calculate_sha256

# 大文件不LFS了，存网盘，不要影响仓库体积性能
stronger_pt = {
    "path": 'weights/stronger.pt',
    "version": '202506080210.best.pt',
    "download_url": "https://gitee.com/nianlingbeige/large_file_repo/raw/master/BabyBus/stronger.pt",
    "sha256": '6447F92987D00E2CF35EF2F467B41A55DCE9CCCEA1E630337190301B752F0166',
    'finish_time': '2025-06-08 02:10:48'
}
abyss_pt = {
    "path": 'weights/abyss.pt',
    "version": 'abyss.04131027.best.pt',
    "download_url": "https://gitee.com/nianlingbeige/large_file_repo/raw/master/BabyBus/abyss.pt",
    "sha256": '4443076FEE74DD97FF26C633EA4104805A375298587114E916EDDCF3AFBA5E89',
    'finish_time': '2025-04-13 10:27:16'
}


def check_init_pt(pt_info):
    pt_file = os.path.join(config_.project_base_path, pt_info["path"])
    if os.path.exists(pt_file):
        sha = calculate_sha256(pt_file)
        if sha.lower() != pt_info["sha256"].lower():
            urllib.request.urlretrieve(pt_info["download_url"], pt_file)
            print(f'{pt_info["path"]} 权重更新完成.')
        else:
            pass
    else:
        old_file = os.path.join(config_.project_base_path, f'weights/{pt_info["version"]}')
        if os.path.exists(old_file):
            old_sha = calculate_sha256(os.path.join(config_.project_base_path, f'weights/{pt_info["version"]}'))
            if old_sha.lower() == pt_info["sha256"].lower():
                shutil.copy(os.path.join(config_.project_base_path, f'weights/{pt_info["version"]}'), pt_file)
            else:
                urllib.request.urlretrieve(pt_info["download_url"], pt_file)
                print(f'{pt_info["path"]} 权重更新完成。')
        else:
            urllib.request.urlretrieve(pt_info["download_url"], pt_file)
            print(f'{pt_info["path"]} 权重更新完成')


check_init_pt(stronger_pt)
check_init_pt(abyss_pt)
