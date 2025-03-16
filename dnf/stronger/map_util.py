# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'


import os
import time

import cv2
import cv2 as cv
import numpy as np
from logger_config import logger
from utils.utilities import match_template_one, match_template
import config as config_
from utils.utilities import compare_images

question_template = cv.imread(os.path.normpath(f'{config_.project_base_path}/assets/img/game/question_mark_bright.png'))


def img_show(img, save_name=None):
    cv.imshow("win", img)
    if save_name:
        cv.imwrite(save_name, img)  # 保存每个小矩形
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_row_count(img):
    # 截取右侧边框，截取其中的距离顶部52像素，距离右侧4像素，宽度为3，从上往下，高度为144的一个矩形。也就是竖排从上往下依次8个3*18的矩形
    # 根据提供的参数计算裁剪区域的位置和大小
    top = 52
    # right = img.shape[1] - 4  # 图像的宽度减去距离右侧的距离
    right = img.shape[1] - 7  # 图像的宽度减去距离右侧的距离
    left = right - 3  # 因为是从右向左，所以left是right减去宽度
    # 计算第一个方格的平均颜色值作为基准
    first_block_top = top
    first_cropped_img = img[first_block_top:first_block_top + 18, left:left + 3]
    average_color_first = np.average(first_cropped_img, axis=(0, 1))
    threshold = 10  # 设定一个合理的差异阈值 30
    # 循环截取横排8个18*18的方形，并计算与第一个方格的差异
    for i in range(1, 8):  # 从第二个方格开始比较
        block_top = top + i * 18
        if block_top > 600:
            break  # 如果超出边界则停止
        cropped_img = img[block_top:block_top + 18, left:left + 3]
        average_color_current = np.average(cropped_img, axis=(0, 1))
        # 计算颜色差异（L2距离）
        color_diff = np.linalg.norm(average_color_first - average_color_current)
        if color_diff > threshold:
            return i


def get_colum_count(img):
    # 根据上方横条，计算有几个格子
    # 截取横条，距离顶部33像素，距离右侧8像素，宽度为18，从右向左，长度为144的一个矩形。也就是横排8个18 * 18的方形

    # 根据提供的参数计算裁剪区域的位置和大小
    top = 33
    # right = img.shape[1] - 8  # 图像的宽度减去距离右侧的距离
    right = img.shape[1] - 11  # 图像的宽度减去距离右侧的距离
    left = right - 18  # 因为是从右向左，所以left是right减去宽度
    # 计算第一个方格的平均颜色值作为基准
    first_block_left = left
    first_cropped_img = img[top:top + 18, first_block_left:first_block_left + 18]
    # cv.imwrite('first_cropped_img.jpg', first_cropped_img)
    average_color_first = np.average(first_cropped_img, axis=(0, 1))
    threshold = 4  # 设定一个合理的差异阈值 30
    # 循环截取横排8个18*18的方形，并计算与第一个方格的差异
    for i in range(1, 10):  # 从第二个方格开始比较
        block_left = left - i * 18
        if block_left < 0:
            break  # 如果超出边界则停止
        cropped_img = img[top:top + 18, block_left:block_left + 18]
        # cv.imwrite(f'first_cropped_img{i}.jpg', cropped_img)
        average_color_current = np.average(cropped_img, axis=(0, 1))
        # 计算颜色差异（L2距离）
        color_diff = np.linalg.norm(average_color_first - average_color_current)
        # print(f"计算颜色差异 Block {i + 1} difference with first block: {color_diff}")
        if color_diff > threshold:
            return i


def get_small_map_region(img):
    """
    获取小地图区域
    :param img:
    :return:
    """
    cols = get_colum_count(img)
    rows = get_row_count(img)

    # x = img.shape[1] - 8 - (cols * 18)
    # y = 52
    # crop = img[y:y + (rows * 18), x:x + (cols * 18)]

    crop = get_small_map_region_img(img, rows, cols)

    # img_show(crop, 'smallMap.jpg')

    return crop, rows, cols


def get_small_map_region_img(img0, rows, cols):
    # x = img0.shape[1] - 8 - (cols * 18)
    x = img0.shape[1] - 11 - (cols * 18)
    y = 52
    crop = img0[y:y + (rows * 18), x:x + (cols * 18)]
    return crop


# 找出该区域内蓝色的点,作为当前所在的房间
def current_room_index_cropped(crop, rows, cols):
    # 获取图像的高度和宽度
    height, width = crop.shape[:2]

    # 分割矩形区域为几行几列的格子
    cell_width = width // cols  # 每个格子的宽度
    cell_height = height // rows  # 每个格子的高度

    # 将图像从BGR转换为HSV
    hsv_crop = cv.cvtColor(crop, cv.COLOR_BGR2HSV)

    # 定义蓝色的 HSV 范围
    lower_blue = np.array([90, 180, 70])  # 蓝色的下限
    upper_blue = np.array([130, 255, 255])  # 蓝色的上限

    # 记录蓝色像素最多的格子信息
    max_blue_pixels = -1
    max_blue_ratio = 0
    bluest_cell = (-1, -1)  # 格子的位置 (row, col)

    # 遍历每个格子，寻找蓝色像素
    for row in range(rows):
        for col in range(cols):
            # 计算每个格子的位置
            cell_x = col * cell_width
            cell_y = row * cell_height
            cell = hsv_crop[cell_y:cell_y + cell_height, cell_x:cell_x + cell_width]

            # 检测蓝色区域
            mask = cv.inRange(cell, lower_blue, upper_blue)

            # 找到所有蓝色像素的坐标
            blue_pixels = np.column_stack(np.where(mask > 0))

            # 获取图像的尺寸
            height, width = mask.shape

            # 定义中间区域 (图像中间 50% 区域)
            x_center_min, x_center_max = int(width * 0.25), int(width * 0.75)
            y_center_min, y_center_max = int(height * 0.25), int(height * 0.75)

            # x_center_min, x_center_max = int(width * 0.15), int(width * 0.85)
            # y_center_min, y_center_max = int(height * 0.15), int(height * 0.85)

            # 计算中间区域内的蓝色像素数量
            center_pixels = [pixel for pixel in blue_pixels
                             if x_center_min <= pixel[1] <= x_center_max and y_center_min <= pixel[0] <= y_center_max]

            # 计算蓝色像素的占比
            total_blue_pixels = len(blue_pixels)
            center_blue_pixels = len(center_pixels)

            if total_blue_pixels > 15:
                center_ratio = center_blue_pixels / total_blue_pixels
            else:
                center_ratio = 0

            # print((row, col), center_blue_pixels, total_blue_pixels, center_ratio)

            # # 判断蓝色像素是否主要集中在中间
            # if center_ratio > 0.5:
            #     if center_ratio > max_blue_ratio:
            #         max_blue_ratio = center_ratio
            #         # print("蓝色像素主要集中在中间区域")
            #         bluest_cell = (row, col)  # 记录行列索引，使用0基索引
            # else:
            #     # print("蓝色像素未集中在中间区域")
            #     pass

            if center_ratio > max_blue_ratio:
                max_blue_ratio = center_ratio
                # print("蓝色像素主要集中在中间区域")
                bluest_cell = (row, col)  # 记录行列索引，使用0基索引

            # # 如果当前格子中的蓝色像素数量比之前的最大值高，更新最大值
            # if blue_pixels > max_blue_pixels:
            #     max_blue_pixels = blue_pixels
            #     bluest_cell = (row, col)  # 记录行列索引，使用0基索引

    # 返回二维数组的结构中的位置
    logger.debug(f"当前房间在 {bluest_cell}")
    return bluest_cell


def get_boss_room_cropped(crop, rows, cols):
    """
    :param crop:
    :param rows:
    :param cols:
    :return: boss房间，0基索引
    """
    # 获取图像的高度和宽度
    height, width = crop.shape[:2]

    # cv.imshow('crop',crop)
    # cv.waitKey(0)
    # cv.imwrite("./map.jpg", crop)

    # 将图像从BGR转换为HSV
    hsv_crop = cv.cvtColor(crop, cv.COLOR_BGR2HSV)

    cell_width = width // cols  # 每个格子的宽度
    cell_height = height // rows  # 每个格子的高度

    # 红色在HSV空间的阈值范围（需要检测两个区间）
    # lower_red_1 = np.array([0, 50, 50])  # 红色的低区间下限
    # upper_red_1 = np.array([10, 255, 255])  # 红色的低区间上限
    # lower_red_2 = np.array([170, 50, 50])  # 红色的高区间下限
    # upper_red_2 = np.array([180, 255, 255])  # 红色的高区间上限

    lower_red_1 = np.array([0, 120, 70])  # 增大饱和度和明度下限
    upper_red_1 = np.array([10, 255, 255])  # 保持色调范围不变
    lower_red_2 = np.array([170, 120, 70])  # 增大饱和度和明度下限
    upper_red_2 = np.array([180, 255, 255])  # 红色的高区间上限

    # 记录红色像素最多的格子信息
    max_red_pixels = 0
    reddest_cell = (-1, -1)  # 格子的位置 (row, col)

    # 遍历每个格子，寻找红色像素
    for row in range(rows):
        for col in range(cols):
            # 计算每个格子的位置
            cell_x = col * cell_width
            cell_y = row * cell_height
            cell = hsv_crop[cell_y:cell_y + cell_height, cell_x:cell_x + cell_width]

            # 检测红色区域（两个区间的掩码）
            mask1 = cv.inRange(cell, lower_red_1, upper_red_1)
            mask2 = cv.inRange(cell, lower_red_2, upper_red_2)

            # 合并两个红色掩码
            red_mask = cv.bitwise_or(mask1, mask2)

            # 计算该格子中红色像素的数量
            red_pixels = cv.countNonZero(red_mask)

            # print((row, col), red_pixels)

            # 如果当前格子中的红色像素数量比之前的最大值高，更新最大值
            if red_pixels > max_red_pixels:
                max_red_pixels = red_pixels
                reddest_cell = (row, col)  # 记录行列索引，使用0基索引

    # 返回二维数组的结构中的位置
    # logger.debug(f"BOOS房间在 {reddest_cell}")
    return reddest_cell


def 根据蓝标找当前房间位置(img):
    # 裁剪出来小地图区域
    crop, rows, cols = get_small_map_region(img)
    # cv.imshow("map", crop)
    # cv.waitKey(0)
    (row, col) = current_room_index_cropped(crop, rows, cols)
    return row, col


def 找BOOS房间(img):
    # 裁剪出来小地图区域
    crop, rows, cols = get_small_map_region(img)

    # 从识别到的的小地图区域中找boss房间
    room = get_boss_room_cropped(crop, rows, cols)

    logger.debug(f"BOOS房间在 {room}")
    return room


def get_boss_from_crop(crop, rows, cols):
    """

    :param crop:
    :param rows:
    :param cols:
    :return: boss房间，0基索引
    """
    # 从识别到的的小地图区域中找boss房间
    room = get_boss_room_cropped(crop, rows, cols)

    logger.debug(f"BOOS房间在 {room}")
    return room


def get_one_grid(img, row, col):
    """
    根据索引，切割出单个格子
    :param img:
    :param row:
    :param col:
    :return:
    """
    one = img[(row - 1) * 18:row * 18, (col - 1) * 18:col * 18]
    return one


def get_allow_directions(crop, cur_row, cur_col):
    # 截取当前位置格子
    cur_img = get_one_grid(crop, cur_row + 1, cur_col + 1)
    # cv.imwrite("cur_grid.jpg",cur_img)

    score_list = []
    found_num = None
    allow_directions = []
    # SSIM查找当前格子，得到允许方向
    for map_npy in npy_list:
        for npy in map_npy['img_num']:
            score = compare_images(npy, cur_img)
            # score_list.append((num, score))
            score_list.append((map_npy['direction'], score))
            if score >= 0.95:
                # print('找到了，序号是：', num, score)
                # found_num = num
                found_num = npy
                allow_directions = map_npy['direction']
                break
        if found_num:
            break

    if not found_num:
        highest_score = max(score_list, key=lambda x: x[1])
        allow_directions = highest_score[0]

    # s = time.time()
    # print(f'找到序号{found_num}, 允许方向：{allow_directions}')
    return allow_directions


def all_question_mark_room_cropped(crop, rows, cols, cur_row, cur_col):
    # 获取图像的高度和宽度
    height, width = crop.shape[:2]

    # 分割矩形区域为几行几列的格子
    cell_width = width // cols  # 每个格子的宽度
    cell_height = height // rows  # 每个格子的高度

    gray_screenshot = cv.cvtColor(crop, cv.COLOR_BGRA2GRAY)
    template_again_gray = cv.cvtColor(question_template, cv.COLOR_BGR2GRAY)
    matches = match_template(gray_screenshot, template_again_gray, threshold=0.7)
    # print(f'问号匹配数量:{len(matches)}')

    res = set()

    if len(matches) > 0:
        # 获取匹配区域的尺寸
        for (x1, y1), (x2, y2) in matches:
            print(x1, y1, x2, y2)

            # 计算图案的中心坐标
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # 计算格子的索引
            grid_x = int(center_x // cell_width)
            grid_y = int(center_y // cell_height)

            # 确保格子的索引在有效范围内
            grid_x = min(max(grid_x, 0), cols - 1)
            grid_y = min(max(grid_y, 0), rows - 1)

            # print(f'图案的中心坐标: ({center_x}, {center_y})')
            # print(f'问号的格子: 行 {grid_y + 1}, 列 {grid_x + 1}')
            # res.add((grid_y, grid_x))

            direction = None
            if grid_x == (cur_col + 1):
                direction = 'right'
            if grid_x == (cur_col - 1):
                direction = 'left'
            if grid_y == (cur_row - 1):
                direction = 'up'
            if grid_y == (cur_row + 1):
                direction = 'down'
            res.add(direction)
    return list(res)


# ------------#######################################################################################

map_list = [
    {'img_num': [4, 5, 6, 7], 'direction': ['right']},
    {'img_num': [12, 12, 14, 15], 'direction': ['up']},
    {'img_num': [20, 21, 22, 23], 'direction': ['up', 'right']},
    {'img_num': [28, 29, 30, 31], 'direction': ['left']},
    {'img_num': [36, 37, 38, 39], 'direction': ['left', 'right']},
    {'img_num': [44, 45, 46, 47], 'direction': ['up', 'left']},
    {'img_num': [52, 53, 54, 55], 'direction': ['up', 'left', 'right']},
    {'img_num': [60, 61, 62, 63], 'direction': ['down']},
    {'img_num': [68, 69, 70, 71], 'direction': ['right', 'down']},
    {'img_num': [76, 77, 78, 79], 'direction': ['up', 'down']},
    {'img_num': [84, 85, 86, 87], 'direction': ['up', 'down', 'right']},
    {'img_num': [92, 93, 94, 95], 'direction': ['left', 'down']},
    {'img_num': [100, 101, 102, 103], 'direction': ['left', 'right', 'down']},
    {'img_num': [108, 109, 110, 111], 'direction': ['up', 'left', 'down']},
    {'img_num': [116, 117, 118, 119], 'direction': ['up', 'down', 'left', 'right']}
]

# 初始化到内存中
npy_list = []
for map_img in map_list:
    item = {'img_num': [], 'direction': map_img['direction']}
    for img_num in map_img['img_num']:
        p = os.path.normpath(f'{config_.project_base_path}/assets/img/game/c_{img_num}.npy')
        image = np.load(p)
        item['img_num'].append(image)
    npy_list.append(item)

# 1.根据蓝标找当前房间位置
# 2 根据当前位置切图，判断是哪个底图，来判断允许的方向
# 3 找问号位置，判断未探索方向
# 4 根据（当前位置，允许方向，为探索方向，boss位置）计算下个方向
# 计算亮度闪烁

#######################################


# if __name__ == "__main__":
#     # 从识别到的的小地图区域中找boss房间
#     crop= cv2.imread('smallMap.jpg')
#     room = get_boss_room_cropped(crop, 3, 6)
#
#     logger.debug(f"BOOS房间在 {room}")
#
#     pass
# # img = cv.imread('all.jpg')
# # # img = cv.imread('all2.jpg')
# # 计算下个方向(img)

#
if __name__ == "__main__":
    img = cv.imread('img0.jpg')  # 1,2
    # img = cv.imread('all2.jpg')  # 1,4
    # img = cv.imread('all3.jpg')  # 1,3

    cols = get_colum_count(img)
    rows = get_row_count(img)

    print('行，列', rows, cols)

    x = img.shape[1] - 8 - (cols * 18)
    y = 52
    crop = img[y:y + (rows * 18), x:x + (cols * 18)]
    # cv.imwrite('rectangle.png',crop)
    # grid_y, grid_x = one_question_mark_room_cropped(crop, rows, cols)
    # print(grid_y, grid_x)
    # grid_y = grid_y+1
    # grid_x = grid_x+1
    #
    # one = crop[(grid_y - 1) * 18:grid_y * 18, (grid_x - 1) * 18:grid_x * 18]
    # cv.imshow("one", one)
    # cv.waitKey(0)
    # cv.imwrite('cccall3.png', one)
#
# # y33, x右8，宽高18
