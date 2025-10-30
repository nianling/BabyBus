# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'

import numpy as np
import pyautogui
import win32con
import win32gui
import win32ui
import win32process
import win32api


def kill_process_by_hwnd(hwnd):
    """
    根据窗口句柄终止对应的进程
    :param hwnd: 窗口句柄 (int)
    """
    if not win32gui.IsWindow(hwnd):
        return False

    # 获取进程 ID
    _, pid = win32process.GetWindowThreadProcessId(hwnd)

    try:
        # 打开进程
        handle = win32api.OpenProcess(win32con.PROCESS_TERMINATE, False, pid)
        # 终止进程
        win32api.TerminateProcess(handle, 0)
        # 关闭句柄
        win32api.CloseHandle(handle)
        print(f"进程 (PID={pid}) 已被终止")
        return True
    except Exception as e:
        print(f"终止进程失败: {e}")
        return False

def get_window_handle(window_title):
    """
    获取窗口句柄
    :param window_title:
    :return:
    """
    handle = win32gui.FindWindow(None, window_title)
    if handle == 0:
        raise Exception(f"根据窗口标题'{window_title}' 没有找到窗口.")
    return handle


def get_window_rect(handle):
    """
    获取窗口位置和大小
    :param handle:
    :return:
    """
    rect = win32gui.GetWindowRect(handle)
    x1, y1, x2, y2 = rect
    width = x2 - x1
    height = y2 - y1
    return x1, y1, width, height


def capture_window_image(handle):
    """
    截取窗口图像
    :param handle:
    :return:
    """
    x, y, width, height = get_window_rect(handle)
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return screenshot


def capture_window_BGRX(hwnd, region=None):
    """
    BGRX
    region(offset_x, offset_y, width, height)
    """

    # 获取窗口大小
    left, top, right, bot = win32gui.GetClientRect(hwnd)
    # left, top, right, bot = win32gui.GetWindowRect(hwnd)
    w = right - left
    h = bot - top

    if region:
        offset_x, offset_y, width, height = region
        left = left + offset_x
        top = top + offset_y
        w = width
        h = height

    # 获取窗口的设备上下文DC（Device Context）
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    # 创建位图对象
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(saveBitMap)

    # 复制窗口内容到位图中
    # result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
    # 复制窗口内容到位图
    saveDC.BitBlt((0, 0), (w, h), mfcDC, (left, top), win32con.SRCCOPY)

    # 将位图转换为PIL Image对象
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    # # 'BGRX' 转换为 'RGB'
    # img = Image.frombuffer(
    #     'RGB',
    #     (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
    #     bmpstr, 'raw', 'BGRX', 0, 1)

    # 不转换了,直接返回win32API的BGRX
    # img = np.frombuffer(bmpstr, dtype='uint8') # 只读的numpy数组
    img = np.fromstring(bmpstr, dtype='uint8')
    img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)

    # 清理资源
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    return img
    # 转换为numpy数组
    # return np.array(img)


def crop_image(image, x, y, width, height):
    """
    从图像中截取指定区域
    :param image: 原始图像 (numpy array)
    :param x: 左上角的x坐标
    :param y: 左上角的y坐标
    :param width: 截取区域的宽度
    :param height: 截取区域的高度
    :return: 截取后的图像
    """
    return image[y:y + height, x:x + width]


class WindowCapture:
    def __init__(self, hwnd):
        self.hwnd = hwnd
        self._init_resources()

    def _init_resources(self):
        # 预获取窗口尺寸（固定不变）
        rect = win32gui.GetWindowRect(self.hwnd)
        self.width = rect[2] - rect[0]
        self.height = rect[3] - rect[1]

        # 创建设备上下文
        self.wdc = win32gui.GetWindowDC(self.hwnd)
        self.dc = win32ui.CreateDCFromHandle(self.wdc)
        self.mem_dc = self.dc.CreateCompatibleDC()

        # 创建位图对象（提前创建避免循环内重复创建）
        self.bitmap = win32ui.CreateBitmap()
        self.bitmap.CreateCompatibleBitmap(self.dc, self.width, self.height)
        self.mem_dc.SelectObject(self.bitmap)

        # 预分配numpy数组内存
        self.buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def capture(self):
        # 直接拷贝到预创建位图
        self.mem_dc.BitBlt((0, 0), (self.width, self.height),
                           self.dc, (0, 0), win32con.SRCCOPY)

        # 获取位图数据（BGRX格式）
        bits = self.bitmap.GetBitmapBits(True)

        # 高效转换BGRX->BGR（去掉X通道）
        img = np.frombuffer(bits, dtype=np.uint8).reshape(
            (self.height, self.width, 4))[..., :3]

        # 使用预分配内存避免重复创建数组
        np.copyto(self.buffer, img)
        return self.buffer

    def release(self):
        win32gui.DeleteObject(self.bitmap.GetHandle())
        self.mem_dc.DeleteDC()
        self.dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, self.wdc)
