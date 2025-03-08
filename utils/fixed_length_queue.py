# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'


from collections import deque


class FixedLengthQueue:
    def __init__(self, max_length=5):
        # 初始化一个最大长度为 max_length 的双端队列
        self.queue = deque(maxlen=max_length)

    def enqueue(self, item):
        # 当队列已满时，最老的元素将自动被移除
        self.queue.append(item)

    def peek(self):
        return self.queue[0] if not self.is_empty() else None

    def dequeue(self):
        # 如果队列为空，则返回 None；否则返回并移除最老的元素
        if not self.is_empty():
            return self.queue.popleft()
        return None

    def is_empty(self):
        # 判断队列是否为空
        return len(self.queue) == 0

    def size(self):
        # 返回队列当前的大小
        return len(self.queue)

    def clear(self):
        self.queue.clear()

    def __repr__(self):
        return f"FixedLengthQueue(max_length={self.queue.maxlen}, current_size={len(self.queue)})"

    # def is_stable(self, threshold=15, window_size=20):
    #     # 获取队列的最后window_size个元素
    #     recent_coords = list(self.queue)[-window_size:]
    #
    #     if len(recent_coords) < window_size:
    #         # 如果不足window_size个元素，直接返回False
    #         return False
    #
    #     # 计算平均位置
    #     avg_x = sum(coord[0] for coord in recent_coords) / window_size
    #     avg_y = sum(coord[1] for coord in recent_coords) / window_size
    #
    #     # print(avg_x, avg_y)
    #
    #     # 计算每个坐标与平均位置的距离，并检查是否超过阈值
    #     for x, y in recent_coords:
    #         if abs(x - avg_x) > threshold or abs(y - avg_y) > threshold:
    #             return False
    #
    #     return True

    def coords_is_stable(self, threshold=15, window_size=20):
        if len(self.queue) < window_size:
            return False

        recent_coords = []
        self.queue.rotate(window_size)
        for _ in range(window_size):
            recent_coords.append(self.queue[0])
            self.queue.rotate(-1)

        avg_x = sum(coord[0] for coord in recent_coords) / window_size
        avg_y = sum(coord[1] for coord in recent_coords) / window_size

        for x, y in recent_coords:
            if abs(x - avg_x) > threshold or abs(y - avg_y) > threshold:
                return False

        return True


    def room_is_same(self, min_size=20):
        if len(self.queue) < min_size:
            return False

        recent_rooms = []
        self.queue.rotate(min_size)
        for _ in range(min_size):
            recent_rooms.append(self.queue[0])
            self.queue.rotate(-1)

        recent_room = recent_rooms[0]

        for room in recent_rooms:
            if recent_room != room:
                return False
        return True

if __name__ == "__main__":
    # fq = FixedLengthQueue(max_length=3)
    # fq.enqueue(1)
    # fq.enqueue(2)
    # fq.enqueue(3)
    # print(fq.size())  # 输出: 3
    #
    # # 当队列满了之后再插入新的元素，最老的一个元素会被移除
    # fq.enqueue(4)
    # print(list(fq.queue))  # 输出: [2, 3, 4]
    #
    # # 移除最老的元素
    # print(fq.dequeue())  # 输出: 2
    # print(list(fq.queue))  # 输出: [3, 4]

    fq = FixedLengthQueue(max_length=50)
    # 假设有一些坐标点
    coordinates = [(100, 100), (101, 101), (102, 102), (110, 110), (110, 110), (110, 110), (110, 110), (110, 110),
                   (110, 110), (110, 110)]

    for coord in coordinates:
        fq.enqueue(coord)

    # 检查最新的20个坐标是否稳定
    print(fq.coords_is_stable(threshold=5, window_size=9))  # 输出取决于实际的坐标数据
