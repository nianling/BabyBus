# -*- coding:utf-8 -*-

__author__ = "廿陵 <wemean66@gmail.com> (GitHub: @nianling)"
__version__ = '1.0'

import concurrent.futures
import time
import threading

class SingleTaskThreadPool:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._task_running = False

    def submit(self, func, *args, **kwargs):
        with self._lock:
            if not self._task_running:
                self._task_running = True
                future = self.executor.submit(func, *args, **kwargs)
                future.add_done_callback(self._on_task_done)
                print("Task submitted and running.")
            else:
                print("Task is already running, new task ignored.")

    def _on_task_done(self, future):
        with self._lock:
            self._task_running = False
            print("Task completed.")


class LimitedTaskThreadPool:
    def __init__(self, max_workers):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._running_tasks = 0
        self.max_workers = max_workers

    def submit(self, func, *args, **kwargs):
        with self._lock:
            if self._running_tasks < self.max_workers:
                self._running_tasks += 1
                future = self.executor.submit(func, *args, **kwargs)
                future.add_done_callback(self._on_task_done)
                print("Task submitted and running.")
            else:
                print("Maximum tasks are running, new task ignored.")

    def _on_task_done(self, future):
        with self._lock:
            self._running_tasks -= 1
            print("Task completed.")

def worker(x):
    print(f'Working on {x}')
    time.sleep(3)  # 模拟耗时操作
    return x * x

def main():
    pool = LimitedTaskThreadPool(max_workers=2)

    # 提交多个任务
    pool.submit(worker, 1)
    time.sleep(1)
    pool.submit(worker, 2)
    time.sleep(1)
    pool.submit(worker, 3)
    time.sleep(1)
    pool.submit(worker, 4)

    # 等待所有任务完成
    time.sleep(10)

if __name__ == '__main__':
    main()