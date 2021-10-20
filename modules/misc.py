import time

__all__ = ['Timer']

class Timer:
    def __init__(self, name):
        self.name = name
        self.tic = 0
        self.num = 0
        self.total = 0
        self.running = False
    def start(self):
        assert not self.running, "cannot start timer twice"
        self.tic = time.time()
        self.running = True
    def pause(self):
        assert self.running, "timer did not start."
        self.num += 1
        self.total += time.time() - self.tic
        self.running = False
    def avg_time(self):
        assert not self.running
        return self.total / self.num
    def print_time(self):
        print(f'timer {self.name}: avg = {self.avg_time()}, tot = {self.total}')

