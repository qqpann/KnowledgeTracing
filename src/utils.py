import time
import math


def sAsMinutes(s):
    return '{:d}m {:d}s'.format(int(s // 60), int(s % 60))


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return f'{sAsMinutes(s)} ( - {sAsMinutes(rs)})'