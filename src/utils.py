import datetime as dt
import math
import time


def sAsMinutes(s):
    return "{:d}m {:d}s".format(int(s // 60), int(s % 60))


def timeSince(since, percent):
    dt_now = dt.datetime.now()
    now_str = dt_now.strftime("%H:%M")
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    fin_str = (dt_now + dt.timedelta(rs)).strftime("%H:%M")
    return f"At {now_str} {sAsMinutes(s)} passed ( - {sAsMinutes(rs)} til {fin_str})"
