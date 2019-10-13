#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 18-10-16 下午7:06
# @Author  : YunSong
# @File    : habse_pool.py
# @Software: PyCharm


import happybase


def make_synchronized(func):
    import threading
    func.__lock__ = threading.Lock()

    def synced_func(*args, **kws):
        with func.__lock__:
            return func(*args, **kws)

    return synced_func


class Singleton(object):
    _instance = None

    @make_synchronized
    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance


class RecsysHBaseCls(Singleton):
    __first_init = False

    def __init__(self):
        if self.__class__.__first_init:
            return
        self.pool = happybase.ConnectionPool(host="127.0.0.1", autoconnect=False, size=3)
        self.__class__.__first_init = True

    def get_pool(self):
        return self.pool


if __name__ == '__main__':
    with RecsysHBaseCls().get_pool().connection() as conn:
        print(conn.tables())
        # print(conn.table('ksai_user_protrait'))
