#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-10-16 下午7:16
# @Author  : YunSong
# @File    : mul_process_put_hbase.py
# @Software: PyCharm
# @verson  : python3


import multiprocessing

from utils.habse_pool import RecsysHBaseCls

N_PROCESS = 3


class InputDataHbaseDemo():
    def __init__(self, table_name=None):
        self.table_name = table_name

    def makeDataList(self):
        rows = []
        value = {
            "click_behavior:cateid": "1",
        }
        for i in range(500):
            row_kes = "rk1" + str(i)
            rows.append((row_kes, value))
        self.data_list = rows

    def put_data(self, data_list):
        with RecsysHBaseCls().get_pool().connection() as conn:
            with conn.table(self.table_name).batch(batch_size=10000) as bat:
                for data in data_list:
                    rowkey = data[0]
                    try:
                        bat.put(rowkey, data[1])
                    except Exception as e:
                        print("ERROR:put data into hbase fail:%s" % e)

    def put_data_multiprocess(self):
        self.makeDataList()
        batch_size = len(self.data_list) // N_PROCESS + 1

        process_list = []
        for idx in range(N_PROCESS):
            proc = multiprocessing.Process(target=self.put_data,
                                           args=(self.data_list[idx * batch_size: (idx + 1) * batch_size],))
            process_list.append(proc)

        map(lambda proc: proc.start(), process_list)
        map(lambda proc: proc.join(), process_list)


if __name__ == '__main__':
    InputDataHbaseDemo(table_name='test_1').put_data_multiprocess()
