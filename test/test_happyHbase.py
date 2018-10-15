#!/usr/bin/env python
# -*- coding: utf-8 -*-


from utils.HappyHbase import HappyHbase


def test():
    hap = HappyHbase(host='127.0.0.1', port=9090)

    for r in hap.list_tables():
        print(r)
    data = {
        "user": dict(),
        "info": dict(),
    }

    table_name = 'test_9'
    # hap.create(table_name, data)

    date = {
        "user:": "ys",
        "info:address": "zhuhai",
        "info:sex": "m",
    }
    for i in range(3):
        hap.put(table_name, "rowkey" + str(i), date)

    print(hap.get_row(table_name=table_name, row_key="rowkey1"))
    print(hap.get_column(table_name=table_name, row_key="rowkey1", columns=["info"]))
    print(hap.get_cell(table_name=table_name, row_key='rowkey1', column="info:sex"))
    for r in hap.scan_table(table_name):
        print(r)


if __name__ == '__main__':
    test()
