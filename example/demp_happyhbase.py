#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-10-16 下午7:20
# @Author  : YunSong
# @File    : demp_happyhbase.py
# @Software: PyCharm

import time
from utils.HappyHbase import HappyHbase

hap = HappyHbase(host='127.0.0.1', port=9090)

for r in hap.list_tables():
    print(r)

date = {
    "property": dict(max_versions=3),
    "click_behavior": dict(max_versions=3),

}

table_name = 'test_2'
hap.create(table_name, date)

date2 = {
    "property:pcGraphic": "123",
    "property:birthday": "86",
    "click_behavior:category": "tiyu",
    "click_behavior:lda_keywords": "lda keyword",
}

for j in range(50):
    hap.put(table_name, "rowkey1", date2)
    time.sleep(2)

res = hap.families(table_name)
print(res)
#
print(hap.get_row(table_name=table_name, row_key="rowkey1"))
print(hap.get_rows(table_name=table_name, row_key_list=["rowkey1", "rowkey2"]))
#
#
print(hap.get_column(table_name=table_name, row_key="rowkey1", columns=["info"]))

i = 0
for item in hap.get_cell(table_name=table_name, row_key='rowkey1', column="property:pcGraphic"):
    i += 1
    print(i)
    print(item)

print(hap.get_cell(table_name=table_name, row_key='rowkey1', column="property:birthday", versions=10))
#

for r in hap.scan_table(table_name):
    print(r)

res = hap.families(table_name)
print(res)
