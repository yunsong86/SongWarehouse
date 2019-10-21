#!/usr/bin/env py_kp
# -*- coding: utf-8 -*-
# @Time    : 18-11-5 下午3:33
# @Author  : song
# @File    : demo_mongodb.py
# @Software: PyCharm
import pymongo

config = {'write_host': '127.0.0.1:27017',
          'read_host': '127.0.0.1:27017',
          'username': None,
          'password': None,
          'db': 'testdb'}

mongo_client_write = pymongo.MongoClient(host=config["write_host"],
                                         username=config["username"],
                                         password=config["password"],
                                         connect=False,
                                         maxPoolSize=1000)

mongo_db_write = mongo_client_write[config["db"]]

data = {'name': 'liming', 'age': '10'}

print('\n=====insert=====')
mongo_db_write['c2'].insert(data)

data_list = [{'name': 'liming2', 'age': '100'},
             {'name': 'liming3', 'age': '20'},
             {'name': 'liming4', 'age': '30'}]

print('\n=====insert_many=====')
mongo_db_write['c2'].insert_many(data_list)

print('\n=====collection_names=====')
for cn in mongo_db_write.collection_names():
    print(cn)

print('\n=====find=====')
for res in mongo_db_write['c2'].find({"name": "liming2"}):
    print(res)

print('\n=====find_one=====')
res = mongo_db_write['c2'].find_one({"name": "liming2"})
print(res)
