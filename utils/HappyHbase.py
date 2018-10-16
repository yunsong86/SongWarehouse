#!/usr/bin/env python
# -*- coding: utf-8 -*-

import happybase

"""
pip install happybase

hbase-daemon.sh start thrift
or 
nohup hbase thrift start &

"""
class HappyHbase(object):
    """
     :param str name:table name
     :param str row: the row key
     :param list_or_tuple columns: list of columns (optional)
    """

    def __init__(self, host, port=9090):
        self.conn = happybase.Connection(host, port=port, autoconnect=False)
        self.conn.open()
        # hbase_pool = happybase.ConnectionPool(host=host, port=port, size=100)
        # with hbase_pool.connection() as connection:
        #     self.conn =connection

    def list_tables(self):
        tabels = self.conn.tables()
        return tabels

    def _table(self, name):
        table = self.conn.table(name)
        return table

    def create(self, table_name=None, row_key=None):
        """
        :param table_name: str
        :param row_key: dict
        ::
            row_key = {
                'cf1': dict(max_versions=10),
                'cf2': dict(max_versions=1, block_cache_enabled=False),
                'cf3': dict(),  # use defaults
            }
        :return:
        """

        self.conn.create_table(table_name, row_key)

    def get_cell(self, table_name=None, row_key=None, column=None, versions=1, include_timestamp=True):
        """
        :return: list
        """
        return self._table(table_name).cells(row_key, column, versions=versions, include_timestamp=include_timestamp)

    def families(self, name):
        """
        :return: dict
        """
        return self.conn.table(name).families()

    def put(self, name, row, kw):
        self._table(name).put(row, kw)

    def get_row(self, table_name=None, row_key=None):
        """
        :return: dict
        """
        return self._table(table_name).row(row_key)

    def get_rows(self, table_name=None, row_key_list=None):
        """
        :return: dict
        """
        return self._table(table_name).rows(row_key_list)

    def get_column(self, table_name, row_key, columns):
        """
        :return: dict
        """
        return self._table(table_name).row(row_key, columns)

    def scan_table(self, table_name=None):
        res = self.conn.table(table_name).scan()
        return res

    def close(self):
        self.conn.close()
