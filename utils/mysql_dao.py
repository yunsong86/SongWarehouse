# _*_ coding: utf-8 _*_
import pymysql
import threading


class MysqlDBDAO(object):
    """
    py2.7, py3.4，注意多线程情况下，有问题
    """

    connection = None
    config = None
    mutex = threading.Lock()

    def __init__(self, config):
        """
        :param config: {'host':'', 'port': 0, 'user':'', 'password':'', 'db':'', 'autocommit': False}
        """
        self.config = config
        self.connection = pymysql.Connect(host=self.config["host"], user=self.config["user"],
                                          password=self.config["password"], database=self.config["db"],
                                          port=self.config["port"], charset="utf8",
                                          autocommit=self.config["autocommit"])

    def __get_connect(self):
        temp_cur = self.connection.cursor()
        if not temp_cur:
            raise (NameError, "Mysql connection failed!")
        else:
            return temp_cur

    def execute_query(self, sql_str, sql_params=()):
        cur = None
        try:
            cur = self.__get_connect()
            cur.execute(sql_str, sql_params)
            res_list = cur.fetchall()
            return res_list
        except Exception as e:
            raise e
        finally:
            if cur is not None:
                cur.close()

    def execute_insert(self, sql_str, sql_params=()):
        cur = None
        try:
            cur = self.__get_connect()
            cur.execute(sql_str, sql_params)
            if not self.config["autocommit"]:
                self.connection.commit()
        except Exception as e:
            raise e
        finally:
            if cur is not None:
                cur.close()

    def execute_delete(self, sql_str, sql_params=()):
        cur = None
        try:
            cur = self.__get_connect()
            cur.execute(sql_str, sql_params)
            if not self.config["autocommit"]:
                self.connection.commit()
        except Exception as e:
            raise e
        finally:
            if cur is not None:
                cur.close()

    def execute_update(self, sql_str, sql_params=()):
        cur = None
        try:
            cur = self.__get_connect()
            cur.execute(sql_str, sql_params)
            if not self.config["autocommit"]:
                self.connection.commit()
        except Exception as e:
            raise e
        finally:
            if cur is not None:
                cur.close()

    def execute_many(self, sql_str, sql_params_list):
        cur = None
        try:
            cur = self.__get_connect()
            cur.executemany(sql_str, sql_params_list)
            if not self.config["autocommit"]:
                self.connection.commit()
        except Exception as e:
            raise e
        finally:
            if cur is not None:
                cur.close()

    def close_conn(self):
        if self.connection:
            self.connection.close()
