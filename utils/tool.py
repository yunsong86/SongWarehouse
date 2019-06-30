# _*_ coding: utf-8 _*_
import re
import random
import codecs
import logging
import base64
import datetime
import os
import hashlib
import time


class Tools(object):
    """
    py2.7, py3.4
    """

    __time_format1 = re.compile("^[1-9]{1}[0-9]{3}-((0[1-9])|(1[0-2]))-((0[1-9])|([1-2][0-9])|(3[0-1])) [0-9]{2}:[0-9]{2}$")
    __time_format2 = re.compile("^[1-9]{1}[0-9]{3}-((0[1-9])|(1[0-2]))-((0[1-9])|([1-2][0-9])|(3[0-1]))$")
    __time_format3 = re.compile("^[1-9]{1}[0-9]{3}-((0[1-9])|(1[0-2]))-((0[1-9])|([1-2][0-9])|(3[0-1])) [0-9]{2}:[0-9]{2}:[0-9]{2}$")

    @staticmethod
    def random_num(size):
        random_num = ""
        numbers = "0123456789"
        for i in range(size):
            random_num += numbers[random.randint(0, len(numbers) - 1)]
        return random_num

    @staticmethod
    def random_char(size):
        random_char = ""
        numbers = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        for i in range(size):
            random_char += numbers[random.randint(0, len(numbers) - 1)]
        return random_char

    @staticmethod
    def random_char_num(size):
        random_char = ""
        numbers = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        for i in range(size):
            random_char += numbers[random.randint(0, len(numbers) - 1)]
        return random_char

    @staticmethod
    def random_range_num(start_num, end_num):
        random_range_num = random.randint(start_num, end_num)
        return random_range_num

    @staticmethod
    def regular_expression_char(start, end, content):
        key_value = None
        pattern = re.compile(r'(?<=%s)(.+?)(?=%s)'%(start, end))
        match = pattern.search(content)
        if match:
            key_value = match.group()
        return key_value

    @staticmethod
    def regular_expression_list(start, end, content):
        pattern = re.compile(r'(?<=%s)(.+?)(?=%s)'%(start, end))
        key_list = pattern.findall(content)
        return key_list

    @staticmethod
    def regular_expression_numbers(content):
        numbers = re.findall("\d+",content)
        return numbers

    @staticmethod
    def write_file(filename, content):
        file_out = codecs.open(filename, "w", "utf-8")
        file_out.write(str(content))
        file_out.close()

    @staticmethod
    def read_file(filename):
        file_out = codecs.open(filename, "r", "utf-8")
        text = file_out.read()
        file_out.close()
        return text

    @staticmethod
    def read_file_rb(filename):
        file_out = open(filename, "rb")
        text = file_out.read()
        file_out.close()
        return text

    @staticmethod
    def set_log_level(level):
        loglevel = eval("logging.%s"%level)
        logger = logging.getLogger()
        logger.setLevel(loglevel)
        ch = logging.StreamHandler()
        ch.setLevel(loglevel)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(filename)s-%(lineno)d %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    @staticmethod
    def base64_encode(intext):
        temp = intext.encode("utf-8")
        encoderstr = base64.b64encode(temp).decode()
        return encoderstr

    @staticmethod
    def current_timestamp_str():
        current_time = datetime.datetime.now()
        return current_time.strftime("%Y%m%d%H%M%S")

    @staticmethod
    def remove_file(file_path):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logging.exception("Removing file issue")

    @staticmethod
    def dumps_msg_to_file(dump_foler, data_text):
        file_path = None
        try:
            temp_path = "%s%s.dat"%(dump_foler, Tools.random_char_num(20))
            Tools.write_file(temp_path, data_text)
            file_path = temp_path
        except Exception as e:
            logging.exception("Error when writing message to file")
        return file_path

    @staticmethod
    def list_folder_files(path):
        filenames = []
        if os.path.exists(path):
            parents = os.listdir(path)
            for item in parents:
                if os.path.isfile(os.path.join(path, item)):
                    filenames.append(item)
        return filenames

    @staticmethod
    def set_md5(char_val):
        """
        Unicode-objects must be encoded before hashing
        :param char_val:
        :return:
        """
        m = hashlib.md5()
        m.update(char_val)
        return m.hexdigest()

    @staticmethod
    def set_sha1(char_val):
        """
        Unicode-objects must be encoded before hashing
        :param char_val:
        :return:
        """
        sh = hashlib.sha1()
        sh.update(char_val)
        return sh.hexdigest()

    @staticmethod
    def isvalid_pub_time(time_str):
        result = False
        if Tools.__time_format1.search(time_str) or Tools.__time_format2.search(time_str) or Tools.__time_format3.search(time_str):
            result = True
        else:
            result = False
        return result

    # 获取指定位数的字符型时间戳
    @staticmethod
    def timestamp(size):
        result = ""
        if isinstance(size, int):
            temp_result = str(time.time()).replace(".", "")
            if size <= 16:
                result = temp_result[:size]
            else:
                result = "{0}{1}".format(temp_result, "0"*(size - 16))
        return result

    @staticmethod
    def days_before(number):
        """
        获取数日前的日期
        :param number:
        :return:
        """
        today = datetime.date.today()    # 获取当前的日期
        timedelta = datetime.timedelta(days=number)    # 一天的时间间隔
        result = (today-timedelta).strftime("%Y-%m-%d %H:%M")
        return result

    @staticmethod
    def trim_char(content):
        keywords = [" ", " ", "　", " ", "\r", "\n", "\t"]
        for key in keywords:
            content = content.replace(key, "")
        return content.strip()
