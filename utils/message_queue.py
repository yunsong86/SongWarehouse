# _*_ coding: utf-8 _*_
import queue
import logging



class MessageQueue(object):
    """
    py2.7, py3.4
    """

    queue_repo = None
    proxy_repo = None

    def __init__(self):
        self.queue_repo = queue.Queue()
        self.proxy_repo = queue.Queue()

    def push(self, obj):
        result = True
        try:
            self.queue_repo.put_nowait(obj)
        except Exception as e:
            logging.exception("Put error")
            result = False
        return result

    def pop(self):
        result = None
        try:
            result = self.queue_repo.get_nowait()
        except queue.Empty as e:
            pass
        except Exception as e:
            logging.exception("Getting message error")
        return result

    def qsize(self):
        return self.queue_repo.qsize()

    def proxy_push(self, obj):
        result = True
        try:
            self.proxy_repo.put_nowait(obj)
        except Exception as e:
            logging.exception("Put error")
            result = False
        return result

    def proxy_pop(self):
        result = None
        try:
            result = self.proxy_repo.get_nowait()
        except queue.Empty as e:
            pass
        except Exception as e:
            logging.exception("Getting message error")
        return result

    def proxy_qsize(self):
        return self.proxy_repo.qsize()


