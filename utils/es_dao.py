# -*- coding: UTF-8
import logging
from elasticsearch import Elasticsearch
import elasticsearch.exceptions as es_exception


class ESDBDAO(object):

    def __init__(self, config):
        """
        建立es链接
        :param config:
        """
        self.config = config
        self._es = Elasticsearch(hosts=self.config["hosts"])

    def insert(self, index, doc_type, id, body={}, params=None):
        return self._es.index(index=index, doc_type=doc_type, id=id, body=body)

    def update(self, index, doc_type, id, body={}, params=None):
        return self._es.update(index=index, doc_type=doc_type, id=id, body=body)

    def delete(self, index, doc_type, id):
        return self._es.delete(index=index, doc_type=doc_type, id=id, ignore=404)
