# _*_ coding: utf-8 _*_
import pymongo


class MongoDBDAO(object):
    """
    py2.7, py3.4
    """

    mongo_client_read = None
    mongo_db_read = None
    mongo_client_write = None
    mongo_db_write = None
    config = None

    def close_connect(self):
        if self.mongo_client_write:
            self.mongo_client_write.close()
        if self.mongo_client_read:
            self.mongo_client_read.close()

    def __init__(self, config):
        """
        :param config: {'write_host': '', 'read_host': '', 'username': '', 'password': '', 'db': ''}
        """
        self.config = config
        if "write_host" in self.config.keys() and self.config["write_host"]:
            self.mongo_client_write = pymongo.MongoClient(host=self.config["write_host"])
            self.mongo_db_write = self.mongo_client_write[self.config["db"]]
            temp_auth = self.mongo_db_write.authenticate(self.config["username"], self.config["password"])

        if "read_host" in self.config.keys() and self.config["read_host"]:
            self.mongo_client_read = pymongo.MongoClient(host=self.config["read_host"])
            self.mongo_db_read = self.mongo_client_read[self.config["db"]]
            temp_auth = self.mongo_db_read.authenticate(self.config["username"], self.config["password"])

    def update_one(self, table, db_where, db_update, upsert=False):
        if not table:
            raise (RuntimeError, "[mongodb] Table name is required!")
        if "$set" not in db_update.keys() and "$addToSet" not in db_update.keys() \
                and "$setOnInsert" not in db_update.keys() and "$inc" not in db_update.keys():
            raise (ValueError, "[mongodb] Pls confirm your update statement!%s" % str(db_update))
        _table = self.mongo_db_write[table]
        write_result = _table.update_one(filter=db_where, update=db_update, upsert=upsert)
        return write_result.raw_result

    def find(self, table, query, sort=None, limit=None):
        _table = self.mongo_db_read[table]
        if limit is None:
            # result = _table.find(query, sort=[('_id', pymongo.ASCENDING)])
            result = _table.find(query, sort=sort)
        else:
            # result = _table.find(query, sort=[('_id', pymongo.ASCENDING)]).limit(limit)
            result = _table.find(query, sort=sort).limit(limit)
        return result

    def find_one(self, table, query):
        _table = self.mongo_db_read[table]
        return _table.find_one(query)

    def delete_one(self, table, query):
        _table = self.mongo_db_write[table]
        return _table.delete_one(query)

    def insert_one(self, table, data):
        _table = self.mongo_db_write[table]
        return _table.insert(data)

    def insert_many(self, table, list_data, ordered=True):
        _table = self.mongo_db_write[table]
        return _table.insert_many(list_data, ordered)

    # return CommandCursor
    def aggregate(self, table, pipeline):
        _table = self.mongo_db_read[table]
        # The aggregate() method always returns a CommandCursor. The pipeline argument must be a list.
        return _table.aggregate(pipeline)

    def find_count(self, table, query):
        _table = self.mongo_db_read[table]
        return _table.count(query)
