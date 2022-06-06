#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/6/5 14:49
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from textMG.APIs.api_loggers.api_logger import logger


#get MYSQL_URL
def get_mysqlURL(pathToUrl):
    with open(pathToUrl, 'r') as f:
        line = f.readlines()
        return line[0].strip()


MYSQL_URL = get_mysqlURL(pathToUrl='/home/victor/mysql_url.txt')
POOL_SIZE = 3
POOL_RECYCLE = 3600
POOL_TIMEOUT = 15
MAX_OVERFLOW = 2
CONNECT_TIMEOUT = 60

class Database():
    def __init__(self) -> None:
        self.connection_is_active = False
        self.engine = None

    def get_db_connection(self):
        if self.connection_is_active == False:
            connect_args = {"connect_timeout":CONNECT_TIMEOUT}
            try:
                self.engine = create_engine(MYSQL_URL, pool_size=POOL_SIZE, pool_recycle=POOL_RECYCLE,
                        pool_timeout=POOL_TIMEOUT, max_overflow=MAX_OVERFLOW, connect_args=connect_args)
                return self.engine
            except Exception as e:
                logger.critical('Error connecting to DB', exc_info=1)
        return self.engine

    def get_db_session(self, engine):
        try:
            Session = sessionmaker(bind=engine)
            session = Session()
            return session
        except Exception as e:
            logger.critical('Error getting DB session', exc_info=1)
            return None