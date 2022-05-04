#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor
@site: http://victor.info/
@email: victor@bai.info
@time: 2022/5/1 00:01
"""
import sys
import logging

def create_logger():
    """
    :return: logger
    """
    # create logger for "excel_processing"
    log_dir = './logs/package_text.log'
    print("created logger for 'tensorflow' at directory: {}".format(log_dir))
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.DEBUG)

    # create file handler which logs_api even debug messages
    file_handler = logging.FileHandler(log_dir, mode='w') #to append message use mode="a"
    file_handler.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s '
                                  ,datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = create_logger()

# logger.debug("Harmless debug Message")
# logger.info("Just an information")
# logger.warning("Its a Warning")
# logger.error("Did you try to divide by zero")
# logger.critical("Internet is down")