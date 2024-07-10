# -*- coding: utf-8 -*-
# @Time    : 2023/5/12 22:41
# @Author  : Guangchen Jiang
# @Email   : guangchen98.jiang@gmail.com
# @File    : src/logger/_get_logger.py
# @Software: PyCharm

import datetime
import logging
from typing import Optional

import pytz
from tzlocal import get_localzone_name

from .logging import get_logger as _get_logger

FmtRegex = "[%(levelname)s] [%(asctime)s] %(message)s - %(filename)s:%(lineno)s"


def converter(timestamp):
    dt = datetime.datetime.fromtimestamp(timestamp)
    tzinfo = pytz.timezone(get_localzone_name())
    return tzinfo.localize(dt)


class Formatter(logging.Formatter):
    """override logging.Formatter to use an aware datetime object"""

    def formatTime(self, record, datefmt=None):
        dt = converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec="milliseconds")
            except TypeError:
                s = dt.isoformat()
        return s


def get_logger(
    name: Optional[str] = None,
    level: Optional[int] = None,
    filename: Optional[str] = None,
    stacklevel: Optional[int] = 3,
) -> logging.Logger:
    logger = _get_logger(name, stacklevel)

    fmtr = Formatter(FmtRegex)
    for hdl in logger.handlers:
        hdl.setFormatter(fmtr)

    if level:
        logger.setLevel(level)
    if filename:
        file_hdlr = logging.FileHandler(filename=filename)
        file_hdlr.setLevel(logging.DEBUG)
        logger.addHandler(file_hdlr)

    return logger
