#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023/10/17 15:40 CST
# @Author  : Guangchen Jiang
# @Email   : guangchen98.jiang@gmail.com
# @File    : src/my_utils/logger/_matching_logger_level.py
# @Software: PyCharm

import logging

info_name_list = [
    "CRITICAL",
    "FATAL",
    "ERROR",
    "WARN",
    "WARNING",
    "INFO",
    "DEBUG",
    "NOTSET",
]


def matching_logger_level(level: str) -> int:
    assert level.upper() in info_name_list, (
        "Logger's level should be set to one of the following "
        "values: 'CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', "
        "'INFO', 'DEBUG', 'NOTSET'."
    )
    str2loglevel = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARN,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "FATAL": logging.FATAL,
        "CRITICAL": logging.CRITICAL,
    }
    return str2loglevel.get(level.upper(), logging.INFO)


if __name__ == "__main__":
    print(matching_logger_level("notset"))
