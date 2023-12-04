#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
import logging


class Logger(object):
    def __init__(self, logger: logging.Logger = None, f_path=None):
        self.logger = logger
        self.f_path = f_path

    def debug(self, msg):
        if self.logger is not None:
            self.logger.debug(msg)
        elif self.f_path is not None:
            with open(self.f_path, "a+") as f:
                f.write("[Debug]:{}\n".format(msg))
        else:
            print("[Debug]:{}".format(msg))

    def info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        elif self.f_path is not None:
            with open(self.f_path, "a+") as f:
                f.write("[Info]:{}\n".format(msg))
        else:
            print("[Info]:{}".format(msg))

    def warn(self, msg):
        if self.logger is not None:
            self.logger.warning(msg)
        elif self.f_path is not None:
            with open(self.f_path, "a+") as f:
                f.write("[Warning]:{}\n".format(msg))
        else:
            print("[Warning]:{}".format(msg))

    def warning(self, msg):
        self.warn(msg)

    def error(self, msg, exc_info=True, stack_info=True):
        if self.logger is not None:
            self.logger.error(msg, exc_info=exc_info, stack_info=stack_info)
        elif self.f_path is not None:
            with open(self.f_path, "a+") as f:
                f.write("[Error]:{}\n".format(msg))
        else:
            print("[Error]:{}".format(msg))
