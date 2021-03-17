'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: ZHANG XIANRUI
LastEditTime: 2021-03-16 17:36:20
'''
import logging
import json
import os
from pathlib import Path
from operator import itemgetter
import shutil
from functools import wraps
import pickle


@wraps(json.load)
def load_json(filenmae, **kwargs):
    try:
        with open(filenmae, 'r') as f:
            return json.load(f, **kwargs)
    except Exception as e:
        logging.exception(e)


@wraps(json.dump)
def dump_json(dict, filename, **kwargs):
    try:
        with open(filename, 'w') as f:
            json.dump(dict, f, **kwargs)
    except Exception as e:
        logging.exception(e)


@wraps(pickle.load)
def load_pickle(filenmae, **kwargs):
    try:
        with open(filenmae, 'rb') as f:
            return pickle.load(f, **kwargs)
    except Exception as e:
        logging.exception(e)


@wraps(pickle.dump)
def dump_pickle(data, filename, **kwargs):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f, **kwargs)
    except Exception as e:
        logging.exception(e)


def get_values(dic, keys):
    return itemgetter(*keys)(dic)


def is_empty(path):
    if not isinstance(path, str):
        path = str(path)
    flag = len(os.listdir(path)) == 0
    return flag


def reset_dir(path, clear_dir):
    '''
    如果文件夹不存在就创建，如果文件存在是否清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    if Path(path).exists():
        if clear_dir:
            shutil.rmtree(path)
            Path(path).mkdir(parents=True)
        else:
            return
    else:
        Path(path).mkdir(parents=True)


class Interval(object):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __contains__(self, item):
        return self.lower <= item <= self.upper


def interval(lower, upper):
    return Interval(lower, upper)
