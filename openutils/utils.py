'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: XRZHANG
LastEditTime: 2020-12-24 15:58:48
'''
import logging
import json
import os
from pathlib import Path
from operator import itemgetter
import shutil
from functools import wraps


@wraps(json.load)
def load_json(filenmae, **kwargs):
    try:
        with open(filenmae, 'r') as f:
            return json.load(f, **kwargs)
    except Exception as e:
        logging.error(e)


@wraps(json.dump)
def dump_json(dict, filename, **kwargs):
    try:
        with open(filename, 'w') as f:
            json.dump(dict, f, **kwargs)
    except Exception as e:
        logging.error(e)


def get_values(dic, keys):
    return itemgetter(*keys)(dic)


def is_empty(path):
    if not isinstance(path, str):
        path = str(path)
    flag = len(os.listdir(path)) == 0
    return flag


def reset_dir(path):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        Path(path).mkdir(parents=True)