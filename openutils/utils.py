'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: XRZHANG
LastEditTime: 2020-12-03 14:16:26
'''
import json
import os
from operator import itemgetter


def load_json(filenmae):
    try:
        with open(filenmae, 'r') as f:
            return json.load(fp=f)
    except:
        return 0


def dump_json(dict, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(dict, f)
        return 1
    except:
        return 0


def get_values(dic, keys):
    return itemgetter(*keys)(dic)


def is_empty(path):
    if not isinstance(path, str):
        path = str(path)
    flag = len(os.listdir(path)) == 0
    return flag