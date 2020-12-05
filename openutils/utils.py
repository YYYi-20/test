'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: XRZHANG
LastEditTime: 2020-12-03 17:38:13
'''
import json
import os
from operator import itemgetter


def load_json(filenmae):
    try:
        with open(filenmae, 'r') as f:
            return json.load(fp=f)
    except:
        print('error in loading json')


def dump_json(dict, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(dict, f)
    except:
        print('error in saving json')


def get_values(dic, keys):
    return itemgetter(*keys)(dic)


def is_empty(path):
    if not isinstance(path, str):
        path = str(path)
    flag = len(os.listdir(path)) == 0
    return flag