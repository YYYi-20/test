'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: XRZHANG
LastEditTime: 2020-12-01 17:48:19
'''
import json


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
