'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: XRZHANG
LastEditTime: 2020-12-16 22:41:41
'''
import xml.etree.ElementTree as ET

path = 'tmp/caseviewer.xml'
tree = ET.parse(path)
root = tree.getroot()

import json
import xml.etree.ElementTree as ET
import numpy as np


class XmlFormater(object):
    """
    Format converter e.g. CAMELYON16 to internal json
    """
    def camelyon16xml2json(self, inxml, outjson):
        """
        Convert an annotation of camelyon16 xml format into a json format.
        Arguments:
            inxml: string, path to the input camelyon16 xml format
            outjson: string, path to the output json format
        """
        root = ET.parse(inxml).getroot()
        annotations_tumor = \
            root.findall('./Annotations/Annotation[@PartOfGroup="Tumor"]')
        annotations_0 = \
            root.findall('./Annotations/Annotation[@PartOfGroup="_0"]')
        annotations_1 = \
            root.findall('./Annotations/Annotation[@PartOfGroup="_1"]')
        annotations_2 = \
            root.findall('./Annotations/Annotation[@PartOfGroup="_2"]')
        annotations_positive = \
            annotations_tumor + annotations_0 + annotations_1
        annotations_negative = annotations_2

        json_dict = {}
        json_dict['positive'] = []
        json_dict['negative'] = []

        for annotation in annotations_positive:
            X = list(
                map(lambda x: float(x.get('X')),
                    annotation.findall('./Coordinates/Coordinate')))
            Y = list(
                map(lambda x: float(x.get('Y')),
                    annotation.findall('./Coordinates/Coordinate')))
            vertices = np.round([X, Y]).astype(int).transpose().tolist()
            name = annotation.attrib['Name']
            json_dict['positive'].append({'name': name, 'vertices': vertices})

        for annotation in annotations_negative:
            X = list(
                map(lambda x: float(x.get('X')),
                    annotation.findall('./Coordinates/Coordinate')))
            Y = list(
                map(lambda x: float(x.get('Y')),
                    annotation.findall('./Coordinates/Coordinate')))
            vertices = np.round([X, Y]).astype(int).transpose().tolist()
            name = annotation.attrib['Name']
            json_dict['negative'].append({'name': name, 'vertices': vertices})

        with open(outjson, 'w') as f:
            json.dump(json_dict, f, indent=1)

    def json2camelyon16xml(self, dict, xml_path, group_color):

        group = ["_" + str(i) for i in range(len(group_color))]
        group_keys = dict.keys()

        assert len(group_keys) == len(group_color)
        # root and its two sub element
        root = ET.Element('ASAP_Annotations')
        sub_01 = ET.SubElement(root, "Annotations")
        sub_02 = ET.SubElement(root, "AnnotationGroups")

        # part of group. e.g. 2 color -- 2 part
        self.partofgroup(sub_02, group_color)

        # for vertices
        for i, key in enumerate(group_keys):
            group_ = group[i]
            cor_ = group_color[i]
            self.plot_area(sub_01, dict[key], group_, cor_)

        tree = ET.ElementTree(root)
        tree.write(xml_path)

    def partofgroup(self, father_node, group_color):

        cor = group_color
        for i in range(len(group_color)):
            title = ET.SubElement(father_node, "Group")
            title.attrib = {
                "Color": cor[i],
                "PartOfGroup": "None",
                "Name": "_" + str(i)
            }
            ET.SubElement(title, "Attributes")

    def plot_area(self, father_node, all_area, group_, cor_):

        for i in range(len(all_area)):
            # print(all_area)
            dict_ = all_area[i]
            title = ET.SubElement(father_node, "Annotation")
            title.attrib = {
                "Color": cor_,
                "PartOfGroup": group_,
                "Type": "Polygon",
                "Name": "_" + str(i)
            }

            coordinates = ET.SubElement(title, "Coordinates")
            dict_point = dict_["vertices"]  # all vertices of the i area

            for j in range(len(dict_point)):
                X = dict_point[j][0]
                Y = dict_point[j][1]
                coordinate = ET.SubElement(coordinates, "Coordinate")
                coordinate.attrib = {"Y": str(Y), "X": str(X), "Order": str(j)}
