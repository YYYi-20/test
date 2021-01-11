import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
from .image_utli import *
from .utils import dump_json, load_json


class AnnotationTranformer():
    """
    Format converter e.g. CAMELYON16 to internal json
    """
    def load_qu(self, json_name='temp/19-00918%20B2.json'):
        """load geojson from qupath. return class names like ['tumor','Lymb'] and coordinates like [[1,1],[2,2],[2,3]].

        Args:
            json_name (str, optional): [description]. Defaults to 'temp/19-00918%20B2.json'.

        Returns:
            classname and coords.
        """
        allobjects = load_json(json_name)
        names = []
        coords = []
        for obj in allobjects:
            class_name = obj.get('properties',
                                 {}).get('classification',
                                         {}).get('name', 'unknown')
            names.append(class_name)
            coordinates = obj['geometry']['coordinates']

            new_list = []

            def split(li):
                for ele in li:
                    if isinstance(ele[0], list):
                        split(ele)
                    else:
                        new_list.append(ele)

            split(coordinates)
            coordinates = np.asarray(new_list)
            assert (len(coordinates.shape) == 2 and coordinates.shape[1] == 2)
            coords.append(coordinates)
        return names, coords

    def export_qu(self, names, coords, out_json, color_dict=None):
        """Transform classname list and coordnations list to genjson for qupath.
        set color for the class name by color_dict like  {'Tumor': -3670016, 'Stroma': -6895466}.

        Args:
            names ([type]): [description]
            coords ([type]): [description]
            out_json ([type]): [description]
            color_dict ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if color_dict is None:
            color_dict = {'Tumor': -3670016, 'Stroma': -6895466}
        result = []
        for name, coord in zip(names, coords):
            tmp = {
                'type': 'Feature',
                'id': 'PathAnnotationObject',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': []
                },
                'properties': {
                    'classification': {
                        'name': 'nan',
                    }
                }
            }
            tmp['geometry']['coordinates'].append(coord)
            tmp['properties']['classification']['name'] = name
            result.append(tmp)
        if out_json is not None:
            dump_json(result, out_json)
        else:
            return result

    def caseview2asap(self, inxml, outxml):
        tmp = 'tmp.json'
        self._caseview2json(inxml, tmp)
        self._json2asap(tmp, outxml)

    def caseview2qupath(self, offset, inxml, outjson=None, color_dict=None):
        """NOTE. In qupath the annotation coords have been subtracted by the offset.

        Args:
            offset ([type]): [description]
            inxml ([type]): [description]
            outjson ([type], optional): [description]. Defaults to None.
            color_dict ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        root = ET.parse(inxml).getroot()
        annotations = root.findall('./destination/annotations/annotation')
        result = []
        offset_x, offset_y = offset
        for anno in annotations:
            tmp = {
                'type': 'Feature',
                'id': 'PathAnnotationObject',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': []
                },
                'properties': {
                    'name': "Unknown",
                    'color': [51, 102, 51],
                    'classification': {
                        'name': 'noname',
                        "colorRGB": -13408717
                    }
                }
            }
            name = anno.get('name')
            anno_id, class_name = name.split(' ')[-1], name[-5:]
            points = [(int(p.get('x')) - offset_x, int(p.get('y')) - offset_y)
                      for p in anno.findall('p')]
            if len(points) == 3:
                logging.info(str(outjson) + 'only 3 poitns')
                # logging.info(str(anno))
                continue
            if points[0] != points[-1]:
                logging.info(str(anno) + 'start end mismatch')
            tmp['geometry']['coordinates'].append(points)
            tmp['properties']['name'] = anno_id
            tmp['properties']['classification']['name'] = class_name
            if color_dict is not None:
                tmp['properties']['classification']['colorRGB'] = color_dict[
                    name]
            result.append(tmp)
        if outjson != None:
            dump_json(result, outjson)
        else:
            return result

    def _caseview2json(self, inxml, outjson=None):
        root = ET.parse(inxml).getroot()
        annotations = root.findall('./destination/annotations/annotation')
        polys = defaultdict(list)
        for anno in annotations:
            name = anno.get('name')
            group = re.findall(r'[a-zA-Z]+', name)[-1]
            if not group in ['Tumor', 'Folli', 'Lymp', 'Fiber']:
                continue
            points = [(int(p.get('x')), int(p.get('y')))
                      for p in anno.findall('p')]
            tmp = {'name': re.findall(r'[0-9]+', name)[-1], 'vertices': points}
            polys[group].append(tmp)
        if outjson is not None:
            dump_json(polys, outjson)
        else:
            return polys

    def _asap2json(self, inxml, outjson):
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
        dump_json(json_dict, outjson, indent=1)

    def _json2asap(self,
                   dict,
                   xml_path,
                   group_color=[
                       '#d0021b', '#F4FA58', '#bd10e0', '#f5a623', '#234df5'
                   ]):

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