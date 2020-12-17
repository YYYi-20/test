'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: XRZHANG
LastEditTime: 2020-12-03 15:19:41
'''
'''
This is a Dataset generator for slide images.
'''
from collections import defaultdict
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from openslide import ImageSlide, OpenSlide, deepzoom
from openslide.lowlevel import OpenSlideUnsupportedFormatError
from torch.utils.data import Dataset

from .image_utli import *
from .normalize_staining import normalize_staining
from .utils import dump_json


def open_slide(filename):
    """Open a whole-slide or regular image.

    Return an OpenSlide object for whole-slide images and an ImageSlide
    object for other types of images."""
    if isinstance(filename, Path):
        filename = str(filename)
    try:
        return OpenSlide(filename)
    except OpenSlideUnsupportedFormatError:
        return ImageSlide(filename)


class WsiDataSet(Dataset):
    """Create a datset generator or iterator for whole slide image

    Args:
        Dataset ([type]): [description]
    """
    def __init__(
        self,
        slide,
        tile_size=256,
        overlap=0,
        limit_bounds=True,
        resolution=0.5,
    ):
        """Create a slide dataset oeject and get tiles in 0.5um. same parameters
         as deepzoom.

        Args:
            slide ([type]): openslide-python object
            tile_size (int, optional): [description]. Defaults to 256.
            overlap (int, optional): [description]. Defaults to 0.
            limit_bounds (bool, optional): [description]. Defaults to True.
        """
        self.slide = slide
        self.tile_size = tile_size
        self.overlap = overlap
        self.limit_bounds = limit_bounds
        self.resolution = resolution
        self._properties = dict(self.slide.properties)
        self._preprocess()

    @property
    def property(self):
        return self._property

    def _preprocess(self):
        mpp_x = float(self._properties['openslide.mpp-x'])
        mpp_y = float(self._properties['openslide.mpp-y'])
        mpp = (mpp_x + mpp_y) / 2
        levels = np.asarray([1, 2, 3, 4, 5])
        self.level = np.abs(np.subtract(levels * mpp,
                                        self.resolution)).argmin()
        if not self.level == 1:
            logging.warn(f'mpp={mpp_x}, so we use L{self.level}')

        self.data_gen = deepzoom.DeepZoomGenerator(self.slide, self.tile_size,
                                                   self.overlap,
                                                   self.limit_bounds)
        self.dz_level = self.data_gen.level_count - 1 - self.level
        self.dz_level_dimensions = self.data_gen.level_dimensions[
            self.dz_level]
        self.num_tiles = self.data_gen.level_tiles[self.dz_level]
        self.W, self.H = np.meshgrid(range(self.num_tiles[0]),
                                     range(self.num_tiles[1]))
        self.W, self.H = self.W.flatten(), self.H.flatten()

        self._properties['l0_offset'] = self.data_gen._l0_offset
        self._properties['l_dimensions'] = self.data_gen._l_dimensions

    def __len__(self):
        return self.num_tiles[0] * self.num_tiles[1]

    def __getitem__(self, index):
        w, h = self.W[index], self.H[index]
        tile = self.data_gen.get_tile(self.dz_level, (w, h))
        return tile

    def save_properties(self, json_file):
        return dump_json(self._properties, json_file)

    def save_tiles(self, out_dir, bw_thres=230, bw_ratio=0.5):
        self.bw_thres = bw_thres
        self.bw_ratio = bw_ratio
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for w, h in zip(self.W, self.H):
            tile = self.data_gen.get_tile(self.dz_level, (w, h))
            if not tile.size == (self.tile_size, self.tile_size):
                # drop bounded tiles
                continue

            gray = pil_to_np(tile.convert('L'))
            bw = np.where(gray < bw_thres, 0, 1)
            if np.average(bw) < bw_ratio:
                try:
                    normalize_staining(pil_to_np(tile),
                                       Path(out_dir, f'w{w}_h{h}.jpeg'))
                except:
                    pass
        logging.info(f'finished tiling and normalizing to -> {out_dir}')


class XmlFormater(object):
    """
    Format converter e.g. CAMELYON16 to internal json
    """
    def caseview2json(self,
                      inxml='../tmp/19-00918 B2_Annotations.xml',
                      outjson='../tmp/19-00918 B2_Annotations.json'):
        root = ET.parse(inxml).getroot()
        annotations = root.findall('destination/annotations/annotation')
        polys = defaultdict(list)
        for anno in annotations:
            name = anno.get('name')
            points = [(int(p.get('x')), int(p.get('y')))
                      for p in anno.findall('p')]
            tmp = {'name': name, 'vertices': points}
            polys[name[-2:]].append(tmp)
        dump_json(polys, outjson)

    def caseview2asap(self,
                      inxml='../tmp/19-00918 B2_Annotations.xml',
                      outxml='../tmp/19-00918 B2_Annotations_asap.json'):
        self._caseview2json(inxml)
        polys = self.polys

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

    def json2camelyon16xml(self,
                           dict,
                           xml_path,
                           group_color=[
                               '#d0021b', '#F4FA58', '#bd10e0', '#f5a623',
                               '#234df5'
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

    def partofgroup(father_node, group_color):

        cor = group_color
        for i in range(len(group_color)):
            title = ET.SubElement(father_node, "Group")
            title.attrib = {
                "Color": cor[i],
                "PartOfGroup": "None",
                "Name": "_" + str(i)
            }
            ET.SubElement(title, "Attributes")

    def plot_area(father_node, all_area, group_, cor_):

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
