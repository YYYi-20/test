'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: XRZHANG
LastEditTime: 2020-12-22 20:35:42
'''
'''
This is a Dataset generator for slide images.
'''
import json
import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from openslide import ImageSlide, OpenSlide, deepzoom
from openslide.lowlevel import OpenSlideUnsupportedFormatError
from torch.utils.data import Dataset
from PIL import Image

from .image_utli import *
from .normalize_staining import normalize_staining
from .utils import dump_json, load_json


def open_slide(filename):
    """Open a whole-slide or regular image.

    Return an OpenSlide object for whole-slide images and an ImageSlide
    object for other types of images."""
    if isinstance(filename, Image.Image):
        return ImageSlide(filename)
    else:
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
        # level_downsamples = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
        mpps = np.asarray(self.slide.level_downsamples) * mpp
        self.level = np.abs(mpps - self.resolution).argmin()
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


class LabeledTile():
    def __init__(self, slide, tile_size, resolution, number=None):
        self.slide = slide
        self.tile_size = tile_size
        self.resolution = resolution
        self._properties = dict(self.slide.properties)
        self.number = number
        self._properties = dict(self.slide.properties)
        self._preprocess()
        self._bg_color = '#' + self.slide.properties.get(
            'openslide.background-color', 'ffffff')

    @property
    def property(self):
        return self._property

    def _preprocess(self):
        mpp_x = float(self._properties['openslide.mpp-x'])
        mpp_y = float(self._properties['openslide.mpp-y'])
        mpp = (mpp_x + mpp_y) / 2
        # level_downsamples = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
        mpps = np.asarray(self.slide.level_downsamples) * mpp
        self.level = np.abs(mpps - self.resolution).argmin()
        if not self.level == 1:
            logging.warn(f'mpp={mpp_x}, so we use L{self.level}')

    def get_mask(self,
                 dicts,
                 metadata=dict(
                     zip(['Tumor', 'Fiber', 'Lymp', 'Folli'], [1, 2, 3, 4])),
                 mask_level=6):
        w, h = self.slide.level_dimensions[mask_level]
        mask = np.zeros((h, w))  # cv2 size is w*h
        # get the factor of level * e.g. level 6 is 2^6
        factor = self.slide.level_downsamples[mask_level]
        # the init mask, and all the value is 0, cv2 shape is w*h
        for group, annotations in dicts.items():
            label = metadata[group]
            for anno in annotations:
                # plot a polygon
                vertices = np.array(anno["vertices"]) / factor
                vertices = vertices.astype('int64')
                cv2.fillPoly(mask, [vertices], (label))
        return mask.astype('int64')

    def save_tile(self, mask, mask_level, out_dir, bw_thres=230, bw_ratio=0.5):
        mask_factor = self.slide.level_downsamples[mask_level]
        tile_factor = self.slide.level_downsamples[self.level]
        for label in np.unique(mask):
            if label == 0:
                continue
            X_idcs, Y_idcs = np.where(mask == label)
            coord = np.c_[X_idcs, Y_idcs]  #shape is sample * 2
            mask_area_lvel_0 = len(X_idcs) * mask_factor
            tile_size_lvel_0 = self.tile_size * tile_factor
            if self.number is None:
                self.number = int(mask_area_lvel_0 / (tile_size_lvel_0**2) *
                                  10)
                # mask 面积 除以 tile面积，得到 tile个数，但是多采样10倍数
            if len(X_idcs) > self.number:
                sampled_index = np.random.randint(0,
                                                  len(X_idcs),
                                                  size=self.number)
                coord = coord[sampled_index, :]

            #  level 0下 采样点中心坐标
            coord_level_0 = (coord * mask_factor).astype('int64')
            # level 0 左上角坐标
            coord_level_0 = (coord_level_0 -
                             tile_size_lvel_0 / 2).astype('int64')

            for i in range(len(coord_level_0)):
                x, y = coord_level_0[i, 0], coord_level_0[i, 1]
                tile = self.slide.read_region((x, y), self.level,
                                              (self.tile_size, self.tile_size))
                # Apply on solid background
                # copy this from deepzoom
                bg = Image.new('RGB', tile.size, self._bg_color)
                tile = Image.composite(tile, bg, tile)

                if tile.size != (self.tile_size, self.tile_size):
                    # drop bounded tiles
                    continue

                gray = pil_to_np(tile.convert('L'))
                bw = np.where(gray < bw_thres, 0, 1)
                if np.average(bw) < bw_ratio:
                    try:
                        normalize_staining(
                            pil_to_np(tile),
                            Path(out_dir) / f'{label}_{x}_{y}.jpeg')
                    except:
                        logging.info('color normalize error')


class AnnotationFormater():
    """
    Format converter e.g. CAMELYON16 to internal json
    """
    def load_qu(self, json_name='temp/19-00918%20B2.json'):
        allobjects = load_json(json_name)
        names = []
        coords = []
        i = 0
        for obj in allobjects:
            class_name = obj['properties'].get('classification',
                                               'nan').get('name', 'nan')
            names.append(class_name)
            coords.append(obj['geometry']['coordinates'][0])
        return names, coords

    def export_qu(self, names, coords, out_json, color_dict=None):
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
                        'colorRGB': -3670016
                    },
                    'isLocked': False,
                    'measurements': []
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
                    'classification': {
                        'name': 'nan',
                        'colorRGB': -3670016
                    },
                    'isLocked': False,
                    'measurements': []
                }
            }
            name = anno.get('name')
            points = [(int(p.get('x')) - offset_x, int(p.get('y')) - offset_y)
                      for p in anno.findall('p')]
            tmp['geometry']['coordinates'].append(points)
            tmp['properties']['classification']['name'] = name
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

        with open(outjson, 'w') as f:
            json.dump(json_dict, f, indent=1)

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