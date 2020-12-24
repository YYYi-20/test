'''
Descripttion: python project
version: 0.1
Author: XRZHANG
LastEditors: XRZHANG
LastEditTime: 2020-12-23 23:32:19
'''
'''
This is a Dataset generator for slide images.
'''
import logging
import math
from pathlib import Path

import cv2
from collections import defaultdict
import numpy as np
from openslide import ImageSlide, OpenSlide, deepzoom
from PIL import Image
from torch.utils.data import Dataset

from .image_utli import *
from .normalize_staining import normalize_staining
from .utils import dump_json


def open_slide(filename, image=False):
    """Open a whole-slide image filenmae, PIL filename/object.

    Return an OpenSlide object for whole-slide or PIL images."""
    if isinstance(filename, Path):
        filename = filename.__fspath__()
    if image:
        return ImageSlide(filename)
    else:
        return OpenSlide(filename)


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


class TrainZoomGenerator():
    #  必须方形
    def __init__(self,
                 slide,
                 tile_size,
                 stride_size,
                 resolution,
                 limit_bounds=True):
        self.slide = slide
        self.tile_size = tile_size
        if stride_size is None:
            stride_size = tile_size  # int
        self.stride_size = stride_size  # must int
        self.resolution = resolution
        self.limit_bounds = limit_bounds
        self._properties = dict(self.slide.properties)
        self._bg_color = '#' + slide.properties['openslide.background-color']

        mpp_x = float(slide.properties['openslide.mpp-x'])
        mpp_y = float(slide.properties['openslide.mpp-y'])
        mpp = (mpp_x + mpp_y) / 2
        # level_downsamples = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
        mpps = np.asarray(slide.level_downsamples) * mpp
        self.level = np.abs(mpps - self.resolution).argmin()
        logging.info(f'mpp={mpp}, so we use L{self.level}')
        self.tile_scale_factor = int(slide.level_downsamples[self.level])
        self.mask_scale_factor = int(self.tile_scale_factor * stride_size)

        self._l0_offset = (0, 0)
        self._l0_dimensions = slide.dimensions
        if limit_bounds:
            self._l0_offset = tuple(
                int(slide.properties[prop])
                for prop in ('openslide.bounds-x', 'openslide.bounds-y'))
            self._l0_dimensions = tuple(
                int(slide.properties[prop])
                for prop in ('openslide.bounds-width',
                             'openslide.bounds-height'))
        self._l0_size = int(self.tile_scale_factor * tile_size)
        # self._l0_stride_size = int(self.tile_scale_factor * stride_size)

        self._mask_offset = tuple(
            int(i / self.mask_scale_factor) for i in self._l0_offset)
        self._mask_dimensions = tuple(
            math.ceil(i / self.mask_scale_factor) for i in self._l0_dimensions)
        self._mask_size = int(self._l0_size / self.mask_scale_factor)
        # self._mask_stride_size = 1

    @property
    def property(self):
        return self._property

    def generate_mask(self,
                      anno_names,
                      anno_coords,
                      name_label_dict=None,
                      anno_offset=(0, 0)):
        """Coordations must be 2-D list or ndarray like [[w1,h1],[w2,h2],[w3,h3]].

        Args:
            names_coords ([type]): [description]
            name_label ([type], optional): [description]. Defaults to dict( zip(['Tumor', 'Fiber', 'Lymp', 'Folli'], [1, 2, 3, 4])).

        Returns:
            [type]: [description]
        """
        w, h = tuple(
            math.ceil(i / self.mask_scale_factor)
            for i in self.slide.dimensions)
        mask = np.zeros((h, w)).astype('uint8')
        # the init mask, and all the value is 0, cv2 shape is w*h
        annotations = defaultdict(list)
        for name, coord in zip(anno_names, anno_coords):
            annotations[name].append(coord)

        name_label = sorted(name_label_dict.items(),
                            key=lambda x: x[1],
                            reverse=False)
        for name, label in name_label:
            coords = annotations[name]
            if len(coords) > 0:
                for coord in coords:
                    abs_coord = np.asarray(coord) + np.asarray(anno_offset)
                    mask_coord = (abs_coord /
                                  self.mask_scale_factor).astype('int64')
                    cv2.fillPoly(mask, [mask_coord], (label))
        # k = 1
        # for name, coord in zip(anno_names, anno_coords):
        #     if name_label_dict is None:
        #         label = k
        #     else:
        #         label = name_label_dict.get(name, 0)
        #     # plot a polygon
        #     # (w,h)坐标列表
        #     abs_coord = np.asarray(coord) + np.asarray(anno_offset)
        #     mask_coord = (abs_coord / self.mask_scale_factor).astype('int64')
        #     cv2.fillPoly(mask, [mask_coord], (label))
        #     k += 1

        self.mask = mask
        if self.limit_bounds:
            w_, h_ = self._mask_offset
            w, h = self._mask_dimensions
            self.mask = mask[h_:h_ + h, w_:w_ + w]
            shape = self.mask.shape
            if shape != (h, w):
                row_num, col_num = max(h - shape[0], 0), max(w - shape[1], 0)
                self.mask = np.pad(self.mask, ((0, row_num), (0, col_num)),
                                   'constant',
                                   constant_values=0)
        return self.mask

    def get_tile(self, address):
        """Get the (w,h) tile.

        Args:
            address ([type]): [description]

        Returns:
            [type]: [description]
        """
        w, h = address
        tile_mask = self.mask[h:h + self._mask_size, w:w + self._mask_size]
        w = w * self.mask_scale_factor + self._l0_offset[0]
        h = h * self.mask_scale_factor + self._l0_offset[1]
        size = (self.tile_size, self.tile_size)
        tile = self.slide.read_region((w, h), self.level, size)
        bg = Image.new('RGB', tile.size, self._bg_color)
        tile = Image.composite(tile, bg, tile)
        return tile_mask, tile

    def save_all_tiles(self,
                       out_dir,
                       bw_thres=230,
                       bw_ratio=0.5,
                       mask_ratio=0.5):
        w_, h_ = self._mask_dimensions
        w_idxs, h_idxs = np.meshgrid(range(w_ - self._mask_size),
                                     range(h_ - self._mask_size))
        w_idxs, h_idxs = w_idxs.flatten(), h_idxs.flatten()
        for w, h in zip(w_idxs, h_idxs):
            tile_mask, tile = self.get_tile((w, h))
            fractions = np.bincount(tile_mask.flatten()) / tile_mask.size
            label = np.argmax(fractions)
            if label != 0 and fractions[label] > mask_ratio:
                assert tile.size == (self.tile_size, self.tile_size)
                gray = pil_to_np(tile.convert('L'))
                bw = np.where(gray < bw_thres, 0, 1)
                if np.average(bw) < bw_ratio:
                    try:
                        normalize_staining(
                            pil_to_np(tile),
                            Path(out_dir) / f'l{label}_w{w}_h{h}.jpeg')
                    except Exception as e:
                        logging.info('{e}\t color normalize error')
