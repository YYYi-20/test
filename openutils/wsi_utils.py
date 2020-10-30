'''
This is a Dataset generator for slide images.
'''
import logging
import os
import sys

import numpy as np
import openslide
from openslide import deepzoom
from torch.utils.data import Dataset

from .image_utli import *
from .normalize_staining import normalize_staining


class WsiDataSet(Dataset):
    """Create a datset generator or iterator for whole slide image

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, slide, tile_size=256, overlap=0, limit_bounds=True):
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
        self._preprocess()

    def _preprocess(self):
        self.mpp_x = eval(
            self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X))
        mpps = np.asarray([1, 2, 3, 4, 5]) * self.mpp_x
        self.level = np.abs(np.subtract(mpps, 0.5)).argmin()
        if not self.level == 1:
            logging.warn(
                f'mpp = {self.mpp_x}, so we use level {self.level} in 0.5um')

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

    def __len__(self):
        return self.num_tiles[0] * self.num_tiles[1]

    def __getitem__(self, index):
        w, h = self.W[index], self.H[index]
        tile = self.data_gen.get_tile(self.dz_level, (w, h))
        return tile

    def save_tiles(self, save_dir, bw_thres=230, bw_ratio=0.5):
        self.bw_thres = bw_thres
        self.bw_ratio = bw_ratio
        out_dir = os.path.join(save_dir, 'um5')
        if os.path.exists(out_dir):
            if len(os.listdir(out_dir)) != 0:
                logging.error(f'files already in directory {save_dir}/um5/')
                return

        os.makedirs(out_dir, exist_ok=True)
        for w, h in zip(self.W, self.H):
            tile = self.data_gen.get_tile(self.dz_level, (w, h))
            if not tile.size == (self.tile_size, self.tile_size):
                continue

            gray = pil_to_np(tile.convert('L'))
            bw = np.where(gray < bw_thres, 0, 1)
            if np.average(bw) < bw_ratio:
                try:
                    normalize_staining(
                        pil_to_np(tile),
                        os.path.join(out_dir, f'w{w}_h{h}.jpeg'))
                except:
                    pass
                #     logging.error(f'normal error in tile w,h=({w},{h})')
        logging.info(f'finished tiling and normalizing to -> {save_dir}')
