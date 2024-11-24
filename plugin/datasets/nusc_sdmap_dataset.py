from.base_dataset import BaseMapDataset
from.nusc_dataset import NuscDataset
from .map_utils.nuscmap_extractor import NuscMapExtractor
from mmdet.datasets import DATASETS
import numpy as np
from .visualize.renderer import Renderer
import mmcv
from time import time
from pyquaternion import Quaternion
import math
import os
import numpy as np

import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from torch.utils.data import Dataset
from .data_osm.rasterize import preprocess_map, preprocess_osm_map
from .data_osm.const import CAMS, NUM_CLASSES, IMG_ORIGIN_H, IMG_ORIGIN_W
from .data_osm.vector_map import VectorizedLocalMap
from .data_osm.lidar import get_lidar_data
from .data_osm.image import normalize_img, img_transform
from .data_osm.utils import label_onehot_encoding
from .data_osm.av2_dataset import AV2PMapNetSemanticDataset

def pad_or_trim_to_np(x, shape, pad_val=0):
  shape = np.asarray(shape)
  pad = shape - np.minimum(np.shape(x), shape)
  zeros = np.zeros_like(pad)
  x = np.pad(x, np.stack([zeros, pad], axis=1), constant_values=pad_val)
  return x[:shape[0], :shape[1]]

@DATASETS.register_module()
class PMapNetDataset(BaseMapDataset):
    """NuScenes map dataset class.

    Args:
        ann_file (str): annotation file path
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        eval_config (Config): evaluation config
        meta (dict): meta information
        pipeline (Config): data processing pipeline config
        interval (int): annotation load interval
        work_dir (str): path to work dir
        test_mode (bool): whether in test mode
    """
    
    def __init__(self, data_root,
                sd_map_path,
                thickness,
                angle_class,
                mask_flag,
                mask_ratio, # random ratio
                mask_patch_h,
                mask_patch_w, **kwargs):
        super().__init__(**kwargs)
        patch_h = self.roi_size[0]
        patch_w = self.roi_size[1]
        canvas_h = 100
        canvas_w = 50

        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.thickness = thickness
        self.angle_class = angle_class
        self.mask_config = [mask_flag, mask_ratio, mask_patch_h, mask_patch_w]

        self.map_extractor = NuscMapExtractor(data_root, self.roi_size)
        self.renderer = Renderer(self.cat2id, self.roi_size, 'nusc')
        self.vector_map = VectorizedLocalMap(data_root, patch_size=self.patch_size, canvas_size=self.canvas_size, sd_map_path=sd_map_path)
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        
        start_time = time()
        ann = mmcv.load(ann_file)
        samples = ann[::self.interval]
        
        print(f'collected {len(samples)} samples in {(time() - start_time):.2f}s')
        self.samples = samples
    
    def get_vectors(self, sample):
        # location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location'] 
        location = sample['location']

        # vectors, polygon_geom, osm_vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
        vectors, polygon_geom, osm_vectors = self.vector_map.gen_vectorized_samples(location, sample['e2g_translation'], sample['e2g_rotation'])
        
        return vectors, polygon_geom, osm_vectors
    
    # todo
    def get_semantic_map(self, sample):
        vectors, _, osm_vectors = self.get_vectors(sample) 
        osm_masks, _ = preprocess_osm_map(osm_vectors, self.patch_size, self.canvas_size)

        # instance_masks, forward_masks, backward_masks, instance_mask_map = preprocess_map(
        #     self.data_conf, vectors, self.patch_size, self.canvas_size, 
        #     NUM_CLASSES, self.thickness, self.angle_class)
        
        instance_masks, forward_masks, backward_masks, instance_mask_map = preprocess_map(
            self.mask_config, vectors, self.patch_size, self.canvas_size, 
            NUM_CLASSES, self.thickness, self.angle_class)

        masked_map = instance_mask_map != 0
        masked_map = torch.cat([(~torch.any(masked_map, axis=0)).unsqueeze(0), masked_map])

        semantic_masks = instance_masks != 0
        semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks]) 
        instance_masks = instance_masks.sum(0)
        forward_oh_masks = label_onehot_encoding(forward_masks, self.angle_class+1)
        backward_oh_masks = label_onehot_encoding(backward_masks, self.angle_class+1)
        direction_masks = forward_oh_masks + backward_oh_masks 
        direction_masks = direction_masks / direction_masks.sum(0)
        return semantic_masks, instance_masks, forward_masks, backward_masks, direction_masks, osm_masks, osm_vectors, masked_map

    def get_sample(self, idx):
        """Get data sample. For each sample, map extractor will be applied to extract 
        map elements. 

        Args:
            idx (int): data index

        Returns:
            result (dict): dict of input
        """

        sample = self.samples[idx]
        location = sample['location']
        
        map_geoms = self.map_extractor.get_map_geom(location, sample['e2g_translation'], 
                sample['e2g_rotation'])

        map_label2geom = {}
        for k, v in map_geoms.items():
            if k in self.cat2id.keys():
                map_label2geom[self.cat2id[k]] = v
        
        ego2img_rts = []
        for c in sample['cams'].values():
            extrinsic, intrinsic = np.array(
                c['extrinsics']), np.array(c['intrinsics'])
            ego2cam_rt = extrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt)
            ego2img_rts.append(ego2cam_rt)

        semantic_masks, instance_masks, _, _, direction_masks, osm_masks, osm_vectors, masked_map = self.get_semantic_map(sample)

        # if sample['sample_idx'] == 0:
        #     is_first_frame = True
        # else:
        #     is_first_frame = self.flag[sample['sample_idx']] > self.flag[sample['sample_idx'] - 1]
        input_dict = {
            'location': location,
            'token': sample['token'],
            'img_filenames': [c['img_fpath'] for c in sample['cams'].values()],
            # intrinsics are 3x3 Ks
            'cam_intrinsics': [c['intrinsics'] for c in sample['cams'].values()],
            # extrinsics are 4x4 tranform matrix, **ego2cam**
            'cam_extrinsics': [c['extrinsics'] for c in sample['cams'].values()],
            'ego2img': ego2img_rts,
            'map_geoms': map_label2geom, # {0: List[ped_crossing(LineString)], 1: ...}
            'ego2global_translation': sample['e2g_translation'], 
            'ego2global_rotation': Quaternion(sample['e2g_rotation']).rotation_matrix.tolist(),
            'sample_idx': sample['sample_idx'],
            'scene_name': sample['scene_name'],
            'osm_masks': osm_masks
        }

        return input_dict
