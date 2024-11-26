import mmcv
import numpy as np
from os import path as osp
from pyquaternion import Quaternion
import argparse
from nusc_split import TRAIN_SCENES, VAL_SCENES

from shapely.geometry import LineString, box, Polygon
from shapely import ops, strtree
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from utils_data_prepare import split_collections, get_drivable_area_contour, \
        get_ped_crossing_contour
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Union
import torch
import matplotlib.pyplot as plt


nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

FAIL_SCENES = ['scene-0499', 'scene-0502', 'scene-0515', 'scene-0517']

def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        help='specify the root path of dataset')
    parser.add_argument(
        '--newsplit',
        action='store_true')
    parser.add_argument(
        '-v','--version',
        choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
        default='v1.0-trainval')
    
    args = parser.parse_args()
    return args
def _union_ped(ped_geoms: List[Polygon]) -> List[Polygon]:
    ''' merge close ped crossings.
        
    Args:
            ped_geoms (list): list of Polygon
    
    Returns:
        union_ped_geoms (Dict): merged ped crossings 
    '''

    def get_rec_direction(geom):
        rect = geom.minimum_rotated_rectangle
        rect_v_p = np.array(rect.exterior.coords)[:3]
        rect_v = rect_v_p[1:]-rect_v_p[:-1]
        v_len = np.linalg.norm(rect_v, axis=-1)
        longest_v_i = v_len.argmax()

        return rect_v[longest_v_i], v_len[longest_v_i]

    tree = strtree.STRtree(ped_geoms)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))

    final_pgeom = []
    remain_idx = [i for i in range(len(ped_geoms))]
    for i, pgeom in enumerate(ped_geoms):

        if i not in remain_idx:
            continue
            # update
        remain_idx.pop(remain_idx.index(i))
        pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
        final_pgeom.append(pgeom)

        for o in tree.query(pgeom):
            o_idx = index_by_id[id(o)]
            if o_idx not in remain_idx:
                continue

            o_v, o_v_norm = get_rec_direction(o)
            cos = pgeom_v.dot(o_v)/(pgeom_v_norm*o_v_norm)
            if 1 - np.abs(cos) < 0.01:  # theta < 8 degrees.
                final_pgeom[-1] =\
                    final_pgeom[-1].union(o)
                    # update
                remain_idx.pop(remain_idx.index(o_idx))

    results = []
    for p in final_pgeom:
        results.extend(split_collections(p))
    return results
'''
def get_map_geom(location: str, 
                 e2g_translation: Union[List, NDArray],
                 e2g_rotation: Union[List, NDArray]) -> Dict[str, List[Union[LineString, Polygon]]]:
    roi_size = (60, 30)
    patch_box = (e2g_translation[0], e2g_translation[1], roi_size[1], roi_size[0])
    rotation = Quaternion(e2g_rotation)
    yaw = quaternion_yaw(rotation) / np.pi * 180

        # get dividers
    lane_dividers = self.map_explorer[location]._get_layer_line(
                patch_box, yaw, 'lane_divider')
        
    road_dividers = self.map_explorer[location]._get_layer_line(
                    patch_box, yaw, 'road_divider')
        
    all_dividers = []
    for line in lane_dividers + road_dividers:
        all_dividers += split_collections(line)

        # get ped crossings
    ped_crossings = []
    ped = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'ped_crossing')
        
    for p in ped:
        ped_crossings += split_collections(p)
        # some ped crossings are split into several small parts
        # we need to merge them
    ped_crossings = self._union_ped(ped_crossings)
        
    ped_crossing_lines = []
    for p in ped_crossings:
            # extract exteriors to get a closed polyline
        line = get_ped_crossing_contour(p, self.local_patch)
        if line is not None:
            ped_crossing_lines.append(line)

        # get boundaries
        # we take the union of road segments and lanes as drivable areas
        # we don't take drivable area layer in nuScenes since its definition may be ambiguous
    road_segments = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'road_segment')
    lanes = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'lane')
    union_roads = ops.unary_union(road_segments)
    union_lanes = ops.unary_union(lanes)
    drivable_areas = ops.unary_union([union_roads, union_lanes])
        
    drivable_areas = split_collections(drivable_areas)
        
        # boundaries are defined as the contour of drivable areas
    boundaries = get_drivable_area_contour(drivable_areas, roi_size)

    return dict(
        divider=all_dividers, # List[LineString]
        ped_crossing=ped_crossing_lines, # List[LineString]
        boundary=boundaries, # List[LineString]
        drivable_area=drivable_areas, # List[Polygon],
    )'''

def sample_fn(line):
    
    thresh = 0.2
    sample_num = 20
    coords_dim = 2
    roi_size = (60, 30)
    attribute = []
    while True:
        line = line.simplify(thresh, preserve_topology=True)
        coords = line.coords.xy
        coords = np.concatenate(coords).reshape((2, -1)).T
        num_pts = coords.shape[0]
        if num_pts <= sample_num:
            break
        thresh += 0.01
        
    assert num_pts <= sample_num, 'num_pts > sample_num after simplify'
        
    line_length = (coords[1:] - coords[:-1])
    line_length = np.sqrt(line_length[:, 0] ** 2 + line_length[:, 1] ** 2)
      
    num_interval = (sample_num - 1) * line_length / line_length.sum()
    num_interval = np.round(num_interval).astype(np.int32)
    num_interval = np.clip(num_interval, 1, None)

    sample_length = line_length / num_interval
    while(num_interval.sum() < sample_num - 1) :
        num_interval[sample_length.argmax()] += 1
        sample_length = line_length / num_interval
    while(num_interval.sum() > sample_num - 1) :
        idxs = np.argsort(sample_length)
        for i in range(len(num_interval)):
            idx = idxs[i]
            if num_interval[idx] == 1:
                pass
            else:
                num_interval[idx] -= 1
                break
        sample_length = line_length / num_interval

    assert num_interval.sum() == sample_num - 1, 'interval not equal to sample_num - 1'
    assert num_interval[0] > 0 and num_interval[-1] > 0, 'start and end point not sampled'

    dis_to_start = np.insert(np.cumsum(line_length), 0, 0)
    distances = []
    for i in range(num_pts - 1):
        distance = np.linspace(dis_to_start[i], dis_to_start[i+1], num_interval[i]+1)
        distances = distances + distance[:-1].tolist()
    distances.append(line.length)
    distances = np.array(distances)

        #sampled_points = np.array([list(line.interpolate(distance).coords)
        #    for distance in distances]).squeeze()
    sampled_points = []
    for distance in distances:
        point = list(line.interpolate(distance).coords)
        if point in coords:
            if abs(point[0][0]) > (roi_size[0]/2 - 0.2) or \
                abs(point[0][1]) > (roi_size[1]/2 - 0.1) :
                attribute.append(3.0)
            else:
                attribute.append(2.0)
        else:
            attribute.append(1.0)
        sampled_points.append(point)
    sampled_points = np.array(sampled_points).reshape(sample_num, coords_dim)#numpy
    attribute = np.array(attribute)
        
    return sampled_points, attribute
def normalize_line(line):
    ''' Convert points to range (0, 1).
        
        Args:
            line (LineString): line
        
        Returns:
            normalized (array): normalized points.
    '''
    roi_size = (60, 30)
    roi_size = np.array(roi_size)
    origin = -np.array([roi_size[0]/2, roi_size[1]/2])

    line[:, :2] = line[:, :2] - origin

        # transform from range [0, 1] to (0, 1)
    eps = 1e-5
    line[:, :2] = line[:, :2] / (roi_size + eps)

    return line

def get_vectorized_lines(map_geoms, divider, ped_crossing, boundary, all_class):

    for label, geom_list in map_geoms.items():
           
        for geom in geom_list:
            if geom.geom_type == 'LineString':
                    
                line, attribute = sample_fn(geom)
                line = line[:, :2]
                line = normalize_line(line)
                line = line.reshape(40)
                
                all_class.append(line)
                if label ==0:
                    ped_crossing.append(line)
                elif label ==1:
                    divider.append(line)
                elif label ==2:
                    boundary.append(line)
                else:
                    raise ValueError('label value error!')
                    
            elif geom.geom_type == 'Polygon':
                    # polygon objects will not be vectorized
                continue
                
            else:
                raise ValueError('map geoms must be either LineString or Polygon!')
            
    return divider, ped_crossing, boundary, all_class

def create_nuscenes_infos_map(root_path,
                            dest_path=None,
                            info_prefix='nuscenes',
                            version='v1.0-trainval',
                            new_split=False):
    """Create info file for map learning task on nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
    """
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits

    MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
    
    cat2id = {
        'ped_crossing': 0,
        'divider': 1,
        'boundary': 2,}
    data_root = root_path
    nusc_maps = {}
    map_explorer = {}
    
    for loc in MAPS:
        nusc_maps[loc] = NuScenesMap(dataroot=data_root, map_name=loc)
        map_explorer[loc] = NuScenesMapExplorer(nusc_maps[loc])

    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    else:
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    
    if new_split:
        train_scenes = TRAIN_SCENES
        val_scenes = VAL_SCENES

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    
    train_samples, val_samples, test_samples = [], [], []
    
    map_matrix = []
    divider_matrix = []
    ped_matrix = []
    bound_matrix = []
    train_sample_idx = 0
    val_sample_idx = 0

    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)
        scene_record = nusc.get('scene', sample['scene_token'])
        log_record = nusc.get('log', scene_record['log_token'])
        location = log_record['location']
        scene_name = scene_record['name']

        #map_geoms = get_map_geom(location, pose_record['translation'], pose_record['rotation'])
        roi_size = (60, 30)
        local_patch = box(-roi_size[0] / 2, -roi_size[1] / 2, 
                roi_size[0] / 2, roi_size[1] / 2)
        e2g_translation = pose_record['translation']
        e2g_rotation = pose_record['rotation']
        patch_box = (e2g_translation[0], e2g_translation[1], roi_size[1], roi_size[0])
        rotation = Quaternion(e2g_rotation)
        yaw = quaternion_yaw(rotation) / np.pi * 180
        lane_dividers = map_explorer[location]._get_layer_line(
                patch_box, yaw, 'lane_divider')
        road_dividers = map_explorer[location]._get_layer_line(
                    patch_box, yaw, 'road_divider')
        
        all_dividers = []
        for line in lane_dividers + road_dividers:
            all_dividers += split_collections(line)
        # get ped crossings
        ped_crossings = []
        ped = map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'ped_crossing')
        for p in ped:
            ped_crossings += split_collections(p)
        # some ped crossings are split into several small parts
        # we need to merge them
        ped_crossings = _union_ped(ped_crossings)
        
        ped_crossing_lines = []
        for p in ped_crossings:
            # extract exteriors to get a closed polyline
            line = get_ped_crossing_contour(p, local_patch)
            if line is not None:
                ped_crossing_lines.append(line)

        # get boundaries
        # we take the union of road segments and lanes as drivable areas
        # we don't take drivable area layer in nuScenes since its definition may be ambiguous
        road_segments = map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'road_segment')
        lanes = map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'lane')
        union_roads = ops.unary_union(road_segments)
        union_lanes = ops.unary_union(lanes)
        drivable_areas = ops.unary_union([union_roads, union_lanes])
        
        drivable_areas = split_collections(drivable_areas)
        
        # boundaries are defined as the contour of drivable areas
        boundaries = get_drivable_area_contour(drivable_areas, roi_size)

        map_geoms = dict(
            divider=all_dividers, # List[LineString]
            ped_crossing=ped_crossing_lines, # List[LineString]
            boundary=boundaries, # List[LineString]
            drivable_area=drivable_areas, # List[Polygon],
        )

        map_label2geom = {}
        for k, v in map_geoms.items():
            if k in cat2id.keys():
                map_label2geom[cat2id[k]] = v
        map_geoms = map_label2geom
        divider_matrix, ped_matrix, bound_matrix, map_matrix = get_vectorized_lines(map_geoms, divider_matrix, ped_matrix, bound_matrix, map_matrix)
        #这里就是每一帧的车道线，只是我把它们用matrix叠起来了，循环一次读到这里就是独立的一帧,下面是画图代码
        plt.figure()
        for i in range(cluster_num):
           name = str(i)
           pts = pts_ap[i].reshape(-1,2)
           plt.plot(pts[:, 0], pts[:, 1], label=name)
           plt.savefig('vis_3class_origin_centroids_bound0.png')
        plt.close()

        if scene_name in FAIL_SCENES: continue

        # obtain 6 image's information per frame
    
    map_matrix_all = np.array(map_matrix)
    divider_all = np.array(divider_matrix)
    ped_all = np.array(ped_matrix)
    bound_all = np.array(bound_matrix)
    '''
    info = {
            'map_all': map_matrix_all,
            'divider_all': divider_all,
            'ped_all': ped_all,
            'bound_all': bound_all,
        }

    dest_path = '/nfs/dataset-ofs-mapping/ivytang_i/nfs_migration/volume-382-121/ivytang_i/datasets/nuscenes/'
    
    info_path = osp.join(dest_path, f'map_normalized_origin_data.pkl')
    print(f'saving set to {info_path}')
    mmcv.dump(info, info_path)'''
        

if __name__ == '__main__':
    data_root = 'datasets/nuScenes/'
    version = 'v1.0-trainval'
    newsplit = False

    create_nuscenes_infos_map(root_path=data_root, version=version, new_split=newsplit)