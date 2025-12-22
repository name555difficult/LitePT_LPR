
import os
import pickle
import numpy as np  
from typing import Dict
import open3d as o3d

from .base_datasets import TrainingTuple, EvaluationTuple, PointCloudLoader
from .builder import DATASETS
from .defaults import DefaultDataset

class CSWildPlacesPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        # Returns Nx3 ndarray
        file_path = os.path.join(file_pathname)
        assert os.path.splitext(file_path)[-1] == ".pcd"
        pc_o3d = o3d.io.read_point_cloud(file_path)
        pc = np.asarray(pc_o3d.points, dtype=np.float64)
        pc = np.float32(pc)
        
        return pc

@DATASETS.register_module()
class WildPlacesDataset(DefaultDataset):
    def __init__(self, **kwargs):
        self.pc_loader = CSWildPlacesPointCloudLoader()
        self.query_filepath = os.path.join(kwargs['data_root'], kwargs['split'])
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        super(WildPlacesDataset, self).__init__(**kwargs)
        self.dense = kwargs.get('dense', False)

    def get_positives(self, idx):
        return self.data_list[idx].positives

    def get_non_negatives(self, idx):
        return self.data_list[idx].non_negatives

    def get_data_list(self):
        queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))
        return queries

    def get_pose(self, idx):
        if hasattr(self.data_list[idx], 'pose'):
            return self.data_list[idx].pose
        else:
            return None

    def get_data(self, idx):
        idx = idx % len(self.data_list)
        # Load point cloud and apply transform
        file_pathname = os.path.join(self.data_root, self.data_list[idx].rel_scan_filepath)
        if self.dense:
            file_pathname = file_pathname.replace('Clouds_downsampled', 'Clouds')
        else:
            pass
            
        query_pc = self.pc_loader(file_pathname)

        strength = np.ones((query_pc.shape[0], 1), dtype=np.float32) #BUG: add strength feature

        data_dict = dict(
            coord=query_pc,
            strength=strength,
            name=self.get_data_name(idx),
            positives=self.get_positives(idx),
            non_negatives=self.get_non_negatives(idx),
            raw_coord = query_pc,
            # pose=self.get_pose(idx),
            label=int(idx),
        )
        
        if self.get_pose(idx) is not None:
            data_dict['pose'] = self.get_pose(idx)

        return data_dict

    def get_data_name(self, idx):
        # return data name for lidar seg, optimize the code when need to support detection
        return self.data_list[idx % len(self.data_list)].timestamp

@DATASETS.register_module()
class WildPlacesEvalDataset(DefaultDataset):
    def __init__(self, **kwargs):
        self.pc_loader = CSWildPlacesPointCloudLoader()
        self.data_dict = kwargs['data_dict']
        self.dense = kwargs.get('dense', False)
        super(WildPlacesEvalDataset, self).__init__(**kwargs)

    def get_data_list(self):
        return self.data_dict

    def get_data(self, idx):
        # Load point cloud and apply transform
        file_pathname = os.path.join(self.data_root, self.data_list[idx]['query'])
        if self.dense:
            file_pathname = file_pathname.replace('Clouds_downsampled', 'Clouds')
        else:
            pass
        query_pc = self.pc_loader(file_pathname)

        strength = np.ones((query_pc.shape[0], 1), dtype=np.float32) #BUG: add strength feature

        data_dict = dict(
            coord=query_pc,
            strength=strength,
            name=self.get_data_name(idx),
            label=int(idx),
        )

        return data_dict

    def get_data_name(self, idx):
        # return data name for lidar seg, optimize the code when need to support detection
        return self.data_list[idx % len(self.data_list)]['timestamp']