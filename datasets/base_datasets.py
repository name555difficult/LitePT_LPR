# Base dataset classes, inherited by dataset-specific classes
import os
import pdb
import pickle
from typing import List
from typing import Dict
import torch
import numpy as np
from torch.utils.data import Dataset

class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray, 
                 pose: np.ndarray = None, multi_scale_positives: List[np.ndarray] = None):
                #  pose: np.ndarray = None, repose_with_positives: np.ndarray = None):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        # pose: 7D pose (x, y, z, qx, qy, qz, qw)
        # repose_with_positives: ndarray of relative poses with positives
        
        assert position.shape == (2,)

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position
        self.pose = pose
        self.multi_scale_positives = multi_scale_positives
        # self.repose_with_positives = repose_with_positives

class EvaluationTuple:
    # Tuple describing an evaluation set element
    def __init__(self, timestamp: int, rel_scan_filepath: str, position: np.array):
        # position: x, y position in meters
        assert position.shape == (2,)
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.position = position

    def to_tuple(self):
        return self.timestamp, self.rel_scan_filepath, self.position


class TrainingDataset(Dataset):
    def __init__(self, dataset_path, query_filename, query_infos=None, transform=None, set_transform=None,
                 load_octree=False, octree_depth=11, full_depth=2, coordinates='cartesian'):
        # remove_zero_points: remove points with all zero coords
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.coordinates = coordinates
        self.load_octree = load_octree
        self.octree_depth = octree_depth
        self.full_depth = full_depth
        
        self.raw_queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))
        print('{} queries in the dataset'.format(len(self.raw_queries)))
        self.raw_sample_indices = list(self.raw_queries.keys())
        self.sample_indices = self.raw_sample_indices
        self.queries = self.raw_queries

        # pc_loader must be set in the inheriting class
        self.pc_loader: PointCloudLoader = None

        if query_infos is not None:
            self.query_infos_path = os.path.join(dataset_path, query_infos)
            assert os.path.exists(self.query_infos_path), 'Cannot access query infos file: {}'.format(self.query_infos_path)
            self.query_infos = pickle.load(open(self.query_infos_path, 'rb')) # dict

    def __len__(self):
        return len(self.queries)

    def _resample_dataset(self, n_samples):
        del self.sample_indices, self.queries
        # Resample the dataset to n_samples
        if n_samples < len(self.raw_queries):
            tmp = np.asarray(self.raw_sample_indices)
            self.sample_indices = tmp[
                np.random.choice(len(self.raw_sample_indices), n_samples, replace=False)
            ].tolist()
            self.queries = {k: self.raw_queries[k] for k in self.sample_indices}
        else:
            self.sample_indices = self.raw_sample_indices
            self.queries = self.raw_queries

    def __getitem__(self, ndx):
        sample_index = self.sample_indices[ndx]
        # Load point cloud and apply transform
        file_pathname = os.path.join(self.dataset_path, self.queries[sample_index].rel_scan_filepath)
        query_pc = self.pc_loader(file_pathname)
        data = torch.tensor(query_pc, dtype=torch.float)
        if self.transform is not None:
            data = self.transform(data)
        if self.load_octree:
            # Ensure no values outside of [-1, 1] exist (see ocnn documentation)
            mask = torch.all(abs(data) <= 1.0, dim=1)
            data = data[mask]
            # Also ensure this will hold if converting coordinate systems
            if self.coordinates == 'cylindrical':
                data_norm = torch.linalg.norm(data[:, :2], dim=1)[:, None]
                mask = torch.all(data_norm <= 1.0, dim=1)
                data = data[mask]
        return data, ndx

    def get_positives(self, ndx):
        sample_index = self.sample_indices[ndx]
        return self.queries[sample_index].positives

    def get_multi_scale_positives(self, ndx):
        sample_index = self.sample_indices[ndx]
        return self.queries[sample_index].multi_scale_positives

    def get_non_negatives(self, ndx):
        sample_index = self.sample_indices[ndx]
        return self.queries[sample_index].non_negatives

    def get_position(self, ndx):
        sample_index = self.sample_indices[ndx]
        return self.queries[sample_index].position.astype(np.float32)

    def get_place_id(self, ndx):
        sample_index = self.sample_indices[ndx]
        place_id = self.query_infos['sample_idx_to_place_id'][sample_index]
        place_center = np.asarray(self.query_infos['place_id_to_grid'][place_id], dtype=np.float32)

        return place_id, place_center


class EvaluationSet:
    # Evaluation set consisting of map and query elements
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set

    def save(self, pickle_filepath: str):
        # Pickle the evaluation set

        # Convert data to tuples and save as tuples
        query_l = []
        for e in self.query_set:
            query_l.append(e.to_tuple())

        map_l = []
        for e in self.map_set:
            map_l.append(e.to_tuple())
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath: str):
        # Load evaluation set from the pickle
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))

        self.query_set = []
        for e in query_l:
            self.query_set.append(EvaluationTuple(e[0], e[1], e[2]))

        self.map_set = []
        for e in map_l:
            self.map_set.append(EvaluationTuple(e[0], e[1], e[2]))

    def get_map_positions(self):
        # Get map positions as (N, 2) array
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            positions[ndx] = pos.position
        return positions

    def get_query_positions(self):
        # Get query positions as (N, 2) array
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            positions[ndx] = pos.position
        return positions


class PointCloudLoader:
    def __init__(self):
        # remove_zero_points: remove points with all zero coordinates
        # remove_ground_plane: remove points on ground plane level and below
        # ground_plane_level: ground plane level
        self.remove_zero_points = True
        self.remove_ground_plane = True
        self.ground_plane_level = None
        self.set_properties()

    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level. Must be defined in inherited classes.
        raise NotImplementedError('set_properties must be defined in inherited classes')

    def __call__(self, file_pathname):
        # Reads the point cloud from a disk and preprocess (optional removal of zero points and points on the ground
        # plane and below
        # file_pathname: relative file path
        assert os.path.exists(file_pathname), f"Cannot open point cloud: {file_pathname}"
        pc = self.read_pc(file_pathname)
        assert pc.shape[1] == 3

        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]

        if self.remove_ground_plane:
            mask = pc[:, 2] > self.ground_plane_level
            pc = pc[mask]

        return pc

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        raise NotImplementedError("read_pc must be overloaded in an inheriting class")

def get_query_database_splits(dataset_name):
    if dataset_name == 'Oxford':
        eval_database_files = ['oxford_evaluation_database.pickle', 'university_evaluation_database.pickle',
                               'residential_evaluation_database.pickle', 'business_evaluation_database.pickle']
        eval_query_files = ['oxford_evaluation_query.pickle', 'university_evaluation_query.pickle',
                            'residential_evaluation_query.pickle', 'business_evaluation_query.pickle']
    elif dataset_name == 'MulRan':
        eval_database_files = ['DCC_database.pickle', 'Sejong_database.pickle']
        eval_query_files = ['DCC_queries.pickle', 'Sejong_queries.pickle']
    elif 'CSWildPlaces' in dataset_name:
        eval_database_files = [
            'CSWildPlaces_Karawatha_evaluation_database.pickle',
            'CSWildPlaces_Venman_evaluation_database.pickle',
            'CSWildPlaces_QCAT_evaluation_database.pickle', 
            'CSWildPlaces_Samford_evaluation_database.pickle',
        ]
        eval_query_files = [
            'CSWildPlaces_Karawatha_evaluation_query.pickle',
            'CSWildPlaces_Venman_evaluation_query.pickle',
            'CSWildPlaces_QCAT_evaluation_query.pickle', 
            'CSWildPlaces_Samford_evaluation_query.pickle',
        ]
    elif 'WildPlaces' in dataset_name:
        eval_database_files = [
            'Karawatha_evaluation_database.pickle',
            'Venman_evaluation_database.pickle',
        ]
        eval_query_files = [
            'Karawatha_evaluation_query.pickle',
            'Venman_evaluation_query.pickle',            
        ]
    elif dataset_name == 'CSCampus3D':
        eval_database_files = ['umd_evaluation_database.pickle']
        eval_query_files = ['umd_evaluation_query_v2.pickle']
    else:
        raise NotImplementedError(f'Dataset {dataset_name} has no splits implemented')
    return eval_database_files, eval_query_files