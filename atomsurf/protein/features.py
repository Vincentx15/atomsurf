import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..'))

from atomsurf.utils.helpers import safe_to_torch


class Features(Data):
    """
    A class to hold features on certain nodes, that can possibly be extended through one hot
     encoding or residue-atom scattering
    """

    def __init__(self, num_nodes, res_map=None, flat_features=None,
                 named_features=None,
                 named_one_hot_features=None,
                 **kwargs):
        super(Features, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        if res_map is not None:
            self.res_map = safe_to_torch(res_map).long()
            self.num_res = int(self.res_map.max()) + 1
            self.possible_nums = {self.num_res, self.num_nodes}
        else:
            self.possible_nums = {self.num_nodes}

        # self.names = names
        self.named_features = named_features
        # self.names_oh = names_oh
        self.named_one_hot_features = named_one_hot_features
        self.flat_features = flat_features

    def sanitize_features(self, value):
        """
        Ensures value is either a Data instance or a torch.Tensor of the right size
        :param value:
        :return:
        """
        assert len(value) in self.possible_nums, (f"Trying to add a feature with {len(value)},"
                                                  f" while possible lengths are {self.possible_nums}")
        return safe_to_torch(value)

    def add_flat_features(self, value):
        assert len(value) in self.possible_nums
        if "flat_features" not in self.keys:
            self.flat_features = value
        else:
            self.flat_features = torch.cat((self.flat_features, value), 0)

    def add_named_features(self, key, value):
        value = self.sanitize_features(value)
        if "named_features" not in self.keys:
            self.named_features = {key: value}
        else:
            self.named_features[key] = value

    def add_named_oh_features(self, key, value, nclasses):
        value = self.sanitize_features(value)
        # OH encoding should involve tensors of shape (n,) or (n,1)
        assert len(value.squeeze().shape) == 1
        if "named_one_hot_features" not in self.keys:
            self.named_one_hot_features = {key: value}
            self.named_one_hot_features_nclasses = {key: nclasses}
        else:
            self.named_one_hot_features[key] = value
            self.named_one_hot_features_nclasses[key] = nclasses

    def add_misc_features(self, key, value):
        """
        Unchecked features for exotic ones
        """
        if "misc_features" not in self.keys:
            self.misc_features = {key: value}
        else:
            self.misc_features[key] = value

    def expand_one(self, res_feat):
        """
        going from residue features to atom ones
        """
        atom_feat = res_feat[self.res_map]
        return atom_feat

    def build_expanded_features(self, feature_keys='all', oh_keys='all'):
        """
        The goal of this function is to return a simple torch matrix to be input in the model
        :param feature_keys: list of features to include in the feature matrix
        :param oh_keys:  list of oh features to include in the feature matrix
        :return:
        """
        all_features = []
        if "flat_features" in self.keys:
            all_features.append(self.flat_features)
        if "named_features" in self.keys:
            if feature_keys == 'all':
                feature_keys = list(self.named_features.keys())
            for feature_key in feature_keys:
                named_feature = self.named_features[feature_key]
                if len(named_feature) != self.num_nodes:
                    named_feature = self.expand_one(named_feature)
                all_features.append(named_feature)
        if "named_one_hot_features" in self.keys:
            if oh_keys == 'all':
                oh_keys = list(self.named_one_hot_features.keys())
            for feature_key in oh_keys:
                type_feature = self.named_one_hot_features[feature_key]
                n_classes = self.named_one_hot_features_nclasses[feature_key]
                encoded_feat = torch.eye(n_classes)[type_feature.long()]
                if len(encoded_feat) != self.num_nodes:
                    encoded_feat = self.expand_one(encoded_feat)
                all_features.append(encoded_feat)
        all_features = [tensor[:, None] if len(tensor.shape) == 1 else tensor for tensor in all_features]
        if len(all_features)>0:
            all_features = torch.hstack(all_features)
        else:all_features=torch.ones(self.num_nodes, 1)
        return all_features

    @staticmethod
    def load(save_path):
        return torch.load(save_path, map_location=torch.device('cpu'))

    def save(self, save_path):
        torch.save(self, save_path)


if __name__ == "__main__":
    pass

    pdb_path = "../../data/example_files/4kt3.pdb"
    feats = Features(num_nodes=9, res_map=[0, 0, 0, 1, 1, 1, 2, 2, 2])
    test_1 = np.random.rand(9, 3)
    test_2 = torch.randn((9, 5))
    test_3 = torch.randn((9, 1))
    test_4 = torch.randn((9, 5))
    feats.add_named_features('test1', test_1)
    feats.add_named_features('test2', test_2)
    feats.add_named_oh_features('test3', test_3, 3)
    # feats.add_named_oh_features('test4', test_4, 4)
    a = 1
