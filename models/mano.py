"""
Mano network used to simplify calculation of mesh for 
datasets without mesh but only with pose and shape parameters.

The code is adapted from the METRO one https://github.com/microsoft/MeshTransformer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn
import numpy as np
from manopth.manolayer import ManoLayer

from configurations import mano_config as cfg

class Mano(nn.Module):
    def __init__(self, args):
        super(Mano, self).__init__()

        self.layer = self.get_layer()
        self.vertex_num = cfg.VERT_NUM
        self.faces = self.layer.th_faces.numpy()
        self.joint_regressor = self.layer.th_J_regressor.numpy()
        
        self.joint_num = cfg.JOIN_NUM
        self.joints_name = cfg.J_NAME
        # Define the connections between joints
        self.skeleton = cfg.SKELETON_DEF
        self.root_joint_idx = cfg.ROOT_INDEX

        # Add fingertips to joint_regressor
        self.fingertip_idx = cfg.FINGERTIPS_RIGHT # fingertips mesh vertices idx (right hand)
        thumbtip_onehot = np.array([1 if i == self.fingertip_idx[0]  else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        indextip_onehot = np.array([1 if i == self.fingertip_idx[1] else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        middletip_onehot = np.array([1 if i == self.fingertip_idx[2] else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        ringtip_onehot = np.array([1 if i == self.fingertip_idx[3] else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        pinkytip_onehot = np.array([1 if i == self.fingertip_idx[4] else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        
        self.joint_regressor = np.concatenate((self.joint_regressor, thumbtip_onehot, indextip_onehot, middletip_onehot, ringtip_onehot, pinkytip_onehot))
        self.joint_regressor = self.joint_regressor[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],:]
        
        joint_regressor_torch = torch.from_numpy(self.joint_regressor).float()
        self.register_buffer('joint_regressor_torch', joint_regressor_torch)

    def get_layer(self):
        return ManoLayer(mano_root=cfg.data_path, # load right hand MANO model
                         flat_hand_mean=False, 
                         use_pca=False) 

         

    def get_3d_joint_from_mesh(self, vertices):
        """
        This method is used to get the joint locations from the mesh
        Input:
            vertices: size = (B, 778, 3)
        Output:
            3D joints: size = (B, 21, 3)
        """
        joints = torch.einsum('bik,ji -> bjk', [vertices, self.joint_regressor_torch])
        return joints
