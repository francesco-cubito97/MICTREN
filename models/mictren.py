"""
Mixed CNN-Transformer Reconstruction Network model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn

from configurations import mano_config as cfg

class MICTREN(nn.Module):
    """
    End-to-end network for hand pose and mesh reconstruction from a single image.
    """
    def __init__(self, args, config, backbone, trans_encoder):
        super(MICTREN, self).__init__()
        self.args = args

        self.config = config
        self.backbone = backbone
        self.trans_encoder = trans_encoder
        
        self.upsampling = nn.Sequential(
            nn.Linear(cfg.VERT_SUB_NUM, cfg.VERT_NUM//2),
            nn.Linear(cfg.VERT_NUM//2, cfg.VERT_NUM)        
        )

        # Predict camera parameters, pose and
        # shape parameters from predicted vertices
        self.n_cam_params = 3
        self.n_pose_params = 48
        self.n_shape_params = 10
        self.n_params = self.n_cam_params + self.n_pose_params + self.n_shape_params
        
        self.parameters_fc1 = nn.Linear(3, 1)
        self.parameters_fc2 = nn.Linear(cfg.JOIN_NUM + cfg.VERT_SUB_NUM, 100)
        self.parameters_fc3 = nn.Linear(100, self.n_params)
        

    def forward(self, images, mano_model, mesh_sampler, meta_masks=None, is_train=False):
        batch_size = images.size(0)
        
        # Generate a template for pose and betas. This will pass through the
        # network and will be modified
        template_pose = torch.zeros((1, self.n_pose_params)).to(self.args.device)
        template_betas = torch.zeros((1, self.n_shape_params)).to(self.args.device)

        template_vertices, template_3d_joints = mano_model.layer(template_pose, template_betas)
        template_vertices = template_vertices/1000.0
        template_3d_joints = template_3d_joints/1000.0
        template_vertices_sub = mesh_sampler.downsample(template_vertices)

        # Normalize results
        template_root = template_3d_joints[:, cfg.ROOT_INDEX, :]
        template_3d_joints = template_3d_joints - template_root[:, None, :]
        template_vertices = template_vertices - template_root[:, None, :]
        template_vertices_sub = template_vertices_sub - template_root[:, None, :]
        
        num_joints = template_3d_joints.shape[1]

        # Concatenate templates and then duplicate to batch size
        # Reference parameters represents the output that I want transformers from transformers
        ref_params = torch.cat([template_3d_joints, template_vertices_sub], dim=1) # shape [1, 195+21, 3]
        ref_params = ref_params.expand(batch_size, -1, -1) # shape [bs, 216, 3]

        # Extract local image features using a CNN backbone
        image_feat = self.backbone(images) # size [bs, 96 + 240 + 576 + 576] = [bs, 1488]

        # Concatenate image features together with template parameters
        image_feat = image_feat.view(batch_size, 1, image_feat.shape[1]).expand(-1, ref_params.shape[1], -1) # shape [bs, 216, 1488]
        features = torch.cat([ref_params, image_feat], dim=2) # shape [bs, 216, 1488+3] 

        if is_train==True:
            # Apply mask vertex/joint modeling
            # meta_masks is a tensor containing all masks, randomly generated in dataloader
            # constant_tensor is a [MASK] token, which is a floating-value vector with 0.01s
            constant_tensor = torch.ones_like(features).cuda()*0.01
            features = features*meta_masks + constant_tensor*(1 - meta_masks)     

        # Forward-pass
        features = self.trans_encoder(features)

        # Get predicted vertices
        pred_3d_joints = features[:, :num_joints, :] 
        pred_vertices_sub = features[:, num_joints:, :]

        predictions = torch.cat([pred_vertices_sub, pred_3d_joints], dim=1) # shape [bs, 216, 3]
        
        # Learn camera/pose/shape parameters from predicted vertices and joints
        pred_params = self.parameters_fc1(predictions)
        pred_params = self.parameters_fc2(pred_params.transpose(1, 2))
        pred_params = self.parameters_fc3(pred_params)
        pred_params = pred_params.transpose(1, 2).squeeze(-1)

        #print("Pred_params= ", pred_params.shape)
        
        pred_cam_params = pred_params[: , :self.n_cam_params]
        pred_pose_params = pred_params[:, self.n_cam_params:self.n_cam_params + self.n_pose_params]
        pred_shape_params = pred_params[:, self.n_cam_params + self.n_pose_params:]
        
        #print("Pred_cam_params = ", pred_cam_params.shape)
        #print("Pred_pose_params = ", pred_pose_params.shape)
        #print("Pred_shape_params = ", pred_shape_params.shape)
        
        # Upsampling
        # [bs, 195, 3] -> [bs, 389, 3] -> [bs, 778, 3]
        pred_vertices = self.upsampling(pred_vertices_sub.transpose(1, 2))
        pred_vertices = pred_vertices.transpose(1, 2)

        return (pred_cam_params, pred_3d_joints, pred_vertices_sub, 
                    pred_vertices, pred_pose_params, pred_shape_params)
            
