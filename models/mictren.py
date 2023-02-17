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
    def __init__(self, args, config, backbone, trans_blocks):
        super(MICTREN, self).__init__()
        self.args = args

        self.config = config
        self.backbone = backbone
        self.trans_blocks = trans_blocks
        
        # Transformers blocks
        self.trans_block1 = trans_blocks.pop(0)
        self.trans_block2 = torch.nn.Sequential(*trans_blocks)

        # Upsampling: 70 -> 216
        self.upsampling_block1 = nn.Linear(cfg.VERT_SUB_NUM_2 + cfg.JOIN_NUM, cfg.VERT_SUB_NUM_1 + cfg.JOIN_NUM)

            

        # Final Upsampling: 195 -> 389 -> 778
        self.final_upsampling = nn.Sequential(
            nn.Linear(cfg.VERT_SUB_NUM_1, cfg.VERT_NUM//2),
            nn.Linear(cfg.VERT_NUM//2, cfg.VERT_NUM)        
        )

        # Predict camera parameters, pose and
        # shape parameters from predicted vertices
        self.n_cam_params = 3
        self.n_pose_params = 48
        self.n_shape_params = 10
        self.n_params = self.n_cam_params + self.n_pose_params + self.n_shape_params
        
        # Get output features to recover parameters
        self.parameters_fc1 = nn.Linear(3, 1)
        self.parameters = nn.Sequential(
            nn.Linear(cfg.JOIN_NUM + cfg.VERT_SUB_NUM_1, 100), # 216 -> 100
            nn.Linear(100, self.n_params)                      # 100 -> 61  
        )
        

    def forward(self, images, mano_model, mesh_sampler, meta_masks=None, is_train=False, iter=None):
        batch_size = images.size(0)
        
        # Generate a template for pose and betas. This will pass through the
        # network and will be modified
        template_pose = torch.zeros((1, self.n_pose_params)).to(self.args.device)
        template_betas = torch.zeros((1, self.n_shape_params)).to(self.args.device)

        template_vertices, template_3d_joints = mano_model.layer(template_pose, template_betas)
        template_vertices = template_vertices/1000.0
        template_3d_joints = template_3d_joints/1000.0

        # Apply a double downsampling
        template_vertices_sub_depth_2 = mesh_sampler.downsample(template_vertices, n2=2)
        #template_vertices_sub_depth_1 = mesh_sampler.downsample(template_vertices, n2=1)
        
        # Normalize results
        template_root = template_3d_joints[:, cfg.ROOT_INDEX, :]
        template_3d_joints = template_3d_joints - template_root[:, None, :]
        template_vertices = template_vertices - template_root[:, None, :]
        template_vertices_sub_depth_2 = template_vertices_sub_depth_2 - template_root[:, None, :]
        #template_vertices_sub_depth_1 = template_vertices_sub_depth_1 - template_root[:, None, :]
        
        num_joints = template_3d_joints.shape[1]

        # Concatenate templates and then duplicate to batch size
        # Reference parameters represents the output that I want transformers from transformers
        ref_params = torch.cat([template_3d_joints, template_vertices_sub_depth_2], dim=1) # shape [1, 21+49, 3]
        ref_params = ref_params.expand(batch_size, -1, -1) # shape [bs, 70, 3]

        # Extract local image features using a CNN backbone
        image_feat_intermediate, image_feat_out = self.backbone(images) # size list[[1, 240],[1, 576],[1, 1024]]

        # Concatenate final image features with coarser mesh template
        image_feat_block1 = image_feat_out.view(batch_size, 1, image_feat_out.shape[1]).expand(-1, ref_params.shape[1], -1) # shape [bs, 70, 1024]
        features_block1 = torch.cat([ref_params, image_feat_block1], dim=2) # shape [bs, 70, 1027]

        if(iter != None and iter == 1):
            print("MICTREN", f"Feature block 1 shape: {features_block1.shape}")   

        # Forward-pass first block
        features_block2 = self.trans_block1(features_block1) # shape [bs, 70, 256]
        # Upsampling 70 -> 216
        features_block2 = self.upsampling_block1(features_block2.transpose(1, 2)) # shape [bs, 216, 256]
        # Concatenate the rest of the image features to the input of the second block
        image_feat_block2 = image_feat_intermediate.view(batch_size, 1, image_feat_intermediate.shape[1]).expand(-1, features_block2.shape[1], -1) # shape [bs, 216, 256] 
        features_block2 = torch.cat([features_block2.transpose(1, 2), image_feat_block2], dim=2) # shape [bs, 216, 1072]
        
        if(iter != None and iter == 1):
            print("MICTREN", f"Feature block 2 shape: {features_block2.shape}")

        if is_train==True:
            # Apply mask vertex/joint modeling
            # meta_masks is a tensor containing all masks, randomly generated in dataloader
            # constant_tensor is a [MASK] token, which is a floating-value vector with 0.01s
            constant_tensor = torch.ones_like(features).cuda()*0.01
            features = features_block2*meta_masks + constant_tensor*(1 - meta_masks)

        # Forward-pass remaining blocks
        features_output = self.trans_block2(features) # shape [bs, 216, 3]

        # Get predicted vertices
        pred_3d_joints = features_output[:, :num_joints, :] 
        pred_vertices_sub = features_output[:, num_joints:, :]

        predictions = torch.cat([pred_vertices_sub, pred_3d_joints], dim=1) # shape [bs, 216, 3]
        
        # Learn camera/pose/shape parameters from predicted vertices and joints
        pred_params = self.parameters_fc1(predictions) # shape [bs, 216, 1]
        pred_params = self.parameters_fc2(pred_params.transpose(1, 2)) # shape [bs, 1, 61]
        pred_params = pred_params.transpose(1, 2).squeeze(-1) # shape [bs, 61]
        
        if(iter != None and iter == 1):
            print("MICTREN", f"Pred_params = {pred_params.shape}")
        
        pred_cam_params = pred_params[:, :self.n_cam_params]
        pred_pose_params = pred_params[:, self.n_cam_params:self.n_cam_params + self.n_pose_params]
        pred_shape_params = pred_params[:, self.n_cam_params + self.n_pose_params:]
        
        if(iter != None and iter == 1):    
            print(f"Pred_cam_params = {pred_cam_params.shape}" )
            print(f"Pred_pose_params = {pred_pose_params.shape}" )
            print(f"Pred_shape_params = {pred_shape_params.shape}")
        
        # Upsampling
        # [bs, 195, 3] -> [bs, 389, 3] -> [bs, 778, 3]
        pred_vertices = self.final_upsampling(pred_vertices_sub.transpose(1, 2))
        pred_vertices = pred_vertices.transpose(1, 2)

        return (pred_cam_params, pred_3d_joints, pred_vertices_sub, 
                    pred_vertices, pred_pose_params, pred_shape_params)
            
