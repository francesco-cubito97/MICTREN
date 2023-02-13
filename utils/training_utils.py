"""
Training useful functions
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path as path
import torch
from torch import nn
import torch.nn.functional as F

from utils.file_utils import create_dir

#---------------------------------USEFUL FUNCTIONS---------------------------------
def save_checkpoint(model, args, epoch, iteration, optimizer, scaler, num_trial=10):
    if args.checkpoint_dir is not None:
        checkpoint_dir = path.join(args.checkpoint_dir, 'checkpoint-{}-{}'.format(
            epoch, iteration))
    else:
        checkpoint_dir = path.join(args.output_dir, 'checkpoint-{}-{}'.format(
            epoch, iteration))

    create_dir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save.state_dict(), path.join(checkpoint_dir, 'model_state_dict.pth'))
            print("SAVE_CHECKPOINT", "Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass

    if i>=num_trial:
        print("SAVE_CHECKPOINT", f"Failed to save checkpoint after {num_trial} trails.")
    
    return checkpoint_dir

# After half epochs decrease learning rate
def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.learning_rate * (0.1 ** (epoch // (args.num_train_epochs/2.0)  ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
#---------------------------------LOSSES FUNCTIONS---------------------------------
def joints_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence is binary and indicates whether the keypoints exist or not.
    """
    
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss

def joints_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d=True):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    
    if len(gt_keypoints_3d) > 0:
        gt_root = gt_keypoints_3d[:, 0, :]
        gt_keypoints_3d = gt_keypoints_3d - gt_root[:, None, :]
        pred_root = pred_keypoints_3d[:, 0, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_root[:, None, :]
        return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).cuda()

def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl=True):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    
    if len(gt_vertices_with_shape) > 0:
        return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).cuda()

# class NormalVectorLoss(nn.Module):
#     def __init__(self, face):
#         super(NormalVectorLoss, self).__init__()
#         self.face = face

#     def forward(self, coord_out, coord_gt):
#         face = torch.LongTensor(self.face).cuda()

#         v1_out = coord_out[:, face[:, 1], :] - coord_out[:, face[:, 0], :]
#         v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
#         v2_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 0], :]
#         v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
#         v3_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 1], :]
#         v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

#         v1_gt = coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 0], :]
#         v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
#         v2_gt = coord_gt[:, face[:, 2], :] - coord_gt[:, face[:, 0], :]
#         v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
#         normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
#         normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

#         cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))
#         cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))
#         cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))
#         loss = torch.cat((cos1, cos2, cos3), 1)
#         return loss.mean()


# class EdgeLengthLoss(nn.Module):
#     def __init__(self, face):
#         super(EdgeLengthLoss, self).__init__()
#         self.face = face

#     def forward(self, coord_out, coord_gt):
#         face = torch.LongTensor(self.face).cuda()

#         d1_out = torch.sqrt(
#             torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :]) ** 2, 2, keepdim=True))
#         d2_out = torch.sqrt(
#             torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))
#         d3_out = torch.sqrt(
#             torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))

#         d1_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
#         d2_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
#         d3_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
 
#         diff1 = torch.abs(d1_out - d1_gt)
#         diff2 = torch.abs(d2_out - d2_gt)
#         diff3 = torch.abs(d3_out - d3_gt)
#         loss = torch.cat((diff1, diff2, diff3), 1)
#         return loss.mean()

def pose_loss(criterion_pose, pred_pose, gt_pose):
    """
    Compute pose parameters loss 
    """
    
    return criterion_pose(pred_pose, gt_pose)
    
    

def betas_loss(criterion_betas, pred_betas, gt_betas):
    """
    Compute betas parameters loss 
    """

    return criterion_betas(pred_betas, gt_betas)