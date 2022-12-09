"""
Training useful functions
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path as path
import torch

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
            torch.save(model_to_save, path.join(checkpoint_dir, 'model.bin'))
            torch.save(model_to_save.state_dict(), path.join(checkpoint_dir, 'state_dict.bin'))
            torch.save(args, path.join(checkpoint_dir, 'training_args.bin'))
            torch.save(optimizer.state_dict(), path.join(checkpoint_dir, 'opt_state_dict.bin'))
            torch.save(scaler.state_dict(), path.join(checkpoint_dir, 'scaler_state_dict.bin'))
            print("SAVE_CHECKPOINT", "Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass

    if i>=num_trial:
        print("SAVE_CHECKPOINT", "Failed to save checkpoint after {} trails.".format(num_trial))
    
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