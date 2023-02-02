"""
Training functions
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import datetime
import numpy as np
import time
import torch
import os.path as path

from utils.metric_utils import AverageMeter
from utils.training_utils import adjust_learning_rate, betas_loss, save_checkpoint, joints_3d_loss, joints_2d_loss, pose_loss, vertices_loss#, NormalVectorLoss
from utils.geometric_utils import orthographic_projection
from utils.render import visualize_mesh
from configurations import mano_config as cfg

def train(args, train_dataloader, Mictren_model, mano_model, renderer, mesh_sampler):

    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs

    optimizer = torch.optim.Adam(params=list(Mictren_model.parameters()),
                                           lr=args.learning_rate,
                                           betas=(0.9, 0.999),
                                           weight_decay=0)
    scaler = torch.cuda.amp.GradScaler()
    
    # Define loss function (criterion) and optimizer
    criterion_2d_keypoints = torch.nn.MSELoss(reduction='none').to(args.device)
    criterion_joints = torch.nn.MSELoss(reduction='none').to(args.device)
    criterion_vertices = torch.nn.L1Loss().to(args.device)
    criterion_pose = torch.nn.L1Loss().to(args.device)
    criterion_betas = torch.nn.L1Loss().to(args.device)
    #normal_vector_loss = NormalVectorLoss(mano_model.faces)

    start_training_time = time.time()
    end = time.time()
    
    Mictren_model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    log_loss_pose = AverageMeter()
    log_loss_betas = AverageMeter()
    log_loss_2djoints = AverageMeter()
    log_loss_3djoints = AverageMeter()
    log_loss_vertices = AverageMeter()
    #log_loss_mesh = AverageMeter()

    istry = True

    for iteration, (img_keys, images, annotations) in enumerate(train_dataloader):
        
        iteration += 1
        
        epoch = iteration // iters_per_epoch
        batch_size = images.size(0)
        adjust_learning_rate(optimizer, epoch, args)
        data_time.update(time.time() - end)
        
        if iteration == 1:
            print("Iteration reached: ", iteration)
            print("Max iter: ", max_iter)
            print("Iterations per epoch: ", iters_per_epoch)
            print("Epoch: ", epoch)
        
        images = images.cuda()

        gt_pose = annotations['pose'].cuda()
        gt_betas = annotations['betas'].cuda()
        gt_2d_joints = annotations['joints_2d'].cuda()

        has_mesh = annotations['has_smpl'].cuda()
        has_3d_joints = has_mesh

        mvm_mask = annotations['mvm_mask'].cuda()
        mjm_mask = annotations['mjm_mask'].cuda()

        # Generate mesh from pose and betas
        gt_vertices, gt_3d_joints = mano_model.layer(gt_pose, gt_betas)
        
        gt_vertices = gt_vertices/1000.0
        gt_3d_joints = gt_3d_joints/1000.0

        gt_vertices_sub = mesh_sampler.downsample(gt_vertices)

        # Normalize ground truth based on hand's root 
        gt_3d_root = gt_3d_joints[:, cfg.ROOT_INDEX, :]
        gt_vertices = gt_vertices - gt_3d_root[:, None, :]
        gt_vertices_sub = gt_vertices_sub - gt_3d_root[:, None, :]
        gt_3d_joints = gt_3d_joints - gt_3d_root[:, None, :]
        gt_3d_joints_with_tag = torch.ones((batch_size, gt_3d_joints.shape[1], 4)).cuda()
        gt_3d_joints_with_tag[:, :, :3] = gt_3d_joints

        # Prepare masks for 3d joints/vertices modeling
        mjm_mask_ = mjm_mask.expand(-1, -1, int(args.input_feat_dim.split(",")[0]))
        mvm_mask_ = mvm_mask.expand(-1, -1, int(args.input_feat_dim.split(",")[0]))
        meta_masks = torch.cat([mjm_mask_, mvm_mask_], dim=1)
        
        # Forward-pass
        pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, pred_pose, pred_betas = Mictren_model(images, mano_model, mesh_sampler, 
                                                                                                               meta_masks=meta_masks, is_train=True)
        
        # Regress 3d joints from the mesh
        pred_3d_joints_from_mesh = mano_model.get_3d_joints_from_mesh(pred_vertices)

        # Obtain 2d joints from 3d MANO ones
        pred_2d_joints = orthographic_projection(pred_3d_joints.contiguous(), pred_camera.contiguous())
        pred_2d_joints_from_mesh = orthographic_projection(pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous())

        # Compute 3d joints loss 
        loss_3d_joints = joints_3d_loss(criterion_joints, pred_3d_joints, gt_3d_joints_with_tag, has_3d_joints) + \
                         joints_3d_loss(criterion_joints, pred_3d_joints_from_mesh, gt_3d_joints_with_tag)

        # Compute 3d vertices loss
        loss_vertices = ( args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub, has_mesh) + \
                          args.vloss_w_full * vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_mesh) )

        # Compute normal vector loss for mesh
        #loss_mesh = args.vloss_w_full * normal_vector_loss(pred_vertices, gt_vertices) 

        # Compute pose and betas losses
        loss_pose = pose_loss(criterion_pose, pred_pose, gt_pose)
        loss_betas = betas_loss(criterion_betas, pred_betas, gt_betas)

        # Compute 2d joints loss
        loss_2d_joints = joints_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints) + \
                         joints_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_mesh, gt_2d_joints)

        loss = args.joints_loss_weight * loss_3d_joints + \
                args.vertices_loss_weight * loss_vertices + \
                args.vertices_loss_weight * loss_2d_joints + \
                args.pose_loss_weight * loss_pose + \
                args.betas_loss_weight * loss_betas
                #args.vertices_loss_weight * loss_mesh + \
                

        # Update logs
        log_loss_pose.update(loss_pose.item(), batch_size)
        log_loss_betas.update(loss_betas.item(), batch_size)
        log_loss_3djoints.update(loss_3d_joints.item(), batch_size)
        log_loss_2djoints.update(loss_2d_joints.item(), batch_size)
        log_loss_vertices.update(loss_vertices.item(), batch_size)
        #log_loss_mesh.update(loss_mesh.item(), batch_size)
        log_losses.update(loss.item(), batch_size)
        
        # Backward-pass
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)

        # Updates the scale for next iteration.
        #scaler.update()
        
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration==1:
            print("Pred camera shape:", pred_camera.shape)
            print("Pred 3d joints shape:", pred_3d_joints.shape)
            print("Pred vertices sub shape:", pred_vertices_sub.shape)
            print("Pred vertices shape:", pred_vertices.shape)
            print("Pred pose shape:", pred_pose.shape)
            print("Pred betas shape:", pred_betas.shape)

        if iteration % args.logging_steps == 0 or iteration == max_iter:
            eta_seconds = batch_time.avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            print("RUN ",
                f"eta: {eta_string}", 
                f"epoch: {epoch}", 
                f"iter: {iteration}", 
                f"max mem : {(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0):.0f}",
                f"loss: {log_losses.avg:.4f}",
                f"vertices loss: {log_loss_vertices.avg:.4f}",
                #f"mesh loss: {log_loss_mesh.avg:.4f}", 
                f"3d joint loss: {log_loss_3djoints.avg:.4f}", 
                f"2d joint loss: {log_loss_2djoints.avg:.4f}", 
                f"compute time avg: {batch_time.avg:.4f}", 
                f"data time avg: {data_time.avg:.4f}", 
                f"lr: {optimizer.param_groups[0]['lr']:.6f}"
            )
                
        # Save a checkpoint and visualize partial results obtained
        if iteration % iters_per_epoch == 0 or istry == True:
            if epoch%5 == 0 or istry == True:
                istry = False

                save_checkpoint(Mictren_model, args, epoch, iteration, optimizer, scaler)

                visual_imgs = visualize_mesh(renderer,
                                            annotations['ori_img'].detach(),
                                            annotations['joints_2d'].detach(),
                                            pred_vertices.detach(), 
                                            pred_camera.detach(),
                                            pred_2d_joints_from_mesh.detach())
                visual_imgs = torch.einsum("abc -> bca", visual_imgs)
                visual_imgs = np.asarray(visual_imgs)

                fname = path.join(args.output_dir, f"visual_{epoch}_{iteration}.jpg")
                # Invert color channels
                cv2.imwrite(fname, np.asarray(visual_imgs[:, :, ::-1]))

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=int(total_training_time)))
    print("RUN", f"Total training time: {total_time_str} ({(total_training_time / max_iter):.4f} s / iter)")
    
    save_checkpoint(Mictren_model, args, epoch, iteration)
