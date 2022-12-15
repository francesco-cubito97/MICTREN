"""
Evaluation functions

Parts of code taken from https://github.com/microsoft/MeshTransformer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import json
import numpy as np
import os
import os.path as path
import torch



from configurations import mano_config as cfg
from utils.geometric_utils import orthographic_projection
from utils.render import visualize_mesh
from utils.training_utils import save_checkpoint


def run_inference_hand_mesh(args, val_loader, Mictren_model, mano_model, mesh_sampler, renderer):
    Mictren_model.eval()
    
    fname_output_save = []
    mesh_output_save = []
    joint_output_save = []
    
    with torch.no_grad():
        for idx, (img_keys, images, annotations) in enumerate(val_loader):
            batch_size = images.size(0)
            
            images = images.cuda()

            # Make a forward-pass to inference
            pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, pred_pose, pred_betas = Mictren_model(images, mano_model, mesh_sampler)

            # Take 3d joints from full mesh
            pred_3d_palm = pred_3d_joints[:, cfg.ROOT_INDEX, :]
            pred_3d_joints_from_mesh = pred_3d_joints - pred_3d_palm[:, None, :]
            pred_vertices = pred_vertices - pred_3d_palm[:, None, :]

            for batch in range(batch_size):
                fname_output_save.append(img_keys[batch])
                
                mesh_output_save.append(pred_vertices[batch].tolist())

                joint_output_save.append(pred_3d_joints[batch].tolist())

            if idx%20==0:
                # Obtain 3d joints, which are regressed from the full mesh
                pred_3d_joints_from_mesh = mano_model.get_3d_joints_from_mesh(pred_vertices)
                # Get 2d joints from orthographic projection of 3d ones taken from mesh
                pred_2d_joints_from_mesh = orthographic_projection(pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous())
                
                # Transform mesh into image
                visual_imgs = visualize_mesh(renderer,
                                             annotations['ori_img'].detach(),
                                             annotations['joints_2d'].detach(),
                                             pred_vertices.detach(), 
                                             pred_camera.detach(),
                                             pred_2d_joints_from_mesh.detach())

                visual_imgs = torch.einsum(visual_imgs, "abc -> bca")
                visual_imgs = np.asarray(visual_imgs)
                
                inference_setting = f"scale{int(args.sc*10):02d}_rot{str(int(args.rot)):s}"
                temp_fname = path.join(args.output_dir, args.saved_checkpoint[0:-9] + "freihand_results_"+inference_setting+"_batch"+str(idx)+".jpg")
                cv2.imwrite(temp_fname, np.asarray(visual_imgs[:, :, ::-1]*255))

    # Saving predictions into a zip file
    print("RUN_INFERENCE", "---------Saving results to 'pred.json'---------")
    
    with open("pred.json", "w") as f:
        json.dump([joint_output_save, mesh_output_save], f)

    run_exp_name = args.saved_checkpoint.split('/')[-3]
    run_ckpt_name = args.saved_checkpoint.split('/')[-2].split('-')[1]
    inference_setting = f"scale{int(args.sc*10):02d}_rot{str(int(args.rot)):s}"
    resolved_submit_cmd = "zip " + args.output_dir + "/" + run_exp_name + "-ckpt" + run_ckpt_name + "-" + inference_setting + "-pred.zip " +  "pred.json"
    
    print("RUN_INFERENCE", f"---------Executing: {resolved_submit_cmd}---------")
    os.system(resolved_submit_cmd)
    
    resolved_submit_cmd = "rm pred.json"
    print("RUN_INFERENCE", f"Executing: {resolved_submit_cmd}")
    os.system(resolved_submit_cmd)
    
    return

def run_eval_and_save(args, val_dataloader, Mictren_model, mano_model, renderer, mesh_sampler):

    run_inference_hand_mesh(args, val_dataloader, Mictren_model, 
                         mano_model, mesh_sampler, renderer)
    
    save_checkpoint(Mictren_model, args, 0, 0)
    
    return
